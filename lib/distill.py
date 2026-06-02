"""
distill.py - Shared KD / OPD training loop for the student model.

KD  (off-policy): train on prefixes sampled from the frozen teacher.
OPD (on-policy):  train on prefixes sampled from the current student.
Both optimize a token-level KL between teacher and student distributions.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.lang import TOKEN2ID
from lib.metrics import compute_metrics, get_flat_params, save_landscape
from lib.model import build_model
from pretrain import get_lr, make_eval_batch

EOS_ID = TOKEN2ID['[EOS]']


def tokenize_prompt(prompt: str) -> List[int]:
    """Convert a space-separated prompt string into token IDs."""
    return [TOKEN2ID[t] for t in prompt.split() if t in TOKEN2ID]


class PromptBuffer:
    """Prompt sampler grouped by prompt length for direct tensor stacking."""
    def __init__(
        self,
        jsonl_path: str,
        context_len: int,
        min_depth: int = 0,
        max_depth: Optional[int] = None,
    ):
        groups: Dict[int, List[Dict]] = defaultdict(list)

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if '?' in rec.get('prompt', ''):
                    continue
                if 'expr' not in rec or 'result' not in rec:
                    continue
                depth = int(rec.get('depth', 0))
                if depth < min_depth:
                    continue
                if max_depth is not None and depth > max_depth:
                    continue
                prompt_ids = tokenize_prompt(rec['prompt'])
                if 0 < len(prompt_ids) < context_len:
                    groups[len(prompt_ids)].append({
                        'prompt_ids': prompt_ids,
                        'expr': rec.get('expr'),
                        'result': rec.get('result'),
                        'depth': depth,
                    })

        if not groups:
            raise ValueError(f"No distillation prompts found in {jsonl_path}")

        self.groups = dict(groups)
        self.lengths = sorted(self.groups)
        sizes = np.array([len(self.groups[L]) for L in self.lengths], dtype=np.float64)
        self.probs = sizes / sizes.sum()
        self.N = int(sizes.sum())

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Sample one same-length prompt batch.

        Returns:
            prompt_ids: [B=batch_size, Lp] int64
            records:    length B
        """
        Lp = int(np.random.choice(self.lengths, p=self.probs))
        group = self.groups[Lp]
        idx = np.random.randint(0, len(group), size=batch_size)
        records = [group[i] for i in idx]
        prompt_ids = torch.tensor(
            [r['prompt_ids'] for r in records],
            dtype=torch.long,
            device=device,
        )
        return prompt_ids, records


def _infer_cfg_from_state_dict(state: Dict[str, torch.Tensor], fallback_cfg: Dict) -> Dict:
    """Infer enough architecture fields from a checkpoint to rebuild the model."""
    d_model = int(state['tok_emb.weight'].shape[1])
    vocab_size = int(state['tok_emb.weight'].shape[0])
    context_len = int(state['pos_emb.weight'].shape[0])
    layer_ids = [
        int(m.group(1))
        for k in state
        for m in [re.match(r'blocks\.(\d+)\.', k)]
        if m is not None
    ]
    n_layers = max(layer_ids) + 1
    d_ffn = int(state['blocks.0.ffn.net.0.weight'].shape[0])
    d_inner = int(state['blocks.0.attn.proj.weight'].shape[1])

    # Prefer the documented project shapes: teacher uses d_head=32, student uses 16.
    for d_head in (32, 16, 8, 4, 1):
        if d_inner % d_head == 0:
            n_heads = d_inner // d_head
            break

    cfg = dict(fallback_cfg)
    cfg.update({
        'vocab_size': vocab_size,
        'context_len': context_len,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'd_head': d_head,
        'd_ffn': d_ffn,
    })
    return cfg


def resolve_teacher_model_cfg(cfg: Dict, teacher_path: str, teacher_state: Dict) -> Dict:
    """
    Resolve teacher architecture without duplicating it in student configs.

    Priority:
      1. train.teacher_config_path
      2. top-level teacher_model block
      3. config/<checkpoint_stem>.yaml or config/<parent_dir>.yaml
      4. checkpoint-shape inference
    """
    train_cfg = cfg['train']
    teacher_cfg_path = train_cfg.get('teacher_config_path')
    if teacher_cfg_path and Path(teacher_cfg_path).exists():
        with open(teacher_cfg_path) as f:
            return yaml.safe_load(f)['model']

    if 'teacher_model' in cfg:
        return cfg['teacher_model']

    candidates = [
        Path('config') / f"{Path(teacher_path).stem}.yaml",
        Path('config') / f"{Path(teacher_path).parent.name}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                loaded = yaml.safe_load(f)
            if 'model' in loaded:
                return loaded['model']

    return _infer_cfg_from_state_dict(teacher_state, fallback_cfg=cfg['model'])


@torch.no_grad()
def sample_rollout_prefixes(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Autoregressively sample continuation tokens from model.

    prompt_ids:  [B=64, Lp=18]
    full_ids:    [B=64, Lp+Tc=74]
    action_mask: [B=64, Tc=56], true through the sampled EOS token
    """
    model.eval()
    raw_model = getattr(model, '_orig_mod', model)
    B, Lp = prompt_ids.shape
    ids = prompt_ids
    done = torch.zeros(B, dtype=torch.bool, device=device)
    gen_cols: List[torch.Tensor] = []
    mask_cols: List[torch.Tensor] = []
    max_new = min(max_new_tokens, raw_model.pos_emb.num_embeddings - Lp)
    temp = max(float(temperature), 1e-6)

    for _ in range(max_new):
        active = ~done
        if not active.any():
            break
        with torch.amp.autocast(
            device_type='cuda' if device == 'cuda' else 'cpu',
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(ids)[:, -1, :]  # logits: [B=64, V=34]

        if temperature == 0:
            next_ids = logits.argmax(dim=-1)  # next_ids: [B=64]
        else:
            probs = F.softmax((logits / temp).float(), dim=-1)  # probs: [B=64, V=34]
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)

        next_ids = torch.where(active, next_ids, torch.full_like(next_ids, EOS_ID))
        gen_cols.append(next_ids)
        mask_cols.append(active)
        done = done | (next_ids == EOS_ID)
        ids = torch.cat([ids, next_ids[:, None]], dim=1)  # ids: [B=64, Lp+t+1]

    if not gen_cols:
        empty = torch.empty((B, 0), dtype=torch.bool, device=device)
        return ids, empty

    action_mask = torch.stack(mask_cols, dim=1)  # action_mask: [B=64, Tc]
    return ids, action_mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over positions where mask is true."""
    return (x * mask).sum() / mask.sum().clamp(min=1)


def distill_loss(
    student: nn.Module,
    teacher: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
    action_mask: torch.Tensor,
    kl_direction: str,
    temperature: float,
    alpha_ce: float,
    loss_on_prompt: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute token-level distillation loss on sampled prefixes.

    full_ids:    [B=64, Lp+Tc=74]
    input_ids:   [B=64, Lp+Tc-1=73]
    target_ids:  [B=64, Lp+Tc-1=73]
    kl_tokens:   [B=64, T_loss=56 or 73]
    """
    device = full_ids.device
    B = full_ids.shape[0]
    input_ids = full_ids[:, :-1]
    target_ids = full_ids[:, 1:]
    Tc = action_mask.shape[1]
    start = prompt_len - 1
    end = start + Tc
    temp = max(float(temperature), 1e-6)

    if Tc == 0:
        zero = student(input_ids).sum() * 0.0
        return zero, zero.detach()

    with torch.amp.autocast(
        device_type='cuda' if device.type == 'cuda' else 'cpu',
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        student_logits_all = student(input_ids)  # student_logits_all: [B=64, Lp+Tc-1, V=34]
        with torch.no_grad():
            teacher_logits_all = teacher(input_ids)  # teacher_logits_all: [B=64, Lp+Tc-1, V=34]

    if loss_on_prompt:
        logits_slice = slice(0, end)
        prompt_mask = torch.ones((B, start), dtype=torch.bool, device=device)
        loss_mask = torch.cat([prompt_mask, action_mask], dim=1)  # loss_mask: [B=64, start+Tc]
    else:
        logits_slice = slice(start, end)
        loss_mask = action_mask  # loss_mask: [B=64, Tc=56]

    student_logits = student_logits_all[:, logits_slice, :]
    teacher_logits = teacher_logits_all[:, logits_slice, :]
    labels = target_ids[:, logits_slice]

    s_logp = F.log_softmax((student_logits / temp).float(), dim=-1)
    t_logp = F.log_softmax((teacher_logits / temp).float(), dim=-1)

    if kl_direction == 'reverse':
        s_prob = s_logp.exp()
        kl_tokens = (s_prob * (s_logp - t_logp)).sum(dim=-1)
    else:
        t_prob = t_logp.exp()
        kl_tokens = (t_prob * (t_logp - s_logp)).sum(dim=-1)

    raw_kl = masked_mean(kl_tokens, loss_mask)
    loss = raw_kl * (temp ** 2)

    if alpha_ce > 0:
        ce_tokens = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
            reduction='none',
        ).view_as(loss_mask)
        ce_loss = masked_mean(ce_tokens, loss_mask)
        loss = (1.0 - alpha_ce) * loss + alpha_ce * ce_loss

    return loss, raw_kl.detach()


@torch.no_grad()
def estimate_prefix_kl(
    student: nn.Module,
    teacher: nn.Module,
    prompt_buf: PromptBuffer,
    prefix_owner: str,
    batch_size: int,
    max_new_tokens: int,
    sample_temperature: float,
    kl_direction: str,
    temperature: float,
    loss_on_prompt: bool,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Estimate teacher/student KL on either teacher-generated or student-generated prefixes."""
    prompt_ids, _ = prompt_buf.sample(batch_size, device)
    owner = teacher if prefix_owner == 'teacher' else student
    full_ids, action_mask = sample_rollout_prefixes(
        model=owner,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=sample_temperature,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    _, kl = distill_loss(
        student=student,
        teacher=teacher,
        full_ids=full_ids,
        prompt_len=prompt_ids.shape[1],
        action_mask=action_mask,
        kl_direction=kl_direction,
        temperature=temperature,
        alpha_ce=0.0,
        loss_on_prompt=loss_on_prompt,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    return round(float(kl.cpu()), 6)


def run_distillation(config_path: str, mode: str):
    if mode not in {'kd', 'opd'}:
        raise ValueError(f"mode must be 'kd' or 'opd', got {mode}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['train']
    out_cfg = cfg['output']
    log_cfg = cfg['logging']

    device = train_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, falling back to CPU")

    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    seed = int(train_cfg.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    context_len = model_cfg['context_len']
    batch_size = int(train_cfg['batch_size'])
    n_steps = int(train_cfg['n_steps'])
    warmup_steps = int(train_cfg['warmup_steps'])
    lr = float(train_cfg['lr'])
    lr_schedule = train_cfg.get('lr_schedule', 'cosine')
    weight_decay = float(train_cfg.get('weight_decay', 0.0))
    grad_clip = float(train_cfg.get('grad_clip', 1.0))
    kl_direction = train_cfg.get('kl_direction', 'forward')
    temperature = float(train_cfg.get('temperature', 1.0))
    alpha_ce = float(train_cfg.get('alpha_ce', 0.0))
    loss_on_prompt = bool(train_cfg.get('loss_on_prompt', False))
    rollout_temperature = float(train_cfg.get('rollout_temperature', temperature))
    max_new_tokens = int(cfg.get('inference', {}).get('max_new_tokens', context_len))
    log_every = int(log_cfg['log_every'])
    eval_every = int(log_cfg.get('eval_every', log_every))
    eval_bs = int(log_cfg['eval_batch_size'])
    kl_metric_bs = int(log_cfg.get('kl_metric_batch_size', batch_size))
    tmp_dir = log_cfg['tmp_ckpt_dir']
    pca_basis_path = log_cfg.get('pca_basis_path')
    log_dir = out_cfg['log_dir']
    model_path = out_cfg['model_path']

    amp_dtype_str = train_cfg.get('amp_dtype', 'bf16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bf16' else torch.float16
    use_amp = train_cfg.get('amp', True) and device == 'cuda'

    for d in [log_dir, tmp_dir, str(Path(model_path).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)

    base_model_path = train_cfg.get('base_model_path', '')
    teacher_model_path = train_cfg.get('teacher_model_path', '')
    if not base_model_path or not Path(base_model_path).exists():
        raise FileNotFoundError(f"base_model_path not found: {base_model_path}")
    if not teacher_model_path or not Path(teacher_model_path).exists():
        raise FileNotFoundError(f"teacher_model_path not found: {teacher_model_path}")

    rollout_min_depth = int(train_cfg.get('rollout_min_depth', 0))
    rollout_max_depth = train_cfg.get('rollout_max_depth')
    rollout_max_depth = int(rollout_max_depth) if rollout_max_depth is not None else None

    print(f"[setup] Loading prompts: {data_cfg['path']}")
    prompt_buf = PromptBuffer(
        data_cfg['path'],
        context_len,
        min_depth=rollout_min_depth,
        max_depth=rollout_max_depth,
    )
    print(f"[setup] Distillation prompts: {prompt_buf.N} records in {len(prompt_buf.lengths)} length groups")
    print(f"[setup] Distillation depth filter: min={rollout_min_depth} max={rollout_max_depth}")

    print(f"[setup] Building student from {base_model_path}")
    student = build_model(model_cfg).to(device)
    student.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))

    print(f"[setup] Building frozen teacher from {teacher_model_path}")
    teacher_state = torch.load(teacher_model_path, map_location=device, weights_only=True)
    teacher_cfg = resolve_teacher_model_cfg(cfg, teacher_model_path, teacher_state)
    teacher = build_model(teacher_cfg).to(device)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if train_cfg.get('compile', False) and hasattr(torch, 'compile'):
        print("[setup] torch.compile() wrapping student...")
        student = torch.compile(student, mode="reduce-overhead")

    eval_data_path = log_cfg.get('eval_data_path', '')
    eval_records: List[Dict] = []
    if Path(eval_data_path).exists():
        with open(eval_data_path) as f:
            eval_records = [json.loads(line) for line in f if line.strip()]
        print(f"[setup] Eval data: {len(eval_records)} records ({eval_data_path})")
    else:
        with open(data_cfg['path']) as f:
            for line in f:
                if line.strip():
                    eval_records.append(json.loads(line))
                if len(eval_records) >= eval_bs:
                    break
        print(f"[setup] Warning: eval file not found, borrowing {len(eval_records)} records from train set")

    eval_batch = make_eval_batch(eval_records, context_len, eval_bs, device, mode='sft')
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    ckpt_interval = max(1, n_steps // max(1, int(log_cfg.get('n_traj_ckpt', 30))))
    pca_d1: Optional[np.ndarray] = None
    pca_d2: Optional[np.ndarray] = None
    traj_proj1: List[float] = []
    traj_proj2: List[float] = []
    raw_student = getattr(student, '_orig_mod', student)
    if pca_basis_path and Path(pca_basis_path).exists():
        with np.load(pca_basis_path) as npz:
            pca_d1 = npz['d1']
            pca_d2 = npz['d2']
        flat_init = get_flat_params(raw_student)
        traj_proj1.append(float(flat_init @ pca_d1))
        traj_proj2.append(float(flat_init @ pca_d2))
        print("[setup] Initial distillation point recorded in shared PCA coordinates")

    metrics_path = Path(log_dir) / "metrics.jsonl"
    metrics_file = open(metrics_path, 'w')
    prev_flat: Optional[np.ndarray] = None
    grad_norm_t: Optional[torch.Tensor] = None
    t_train_s = 0.0
    t_start = time.time()
    prefix_owner = 'teacher' if mode == 'kd' else 'student'
    stage_name = 'KD' if mode == 'kd' else 'OPD'

    print(f"\n{'=' * 60}")
    print(f"{stage_name}  n_steps={n_steps}  batch={batch_size}  prefix_owner={prefix_owner}  device={device}")
    print(f"  data={prompt_buf.N}  lr={lr:.2e}  kl_direction={kl_direction}  T={temperature:.2f}")
    print(f"  loss_on_prompt={loss_on_prompt}  alpha_ce={alpha_ce:.2f}  AMP={use_amp}")
    print(f"  log_every={log_every}  eval_every={eval_every}")
    print(f"{'=' * 60}\n")

    for step in range(1, n_steps + 1):
        student.train()
        cur_lr = get_lr(step, n_steps, lr, warmup_steps, lr_schedule)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        t_step = time.time()
        prompt_ids, _ = prompt_buf.sample(batch_size, device)  # prompt_ids: [B=64, Lp]
        rollout_model = teacher if prefix_owner == 'teacher' else student
        sample_temp = temperature if prefix_owner == 'teacher' else rollout_temperature
        full_ids, action_mask = sample_rollout_prefixes(
            model=rollout_model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=sample_temp,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        # full_ids: [B=64, Lp+Tc], action_mask: [B=64, Tc]
        student.train()

        loss, _ = distill_loss(
            student=student,
            teacher=teacher,
            full_ids=full_ids,
            prompt_len=prompt_ids.shape[1],
            action_mask=action_mask,
            kl_direction=kl_direction,
            temperature=temperature,
            alpha_ce=alpha_ce,
            loss_on_prompt=loss_on_prompt,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_t = nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        t_train_s += time.time() - t_step

        if step == 1:
            print(f"[timing] First step: {t_train_s * 1000:.0f}ms")

        raw_student = getattr(student, '_orig_mod', student)
        if (step % ckpt_interval == 0 or step == 1) and pca_d1 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            flat = get_flat_params(raw_student)
            traj_proj1.append(float(flat @ pca_d1))
            traj_proj2.append(float(flat @ pca_d2))

        if step % log_every == 0 or step == n_steps:
            train_loss_val = float(loss.detach().cpu())
            grad_norm_val = float(grad_norm_t.detach().cpu()) if grad_norm_t is not None else 0.0
            tok_per_step = batch_size * max(1, action_mask.shape[1])
            tps = tok_per_step * log_every / max(t_train_s, 1e-6)

            raw_student = getattr(student, '_orig_mod', student)
            kl_teacher_prefix = estimate_prefix_kl(
                student=raw_student,
                teacher=teacher,
                prompt_buf=prompt_buf,
                prefix_owner='teacher',
                batch_size=kl_metric_bs,
                max_new_tokens=max_new_tokens,
                sample_temperature=temperature,
                kl_direction='forward',
                temperature=temperature,
                loss_on_prompt=loss_on_prompt,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            kl_student_prefix = estimate_prefix_kl(
                student=raw_student,
                teacher=teacher,
                prompt_buf=prompt_buf,
                prefix_owner='student',
                batch_size=kl_metric_bs,
                max_new_tokens=max_new_tokens,
                sample_temperature=rollout_temperature,
                kl_direction='forward',
                temperature=temperature,
                loss_on_prompt=loss_on_prompt,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            do_eval = (step % eval_every == 0) or (step == n_steps)
            if do_eval:
                t_eval = time.time()
                m, prev_flat = compute_metrics(
                    model=raw_student,
                    eval_batch=eval_batch,
                    eval_records=eval_records[:eval_bs],
                    device=device,
                    step=step,
                    train_loss=train_loss_val,
                    grad_norm=grad_norm_val,
                    prev_flat_params=prev_flat,
                    kl_teacher_prefix=kl_teacher_prefix,
                    kl_student_prefix=kl_student_prefix,
                )
                t_eval_s = time.time() - t_eval
                print(
                    f"step {step:6d}/{n_steps} | lr={cur_lr:.2e} | "
                    f"loss={m['train_loss']:.4f} | kl_T={kl_teacher_prefix:.4f} | "
                    f"kl_S={kl_student_prefix:.4f} | val={m['val_loss']:.4f} | "
                    f"acc={m['task_acc']:.3f} | gnorm={grad_norm_val:.2f} | "
                    f"{tps/1000:.1f}k tok/s | eval={t_eval_s*1000:.0f}ms"
                )
            else:
                m = {
                    "step": step,
                    "train_loss": round(train_loss_val, 6),
                    "val_loss": None,
                    "task_acc": None,
                    "task_acc_by_depth": None,
                    "kl_teacher_prefix": kl_teacher_prefix,
                    "kl_student_prefix": kl_student_prefix,
                    "mean_reward": None,
                    "kl_to_ref": None,
                    "grad_norm": round(grad_norm_val, 6),
                    "param_step_norm": None,
                }
                print(
                    f"step {step:6d}/{n_steps} | lr={cur_lr:.2e} | "
                    f"loss={m['train_loss']:.4f} | kl_T={kl_teacher_prefix:.4f} | "
                    f"kl_S={kl_student_prefix:.4f} | gnorm={grad_norm_val:.2f} | "
                    f"{tps/1000:.1f}k tok/s"
                )

            metrics_file.write(json.dumps(m) + '\n')
            metrics_file.flush()
            t_train_s = 0.0

    metrics_file.close()

    raw_student = getattr(student, '_orig_mod', student)
    torch.save(raw_student.state_dict(), model_path)
    print(f"\n[timing] Final model saved: {model_path}")

    precomputed_traj = None
    if traj_proj1 and pca_d1 is not None and pca_d2 is not None:
        flat_star = get_flat_params(raw_student)
        precomputed_traj = (
            np.array([p - float(flat_star @ pca_d1) for p in traj_proj1], dtype=np.float32),
            np.array([q - float(flat_star @ pca_d2) for q in traj_proj2], dtype=np.float32),
        )
        print(f"[timing] Trajectory precomputed: {len(traj_proj1)} points")

    print("\n[timing] Computing loss landscape...")
    save_landscape(
        model=raw_student,
        tmp_ckpt_dir=tmp_dir,
        eval_batch=eval_batch,
        device=device,
        log_dir=log_dir,
        grid_res=log_cfg.get('grid_res', 31),
        alpha_range=tuple(log_cfg.get('landscape_alpha_range', [-1.0, 1.0])),
        beta_range=tuple(log_cfg.get('landscape_beta_range', [-1.0, 1.0])),
        pca_basis_path=pca_basis_path,
        precomputed_traj=precomputed_traj,
    )

    total_min = (time.time() - t_start) / 60
    print(f"\n{stage_name} complete. Total time: {total_min:.1f} min. Log: {log_dir}")


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True, help='Path to training config yaml')
    return parser.parse_args()
