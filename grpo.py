"""
grpo.py — GRPO training for teacher and student models.

Starts from train.base_model_path, samples multiple completions per prompt,
scores them with the interpreter, and updates the policy with group-normalized
advantages plus a KL penalty to train.reference_model_path.

Usage:
    python grpo.py --config config/teacher_grpo.yaml
    python grpo.py --config config/student_grpo.yaml
"""
import argparse
import json
import os
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

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang import TOKEN2ID
from lib.metrics import compute_metrics, get_flat_params, save_landscape
from lib.model import build_model
from pretrain import get_lr, make_eval_batch

ID2TOK = {v: k for k, v in TOKEN2ID.items()}
EOS_ID = TOKEN2ID['[EOS]']


def tokenize_prompt(prompt: str) -> List[int]:
    """Convert a space-separated prompt string into token IDs."""
    return [TOKEN2ID[t] for t in prompt.split() if t in TOKEN2ID]


def extract_result(gen_ids: List[int]) -> str:
    """
    Extract the final numeric answer from generated token IDs.

    Supports direct output '= RESULT [EOS]' and CoT output
    '<think> ... </think> = RESULT [EOS]'.
    """
    toks = [ID2TOK.get(i, '?') for i in gen_ids]
    last_eq = max((i for i, t in enumerate(toks) if t == '='), default=-1)
    if last_eq < 0:
        return ''

    parts = []
    for t in toks[last_eq + 1:]:
        if t == '[EOS]':
            break
        if t.isdigit():
            parts.append(t)
    return ''.join(parts)


def reward_completion(gen_ids: List[int], expected_result: str, reward_fn: str) -> float:
    """Score one generated completion against the interpreter result."""
    pred = extract_result(gen_ids)
    if pred == expected_result:
        return 1.0
    if reward_fn != 'partial_match' or not pred:
        return 0.0

    n = min(len(pred), len(expected_result))
    prefix = 0
    while prefix < n and pred[prefix] == expected_result[prefix]:
        prefix += 1
    return 0.25 * prefix / max(1, len(expected_result))


class PromptBuffer:
    """
    Prompt sampler for GRPO rollouts.

    Records are grouped by prompt length so each sampled batch can be stacked
    without padding inside the prompt.
    """
    def __init__(self, jsonl_path: str, context_len: int):
        groups: Dict[int, List[Dict]] = defaultdict(list)

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if '?' in rec.get('prompt', ''):
                    continue
                if 'expr' not in rec or 'result' not in rec or rec['result'] is None:
                    continue
                prompt_ids = tokenize_prompt(rec['prompt'])
                if 0 < len(prompt_ids) < context_len:
                    groups[len(prompt_ids)].append({
                        'prompt_ids': prompt_ids,
                        'expr': rec['expr'],
                        'result': rec['result'],
                        'depth': rec.get('depth', 0),
                    })

        if not groups:
            raise ValueError(f"No rollout prompts found in {jsonl_path}")

        self.groups = dict(groups)
        self.lengths = sorted(self.groups)
        sizes = np.array([len(self.groups[L]) for L in self.lengths], dtype=np.float64)
        self.probs = sizes / sizes.sum()
        self.N = int(sizes.sum())
        self.context_len = context_len

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Sample prompts with replacement from one prompt-length group.

        Returns:
            prompt_ids: [B=batch_size, Lp] int64
            records:    length B, each with expr/result/depth
        """
        Lp = int(np.random.choice(self.lengths, p=self.probs))
        group = self.groups[Lp]
        idx = np.random.randint(0, len(group), size=batch_size)
        records = [group[i] for i in idx]
        ids = torch.tensor(
            [r['prompt_ids'] for r in records],
            dtype=torch.long,
            device=device,
        )
        return ids, records


@torch.no_grad()
def sample_rollouts(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    n_rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sampled completions from the current policy.

    prompt_ids: [B, Lp]
    Returns:
        full_ids:     [R=B*G, Lp+Tc] where Tc<=max_new_tokens
        action_mask:  [R, Tc] true for generated tokens before/including EOS
        old_logprobs: [R, Tc] log probs under the rollout policy
    """
    model.eval()
    B, Lp = prompt_ids.shape
    G = n_rollouts_per_prompt
    R = B * G

    ids = prompt_ids.repeat_interleave(G, dim=0)  # [R, Lp]
    done = torch.zeros(R, dtype=torch.bool, device=device)  # [R]
    gen_cols: List[torch.Tensor] = []
    mask_cols: List[torch.Tensor] = []
    logp_cols: List[torch.Tensor] = []

    max_new = min(max_new_tokens, model.pos_emb.num_embeddings - Lp)
    temp = max(temperature, 1e-6)

    for _ in range(max_new):
        active = ~done  # [R]
        if not active.any():
            break

        with torch.amp.autocast(
            device_type='cuda' if device == 'cuda' else 'cpu',
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(ids)[:, -1, :]  # [R, V]

        policy_logits = logits / temp  # [R, V]
        probs = F.softmax(policy_logits.float(), dim=-1)  # [R, V]
        next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # [R]
        next_ids = torch.where(active, next_ids, torch.full_like(next_ids, EOS_ID))

        logp = F.log_softmax(policy_logits.float(), dim=-1).gather(
            1, next_ids[:, None]
        ).squeeze(1)  # [R]

        gen_cols.append(next_ids)
        mask_cols.append(active)
        logp_cols.append(logp)

        done = done | (next_ids == EOS_ID)
        ids = torch.cat([ids, next_ids[:, None]], dim=1)  # [R, current_len+1]

    if not gen_cols:
        empty = torch.empty((R, 0), dtype=torch.long, device=device)
        return ids, empty.bool(), empty.float()

    action_mask = torch.stack(mask_cols, dim=1)  # [R, Tc]
    old_logprobs = torch.stack(logp_cols, dim=1).detach()  # [R, Tc]
    return ids, action_mask, old_logprobs


def sequence_logprobs_and_kl(
    model: nn.Module,
    ref_model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
    action_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    temperature: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recompute trainable logprobs and exact per-token KL on sampled contexts.

    full_ids:      [R, Lp+Tc]
    action_mask:   [R, Tc]
    old_logprobs:  [R, Tc]
    Returns:
        cur_logprobs: [R, Tc]
        old_logprobs: [R, Tc]
        kl_tokens:    [R, Tc]
    """
    device = full_ids.device
    temp = max(temperature, 1e-6)

    input_ids = full_ids[:, :-1]  # [R, Lp+Tc-1]
    action_ids = full_ids[:, 1:]  # [R, Lp+Tc-1]
    start = prompt_len - 1
    end = start + action_mask.shape[1]

    with torch.amp.autocast(
        device_type='cuda' if device.type == 'cuda' else 'cpu',
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        logits = model(input_ids)  # [R, Lp+Tc-1, V]
        with torch.no_grad():
            ref_logits = ref_model(input_ids)  # [R, Lp+Tc-1, V]

    action_slice = action_ids[:, start:end]  # [R, Tc]
    policy_logits = logits[:, start:end, :] / temp  # [R, Tc, V]
    cur_logp_all = F.log_softmax(policy_logits.float(), dim=-1)  # [R, Tc, V]
    cur_logprobs = cur_logp_all.gather(2, action_slice[:, :, None]).squeeze(2)  # [R, Tc]

    raw_logp = F.log_softmax(logits[:, start:end, :].float(), dim=-1)  # [R, Tc, V]
    ref_logp = F.log_softmax(ref_logits[:, start:end, :].float(), dim=-1)  # [R, Tc, V]
    raw_prob = raw_logp.exp()  # [R, Tc, V]
    kl_tokens = (raw_prob * (raw_logp - ref_logp)).sum(dim=-1)  # [R, Tc]

    return cur_logprobs, old_logprobs, kl_tokens


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over true mask positions."""
    return (x * mask).sum() / mask.sum().clamp(min=1)


def main(config_path: str):
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

    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    context_len = model_cfg['context_len']
    batch_size = train_cfg['batch_size']
    n_steps = train_cfg['n_steps']
    warmup_steps = train_cfg['warmup_steps']
    lr = train_cfg['lr']
    lr_schedule = train_cfg.get('lr_schedule', 'cosine')
    weight_decay = train_cfg.get('weight_decay', 0.0)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    G = train_cfg['n_rollouts_per_prompt']
    clip_eps = train_cfg.get('clip_eps', 0.2)
    kl_coeff = train_cfg.get('kl_coeff', 0.05)
    reward_fn = train_cfg.get('reward_fn', 'interpreter')
    rollout_temperature = train_cfg.get('rollout_temperature', 1.0)
    max_new_tokens = cfg.get('inference', {}).get('max_new_tokens', context_len)
    log_every = log_cfg['log_every']
    eval_every = log_cfg.get('eval_every', log_every)
    eval_bs = log_cfg['eval_batch_size']
    log_dir = out_cfg['log_dir']
    model_path = out_cfg['model_path']
    tmp_dir = log_cfg['tmp_ckpt_dir']
    pca_basis_path = log_cfg.get('pca_basis_path')

    amp_dtype_str = train_cfg.get('amp_dtype', 'bf16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bf16' else torch.float16
    use_amp = train_cfg.get('amp', True) and device == 'cuda'

    for d in [log_dir, tmp_dir, str(Path(model_path).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)

    base_model_path = train_cfg.get('base_model_path', '')
    ref_model_path = train_cfg.get('reference_model_path', base_model_path)
    if not base_model_path or not Path(base_model_path).exists():
        raise FileNotFoundError(f"base_model_path not found: {base_model_path}")
    if not ref_model_path or not Path(ref_model_path).exists():
        raise FileNotFoundError(f"reference_model_path not found: {ref_model_path}")

    print(f"[setup] Loading rollout prompts: {data_cfg['path']}")
    prompt_buf = PromptBuffer(data_cfg['path'], context_len)
    print(f"[setup] Rollout prompts: {prompt_buf.N} records in {len(prompt_buf.lengths)} length groups")

    print(f"[setup] Building policy model from {base_model_path}")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))

    print(f"[setup] Building frozen reference model from {ref_model_path}")
    ref_model = build_model(model_cfg).to(device)
    ref_model.load_state_dict(torch.load(ref_model_path, map_location=device, weights_only=True))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

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
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    ckpt_interval = max(1, n_steps // max(1, log_cfg.get('n_traj_ckpt', 30)))
    pca_d1: Optional[np.ndarray] = None
    pca_d2: Optional[np.ndarray] = None
    traj_proj1: List[float] = []
    traj_proj2: List[float] = []
    if pca_basis_path and Path(pca_basis_path).exists():
        with np.load(pca_basis_path) as npz:
            pca_d1 = npz['d1']
            pca_d2 = npz['d2']
        print("[setup] PCA basis loaded for lightweight trajectory tracking")

    metrics_path = Path(log_dir) / "metrics.jsonl"
    metrics_file = open(metrics_path, 'w')
    prev_flat: Optional[np.ndarray] = None
    grad_norm_t: Optional[torch.Tensor] = None
    t_start = time.time()
    t_train_s = 0.0

    print(f"\n{'=' * 60}")
    print(f"GRPO  n_steps={n_steps}  prompts/step={batch_size}  rollouts/prompt={G}  device={device}")
    print(f"  data={prompt_buf.N}  lr={lr:.2e}  clip_eps={clip_eps:.2f}  kl_coeff={kl_coeff:.3f}")
    print(f"  reward_fn={reward_fn}  rollout_temperature={rollout_temperature:.2f}")
    print(f"  log_every={log_every}  eval_every={eval_every}  AMP={use_amp}")
    print(f"{'=' * 60}\n")

    for step in range(1, n_steps + 1):
        model.train()
        cur_lr = get_lr(step, n_steps, lr, warmup_steps, lr_schedule)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        t_step = time.time()
        prompt_ids, records = prompt_buf.sample(batch_size, device)  # prompt_ids: [B, Lp]
        B, Lp = prompt_ids.shape

        full_ids, action_mask, old_logprobs = sample_rollouts(
            model=model,
            prompt_ids=prompt_ids,
            n_rollouts_per_prompt=G,
            max_new_tokens=max_new_tokens,
            temperature=rollout_temperature,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        # full_ids: [R=B*G, Lp+Tc], action_mask/old_logprobs: [R, Tc]
        model.train()

        completion_ids = full_ids[:, Lp:].detach().cpu().tolist()
        expected = [r['result'] for r in records for _ in range(G)]
        reward_vals = [
            reward_completion(gen, exp, reward_fn)
            for gen, exp in zip(completion_ids, expected)
        ]
        rewards = torch.tensor(reward_vals, dtype=torch.float32, device=device).view(B, G)  # [B, G]
        reward_mean = rewards.mean(dim=1, keepdim=True)  # [B, 1]
        reward_std = rewards.std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
        advantages = ((rewards - reward_mean) / (reward_std + 1e-6)).view(B * G, 1)  # [R, 1]

        cur_logprobs, old_logprobs, kl_tokens = sequence_logprobs_and_kl(
            model=model,
            ref_model=ref_model,
            full_ids=full_ids,
            prompt_len=Lp,
            action_mask=action_mask,
            old_logprobs=old_logprobs,
            temperature=rollout_temperature,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        ratio = torch.exp(cur_logprobs - old_logprobs)  # [R, Tc]
        unclipped = ratio * advantages  # [R, Tc]
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages  # [R, Tc]
        policy_loss = -masked_mean(torch.minimum(unclipped, clipped), action_mask)  # scalar
        kl_loss = masked_mean(kl_tokens, action_mask)  # scalar
        loss = policy_loss + kl_coeff * kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_t = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        t_train_s += time.time() - t_step

        if step == 1:
            print(f"[timing] First step: {t_train_s * 1000:.0f}ms")

        if (step % ckpt_interval == 0 or step == 1) and pca_d1 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            flat = get_flat_params(model)
            traj_proj1.append(float(flat @ pca_d1))
            traj_proj2.append(float(flat @ pca_d2))

        if step % log_every == 0 or step == n_steps:
            train_loss_val = float(loss.detach().cpu())
            grad_norm_val = float(grad_norm_t.detach().cpu()) if grad_norm_t is not None else 0.0
            mean_reward_val = float(rewards.mean().detach().cpu())
            kl_to_ref_val = float(kl_loss.detach().cpu())
            tps = batch_size * G * max(1, action_mask.shape[1]) * log_every / max(t_train_s, 1e-6)

            do_eval = (step % eval_every == 0) or (step == n_steps)
            if do_eval:
                t_eval = time.time()
                m, prev_flat = compute_metrics(
                    model=model,
                    eval_batch=eval_batch,
                    eval_records=eval_records[:eval_bs],
                    device=device,
                    step=step,
                    train_loss=train_loss_val,
                    grad_norm=grad_norm_val,
                    prev_flat_params=prev_flat,
                    mean_reward=round(mean_reward_val, 6),
                    kl_to_ref=round(kl_to_ref_val, 6),
                )
                t_eval_s = time.time() - t_eval
                print(
                    f"step {step:6d}/{n_steps} | lr={cur_lr:.2e} | "
                    f"loss={m['train_loss']:.4f} | reward={mean_reward_val:.3f} | "
                    f"kl_ref={kl_to_ref_val:.4f} | val={m['val_loss']:.4f} | "
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
                    "kl_teacher_prefix": None,
                    "kl_student_prefix": None,
                    "mean_reward": round(mean_reward_val, 6),
                    "kl_to_ref": round(kl_to_ref_val, 6),
                    "grad_norm": round(grad_norm_val, 6),
                    "param_step_norm": None,
                }
                print(
                    f"step {step:6d}/{n_steps} | lr={cur_lr:.2e} | "
                    f"loss={m['train_loss']:.4f} | reward={mean_reward_val:.3f} | "
                    f"kl_ref={kl_to_ref_val:.4f} | gnorm={grad_norm_val:.2f} | "
                    f"{tps/1000:.1f}k tok/s"
                )

            metrics_file.write(json.dumps(m) + '\n')
            metrics_file.flush()
            t_train_s = 0.0

    metrics_file.close()

    torch.save(model.state_dict(), model_path)
    print(f"\n[timing] Final model saved: {model_path}")

    precomputed_traj = None
    if traj_proj1 and pca_d1 is not None and pca_d2 is not None:
        flat_star = get_flat_params(model)
        precomputed_traj = (
            np.array([p - float(flat_star @ pca_d1) for p in traj_proj1], dtype=np.float32),
            np.array([q - float(flat_star @ pca_d2) for q in traj_proj2], dtype=np.float32),
        )
        print(f"[timing] Trajectory precomputed: {len(traj_proj1)} points")

    print("\n[timing] Computing loss landscape...")
    save_landscape(
        model=model,
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
    print(f"\nGRPO complete. Total time: {total_min:.1f} min. Log: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRPO training')
    parser.add_argument('--config', required=True,
                        help='Path to training config yaml (e.g. config/teacher_grpo.yaml)')
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    main(args.config)
