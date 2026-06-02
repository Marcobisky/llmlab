"""
sft_lora.py — LoRA (Low-Rank Adaptation) SFT for teacher model.

Freezes the pretrained base model and injects low-rank adapter matrices into
selected Linear layers. Only adapter params (A, B) are trained.

LoRA formula:
    h = W0 x + (alpha/r) * B A x
    W0 : frozen pretrained weight   [d_out, d_in]
    A  : trainable, init Normal     [r, d_in]       (r = rank)
    B  : trainable, init zeros      [d_out, r]
    Scaling alpha/r ensures the LoRA branch is near-zero at init regardless of r.

After training, adapters are merged: W_merged = W0 + (alpha/r) * B @ A
Saved weights use the standard Transformer state_dict format (same as full SFT),
so all downstream eval/inference code works without modification.

Tmp checkpoints also store merged weights, making save_landscape() compatible
with the base-model state_dict format it expects.

Usage:
    python sft_lora.py --config config/teacher_sft_lora.yaml
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang    import TOKEN2ID
from lib.model   import build_model, Transformer
from lib.metrics import compute_metrics, save_landscape

# Reuse data utilities from pretrain.py (data loading, lr schedule, eval batch)
from pretrain import GPUDataBuffer, make_eval_batch, get_lr

PAD_ID = TOKEN2ID['[EOS]']
EOS_ID = TOKEN2ID['[EOS]']


# ─────────────────────────────────────────────────────────────────────────────
# LoRA layer
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Replaces one nn.Linear with a frozen base weight + trainable low-rank adapter.

    Forward:
        out = W0 @ x  +  (alpha/r) * (B @ A) @ x
            = F.linear(x, W0 + scale * B @ A)

    State dict keys match the original nn.Linear ('weight', optionally 'bias'),
    so get_merged_state_dict() can produce a base-model-compatible checkpoint.
    """
    def __init__(
        self,
        linear: nn.Linear,   # original frozen layer to wrap
        rank: int,            # r: adapter bottleneck dimension
        alpha: float,         # scaling numerator; scale = alpha/r
        dropout: float = 0.0, # dropout on x before LoRA branch (usually 0)
    ):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.scale = alpha / rank

        # Frozen base weight: registered as non-trainable Parameter so it
        # appears in state_dict() with key '...weight' (same as nn.Linear)
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Trainable adapters: A ~ N(0, 0.02), B = 0 (net output = 0 at init)
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.02)  # [r, d_in]
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))          # [d_out, r]

        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_in]
        # Fuse base + adapter into one matmul (avoids two separate linear passes)
        # lora_B @ lora_A: [d_out, r] @ [r, d_in] -> [d_out, d_in]
        w = self.weight + self.scale * (self.lora_B @ self.lora_A)
        return F.linear(self.lora_drop(x), w, self.bias)

    def merged_weight(self) -> torch.Tensor:
        """W_merged = W0 + (alpha/r) * B @ A  (detached, for checkpoint saving)."""
        return (self.weight + self.scale * (self.lora_B @ self.lora_A)).detach()


# ─────────────────────────────────────────────────────────────────────────────
# LoRA model surgery
# ─────────────────────────────────────────────────────────────────────────────

def _set_module(root: nn.Module, name: str, new_module: nn.Module):
    """Replace root.{name} (dot-separated path, supports ModuleList indices)."""
    parts  = name.split('.')
    parent = root
    for p in parts[:-1]:
        # PyTorch ModuleList supports both getattr and _modules[str_idx]
        parent = parent._modules[p] if p.isdigit() else getattr(parent, p)
    parent._modules[parts[-1]] = new_module


def apply_lora(model: nn.Module, lora_cfg: Dict) -> nn.Module:
    """
    Walk all named nn.Linear layers; replace those whose name ends with any
    target string with a LoRALinear. Freeze all other parameters.

    Returns the same model object (modified in-place).

    Example — targets ['attn.qkv', 'attn.proj'] matches:
        blocks.0.attn.qkv,  blocks.1.attn.qkv,  ..., blocks.5.attn.qkv
        blocks.0.attn.proj, blocks.1.attn.proj,  ..., blocks.5.attn.proj
    """
    rank     = lora_cfg['rank']
    alpha    = float(lora_cfg['alpha'])
    dropout  = float(lora_cfg.get('dropout', 0.0))
    targets  = lora_cfg['targets']   # e.g. ['attn.qkv', 'attn.proj']

    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Then inject LoRA adapters (which have requires_grad=True by default)
    n_replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in targets):
            continue
        lora_layer = LoRALinear(module, rank, alpha, dropout)
        _set_module(model, name, lora_layer)
        n_replaced += 1
        d_out, d_in = module.weight.shape
        n_adapter = rank * (d_in + d_out)
        print(f"  LoRA  {name:<35}  [{d_in}->{d_out}]  r={rank}  +{n_adapter:,} params")

    # Print trainable param count
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,}  ({100*n_trainable/n_total:.2f}%)")
    return model


def get_merged_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Build a state_dict compatible with the base Transformer (no lora_A/lora_B keys).
    For LoRALinear layers, 'weight' is replaced with the merged value W0 + scale*B@A.
    All other keys are passed through unchanged.
    """
    # Start from full state_dict, strip adapter keys
    state = {k: v for k, v in model.state_dict().items()
             if '.lora_A' not in k and '.lora_B' not in k}

    # Overwrite base weights with merged weights for LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name + '.weight'] = module.merged_weight()

    return state


def merge_lora_inplace(model: nn.Module) -> nn.Module:
    """
    Replace every LoRALinear with a plain nn.Linear (merged weights) in-place.
    After this, model is a standard Transformer compatible with load_state_dict.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        d_out, d_in = module.weight.shape
        merged = nn.Linear(d_in, d_out, bias=module.bias is not None)
        merged.weight = nn.Parameter(module.merged_weight().float())
        if module.bias is not None:
            merged.bias = nn.Parameter(module.bias.detach().float())
        _set_module(model, name, merged)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tmp checkpoint (saves MERGED weights for landscape compatibility)
# ─────────────────────────────────────────────────────────────────────────────

_ckpt_thread: threading.Thread = None


def save_tmp_ckpt_lora(model: nn.Module, tmp_dir: str, step: int,
                       dtype: str = 'float16') -> float:
    """
    Save merged weights (W0 + LoRA delta) to tmp_dir/<step>.pt.
    Checkpoint format matches base Transformer — compatible with save_landscape().
    GPU sync + CPU copy done in calling thread; disk write dispatched to background.
    Returns time spent in calling thread (GPU copy only).
    """
    global _ckpt_thread

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    raw = getattr(model, '_orig_mod', model)   # unwrap torch.compile if needed

    merged = get_merged_state_dict(raw)
    if dtype == 'float16':
        merged = {k: v.half().cpu() for k, v in merged.items()}
    else:
        merged = {k: v.cpu() for k, v in merged.items()}
    t_copy = time.time() - t0

    if _ckpt_thread is not None and _ckpt_thread.is_alive():
        _ckpt_thread.join()

    out_path = Path(tmp_dir) / f"{step:06d}.pt"

    def _write(state, path):
        torch.save(state, path)

    _ckpt_thread = threading.Thread(target=_write, args=(merged, out_path), daemon=True)
    _ckpt_thread.start()
    return t_copy


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg['data']
    model_cfg = cfg['model']
    lora_cfg  = cfg['lora']
    train_cfg = cfg['train']
    out_cfg   = cfg['output']
    log_cfg   = cfg['logging']

    device = train_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, falling back to CPU")

    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    context_len  = model_cfg['context_len']
    batch_size   = train_cfg['batch_size']
    n_steps      = train_cfg['n_steps']
    warmup_steps = train_cfg['warmup_steps']
    lr           = train_cfg['lr']
    lr_schedule  = train_cfg.get('lr_schedule', 'cosine')
    weight_decay = train_cfg.get('weight_decay', 0.1)
    grad_clip    = train_cfg.get('grad_clip', 2.0)
    _amp_dtype_str = train_cfg.get('amp_dtype', 'bf16')
    _amp_dtype     = torch.bfloat16 if _amp_dtype_str == 'bf16' else torch.float16
    use_amp        = train_cfg.get('amp', True) and device == 'cuda'
    log_every    = log_cfg['log_every']
    eval_bs      = log_cfg['eval_batch_size']
    n_traj_ckpt = log_cfg['n_traj_ckpt']
    ckpt_dtype      = log_cfg.get('ckpt_dtype', 'float16')
    tmp_dir         = log_cfg['tmp_ckpt_dir']
    pca_basis_path  = log_cfg.get('pca_basis_path')
    log_dir         = out_cfg['log_dir']
    model_path      = out_cfg['model_path']
    eval_every      = log_cfg.get('eval_every', log_every)

    ckpt_interval = max(1, n_steps // n_traj_ckpt)

    for d in [log_dir, tmp_dir, str(Path(model_path).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Build base model and load pretrain checkpoint ────────────────────────
    _t = time.time()
    print(f"[setup] Building model...")
    model = build_model(model_cfg).to(device)

    base_path = train_cfg.get('base_model_path')
    if base_path and Path(base_path).exists():
        print(f"[setup] Loading base model: {base_path}")
        model.load_state_dict(
            torch.load(base_path, map_location=device, weights_only=True)
        )
    else:
        print(f"[setup] Warning: base_model_path not found ({base_path}), starting from scratch")
    print(f"[setup] Base model ready in {time.time()-_t:.2f}s")

    # ── Inject LoRA adapters and freeze base weights ─────────────────────────
    print(f"[setup] Applying LoRA (rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}):")
    apply_lora(model, lora_cfg)
    model = model.to(device)

    # torch.compile works with LoRA since LoRALinear uses standard ops
    if train_cfg.get('compile', True) and hasattr(torch, 'compile'):
        print("[setup] torch.compile() wrapping model...")
        model = torch.compile(model, mode="reduce-overhead")

    # ── Data loading ─────────────────────────────────────────────────────────
    buf = GPUDataBuffer(data_cfg['path'], context_len, mode='sft', device=device)

    # ── Eval data ────────────────────────────────────────────────────────────
    eval_data_path = log_cfg.get('eval_data_path', '')
    eval_records: List[Dict] = []
    if Path(eval_data_path).exists():
        with open(eval_data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
        print(f"[setup] Eval: {len(eval_records)} records ({eval_data_path})")
    else:
        with open(data_cfg['path']) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
                if len(eval_records) >= eval_bs:
                    break
        print(f"[setup] Warning: eval file not found, borrowing {len(eval_records)} train records")

    eval_batch = make_eval_batch(eval_records, context_len, eval_bs, device, mode='sft')
    # eval_batch: (inp [eval_bs, C-1], lbl [eval_bs, C-1]), fixed

    # ── Optimizer: only LoRA adapter params ──────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    # lora_params: only lora_A and lora_B tensors (all others frozen)
    optimizer = torch.optim.AdamW(
        lora_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    scaler = (torch.cuda.amp.GradScaler()
              if use_amp and _amp_dtype == torch.float16 else None)

    # ── Training loop ─────────────────────────────────────────────────────────
    metrics_path = Path(log_dir) / "metrics.jsonl"
    metrics_file = open(metrics_path, 'w')

    prev_flat: Optional[np.ndarray]     = None
    grad_norm_t: Optional[torch.Tensor] = None
    t_start    = time.time()
    t_train_s  = 0.0
    t_ckpt_s   = 0.0
    n_ckpts_in_interval = 0

    n_epoch_steps = max(1, buf.N // batch_size)
    n_trainable   = sum(p.numel() for p in lora_params)
    print(f"\n{'='*60}")
    print(f"LoRA SFT  n_steps={n_steps}  batch={batch_size}  device={device}")
    print(f"  data={buf.N}  steps/epoch≈{n_epoch_steps}  "
          f"total≈{n_steps/n_epoch_steps:.1f} epochs")
    print(f"  trainable={n_trainable:,}  AMP={use_amp}  lr={lr:.2e}")
    print(f"{'='*60}\n")

    for step in range(1, n_steps + 1):
        model.train()

        cur_lr = get_lr(step, n_steps, lr, warmup_steps, lr_schedule)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        t_step = time.time()
        inp_ids, labels = buf.sample(batch_size)
        # inp_ids: [B=batch_size, T=C-1]
        # labels:  [B, T], -100 positions masked

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type='cuda' if device == 'cuda' else 'cpu',
            dtype=_amp_dtype, enabled=use_amp
        ):
            _, loss = model(inp_ids, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm_t = nn.utils.clip_grad_norm_(lora_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm_t = nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

        t_train_s += time.time() - t_step

        if step == 1:
            compile_note = " (incl. torch.compile JIT)" if train_cfg.get('compile', True) else ""
            print(f"[timing] First step: {t_train_s*1000:.0f}ms{compile_note}")

        # ── Save tmp checkpoint (merged weights for landscape compat) ────────
        if step % ckpt_interval == 0 or step == 1:
            elapsed_ckpt = save_tmp_ckpt_lora(model, tmp_dir, step, ckpt_dtype)
            t_ckpt_s += elapsed_ckpt
            n_ckpts_in_interval += 1

        # ── Logging ──────────────────────────────────────────────────────────
        if step % log_every == 0 or step == n_steps:
            train_loss_val = loss.item()
            grad_norm_val  = grad_norm_t.item() if grad_norm_t is not None else 0.0
            tps = log_every * batch_size * (context_len - 1) / max(t_train_s, 1e-6)

            timing_suffix = (
                f"[train={t_train_s*1000/log_every:.1f}ms/step "
                f"ckpt={t_ckpt_s*1000:.0f}ms×{n_ckpts_in_interval}"
            )

            do_eval = (step % eval_every == 0) or (step == n_steps)

            if do_eval:
                t_eval = time.time()
                # compute_metrics needs the unwrapped model (no torch.compile wrapper)
                raw_model = getattr(model, '_orig_mod', model)
                m, prev_flat = compute_metrics(
                    model            = raw_model,
                    eval_batch       = eval_batch,
                    eval_records     = eval_records[:eval_bs],
                    device           = device,
                    step             = step,
                    train_loss       = train_loss_val,
                    grad_norm        = grad_norm_val,
                    prev_flat_params = prev_flat,
                )
                t_eval_s = time.time() - t_eval
                timing_suffix += f" eval={t_eval_s*1000:.0f}ms]"
                print(
                    f"step {step:6d}/{n_steps} | lr={cur_lr:.2e} | "
                    f"loss={m['train_loss']:.4f} | val={m['val_loss']:.4f} | "
                    f"acc={m['task_acc']:.3f} | gnorm={grad_norm_val:.2f} | "
                    f"{tps/1000:.1f}k tok/s | {timing_suffix}"
                )
            else:
                m = {
                    "step":              step,
                    "train_loss":        round(train_loss_val, 6),
                    "val_loss":          None,
                    "task_acc":          None,
                    "task_acc_by_depth": None,
                    "kl_teacher_prefix": None,
                    "kl_student_prefix": None,
                    "mean_reward":       None,
                    "kl_to_ref":         None,
                    "grad_norm":         round(grad_norm_val, 6),
                    "param_step_norm":   None,
                }
                timing_suffix += "]"
                print(
                    f"step {step:6d}/{n_steps} | lr={cur_lr:.2e} | "
                    f"loss={m['train_loss']:.4f} | gnorm={grad_norm_val:.2f} | "
                    f"{tps/1000:.1f}k tok/s | {timing_suffix}"
                )

            metrics_file.write(json.dumps(m) + '\n')
            metrics_file.flush()

            t_train_s  = 0.0
            t_ckpt_s   = 0.0
            n_ckpts_in_interval = 0

    metrics_file.close()

    if _ckpt_thread is not None and _ckpt_thread.is_alive():
        print("[timing] Waiting for background checkpoint write...")
        _ckpt_thread.join()

    # ── Merge LoRA into base weights and save ────────────────────────────────
    # After merge, model becomes a standard Transformer (same format as full SFT).
    # This makes the checkpoint compatible with all eval/inference/KD scripts.
    _t = time.time()
    raw_model = getattr(model, '_orig_mod', model)
    print("\n[setup] Merging LoRA adapters into base weights...")
    merge_lora_inplace(raw_model)
    torch.save(raw_model.state_dict(), model_path)
    print(f"[timing] Merged model saved: {model_path}  ({time.time()-_t:.2f}s)")

    # ── Loss landscape ───────────────────────────────────────────────────────
    # tmp checkpoints store merged weights -> compatible with standard load_state_dict
    # raw_model is now a plain Transformer (LoRA merged) -> landscape works normally
    print("\n[timing] Computing loss landscape...")
    save_landscape(
        model           = raw_model,
        tmp_ckpt_dir    = tmp_dir,
        eval_batch      = eval_batch,
        device          = device,
        log_dir         = log_dir,
        grid_res        = log_cfg.get('grid_res', 31),
        alpha_range     = tuple(log_cfg.get('landscape_alpha_range', [-1.0, 1.0])),
        beta_range      = tuple(log_cfg.get('landscape_beta_range',  [-1.0, 1.0])),
        pca_basis_path  = pca_basis_path,
    )

    total_min = (time.time() - t_start) / 60
    print(f"\nLoRA SFT complete. Total time: {total_min:.1f} min. Log: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoRA SFT')
    parser.add_argument('--config', required=True,
                        help='Path to config yaml (e.g. config/teacher_sft_lora.yaml)')
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    main(args.config)
