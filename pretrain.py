"""
pretrain.py — Pretraining script, shared by teacher and student via config.
All behaviour is controlled by the five config blocks: data / model / train / output / logging.

Usage:
    python pretrain.py --config config/teacher_pretrain.yaml
    python pretrain.py --config config/student_pretrain.yaml

Outputs (all paths specified in config):
    log/<name>/<name>.pt           final model weights
    log/<name>/metrics.jsonl       one scalar-metrics row per log_every steps
    log/<name>/landscape.npz       loss landscape computed at end of training

GPU utilization optimizations:
    1. GPUDataBuffer — tokenizes the entire dataset once and pins it on GPU;
       sampling = torch.randint indexing, zero CPU-GPU transfer overhead.
    2. AMP (torch.amp.autocast) — float16/bf16 forward/backward, tensor core acceleration.
    3. torch.compile (PyTorch >= 2.0) — graph compilation for further throughput gains.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang    import TOKEN2ID
from lib.model   import build_model
from lib.metrics import compute_metrics, save_landscape

PAD_ID = TOKEN2ID['[EOS]']   # padding token (labels use -100 mask)
EOS_ID = TOKEN2ID['[EOS]']


# ─────────────────────────────────────────────────────────────────────────────
# GPU data preloading (core speed-up mechanism)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_record(rec: Dict, context_len: int, mode: str) -> Tuple[List[int], List[int]]:
    """
    Single jsonl record -> (inp, lbl) padded to context_len-1.
    mode='pretrain': full sequence loss.
    mode='sft': prompt labels set to -100 (only target tokens count).
    """
    C = context_len
    prompt_ids = [TOKEN2ID.get(t, 0) for t in rec['prompt'].split()]
    target_ids = [TOKEN2ID.get(t, 0) for t in rec['target'].split()]
    full_ids   = (prompt_ids + target_ids)[:C]   # truncate (rarely triggered)

    inp = full_ids[:-1]   # [BOS] EXPR = RESULT      length L-1
    lbl = full_ids[1:]    # EXPR = RESULT [EOS]       length L-1

    if mode == 'sft':
        # mask prompt positions so only target tokens contribute to loss
        n_mask = min(len(prompt_ids) - 1, len(inp))
        for i in range(n_mask):
            lbl[i] = -100

    pad = (C - 1) - len(inp)
    inp += [PAD_ID] * pad
    lbl += [-100]   * pad
    return inp, lbl


class GPUDataBuffer:
    """
    Preloads the entire training set onto GPU memory.

    Why this works:
      - Dataset is small (50k x 95 tokens x int64 ≈ 38 MB), fits easily on GPU.
      - Sampling becomes a pure GPU op: torch.randint -> tensor indexing,
        eliminating DataLoader Python overhead, CPU->GPU transfer, and worker sync.
      - Lifts GPU utilization from ~17% to ~85%+ in practice.

    inp:  [N, C-1]  int64, token IDs (padding = PAD_ID)
    lbl:  [N, C-1]  int64, labels (padding = -100)
    """
    def __init__(self, jsonl_path: str, context_len: int, mode: str, device: str):
        print(f"  Preloading dataset to {device} ({jsonl_path}) ...")
        t0 = time.time()

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        all_inp, all_lbl = [], []
        for rec in records:
            inp, lbl = _tokenize_record(rec, context_len, mode)
            all_inp.append(inp)
            all_lbl.append(lbl)

        self.inp = torch.tensor(all_inp, dtype=torch.long, device=device)  # [N, C-1]
        self.lbl = torch.tensor(all_lbl, dtype=torch.long, device=device)  # [N, C-1]
        self.N   = self.inp.shape[0]

        mb = self.inp.numel() * 2 * 8 / 1e6   # inp + lbl, int64
        print(f"  Loaded {self.N} records, GPU memory ≈ {mb:.1f} MB, time {time.time()-t0:.1f}s")

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample batch_size records with replacement, pure GPU op.
        Returns (inp [B, C-1], lbl [B, C-1]) already on GPU.
        """
        idx = torch.randint(self.N, (batch_size,), device=self.inp.device)
        return self.inp[idx], self.lbl[idx]


def make_eval_batch(
    eval_records: List[Dict], context_len: int, max_samples: int, device: str,
    mode: str = 'sft',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert eval records to a fixed batch (not shuffled, for stable val_loss)."""
    all_inp, all_lbl = [], []
    for rec in eval_records[:max_samples]:
        inp, lbl = _tokenize_record(rec, context_len, mode=mode)
        all_inp.append(inp)
        all_lbl.append(lbl)
    inp_t = torch.tensor(all_inp, dtype=torch.long, device=device)  # [B, C-1]
    lbl_t = torch.tensor(all_lbl, dtype=torch.long, device=device)  # [B, C-1]
    return inp_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, n_steps: int, lr: float, warmup_steps: int,
           schedule: str = 'cosine') -> float:
    """
    Linear warmup (0 -> lr over warmup_steps) then cosine decay to lr*0.1.
    step starts at 1.

    Decays to lr*0.1 rather than 0 to avoid the training stall caused by
    near-zero learning rates at the end (cosine to 0 freezes updates).
    """
    if step <= warmup_steps:
        return lr * step / max(1, warmup_steps)
    if schedule == 'constant':
        return lr
    min_ratio = 0.1
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_ratio + (1 - min_ratio) * cosine)


# ─────────────────────────────────────────────────────────────────────────────
# Temporary checkpoint (for landscape computation)
# ─────────────────────────────────────────────────────────────────────────────

_ckpt_thread: threading.Thread = None   # global handle to the last ckpt background thread


def save_tmp_ckpt(model: nn.Module, tmp_dir: str, step: int, dtype: str = 'float16') -> float:
    """
    Save current parameters to tmp_dir/<step>.pt (float16 by default).

    The GPU->CPU copy is done synchronously in the calling thread (fast, ~1ms).
    The actual disk write is dispatched to a background thread so it overlaps
    with the next training steps instead of blocking them.

    Returns the time spent in the calling thread (GPU copy only, not disk write).
    """
    global _ckpt_thread

    # With torch.compile(reduce-overhead), CUDA ops are submitted asynchronously.
    # Sync the stream BEFORE timing so the measured time is only the CPU copy,
    # not the accumulated async GPU work from the training steps.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    raw = getattr(model, '_orig_mod', model)  # unwrap torch.compile if needed

    # Copy parameters to CPU first (in calling thread: fast after sync above)
    if dtype == 'float16':
        state_cpu = {k: v.half().cpu() for k, v in raw.state_dict().items()}
    else:
        state_cpu = {k: v.cpu() for k, v in raw.state_dict().items()}
    t_copy = time.time() - t0

    # Wait for any previous background save to finish before starting a new one
    # (avoids two concurrent writes to the same directory)
    if _ckpt_thread is not None and _ckpt_thread.is_alive():
        _ckpt_thread.join()

    out_path = Path(tmp_dir) / f"{step:06d}.pt"

    def _write(state, path):
        torch.save(state, path)

    _ckpt_thread = threading.Thread(target=_write, args=(state_cpu, out_path), daemon=True)
    _ckpt_thread.start()

    return t_copy   # only report GPU copy time; disk write runs in background


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['train']
    out_cfg   = cfg['output']
    log_cfg   = cfg['logging']

    device = train_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, falling back to CPU")

    # NVIDIA GPU settings
    if device == 'cuda':
        # TF32: Ampere+ uses TF32 for FP32 matmul (negligible precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        # auto-select fastest cuDNN conv algorithm for fixed input sizes
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
    pca_basis_path  = log_cfg.get('pca_basis_path')   # optional: path to pca_basis.npz from an earlier stage
    log_dir         = out_cfg['log_dir']
    model_path      = out_cfg['model_path']
    # eval_every: how often to run the full (slow) eval: val_loss + greedy-decode task_acc.
    # Defaults to log_every so existing configs keep their current behaviour unchanged.
    # Set to a larger multiple of log_every to reduce eval overhead.
    eval_every      = log_cfg.get('eval_every', log_every)

    ckpt_interval = max(1, n_steps // n_traj_ckpt)

    # create output directories
    for d in [log_dir, tmp_dir, str(Path(model_path).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── build model ──────────────────────────────────────────────────────────
    _t = time.time()
    print(f"[setup] Building model (context_len={context_len})...")
    model = build_model(model_cfg).to(device)
    print(f"[setup] Model built in {time.time()-_t:.2f}s")

    base_path = train_cfg.get('base_model_path')
    if base_path and Path(base_path).exists():
        _t = time.time()
        print(f"[setup] Loading base model: {base_path}")
        model.load_state_dict(
            torch.load(base_path, map_location=device, weights_only=True)
        )
        print(f"[setup] Base model loaded in {time.time()-_t:.2f}s")

    # torch.compile: ~30-50% additional speedup for small models
    if train_cfg.get('compile', True) and hasattr(torch, 'compile'):
        _t = time.time()
        print("[setup] torch.compile() wrapping model...")
        model = torch.compile(model, mode="reduce-overhead")
        # NOTE: actual kernel compilation is deferred to the first forward pass
        print(f"[setup] torch.compile() wrap done in {time.time()-_t:.2f}s "
              f"(first step will be slow due to JIT compilation)")

    # ── GPU data preloading ──────────────────────────────────────────────────
    data_path = data_cfg['path']
    # mode='sft': prompt labels are -100; loss computed only on target tokens.
    # Both pretrain and SFT use this mode — the difference is in the data:
    #   pretrain: mixed stmt/check/cot, teaches all sentence patterns
    #   SFT: only one type (e.g. CoT), specializes generation format
    buf = GPUDataBuffer(data_path, context_len, mode='sft', device=device)

    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"[setup] GPU memory after data load: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")

    # ── eval data ────────────────────────────────────────────────────────────
    eval_data_path = log_cfg.get('eval_data_path', '')
    _t = time.time()
    eval_records: List[Dict] = []
    if Path(eval_data_path).exists():
        with open(eval_data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
        print(f"[setup] Eval data: {len(eval_records)} records ({eval_data_path}), "
              f"loaded in {time.time()-_t:.2f}s")
    else:
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
                if len(eval_records) >= eval_bs:
                    break
        print(f"[setup] Warning: eval file not found, borrowing {len(eval_records)} records "
              f"from train set in {time.time()-_t:.2f}s")

    _t = time.time()
    eval_batch = make_eval_batch(eval_records, context_len, eval_bs, device, mode='sft')
    print(f"[setup] Eval batch ({eval_bs} records) built in {time.time()-_t:.2f}s")
    # eval_batch: (inp [B, C-1], lbl [B, C-1]), fixed throughout training

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    # BF16 does not need GradScaler (dynamic range is sufficient).
    # FP16 still needs scaler for legacy GPU support.
    scaler = (torch.cuda.amp.GradScaler()
              if use_amp and _amp_dtype == torch.float16 else None)

    # training loop
    metrics_path = Path(log_dir) / "metrics.jsonl"
    metrics_file = open(metrics_path, 'w')

    prev_flat: Optional[np.ndarray]     = None
    grad_norm_t: Optional[torch.Tensor] = None  # avoid .item() in hot path
    t_start    = time.time()
    t_interval = t_start

    # per-interval timing accumulators (reset every log_every steps)
    t_train_s   = 0.0   # pure forward + backward + optimizer time
    t_ckpt_s    = 0.0   # checkpoint save time
    n_ckpts_in_interval = 0

    n_epoch_steps = max(1, buf.N // batch_size)
    print(f"\n{'='*60}")
    print(f"Pretrain  n_steps={n_steps}  batch={batch_size}  device={device}")
    print(f"  data={buf.N}  steps/epoch≈{n_epoch_steps}  "
          f"total≈{n_steps/n_epoch_steps:.1f} epochs")
    print(f"  AMP={use_amp}  grad_clip={grad_clip}  lr={lr:.2e}")
    print(f"  ckpt_interval={ckpt_interval}  log_every={log_every}  eval_every={eval_every}")
    print(f"{'='*60}\n")

    for step in range(1, n_steps + 1):
        model.train()

        cur_lr = get_lr(step, n_steps, lr, warmup_steps, lr_schedule)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        # ── forward + backward + optimizer (pure training time) ──────────────
        t_step = time.time()
        inp_ids, labels = buf.sample(batch_size)
        # inp_ids: [B=batch_size, T=C-1]
        # labels:  [B, T], -100 positions ignored by CE

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type='cuda' if device == 'cuda' else 'cpu',
            dtype=_amp_dtype, enabled=use_amp
        ):
            _, loss = model(inp_ids, labels)

        if scaler is not None:          # FP16 path
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm_t = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:                           # BF16 / no AMP: skip Scaler overhead
            loss.backward()
            grad_norm_t = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        # avoid .item() in hot path; CPU-GPU sync deferred to log steps
        t_train_s += time.time() - t_step

        # first step: report compile JIT overhead
        if step == 1:
            compile_note = " (incl. torch.compile JIT)" if train_cfg.get('compile', True) else ""
            print(f"[timing] First step: {t_train_s*1000:.0f}ms{compile_note}")

        # ── save trajectory checkpoint ───────────────────────────────────────
        if step % ckpt_interval == 0 or step == 1:
            elapsed_ckpt = save_tmp_ckpt(model, tmp_dir, step, ckpt_dtype)
            t_ckpt_s += elapsed_ckpt
            n_ckpts_in_interval += 1

        # ── metrics logging ──────────────────────────────────────────────────
        if step % log_every == 0 or step == n_steps:
            t_interval = time.time()   # reset interval clock

            train_loss_val = loss.item()
            grad_norm_val  = grad_norm_t.item() if grad_norm_t is not None else 0.0
            # tps uses pure training time (excludes ckpt + eval overhead)
            tps = log_every * batch_size * (context_len - 1) / max(t_train_s, 1e-6)

            timing_suffix = (
                f"[train={t_train_s*1000/log_every:.1f}ms/step "
                f"ckpt={t_ckpt_s*1000:.0f}ms×{n_ckpts_in_interval}"
            )

            # full eval (val_loss + greedy-decode task_acc): only every eval_every steps
            do_eval = (step % eval_every == 0) or (step == n_steps)

            if do_eval:
                t_eval = time.time()
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
                # lightweight row: only train_loss and grad_norm; val/acc fields are null
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

            # reset interval accumulators
            t_train_s  = 0.0
            t_ckpt_s   = 0.0
            n_ckpts_in_interval = 0

    metrics_file.close()

    # wait for any in-flight background checkpoint write to finish
    if _ckpt_thread is not None and _ckpt_thread.is_alive():
        print("[timing] Waiting for background checkpoint write to finish...")
        _ckpt_thread.join()

    # save final weights
    _t = time.time()
    raw_model = getattr(model, '_orig_mod', model)
    torch.save(raw_model.state_dict(), model_path)
    print(f"\n[timing] Final model saved: {model_path}  ({time.time()-_t:.2f}s)")

    # loss landscape
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
    print(f"\nTraining complete. Total time: {total_min:.1f} min. Log: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument('--config', required=True,
                        help='Path to training config yaml (e.g. config/teacher_pretrain.yaml)')
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    main(args.config)
