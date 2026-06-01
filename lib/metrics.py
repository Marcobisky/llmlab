"""
lib/metrics.py — Shared evaluation logic, imported by all training scripts.

Functions:
  1. compute_metrics()  — compute scalar metrics on a fixed eval set (val_loss, task_acc, etc.)
  2. save_landscape()   — compute loss landscape at the end of training, save as log/<stage>/landscape.npz
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Token utilities (get vocab mapping from lang.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_vocab():
    from lib.lang import TOKEN2ID
    id2tok = {v: k for k, v in TOKEN2ID.items()}
    return TOKEN2ID, id2tok, TOKEN2ID.get('[EOS]', 31)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy decoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model,
    prompt_ids: List[int],   # tokenized prompt of length L_p
    eos_id: int,
    max_new: int,
    device: str,
) -> List[int]:
    """
    Greedy decode starting from prompt_ids until [EOS] or max_new steps.
    Returns newly generated token IDs (excluding prompt and tokens after EOS).
    """
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    # ids: [1, L_p]

    generated = []
    context_len = model.pos_emb.num_embeddings  # e.g. 96

    for _ in range(max_new):
        if ids.shape[1] >= context_len:
            break
        logits = model(ids)                       # [1, L, V]
        next_id = logits[0, -1].argmax().item()   # greedy argmax
        generated.append(next_id)
        if next_id == eos_id:
            break
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1
        )                                         # [1, L+1]

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Task accuracy (broken down by depth)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_result(toks: List[str]) -> str:
    """
    Extract the final numeric result from a generated token sequence.
    Supports two formats:

    Direct format : '= 1 2 3 [EOS]'
    CoT format    : '<think> ... </think> = 1 2 3 [EOS]'

    Strategy: find the last '=' in the sequence, collect tokens until '[EOS]'.
    The last '=' is always the one in '= RESULT [EOS]' (CoT trace '='s are
    all before '</think>').
    """
    last_eq = max((i for i, t in enumerate(toks) if t == '='), default=-1)
    if last_eq < 0:
        return ''
    parts = []
    for t in toks[last_eq + 1:]:
        if t == '[EOS]':
            break
        parts.append(t)
    return ''.join(parts)


@torch.no_grad()
def compute_task_acc(
    model,
    eval_records: List[Dict],
    token2id: Dict[str, int],
    id2tok: Dict[int, str],
    eos_id: int,
    device: str,
    max_depth: int = 5,
) -> Tuple[float, List[float]]:
    """
    Batched greedy decoding grouped by prompt length; same-length prompts are
    stacked into a batch (no padding needed), greatly improving GPU utilization.

    Old approach: N samples x max_new steps x batch=1 = N*max_new sequential forwards
    New approach: ~K groups x max_new steps x batch=B_k, total forwards ~ K*max_new
    """
    from collections import defaultdict

    model.eval()
    context_len = model.pos_emb.num_embeddings  # e.g. 96

    stmt_recs = [r for r in eval_records
                 if r.get('type') == 'stmt' and r.get('depth', 0) <= max_depth]
    if not stmt_recs:
        return 0.0, [0.0] * (max_depth + 1)

    # group by prompt token count (same length -> stack directly, no padding)
    by_len: Dict[int, list] = defaultdict(list)
    for rec in stmt_recs:
        p = [token2id[t] for t in rec['prompt'].split() if t in token2id]
        by_len[len(p)].append((p, rec))

    correct_by_depth = [0] * (max_depth + 1)
    total_by_depth   = [0] * (max_depth + 1)

    for prompt_len, items in by_len.items():
        batch_p   = [x[0] for x in items]   # List[List[int]], each of length prompt_len
        batch_rec = [x[1] for x in items]
        B = len(batch_p)

        # [B, prompt_len] — same length, direct stack, no padding
        ids  = torch.tensor(batch_p, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        gen  = [[] for _ in range(B)]       # gen[b] = list of generated token ids

        max_new = context_len - prompt_len  # use up all remaining context window
        for _ in range(max_new):
            if done.all() or ids.shape[1] >= context_len:
                break
            logits = model(ids)             # [B, T, V]
            nxt    = logits[:, -1].argmax(dim=-1)   # [B] — greedy argmax
            ids    = torch.cat([ids, nxt.unsqueeze(1)], dim=1)  # [B, T+1]

            eos_hit = (nxt == eos_id)
            for b in range(B):
                if not done[b]:
                    gen[b].append(nxt[b].item())
                    if eos_hit[b]:
                        done[b] = True

        # parse each output (compatible with direct and CoT formats)
        for b, rec in enumerate(batch_rec):
            depth  = rec.get('depth', 0)
            toks   = [id2tok.get(i, '?') for i in gen[b]]
            result = _extract_result(toks)
            total_by_depth[depth]   += 1
            correct_by_depth[depth] += int(result == rec['result'])

    acc_by_depth = [c / t if t > 0 else 0.0
                    for c, t in zip(correct_by_depth, total_by_depth)]
    total_t = sum(total_by_depth)
    overall = sum(correct_by_depth) / total_t if total_t > 0 else 0.0
    return overall, acc_by_depth


# ─────────────────────────────────────────────────────────────────────────────
# val_loss computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_loss(
    model,
    eval_batch: Tuple[torch.Tensor, torch.Tensor],  # (input_ids, labels)
    device: str,
) -> float:
    """
    Compute CE loss on eval_batch (no gradient update).
    eval_batch = (input_ids [B,T], labels [B,T]), -100 positions are ignored.
    """
    model.eval()
    input_ids, labels = eval_batch
    input_ids = input_ids.to(device)
    labels    = labels.to(device)
    _, loss   = model(input_ids, labels)
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Unified compute_metrics interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    model,
    eval_batch: Tuple[torch.Tensor, torch.Tensor],
    eval_records: List[Dict],
    device: str,
    step: int,
    train_loss: float,
    grad_norm: float,
    prev_flat_params: Optional[np.ndarray] = None,   # previous step params for param_step_norm
    # optional (only meaningful in KD/OPD/GRPO stages)
    kl_teacher_prefix: Optional[float] = None,
    kl_student_prefix: Optional[float] = None,
    mean_reward: Optional[float] = None,
    kl_to_ref: Optional[float] = None,
) -> Tuple[Dict, np.ndarray]:
    """
    Compute all metrics and return (metrics_dict, current_flat_params).
    metrics_dict is written directly to metrics.jsonl.
    """
    token2id, id2tok, eos_id = _get_vocab()

    val_loss = compute_val_loss(model, eval_batch, device)
    task_acc, task_acc_by_depth = compute_task_acc(
        model, eval_records, token2id, id2tok, eos_id, device
    )

    # param_step_norm = ||theta_t - theta_{t-1}||
    cur_flat = get_flat_params(model)
    param_step_norm = float(np.linalg.norm(cur_flat - prev_flat_params)) \
        if prev_flat_params is not None else 0.0

    m = {
        "step":               step,
        "train_loss":         round(train_loss, 6),
        "val_loss":           round(val_loss, 6),
        "task_acc":           round(task_acc, 6),
        "task_acc_by_depth":  [round(a, 6) for a in task_acc_by_depth],
        "kl_teacher_prefix":  kl_teacher_prefix,
        "kl_student_prefix":  kl_student_prefix,
        "mean_reward":        mean_reward,
        "kl_to_ref":          kl_to_ref,
        "grad_norm":          round(grad_norm, 6),
        "param_step_norm":    round(param_step_norm, 6),
    }
    return m, cur_flat


# ─────────────────────────────────────────────────────────────────────────────
# Parameter vector utilities (used by landscape)
# ─────────────────────────────────────────────────────────────────────────────

def get_flat_params(model) -> np.ndarray:
    """Flatten all model parameters into a single float32 numpy vector."""
    return np.concatenate(
        [p.data.float().cpu().numpy().flatten() for p in model.parameters()]
    )


def set_flat_params(model, flat: np.ndarray):
    """Write flat vector (float32) back into model parameters (in-place)."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(
            torch.tensor(flat[idx: idx + n], dtype=torch.float32).view(p.shape)
        )
        idx += n


# ─────────────────────────────────────────────────────────────────────────────
# Loss Landscape
# ─────────────────────────────────────────────────────────────────────────────

def save_landscape(
    model,                          # final trained model (theta*)
    tmp_ckpt_dir: str,              # directory of float16 .pt checkpoints (kept after training)
    eval_batch: Tuple[torch.Tensor, torch.Tensor],  # fixed eval batch
    device: str,
    log_dir: str,                   # where to save landscape.npz and pca_basis.npz
    grid_res: int = 31,             # grid resolution (grid_res x grid_res)
    alpha_range: Tuple[float,float] = (-1.0, 1.0),
    beta_range:  Tuple[float,float] = (-1.0, 1.0),
    loss_fn = None,                 # None -> CE; GRPO stages can pass custom loss
    pca_basis_path: Optional[str] = None,  # path to pca_basis.npz from an earlier stage;
                                           # if set, reuse those d1/d2 for a shared coordinate system
):
    """
    Compute and save the loss landscape for one training stage.

    PCA directions (d1, d2):
      - If pca_basis_path is None (base stage, e.g. pretrain):
          compute d1, d2 via SVD on the checkpoint deltas and save them to
          log_dir/pca_basis.npz so downstream stages can reuse the same axes.
      - If pca_basis_path points to an existing pca_basis.npz (downstream stages):
          load d1, d2 from that file. The trajectory is projected onto the same
          coordinate system, making landscapes from different stages directly comparable.

    Steps:
      1. Load all float16 checkpoints from tmp_ckpt_dir.
      2. Obtain d1, d2 (compute or load, see above).
      3. Project each checkpoint to (alpha, beta) relative to theta* -> trajectory.
      4. Evaluate loss on theta* + alpha*d1 + beta*d2 grid -> Z [grid_res, grid_res].
      5. Save log_dir/landscape.npz. Checkpoints are retained for further analysis.
    """
    tmp_dir = Path(tmp_ckpt_dir)
    ckpt_files = sorted(tmp_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"  [landscape] No checkpoints in {tmp_ckpt_dir}, skipping.")
        return

    print(f"  [landscape] Loading {len(ckpt_files)} checkpoints...")
    theta_star = get_flat_params(model)   # final parameters, shape [P]
    P = len(theta_star)

    # Load all checkpoints as float32 flat vectors.
    # Must go through load_state_dict -> get_flat_params to handle tied embeddings:
    # state_dict has separate keys for tok_emb.weight and head.weight, but
    # model.parameters() counts shared params only once — direct flattening
    # of state.values() would mismatch dimensions.
    thetas = []
    for f in ckpt_files:
        state = torch.load(f, map_location='cpu', weights_only=True)
        state_f32 = {k: v.float() for k, v in state.items()}
        model.load_state_dict(state_f32, strict=True)
        thetas.append(get_flat_params(model))

    # restore final parameters before we start modifying them for the grid
    set_flat_params(model, theta_star)

    # compute offsets relative to theta_star
    deltas = np.stack([t - theta_star for t in thetas], axis=0)  # [N, P]

    # ── PCA directions ──────────────────────────────────────────────────────
    if pca_basis_path and Path(pca_basis_path).exists():
        # reuse pre-computed basis from an earlier stage (e.g. pretrain)
        with np.load(pca_basis_path) as npf:
            d1 = npf['d1']   # [P]
            d2 = npf['d2']   # [P]
        print(f"  [landscape] Reusing PCA basis from {pca_basis_path}")
        if d1.shape[0] != P:
            raise ValueError(
                f"PCA basis dimension {d1.shape[0]} != model param count {P}. "
                "Only reuse pca_basis across stages with identical architecture."
            )
    else:
        # compute fresh PCA via SVD and save for downstream stages
        print(f"  [landscape] Computing PCA ({deltas.shape[0]} x {P})...")
        _, _, Vt = np.linalg.svd(deltas, full_matrices=False)   # Vt: [min(N,P), P]
        d1 = Vt[0]                                                # [P], 1st principal direction
        d2 = Vt[1] if Vt.shape[0] > 1 else np.random.randn(P)   # [P]
        basis_out = Path(log_dir) / 'pca_basis.npz'
        basis_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(basis_out, d1=d1, d2=d2)
        print(f"  [landscape] PCA basis saved to {basis_out} (use as pca_basis_path in downstream configs)")

    # ── Project trajectory ──────────────────────────────────────────────────
    traj_alpha = deltas @ d1  # [N]
    traj_beta  = deltas @ d2  # [N]

    # auto-set grid range to cover the full trajectory + 15% margin
    def _auto_range(vals: np.ndarray, pad_ratio: float = 0.15) -> Tuple[float, float]:
        lo, hi  = float(vals.min()), float(vals.max())
        span    = hi - lo if hi > lo else 1.0
        pad     = span * pad_ratio
        return lo - pad, hi + pad

    alpha_range = _auto_range(traj_alpha)
    beta_range  = _auto_range(traj_beta)
    print(f"  [landscape] Trajectory alpha=[{traj_alpha.min():.2f},{traj_alpha.max():.2f}] "
          f"beta=[{traj_beta.min():.2f},{traj_beta.max():.2f}]")
    print(f"  [landscape] Grid range  alpha={alpha_range}  beta={beta_range}")

    # ── Grid loss computation ───────────────────────────────────────────────
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_res)
    beta_vals  = np.linspace(beta_range[0],  beta_range[1],  grid_res)
    Z = np.zeros((grid_res, grid_res), dtype=np.float32)

    print(f"  [landscape] Computing {grid_res}x{grid_res} grid loss...")
    model.eval()
    input_ids, labels = eval_batch
    input_ids = input_ids.to(device)
    labels    = labels.to(device)

    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            theta_grid = theta_star + alpha * d1 + beta * d2
            set_flat_params(model, theta_grid)
            with torch.no_grad():
                if loss_fn is None:
                    _, loss = model(input_ids, labels)
                    Z[i, j] = loss.item()
                else:
                    Z[i, j] = loss_fn(model, input_ids, labels)

    # restore final parameters
    set_flat_params(model, theta_star)

    # ── Save landscape.npz ──────────────────────────────────────────────────
    out_path = Path(log_dir) / "landscape.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        alpha_grid = alpha_vals,     # [grid_res]
        beta_grid  = beta_vals,      # [grid_res]
        Z          = Z,              # [grid_res, grid_res]
        traj_alpha = traj_alpha,     # [N]
        traj_beta  = traj_beta,      # [N]
    )
    print(f"  [landscape] Saved {out_path}")
    print(f"  [landscape] Checkpoints retained in {tmp_dir}")
