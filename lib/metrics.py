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
    tmp_ckpt_dir: str,              # directory of float16 .pt checkpoints (pretrain only)
    eval_batch: Tuple[torch.Tensor, torch.Tensor],  # fixed eval batch
    device: str,
    log_dir: str,                   # where to save landscape.npz and pca_basis.npz
    grid_res: int = 31,             # grid resolution (grid_res x grid_res)
    alpha_range: Tuple[float,float] = (-1.0, 1.0),
    beta_range:  Tuple[float,float] = (-1.0, 1.0),
    loss_fn = None,                 # None -> CE; GRPO stages can pass custom loss
    pca_basis_path: Optional[str] = None,  # path to pca_basis.npz from an earlier stage;
                                           # if set, reuse those d1/d2 for a shared coordinate system
    precomputed_traj: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                    # (traj_alpha [N], traj_beta [N]) pre-projected by caller.
                                    # Used when save_ckpt=False: pretrain.py tracks dot-product
                                    # snapshots during training instead of saving full checkpoints.
):
    """
    Compute and save the loss landscape for one training stage.

    PCA directions (d1, d2):
      - If pca_basis_path is None (pretrain): compute d1, d2 via SVD on checkpoint
        deltas and save to log_dir/pca_basis.npz for downstream stages.
      - If pca_basis_path is set (post-training): load d1, d2 from that file so all
        stages share the same coordinate system.

    Trajectory:
      - save_ckpt=True  (pretrain): load checkpoint files from tmp_ckpt_dir, project
        each delta onto (d1, d2), auto-range the grid around the trajectory.
      - save_ckpt=False (post-training): caller passes precomputed_traj with dot-product
        snapshots already projected; no checkpoint files needed.

    Grid:
      Always computed: loss at theta* + alpha*d1 + beta*d2 for each (alpha, beta) in grid.
    """
    tmp_dir    = Path(tmp_ckpt_dir)
    ckpt_files = sorted(tmp_dir.glob("*.pt"))
    has_ckpts  = bool(ckpt_files)

    theta_star = get_flat_params(model)   # final parameters [P]
    P = len(theta_star)

    # ── Require at least one of: checkpoints (to compute PCA) or pca_basis_path ──
    has_pca_basis = pca_basis_path and Path(pca_basis_path).exists()
    if not has_ckpts and not has_pca_basis:
        print(f"  [landscape] No checkpoints in {tmp_ckpt_dir} and no pca_basis_path; skipping.")
        return

    # ── Load trajectory checkpoints (optional) ───────────────────────────────
    # Checkpoints are needed to: (a) compute PCA directions when pca_basis_path
    # is absent; (b) plot the training trajectory on the landscape.
    # When save_ckpt=False (post-training stages), neither is needed — the grid
    # itself only requires theta*, d1, d2.
    deltas: Optional[np.ndarray] = None
    if has_ckpts:
        print(f"  [landscape] Loading {len(ckpt_files)} checkpoints...")
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
        set_flat_params(model, theta_star)   # restore final params
        deltas = np.stack([t - theta_star for t in thetas], axis=0)  # [N, P]

    # ── PCA directions ──────────────────────────────────────────────────────
    if has_pca_basis:
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
        # compute fresh PCA via SVD on trajectory deltas; save for downstream stages
        print(f"  [landscape] Computing PCA ({deltas.shape[0]} x {P})...")
        _, _, Vt = np.linalg.svd(deltas, full_matrices=False)   # Vt: [min(N,P), P]
        d1 = Vt[0]                                                # [P], 1st principal direction
        d2 = Vt[1] if Vt.shape[0] > 1 else np.random.randn(P)   # [P]
        basis_out = Path(log_dir) / 'pca_basis.npz'
        basis_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(basis_out, d1=d1, d2=d2)
        print(f"  [landscape] PCA basis saved to {basis_out} (use as pca_basis_path in downstream configs)")

    # ── Trajectory ───────────────────────────────────────────────────────────
    def _auto_range(vals: np.ndarray, pad_ratio: float = 0.15) -> Tuple[float, float]:
        lo, hi = float(vals.min()), float(vals.max())
        span   = hi - lo if hi > lo else 1.0
        pad    = span * pad_ratio
        return lo - pad, hi + pad

    if precomputed_traj is not None:
        # post-training path: caller tracked dot-product snapshots, no checkpoint files
        traj_alpha, traj_beta = precomputed_traj   # [N], [N]
        alpha_range = _auto_range(traj_alpha)
        beta_range  = _auto_range(traj_beta)
        print(f"  [landscape] Trajectory (precomputed, {len(traj_alpha)} points) "
              f"alpha=[{traj_alpha.min():.2f},{traj_alpha.max():.2f}] "
              f"beta=[{traj_beta.min():.2f},{traj_beta.max():.2f}]")
    elif deltas is not None:
        # pretrain path: project checkpoint deltas onto PCA directions
        traj_alpha = deltas @ d1   # [N]
        traj_beta  = deltas @ d2   # [N]
        alpha_range = _auto_range(traj_alpha)
        beta_range  = _auto_range(traj_beta)
        print(f"  [landscape] Trajectory ({len(traj_alpha)} checkpoints) "
              f"alpha=[{traj_alpha.min():.2f},{traj_alpha.max():.2f}] "
              f"beta=[{traj_beta.min():.2f},{traj_beta.max():.2f}]")
    else:
        traj_alpha = np.empty(0, dtype=np.float32)
        traj_beta  = np.empty(0, dtype=np.float32)
        print(f"  [landscape] No trajectory; using config ranges alpha={alpha_range} beta={beta_range}")
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
        alpha_grid = alpha_vals,   # [grid_res]
        beta_grid  = beta_vals,    # [grid_res]
        Z          = Z,            # [grid_res, grid_res]
        traj_alpha = traj_alpha,   # [N] or empty when save_ckpt=False
        traj_beta  = traj_beta,    # [N] or empty when save_ckpt=False
    )
    print(f"  [landscape] Saved {out_path}")

    # Delete tmp checkpoints: traj_alpha/beta are now in landscape.npz.
    if has_ckpts:
        for f in ckpt_files:
            f.unlink()
        print(f"  [landscape] Tmp checkpoints deleted ({len(ckpt_files)} files, {tmp_dir})")
