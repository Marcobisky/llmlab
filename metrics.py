"""
metrics.py — 共享评估逻辑，被所有训练脚本 import。

功能：
  1. compute_metrics()  — 在固定 eval 集上算标量指标（val_loss、task_acc 等）
  2. save_landscape()   — 训练末一次性算 loss landscape，存 log/<stage>/landscape.npz
"""
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Token 工具（从 lang.py 获取词表映射）
# ─────────────────────────────────────────────────────────────────────────────

def _get_vocab():
    from lib.lang import VOCAB_COT, TOKEN2ID
    id2tok = {v: k for k, v in TOKEN2ID.items()}
    return TOKEN2ID, id2tok, TOKEN2ID.get('[EOS]', 31)


# ─────────────────────────────────────────────────────────────────────────────
# 贪心解码
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model,
    prompt_ids: List[int],   # 已经 token 化的 prompt，长度 L_p
    eos_id: int,
    max_new: int,
    device: str,
) -> List[int]:
    """
    从 prompt_ids 开始贪心解码，直到生成 [EOS] 或达到 max_new 步。
    返回新生成的 token ID 列表（不含 prompt，不含 EOS 之后的内容）。
    """
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    # ids: [1, L_p]

    generated = []
    context_len = model.pos_emb.num_embeddings  # 48

    for _ in range(max_new):
        if ids.shape[1] >= context_len:
            break
        logits = model(ids)                       # [1, L, V]
        next_id = logits[0, -1].argmax().item()   # 贪心取 argmax
        generated.append(next_id)
        if next_id == eos_id:
            break
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1
        )                                         # [1, L+1]

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# 任务正确率（按 depth 分）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_task_acc(
    model,
    eval_records: List[Dict],   # eval.jsonl 中的记录列表（已过滤为 stmt 类型）
    token2id: Dict[str, int],
    id2tok: Dict[int, str],
    eos_id: int,
    device: str,
    max_depth: int = 5,
) -> Tuple[float, List[float]]:
    """
    对 eval_records（stmt 类型）做贪心解码，用解释器判断正确性。
    返回 (overall_acc, acc_by_depth[0..max_depth])。

    判断逻辑：
      - prompt = '[BOS] EXPR'，model 应生成 '= RESULT [EOS]'
      - 从生成序列里提取 RESULT（'=' 之后到 '[EOS]' 之前的数字 token 拼接）
      - 与 record['result'] 比对
    """
    correct_by_depth = [0] * (max_depth + 1)
    total_by_depth   = [0] * (max_depth + 1)

    stmt_recs = [r for r in eval_records if r.get('type') == 'stmt']

    for rec in stmt_recs:
        depth = rec.get('depth', 0)
        if depth > max_depth:
            continue

        prompt_ids = [token2id[t] for t in rec['prompt'].split()
                      if t in token2id]
        generated  = greedy_decode(model, prompt_ids, eos_id,
                                   max_new=32, device=device)
        gen_toks   = [id2tok.get(i, '?') for i in generated]

        # 解析：'= t1 t2 ... [EOS]' → 拼出 result 字符串
        decoded_result = ''
        if gen_toks and gen_toks[0] == '=':
            parts = []
            for t in gen_toks[1:]:
                if t == '[EOS]':
                    break
                parts.append(t)
            decoded_result = ''.join(parts)  # 每个 part 是单字符数字 token

        total_by_depth[depth] += 1
        if decoded_result == rec['result']:
            correct_by_depth[depth] += 1

    acc_by_depth = [
        c / t if t > 0 else 0.0
        for c, t in zip(correct_by_depth, total_by_depth)
    ]
    total_c = sum(correct_by_depth)
    total_t = sum(total_by_depth)
    overall  = total_c / total_t if total_t > 0 else 0.0
    return overall, acc_by_depth


# ─────────────────────────────────────────────────────────────────────────────
# val_loss 计算
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_loss(
    model,
    eval_batch: Tuple[torch.Tensor, torch.Tensor],  # (input_ids, labels)
    device: str,
) -> float:
    """
    在 eval_batch 上计算 CE loss（不更新梯度）。
    eval_batch = (input_ids [B,T], labels [B,T])，labels 中 -100 位置被忽略。
    """
    model.eval()
    input_ids, labels = eval_batch
    input_ids = input_ids.to(device)
    labels    = labels.to(device)
    _, loss   = model(input_ids, labels)
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# 统一 compute_metrics 接口
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    model,
    eval_batch: Tuple[torch.Tensor, torch.Tensor],
    eval_records: List[Dict],
    device: str,
    step: int,
    train_loss: float,
    grad_norm: float,
    prev_flat_params: Optional[np.ndarray] = None,   # 上一步参数，算 param_step_norm
    # 可选（KD/OPD/GRPO 阶段才有意义）
    kl_teacher_prefix: Optional[float] = None,
    kl_student_prefix: Optional[float] = None,
    mean_reward: Optional[float] = None,
    kl_to_ref: Optional[float] = None,
) -> Tuple[Dict, np.ndarray]:
    """
    综合指标，返回 (metrics_dict, current_flat_params)。
    metrics_dict 结构与 README §6.2 完全对应，直接写入 metrics.jsonl。
    """
    token2id, id2tok, eos_id = _get_vocab()

    val_loss = compute_val_loss(model, eval_batch, device)
    task_acc, task_acc_by_depth = compute_task_acc(
        model, eval_records, token2id, id2tok, eos_id, device
    )

    # param_step_norm = ||θ_t − θ_{t-1}||
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
# 参数向量工具（供 landscape 使用）
# ─────────────────────────────────────────────────────────────────────────────

def get_flat_params(model) -> np.ndarray:
    """把模型所有参数展平成一个 float32 numpy 向量。"""
    return np.concatenate(
        [p.data.float().cpu().numpy().flatten() for p in model.parameters()]
    )


def set_flat_params(model, flat: np.ndarray):
    """把 flat 向量（float32）写回模型参数（in-place）。"""
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
    model,                          # 训练完毕的最终模型（theta*）
    tmp_ckpt_dir: str,              # 临时 checkpoint 目录（float16 .pt 文件）
    eval_batch: Tuple[torch.Tensor, torch.Tensor],  # 固定 eval batch
    device: str,
    log_dir: str,                   # landscape.npz 保存目录
    grid_res: int = 31,             # 网格分辨率（grid_res × grid_res）
    alpha_range: Tuple[float,float] = (-1.0, 1.0),
    beta_range:  Tuple[float,float] = (-1.0, 1.0),
    loss_fn = None,                 # None → CE；GRPO 阶段会传入自定义 loss
):
    """
    1. 加载 tmp_ckpt_dir 中所有 float16 checkpoint
    2. PCA 取前 2 个方向 d1, d2
    3. 在 theta* + α*d1 + β*d2 网格上计算 loss → Z [grid_res, grid_res]
    4. 投影各 checkpoint 到 (α, β) 平面 → 轨迹
    5. 存 log_dir/landscape.npz，删除 tmp checkpoint
    """
    tmp_dir = Path(tmp_ckpt_dir)
    ckpt_files = sorted(tmp_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"  [landscape] {tmp_ckpt_dir} 中无 checkpoint，跳过。")
        return

    print(f"  [landscape] 加载 {len(ckpt_files)} 个临时 checkpoint...")
    theta_star = get_flat_params(model)   # 最终参数，shape [P]
    P = len(theta_star)

    # 加载所有 checkpoint 并转为 float32 flat 向量
    # 注意：必须通过 load_state_dict 加载回模型再调 get_flat_params，
    # 而不是直接展平 state.values()。原因：tie_embedding 时 state_dict 里
    # tok_emb.weight 和 head.weight 是两个 key，但 model.parameters()
    # 对共享参数只计一次，两种展平方式维度不同会导致 broadcast 报错。
    thetas = []
    for f in ckpt_files:
        state = torch.load(f, map_location='cpu', weights_only=True)
        # float16 → float32，然后通过模型接口去重
        state_f32 = {k: v.float() for k, v in state.items()}
        model.load_state_dict(state_f32, strict=True)
        thetas.append(get_flat_params(model))

    # 计算相对于 theta_star 的偏移
    deltas = np.stack([t - theta_star for t in thetas], axis=0)  # [N, P]

    # PCA：SVD 取前 2 个右奇异向量作为主方向
    print(f"  [landscape] PCA ({deltas.shape[0]} x {P})...")
    # 用 float32 做 SVD；P 很大时用 full_matrices=False 节省内存
    _, _, Vt = np.linalg.svd(deltas, full_matrices=False)   # Vt: [min(N,P), P]
    d1 = Vt[0]  # [P]，第 1 主方向
    d2 = Vt[1] if Vt.shape[0] > 1 else np.random.randn(P)  # [P]

    # 投影各 checkpoint 到 (d1, d2) 平面（相对于 theta_star）
    traj_alpha = deltas @ d1  # [N]
    traj_beta  = deltas @ d2  # [N]

    # 构建网格，计算 loss
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_res)
    beta_vals  = np.linspace(beta_range[0],  beta_range[1],  grid_res)
    Z = np.zeros((grid_res, grid_res), dtype=np.float32)

    print(f"  [landscape] 计算 {grid_res}x{grid_res} 网格 loss...")
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

    # 恢复最终参数
    set_flat_params(model, theta_star)

    # 存 landscape.npz
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
    print(f"  [landscape] 已保存 {out_path}")

    # 删除临时 checkpoint
    shutil.rmtree(tmp_dir)
    print(f"  [landscape] 已删除临时目录 {tmp_dir}")
