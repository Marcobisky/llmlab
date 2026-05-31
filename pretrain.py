"""
pretrain.py — 预训练脚本，teacher 和 student 共用同一套代码。
通过 config yaml 的 data / model / train / output / logging 五个块控制所有行为。

用法：
    python pretrain.py                             # 使用顶部 CONFIG_YAML
    python pretrain.py config/teacher_pretrain.yaml

输出：
    model/<name>.pt          最终模型权重
    log/<stage>/config.json  超参记录
    log/<stage>/metrics.jsonl  每 log_every 步一行标量指标
    log/<stage>/landscape.npz  训练末 loss landscape（同时删除 tmp/ 下临时权重）
"""
import json
import math
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# ── 顶部配置（用户修改此处选择要运行的 stage） ──────────────────────────────
CONFIG_YAML = "config/teacher_pretrain.yaml"
# ─────────────────────────────────────────────────────────────────────────────

# 把项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent))
from lib.lang import VOCAB_COT, TOKEN2ID
from model   import build_model
from metrics import compute_metrics, save_landscape, get_flat_params

PAD_ID = TOKEN2ID['[EOS]']   # 用 [EOS] 做 padding token（label 处 -100 mask）
EOS_ID = TOKEN2ID['[EOS]']


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    从 .jsonl 读取样本，按 mode 决定 labels 掩码方式：
      mode='pretrain' → 全序列计 loss（prompt + target）
      mode='sft'      → 只对 target 部分计 loss（prompt 部分 label=-100）
    """
    def __init__(self, jsonl_path: str, context_len: int, mode: str = 'pretrain'):
        super().__init__()
        self.context_len = context_len
        self.mode        = mode

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        C   = self.context_len

        # Token 化 prompt 和 target（空格分隔 → token ID 列表）
        prompt_ids = [TOKEN2ID.get(t, 0) for t in rec['prompt'].split()]
        target_ids = [TOKEN2ID.get(t, 0) for t in rec['target'].split()]
        full_ids   = prompt_ids + target_ids  # [BOS] EXPR = RESULT [EOS] 等

        # 截断（通常不会超，data.py 已保证 ≤ context_len）
        full_ids = full_ids[:C]
        L = len(full_ids)

        # next-token prediction：input = full[:-1], label = full[1:]
        inp = full_ids[:-1]    # 长度 L-1
        lbl = full_ids[1:]     # 长度 L-1

        if self.mode == 'sft':
            # prompt 部分（前 len(prompt_ids)-1 个位置）label = -100
            n_prompt_in_inp = min(len(prompt_ids) - 1, len(inp))
            for i in range(n_prompt_in_inp):
                lbl[i] = -100

        # 右 padding 到 C-1
        pad_len = (C - 1) - len(inp)
        inp = inp + [PAD_ID] * pad_len
        lbl = lbl + [-100]   * pad_len

        return (
            torch.tensor(inp, dtype=torch.long),  # [C-1]
            torch.tensor(lbl, dtype=torch.long),  # [C-1]
        )


def make_loader(
    jsonl_path: str, context_len: int, batch_size: int, mode: str = 'pretrain'
) -> DataLoader:
    ds = TokenDataset(jsonl_path, context_len, mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      drop_last=True, num_workers=0, pin_memory=False)


def make_eval_batch(
    eval_records: List[Dict], context_len: int, max_samples: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 eval 记录转成固定 batch（不 shuffle，用于稳定指标计算）。"""
    inp_list, lbl_list = [], []
    for rec in eval_records[:max_samples]:
        full_ids = (
            [TOKEN2ID.get(t, 0) for t in rec['prompt'].split()] +
            [TOKEN2ID.get(t, 0) for t in rec['target'].split()]
        )[:context_len]
        L = len(full_ids)
        inp = full_ids[:-1]
        lbl = full_ids[1:]
        pad = (context_len - 1) - len(inp)
        inp = inp + [PAD_ID] * pad
        lbl = lbl + [-100]   * pad
        inp_list.append(inp)
        lbl_list.append(lbl)

    inp_t = torch.tensor(inp_list, dtype=torch.long, device=device)  # [B, C-1]
    lbl_t = torch.tensor(lbl_list, dtype=torch.long, device=device)  # [B, C-1]
    return inp_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, n_steps: int, lr: float, warmup_steps: int,
           schedule: str = 'cosine') -> float:
    """
    Warmup（线性 0→lr）+ 主阶段（cosine 衰减 lr→0 或 constant）。
    step 从 1 开始计数。
    """
    if step <= warmup_steps:
        return lr * step / max(1, warmup_steps)
    if schedule == 'constant':
        return lr
    # cosine decay
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Tmp Checkpoint 保存
# ─────────────────────────────────────────────────────────────────────────────

def save_tmp_ckpt(model: nn.Module, tmp_dir: str, step: int, dtype: str = 'float16'):
    """以 float16 保存当前参数到 tmp_dir/<step>.pt。"""
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    state = {k: v.half() if dtype == 'float16' else v.clone()
             for k, v in model.state_dict().items()}
    torch.save(state, Path(tmp_dir) / f"{step:06d}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str):
    # ── 读配置 ─────────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['train']
    out_cfg   = cfg['output']
    log_cfg   = cfg['logging']

    device     = train_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠ CUDA 不可用，回退到 CPU")

    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    context_len   = model_cfg['context_len']   # 48
    batch_size    = train_cfg['batch_size']
    n_steps       = train_cfg['n_steps']
    warmup_steps  = train_cfg['warmup_steps']
    lr            = train_cfg['lr']
    lr_schedule   = train_cfg.get('lr_schedule', 'cosine')
    weight_decay  = train_cfg.get('weight_decay', 0.1)
    grad_clip     = train_cfg.get('grad_clip', 1.0)
    log_every     = log_cfg['log_every']
    eval_bs       = log_cfg['eval_batch_size']
    n_traj_ckpt  = log_cfg['n_traj_ckpt']
    ckpt_dtype    = log_cfg.get('ckpt_dtype', 'float16')
    tmp_dir       = log_cfg['tmp_ckpt_dir']
    log_dir       = out_cfg['log_dir']
    model_path    = out_cfg['model_path']

    # tmp checkpoint 保存间隔
    ckpt_interval = max(1, n_steps // n_traj_ckpt)

    # ── 目录准备 ────────────────────────────────────────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # ── 保存 config.json ────────────────────────────────────────────────────
    with open(Path(log_dir) / "config.json", 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ── 构建模型 ────────────────────────────────────────────────────────────
    print(f"构建模型（context_len={context_len}）...")
    model = build_model(model_cfg).to(device)

    # 若指定 base_model_path（SFT / GRPO 等），从此处加载
    base_path = train_cfg.get('base_model_path')
    if base_path and Path(base_path).exists():
        print(f"加载 base model: {base_path}")
        model.load_state_dict(torch.load(base_path, map_location=device,
                                          weights_only=True))

    # ── DataLoader ──────────────────────────────────────────────────────────
    data_path = data_cfg['path']
    print(f"加载训练数据: {data_path}")
    loader = make_loader(data_path, context_len, batch_size, mode='pretrain')
    loader_iter = cycle(loader)   # 无限循环，允许多 epoch

    # ── Eval 数据 ────────────────────────────────────────────────────────────
    eval_data_path = log_cfg.get('eval_data_path', '')
    eval_records: List[Dict] = []
    if Path(eval_data_path).exists():
        with open(eval_data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
        print(f"eval 数据: {len(eval_records)} 条  ({eval_data_path})")
    else:
        # fallback：从训练集取前 eval_bs 条
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
                if len(eval_records) >= eval_bs:
                    break
        print(f"⚠ eval 文件不存在，从训练集取 {len(eval_records)} 条作 fallback")

    eval_batch = make_eval_batch(eval_records, context_len,
                                 max_samples=eval_bs, device=device)
    # eval_batch: (input_ids [B, C-1], labels [B, C-1])

    # ── 优化器 ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    metrics_path = Path(log_dir) / "metrics.jsonl"
    metrics_file = open(metrics_path, 'w')

    prev_flat: Optional[np.ndarray] = None
    grad_norm_val = 0.0

    print(f"\n{'='*55}")
    print(f"开始预训练  n_steps={n_steps}  device={device}")
    print(f"{'='*55}\n")

    for step in range(1, n_steps + 1):
        model.train()

        # ── 调整 LR ──────────────────────────────────────────────────────
        cur_lr = get_lr(step, n_steps, lr, warmup_steps, lr_schedule)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        # ── 前向 + 反向 ───────────────────────────────────────────────────
        inp_ids, labels = next(loader_iter)
        # inp_ids: [B, C-1]  labels: [B, C-1]
        inp_ids = inp_ids.to(device)
        labels  = labels.to(device)

        _, loss = model(inp_ids, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪 + 记录 grad_norm
        grad_norm_val = nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

        optimizer.step()

        # ── 保存 tmp checkpoint ────────────────────────────────────────────
        if step % ckpt_interval == 0 or step == 1:
            save_tmp_ckpt(model, tmp_dir, step, ckpt_dtype)

        # ── 指标记录 ──────────────────────────────────────────────────────
        if step % log_every == 0 or step == n_steps:
            m, prev_flat = compute_metrics(
                model        = model,
                eval_batch   = eval_batch,
                eval_records = eval_records[:eval_bs],
                device       = device,
                step         = step,
                train_loss   = loss.item(),
                grad_norm    = grad_norm_val,
                prev_flat_params = prev_flat,
            )
            metrics_file.write(json.dumps(m) + '\n')
            metrics_file.flush()

            print(
                f"step {step:5d}/{n_steps} | lr={cur_lr:.2e} | "
                f"loss={m['train_loss']:.4f} | val={m['val_loss']:.4f} | "
                f"acc={m['task_acc']:.3f} | gnorm={m['grad_norm']:.3f}"
            )

    metrics_file.close()

    # ── 保存最终模型权重 ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存: {model_path}")

    # ── 计算 Loss Landscape ──────────────────────────────────────────────────
    print("\n计算 Loss Landscape...")
    save_landscape(
        model        = model,
        tmp_ckpt_dir = tmp_dir,
        eval_batch   = eval_batch,
        device       = device,
        log_dir      = log_dir,
        grid_res     = log_cfg.get('grid_res', 31),
        alpha_range  = tuple(log_cfg.get('landscape_alpha_range', [-1.0, 1.0])),
        beta_range   = tuple(log_cfg.get('landscape_beta_range',  [-1.0, 1.0])),
    )

    print(f"\n训练完成。日志: {log_dir}")


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_YAML
    os.chdir(Path(__file__).parent)  # 确保相对路径从项目根目录解析
    main(config_path)
