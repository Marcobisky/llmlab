"""
pretrain.py — 预训练脚本，teacher 和 student 共用同一套代码。
通过 config yaml 的 data / model / train / output / logging 五个块控制所有行为。

用法：
    python pretrain.py                             # 使用顶部 CONFIG_YAML
    python pretrain.py config/teacher_pretrain.yaml

输出：
    model/<name>.pt            最终模型权重
    log/<stage>/config.json    超参记录
    log/<stage>/metrics.jsonl  每 log_every 步一行标量指标
    log/<stage>/landscape.npz  训练末 loss landscape（同时删除 tmp/ 下临时权重）

GPU 利用率优化：
    1. GPUDataBuffer — 把整个数据集 tokenize 后一次性放到 GPU，
       采样 = torch.randint 索引，零 CPU-GPU 数据传输开销。
    2. AMP (torch.amp.autocast) — float16 前向/反向，tensor core 加速。
    3. torch.compile (PyTorch ≥ 2.0) — 图编译，进一步提升吞吐。
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# ── 顶部配置（修改此处选择要运行的 stage） ──────────────────────────────────
CONFIG_YAML = "config/teacher_pretrain.yaml"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang import TOKEN2ID
from model   import build_model
from metrics import compute_metrics, save_landscape

PAD_ID = TOKEN2ID['[EOS]']   # padding token（labels 处用 -100 mask）
EOS_ID = TOKEN2ID['[EOS]']


# ─────────────────────────────────────────────────────────────────────────────
# GPU 数据预加载（核心加速手段）
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_record(rec: Dict, context_len: int, mode: str) -> Tuple[List[int], List[int]]:
    """
    单条 jsonl 记录 → (inp, lbl) 两个 int 列表，已右 padding 到 context_len-1。
    mode='pretrain': 全序列计 loss；mode='sft': prompt 部分 label=-100。
    """
    C = context_len
    prompt_ids = [TOKEN2ID.get(t, 0) for t in rec['prompt'].split()]
    target_ids = [TOKEN2ID.get(t, 0) for t in rec['target'].split()]
    full_ids   = (prompt_ids + target_ids)[:C]   # 截断（通常不触发）

    inp = full_ids[:-1]   # [BOS] EXPR = RESULT      长度 L-1
    lbl = full_ids[1:]    # EXPR = RESULT [EOS]       长度 L-1

    if mode == 'sft':
        # prompt 对应的 input 位置不计 loss（mask 掉）
        n_mask = min(len(prompt_ids) - 1, len(inp))
        for i in range(n_mask):
            lbl[i] = -100

    pad = (C - 1) - len(inp)
    inp += [PAD_ID] * pad
    lbl += [-100]   * pad
    return inp, lbl


class GPUDataBuffer:
    """
    把整个训练集一次性加载到 GPU 显存。

    为什么这样做：
      - 数据集很小（50k × 47 tokens × int64 ≈ 18.8 MB），完全放得下。
      - 采样变成纯 GPU 操作：torch.randint → tensor 索引，
        消除 DataLoader 的 Python 开销、CPU→GPU 传输、worker 同步等瓶颈。
      - 实测可把 GPU 利用率从 ~17% 提升到 ~85%+。

    inp:  [N, C-1]  int64，token IDs（padding = PAD_ID）
    lbl:  [N, C-1]  int64，labels（padding = -100）
    """
    def __init__(self, jsonl_path: str, context_len: int, mode: str, device: str):
        print(f"  预加载数据集到 {device} ({jsonl_path}) ...")
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

        # 一次性转 GPU tensor
        self.inp = torch.tensor(all_inp, dtype=torch.long, device=device)  # [N, C-1]
        self.lbl = torch.tensor(all_lbl, dtype=torch.long, device=device)  # [N, C-1]
        self.N   = self.inp.shape[0]

        mb = self.inp.numel() * 2 * 8 / 1e6   # inp + lbl, int64
        print(f"  已加载 {self.N} 条，显存占用 ≈ {mb:.1f} MB，耗时 {time.time()-t0:.1f}s")

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机采 batch_size 条（有放回），纯 GPU 操作。
        返回 (inp [B, C-1], lbl [B, C-1])，已在 GPU 上。
        """
        idx = torch.randint(self.N, (batch_size,), device=self.inp.device)
        return self.inp[idx], self.lbl[idx]


def make_eval_batch(
    eval_records: List[Dict], context_len: int, max_samples: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 eval 记录转成固定 batch（不 shuffle，用于稳定指标计算）。"""
    all_inp, all_lbl = [], []
    for rec in eval_records[:max_samples]:
        inp, lbl = _tokenize_record(rec, context_len, mode='pretrain')
        all_inp.append(inp)
        all_lbl.append(lbl)
    inp_t = torch.tensor(all_inp, dtype=torch.long, device=device)  # [B, C-1]
    lbl_t = torch.tensor(all_lbl, dtype=torch.long, device=device)  # [B, C-1]
    return inp_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, n_steps: int, lr: float, warmup_steps: int,
           schedule: str = 'cosine') -> float:
    """
    Warmup（线性 0→lr）+ cosine 衰减到 lr_min（默认 lr/10）。
    step 从 1 开始。

    改进：cosine 衰减到 lr/10 而非 0，避免训练末期 LR 砸地板、
    模型更新完全停止（原来 3000 步末 LR ≈ 2.7e-6，导致 loss 冻住）。
    """
    if step <= warmup_steps:
        return lr * step / max(1, warmup_steps)
    if schedule == 'constant':
        return lr
    # cosine decay：lr → lr * min_ratio
    min_ratio = 0.1
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_ratio + (1 - min_ratio) * cosine)


# ─────────────────────────────────────────────────────────────────────────────
# Tmp Checkpoint 保存
# ─────────────────────────────────────────────────────────────────────────────

def save_tmp_ckpt(model: nn.Module, tmp_dir: str, step: int, dtype: str = 'float16'):
    """以 float16 保存当前参数到 tmp_dir/<step>.pt（landscape 用）。"""
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    # 若 model 被 torch.compile 包装，通过 _orig_mod 取原始模型
    raw = getattr(model, '_orig_mod', model)
    state = {k: v.half() if dtype == 'float16' else v.clone()
             for k, v in raw.state_dict().items()}
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

    device = train_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠ CUDA 不可用，回退到 CPU")

    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    context_len  = model_cfg['context_len']   # 48
    batch_size   = train_cfg['batch_size']
    n_steps      = train_cfg['n_steps']
    warmup_steps = train_cfg['warmup_steps']
    lr           = train_cfg['lr']
    lr_schedule  = train_cfg.get('lr_schedule', 'cosine')
    weight_decay = train_cfg.get('weight_decay', 0.1)
    grad_clip    = train_cfg.get('grad_clip', 2.0)
    use_amp      = train_cfg.get('amp', True) and device == 'cuda'
    log_every    = log_cfg['log_every']
    eval_bs      = log_cfg['eval_batch_size']
    n_traj_ckpt = log_cfg['n_traj_ckpt']
    ckpt_dtype   = log_cfg.get('ckpt_dtype', 'float16')
    tmp_dir      = log_cfg['tmp_ckpt_dir']
    log_dir      = out_cfg['log_dir']
    model_path   = out_cfg['model_path']

    ckpt_interval = max(1, n_steps // n_traj_ckpt)

    # ── 目录准备 ────────────────────────────────────────────────────────────
    for d in [log_dir, tmp_dir, str(Path(model_path).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)

    with open(Path(log_dir) / "config.json", 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ── 构建模型 ────────────────────────────────────────────────────────────
    print(f"构建模型（context_len={context_len}）...")
    model = build_model(model_cfg).to(device)

    base_path = train_cfg.get('base_model_path')
    if base_path and Path(base_path).exists():
        print(f"加载 base model: {base_path}")
        model.load_state_dict(
            torch.load(base_path, map_location=device, weights_only=True)
        )

    # torch.compile：若可用则启用，对小模型约有 30-50% 额外提速
    if train_cfg.get('compile', True) and hasattr(torch, 'compile'):
        print("  torch.compile() 编译中（首次 step 会慢）...")
        model = torch.compile(model)

    # ── GPU 数据预加载 ───────────────────────────────────────────────────────
    data_path = data_cfg['path']
    buf = GPUDataBuffer(data_path, context_len, mode='pretrain', device=device)

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
        # fallback：复用训练集前 eval_bs 条
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
                if len(eval_records) >= eval_bs:
                    break
        print(f"⚠ eval 文件不存在，从训练集借用 {len(eval_records)} 条")

    eval_batch = make_eval_batch(eval_records, context_len, eval_bs, device)
    # eval_batch: (inp [B, C-1], lbl [B, C-1])，固定不变

    # ── 优化器 + AMP ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    metrics_path = Path(log_dir) / "metrics.jsonl"
    metrics_file = open(metrics_path, 'w')

    prev_flat: Optional[np.ndarray] = None
    grad_norm_val = 0.0
    t_start = time.time()

    n_epoch_steps = max(1, buf.N // batch_size)  # 每 epoch 步数（仅用于显示）
    print(f"\n{'='*60}")
    print(f"开始预训练  n_steps={n_steps}  batch={batch_size}  device={device}")
    print(f"  数据量={buf.N}  每 epoch≈{n_epoch_steps} steps  "
          f"总≈{n_steps/n_epoch_steps:.1f} epochs")
    print(f"  AMP={use_amp}  grad_clip={grad_clip}  lr={lr:.2e}")
    print(f"{'='*60}\n")

    for step in range(1, n_steps + 1):
        model.train()

        cur_lr = get_lr(step, n_steps, lr, warmup_steps, lr_schedule)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        # ── 采样 + 前向（AMP） ───────────────────────────────────────────
        inp_ids, labels = buf.sample(batch_size)
        # inp_ids: [B=batch_size, T=C-1=47]
        # labels:  [B, T]，-100 位置被 CE 忽略

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                                 enabled=use_amp):
            _, loss = model(inp_ids, labels)

        scaler.scale(loss).backward()

        # unscale 后再裁剪，保证 grad_norm 是真实值
        scaler.unscale_(optimizer)
        grad_norm_val = nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip
        ).item()

        scaler.step(optimizer)
        scaler.update()

        # ── 保存 tmp checkpoint ─────────────────────────────────────────
        if step % ckpt_interval == 0 or step == 1:
            save_tmp_ckpt(model, tmp_dir, step, ckpt_dtype)

        # ── 指标记录 ────────────────────────────────────────────────────
        if step % log_every == 0 or step == n_steps:
            # 计算吞吐：tokens/s
            elapsed   = time.time() - t_start
            tps       = step * batch_size * (context_len - 1) / elapsed

            # 取原始模型（compile 包装后需通过 _orig_mod）
            raw_model = getattr(model, '_orig_mod', model)
            m, prev_flat = compute_metrics(
                model            = raw_model,
                eval_batch       = eval_batch,
                eval_records     = eval_records[:eval_bs],
                device           = device,
                step             = step,
                train_loss       = loss.item(),
                grad_norm        = grad_norm_val,
                prev_flat_params = prev_flat,
            )
            metrics_file.write(json.dumps(m) + '\n')
            metrics_file.flush()

            print(
                f"step {step:6d}/{n_steps} | lr={cur_lr:.2e} | "
                f"loss={m['train_loss']:.4f} | val={m['val_loss']:.4f} | "
                f"acc={m['task_acc']:.3f} | gnorm={m['grad_norm']:.2f} | "
                f"{tps/1000:.1f}k tok/s"
            )

    metrics_file.close()

    # ── 保存最终权重 ────────────────────────────────────────────────────────
    raw_model = getattr(model, '_orig_mod', model)
    torch.save(raw_model.state_dict(), model_path)
    print(f"\n模型已保存: {model_path}")

    # ── Loss Landscape ──────────────────────────────────────────────────────
    print("\n计算 Loss Landscape...")
    save_landscape(
        model        = raw_model,
        tmp_ckpt_dir = tmp_dir,
        eval_batch   = eval_batch,
        device       = device,
        log_dir      = log_dir,
        grid_res     = log_cfg.get('grid_res', 31),
        alpha_range  = tuple(log_cfg.get('landscape_alpha_range', [-1.0, 1.0])),
        beta_range   = tuple(log_cfg.get('landscape_beta_range',  [-1.0, 1.0])),
    )

    total_min = (time.time() - t_start) / 60
    print(f"\n训练完成，总耗时 {total_min:.1f} min。日志: {log_dir}")


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_YAML
    os.chdir(Path(__file__).parent)
    main(config_path)
