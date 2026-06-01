"""
visualize_llm.py — Transformer attention 可解释性可视化。

给定表达式，先自回归生成完整序列（含 CoT），
再用 forward hook 提取各层 attention weights，
输出 2×4 布局热力图：

  [ Head 0 ]  [ Head 1 ]  [ Head 2 ]  [ Mean (all heads) ]
  [ Head 3 ]  [ Head 4 ]  [ Head 5 ]  [ Entropy Map H×T  ]

橙色虚线分隔 prompt / 生成部分；特殊 token 标签红色高亮。

用法：
    python visualize_llm.py rs1234                    # 全部 6 层
    python visualize_llm.py rs1234 --layer 2          # 只看第 2 层
    python visualize_llm.py rs1234 --layer 0,3,5      # 多层
    python visualize_llm.py rs1234 --prompt-only      # 只可视化 prompt
    python visualize_llm.py rs1234 --config config/inference.yaml

输出：
    log/figures/attn_L{layer}_{expr}.png
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang import TOKEN2ID
from model import build_model

CONFIG_YAML = 'config/inference.yaml'

ID2TOK = {v: k for k, v in TOKEN2ID.items()}
BOS_ID = TOKEN2ID['[BOS]']

# 显示时缩写长 token
_ABBR = {'[BOS]': 'BOS', '[EOS]': 'EOS', '<think>': '<th>', '</think>': '</th>'}
_SPECIAL_DISPLAY = {'BOS', 'EOS', '<th>', '</th>'}


# ─────────────────────────────────────────────────────────────────────────────
# Attention Hook
# ─────────────────────────────────────────────────────────────────────────────

def register_attn_hooks(model):
    """
    在每层的 CausalSelfAttention.qkv (nn.Linear) 上注册 forward hook。
    hook 截获 qkv 的输出 [B, T, 3*H*dh]，手动计算显式 softmax attention weights，
    绕过 F.scaled_dot_product_attention 不返回 attention weights 的问题。

    返回：
        storage : dict  {layer_idx: np.ndarray [H, T, T]}
        hooks   : list  (调用 remove_hooks 清理)
    """
    storage = {}
    hooks   = []

    for layer_idx, block in enumerate(model.blocks):
        def make_hook(idx, attn_mod):
            def hook(module, inputs, output):
                # output: [B, T, 3*H*dh]
                with torch.no_grad():
                    B, T, _ = output.shape
                    H, dh   = attn_mod.n_heads, attn_mod.d_head
                    q, k, _ = output.split(H * dh, dim=-1)
                    q = q.view(B, T, H, dh).transpose(1, 2)   # [B, H, T, dh]
                    k = k.view(B, T, H, dh).transpose(1, 2)   # [B, H, T, dh]

                    scores = (q @ k.transpose(-2, -1)) * (dh ** -0.5)  # [B, H, T, T]
                    # causal mask: position t cannot attend to t+1, t+2, ...
                    causal = torch.triu(
                        torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1
                    )
                    scores = scores.masked_fill(causal, float('-inf'))
                    weights = F.softmax(scores, dim=-1)          # [B, H, T, T]
                    weights = torch.nan_to_num(weights, nan=0.0) # t=0 的行仍保留 1.0
                    storage[idx] = weights[0].cpu().numpy()       # [H, T, T]
            return hook

        h = block.attn.qkv.register_forward_hook(make_hook(layer_idx, block.attn))
        hooks.append(h)

    return storage, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Token 标签工具
# ─────────────────────────────────────────────────────────────────────────────

def tok_labels(token_ids):
    """token ID list → 显示标签 list（特殊 token 缩写）。"""
    return [_ABBR.get(ID2TOK.get(i, f'?{i}'), ID2TOK.get(i, f'?{i}')) for i in token_ids]


def color_special_ticks(ax, labels):
    """将 ax x/y 轴上属于 _SPECIAL_DISPLAY 的 tick label 改为红色加粗。"""
    for tl in ax.get_xticklabels():
        if tl.get_text() in _SPECIAL_DISPLAY:
            tl.set_color('#d62728')
            tl.set_fontweight('bold')
    for tl in ax.get_yticklabels():
        if tl.get_text() in _SPECIAL_DISPLAY:
            tl.set_color('#d62728')
            tl.set_fontweight('bold')


def tick_params(T):
    """返回 (ticks, step)，序列较长时间隔采样以避免标签重叠。"""
    step = 1 if T <= 24 else (2 if T <= 48 else 3)
    ticks = list(range(0, T, step))
    return ticks, step


# ─────────────────────────────────────────────────────────────────────────────
# 单张热力图绘制（复用）
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_ATTN = plt.cm.Blues.copy()
_CMAP_ATTN.set_bad(color='#eeeeee')   # masked upper-triangle: 浅灰


def draw_attn_heatmap(ax, w, labels, title, ticks, cmap=_CMAP_ATTN, vmax=1.0,
                      prompt_end=None):
    """
    w       : [T, T] float，上三角为 masked（0），会被替换为 NaN 以区别「无效」和「低权重」
    labels  : length-T 字符串列表
    prompt_end : prompt 长度，若非 None 则画橙色虚线分隔 prompt / 生成部分
    """
    T = w.shape[0]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    w_plot = np.where(mask, np.nan, w)

    im = ax.imshow(w_plot, cmap=cmap, vmin=0, vmax=vmax,
                   aspect='auto', interpolation='nearest')

    tick_labels = [labels[i] for i in ticks]
    ax.set_xticks(ticks);  ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(ticks);  ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Key (attended-to)',  fontsize=7)
    ax.set_ylabel('Query (current pos)', fontsize=7)

    # prompt / generated 分隔线
    if prompt_end is not None and 0 < prompt_end < T:
        sep = prompt_end - 0.5
        ax.axvline(sep, color='#ff7f0e', linewidth=1.2, linestyle='--', alpha=0.8)
        ax.axhline(sep, color='#ff7f0e', linewidth=1.2, linestyle='--', alpha=0.8)

    color_special_ticks(ax, labels)
    return im


# ─────────────────────────────────────────────────────────────────────────────
# 每层主绘图函数
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_attention(attn, token_ids, layer_idx, prompt_len, out_path):
    """
    attn       : [H, T, T]  从 hook 中取到的 attention weights
    token_ids  : length-T  完整 token ID 序列
    prompt_len : prompt 部分长度（用于画分隔线）
    out_path   : 输出 PNG 路径
    """
    H, T, _ = attn.shape
    labels   = tok_labels(token_ids)
    ticks, _ = tick_params(T)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(
        f'Layer {layer_idx}  —  Attention Weights'
        f'  (seq_len={T}, {H} heads)',
        fontsize=14, fontweight='bold'
    )

    # ── Head 0-5 heatmaps ────────────────────────────────────────────────────
    for h in range(H):
        row, col = h // 3, h % 3
        ax = axes[row, col]

        # mean entropy of this head (over valid positions only)
        ent_vals = []
        for t in range(T):
            p = attn[h, t, :t + 1] + 1e-9
            p /= p.sum()
            ent_vals.append(-np.sum(p * np.log(p)))
        mean_ent = np.mean(ent_vals)

        im = draw_attn_heatmap(
            ax, attn[h], labels,
            title=f'Head {h}  (mean entropy={mean_ent:.2f})',
            ticks=ticks, prompt_end=prompt_len
        )
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    # ── Mean attention （所有 head 平均） ─────────────────────────────────────
    ax_mean = axes[0, 3]
    mean_w  = attn.mean(axis=0)   # [T, T]
    im_mean = draw_attn_heatmap(
        ax_mean, mean_w, labels,
        title='Mean (all heads)', ticks=ticks, prompt_end=prompt_len
    )
    plt.colorbar(im_mean, ax=ax_mean, shrink=0.75, pad=0.02)

    # ── Entropy Map：[H, T] 归一化熵 ─────────────────────────────────────────
    # norm_entropy[h, t] = entropy(head h, query t) / log(t+1)
    #   → 0 = 完全聚焦（sharp），1 = 完全均匀（uniform）
    # 意义：绿色 = 该 head 在该位置有明确的关注目标；红色 = 无方向性
    ax_ent = axes[1, 3]

    entropy     = np.zeros((H, T))
    norm_entropy = np.zeros((H, T))
    for h in range(H):
        for t in range(T):
            p   = attn[h, t, :t + 1] + 1e-9
            p  /= p.sum()
            ent = -np.sum(p * np.log(p))
            entropy[h, t] = ent
            max_ent = np.log(t + 1) if t > 0 else 1.0   # t=0 固定为 1.0（entropy=0）
            norm_entropy[h, t] = ent / max_ent

    cmap_ent = plt.cm.RdYlGn_r.copy()
    im_ent = ax_ent.imshow(
        norm_entropy, cmap=cmap_ent, vmin=0, vmax=1,
        aspect='auto', interpolation='nearest'
    )
    ax_ent.set_xticks(ticks)
    ax_ent.set_xticklabels([labels[i] for i in ticks], rotation=45, ha='right', fontsize=7)
    ax_ent.set_yticks(range(H))
    ax_ent.set_yticklabels([f'H{h}' for h in range(H)], fontsize=9)
    ax_ent.set_title(
        'Normalized Entropy  [H × T]\n'
        'green = sharp/focused,  red = uniform/diffuse',
        fontsize=9
    )
    ax_ent.set_xlabel('Query position', fontsize=7)
    ax_ent.set_ylabel('Head', fontsize=7)

    if prompt_len is not None and 0 < prompt_len < T:
        ax_ent.axvline(prompt_len - 0.5, color='#ff7f0e',
                       linewidth=1.2, linestyle='--', alpha=0.8)

    color_special_ticks(ax_ent, labels)
    plt.colorbar(im_ent, ax=ax_ent, shrink=0.75, pad=0.02,
                 label='entropy / log(pos)')

    # ── 图例注释 ──────────────────────────────────────────────────────────────
    fig.text(0.01, 0.01,
             'Orange dashed line = prompt / generated boundary  |  '
             'Red bold labels = special tokens',
             fontsize=8, color='#555555')

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Layer {layer_idx:2d}  →  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LLMlab Attention Visualizer')
    parser.add_argument('expr',
                        help='表达式，如 rs1234 或 12c34')
    parser.add_argument('--config',  default=CONFIG_YAML,
                        help='yaml 配置文件（默认 config/inference.yaml）')
    parser.add_argument('--layer',   default='all',
                        help='要可视化的层，如 "2"、"0,3,5" 或 "all"')
    parser.add_argument('--prompt-only', action='store_true',
                        help='只可视化 prompt tokens，不自动生成')
    parser.add_argument('--out',     default='',
                        help='输出目录（默认 log/figures/）')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg.get('model_path', 'model/teacher_pretrain.pt')
    model_cfg  = cfg['model']
    infer_cfg  = cfg.get('inference', {})
    device     = infer_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = build_model(model_cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"加载模型: {model_path}  device={device}")

    # ── 选定层 ────────────────────────────────────────────────────────────────
    n_layers = len(model.blocks)
    if args.layer == 'all':
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layer.split(',')]
        layers = [l for l in layers if 0 <= l < n_layers]

    # ── 构建 token ID 序列 ────────────────────────────────────────────────────
    # 延迟导入，避免 inference.py 初始化副作用
    from inference import tokenize_expr, generate

    prompt_ids = [BOS_ID] + tokenize_expr(args.expr)

    if args.prompt_only:
        token_ids  = prompt_ids
        prompt_len = len(prompt_ids)   # 不画分隔线
    else:
        temperature = float(infer_cfg.get('temperature',  0.0))
        top_p       = float(infer_cfg.get('top_p',        1.0))
        max_new     = int(infer_cfg.get('max_new_tokens', 96))
        gen_ids     = generate(model, prompt_ids, max_new, temperature, top_p, device)
        token_ids   = prompt_ids + gen_ids
        prompt_len  = len(prompt_ids)
        seq_str     = ' '.join(ID2TOK.get(i, '?') for i in token_ids)
        print(f"生成序列 (len={len(token_ids)}): {seq_str}\n")

    # ── 注册 hook → 跑一次 forward → 撤销 hook ───────────────────────────────
    storage, hooks = register_attn_hooks(model)
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(ids_t)
    remove_hooks(hooks)

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out) if args.out else Path('log/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 逐层生成图 ────────────────────────────────────────────────────────────
    safe_expr = args.expr.replace('/', '_')
    for layer_idx in layers:
        if layer_idx not in storage:
            print(f"  Layer {layer_idx}: hook 未捕获数据，跳过。")
            continue
        out_path = out_dir / f'attn_L{layer_idx}_{safe_expr}.png'
        plot_layer_attention(
            attn=storage[layer_idx],     # [H, T, T]
            token_ids=token_ids,
            layer_idx=layer_idx,
            prompt_len=prompt_len,
            out_path=out_path,
        )

    print(f"\n完成，共 {len(layers)} 层，图片保存到 {out_dir}/")


if __name__ == '__main__':
    main()
