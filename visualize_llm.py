"""
visualize_llm.py — Transformer attention interpretability visualization.

Given an expression, auto-regressively generates the full sequence (with CoT),
then uses forward hooks to extract per-layer attention weights and saves
2x4 heatmap grids:

    [ Head 0 ]  [ Head 1 ]  [ Head 2 ]  [ Mean (all heads) ]
    [ Head 3 ]  [ Head 4 ]  [ Head 5 ]  [ Entropy Map H x T ]

Orange dashed lines separate prompt / generated parts; special token labels are red.
Output images go to log/<config_name>/fig/ (derived from the --config argument).

Usage:
    python visualize_llm.py --config config/teacher_pretrain.yaml rs1234
    python visualize_llm.py --config config/teacher_sft.yaml rs1234 --layer 2
    python visualize_llm.py --config config/teacher_pretrain.yaml rs1234 --layer 0,3,5
    python visualize_llm.py --config config/teacher_pretrain.yaml rs1234 --prompt-only
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
from lib.lang  import TOKEN2ID
from lib.model import build_model

ID2TOK = {v: k for k, v in TOKEN2ID.items()}
BOS_ID = TOKEN2ID['[BOS]']

# abbreviate long special token labels
_ABBR = {'[BOS]': 'BOS', '[EOS]': 'EOS', '<think>': '<th>', '</think>': '</th>'}
_SPECIAL_DISPLAY = {'BOS', 'EOS', '<th>', '</th>'}


# ─────────────────────────────────────────────────────────────────────────────
# Attention hook
# ─────────────────────────────────────────────────────────────────────────────

def register_attn_hooks(model):
    """
    Register forward hooks on each layer's CausalSelfAttention.qkv Linear.
    The hook intercepts qkv output [B, T, 3*H*dh] and manually computes
    softmax attention weights (workaround: F.scaled_dot_product_attention
    does not return attention weights).

    Returns:
        storage : dict  {layer_idx: np.ndarray [H, T, T]}
        hooks   : list  (call remove_hooks to clean up)
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
                    weights = torch.nan_to_num(weights, nan=0.0) # t=0 row: entropy=0
                    storage[idx] = weights[0].cpu().numpy()       # [H, T, T]
            return hook

        h = block.attn.qkv.register_forward_hook(make_hook(layer_idx, block.attn))
        hooks.append(h)

    return storage, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Token label utilities
# ─────────────────────────────────────────────────────────────────────────────

def tok_labels(token_ids):
    """token ID list -> display label list (special tokens abbreviated)."""
    return [_ABBR.get(ID2TOK.get(i, f'?{i}'), ID2TOK.get(i, f'?{i}')) for i in token_ids]


def color_special_ticks(ax, labels):
    """Color x/y axis tick labels that are special tokens red+bold."""
    for tl in ax.get_xticklabels():
        if tl.get_text() in _SPECIAL_DISPLAY:
            tl.set_color('#d62728')
            tl.set_fontweight('bold')
    for tl in ax.get_yticklabels():
        if tl.get_text() in _SPECIAL_DISPLAY:
            tl.set_color('#d62728')
            tl.set_fontweight('bold')


def tick_params(T):
    """Returns (ticks, step) — subsample for long sequences to avoid overlap."""
    step = 1 if T <= 24 else (2 if T <= 48 else 3)
    ticks = list(range(0, T, step))
    return ticks, step


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap drawing
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_ATTN = plt.cm.Blues.copy()
_CMAP_ATTN.set_bad(color='#eeeeee')   # masked upper-triangle: light gray


def draw_attn_heatmap(ax, w, labels, title, ticks, cmap=_CMAP_ATTN, vmax=1.0,
                      prompt_end=None):
    """
    w          : [T, T] float, upper triangle is masked (replaced with NaN)
    labels     : length-T string list
    prompt_end : if not None, draw an orange dashed line at this position
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
    ax.set_xlabel('Key (attended-to)',   fontsize=7)
    ax.set_ylabel('Query (current pos)', fontsize=7)

    if prompt_end is not None and 0 < prompt_end < T:
        sep = prompt_end - 0.5
        ax.axvline(sep, color='#ff7f0e', linewidth=1.2, linestyle='--', alpha=0.8)
        ax.axhline(sep, color='#ff7f0e', linewidth=1.2, linestyle='--', alpha=0.8)

    color_special_ticks(ax, labels)
    return im


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_attention(attn, token_ids, layer_idx, prompt_len, out_path):
    """
    attn       : [H, T, T]  attention weights from hook
    token_ids  : length-T complete token ID sequence
    prompt_len : number of prompt tokens (used for separator line)
    out_path   : output PNG path
    """
    H, T, _ = attn.shape
    labels   = tok_labels(token_ids)
    ticks, _ = tick_params(T)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(
        f'Layer {layer_idx}  -  Attention Weights'
        f'  (seq_len={T}, {H} heads)',
        fontsize=14, fontweight='bold'
    )

    for h in range(H):
        row, col = h // 3, h % 3
        ax = axes[row, col]

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

    # mean attention across all heads
    ax_mean = axes[0, 3]
    mean_w  = attn.mean(axis=0)   # [T, T]
    im_mean = draw_attn_heatmap(
        ax_mean, mean_w, labels,
        title='Mean (all heads)', ticks=ticks, prompt_end=prompt_len
    )
    plt.colorbar(im_mean, ax=ax_mean, shrink=0.75, pad=0.02)

    # entropy map: [H, T] normalized entropy
    # norm_entropy[h, t] = entropy(head h, query t) / log(t+1)
    #   0 = fully focused (sharp), 1 = fully uniform (diffuse)
    ax_ent = axes[1, 3]

    entropy      = np.zeros((H, T))
    norm_entropy = np.zeros((H, T))
    for h in range(H):
        for t in range(T):
            p   = attn[h, t, :t + 1] + 1e-9
            p  /= p.sum()
            ent = -np.sum(p * np.log(p))
            entropy[h, t] = ent
            max_ent = np.log(t + 1) if t > 0 else 1.0
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
        'Normalized Entropy  [H x T]\n'
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

    fig.text(0.01, 0.01,
             'Orange dashed line = prompt / generated boundary  |  '
             'Red bold labels = special tokens',
             fontsize=8, color='#555555')

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Layer {layer_idx:2d}  ->  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LLMlab Attention Visualizer')
    parser.add_argument('--config', required=True,
                        help='Training config yaml (e.g. config/teacher_pretrain.yaml)')
    parser.add_argument('expr',
                        help='Expression to visualize, e.g. rs1234 or 12c34')
    parser.add_argument('--layer',   default='all',
                        help='Layers to visualize: "2", "0,3,5", or "all"')
    parser.add_argument('--prompt-only', action='store_true',
                        help='Visualize only prompt tokens, skip generation')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg['output']['model_path']
    model_cfg  = cfg['model']
    infer_cfg  = cfg.get('inference', {})
    device     = infer_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    log_dir = cfg['output']['log_dir']
    out_dir = Path(log_dir) / 'fig'
    out_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = build_model(model_cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {model_path}  device={device}")

    # select layers
    n_layers = len(model.blocks)
    if args.layer == 'all':
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layer.split(',')]
        layers = [l for l in layers if 0 <= l < n_layers]

    # build token ID sequence
    from inference import tokenize_expr, generate

    prompt_ids = [BOS_ID] + tokenize_expr(args.expr)

    if args.prompt_only:
        token_ids  = prompt_ids
        prompt_len = len(prompt_ids)
    else:
        temperature = float(infer_cfg.get('temperature',  0.0))
        top_p       = float(infer_cfg.get('top_p',        1.0))
        max_new     = int(infer_cfg.get('max_new_tokens', 96))
        gen_ids     = generate(model, prompt_ids, max_new, temperature, top_p, device)
        token_ids   = prompt_ids + gen_ids
        prompt_len  = len(prompt_ids)
        seq_str     = ' '.join(ID2TOK.get(i, '?') for i in token_ids)
        print(f"Generated sequence (len={len(token_ids)}): {seq_str}\n")

    # register hooks -> one forward pass -> remove hooks
    storage, hooks = register_attn_hooks(model)
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(ids_t)
    remove_hooks(hooks)

    # generate per-layer attention plots
    safe_expr = args.expr.replace('/', '_')
    for layer_idx in layers:
        if layer_idx not in storage:
            print(f"  Layer {layer_idx}: hook data missing, skipped.")
            continue
        out_path = out_dir / f'attn_L{layer_idx}_{safe_expr}.png'
        plot_layer_attention(
            attn=storage[layer_idx],     # [H, T, T]
            token_ids=token_ids,
            layer_idx=layer_idx,
            prompt_len=prompt_len,
            out_path=out_path,
        )

    print(f"\nDone. {len(layers)} layer(s). Plots saved to {out_dir}/")


if __name__ == '__main__':
    main()
