"""
visualize_weight.py — 只读 log/*/landscape.npz，生成 loss landscape 等高线 + 轨迹图。

每个 stage 一张子图：等高线 = loss landscape，散点+折线 = 训练轨迹，
★ = 训练终点（theta*）对应坐标原点（0,0）。

用法：
    python visualize_weight.py
    python visualize_weight.py --log log/ --out figures/
"""
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# stage 顺序（与 visualize_loss.py 保持一致）
STAGE_ORDER = [
    'teacher_pretrain', 'teacher_sft', 'teacher_grpo', 'teacher_sdpo',
    'student_pretrain', 'student_sft',
    'student_kd', 'student_opd', 'student_grpo',
]


# ─────────────────────────────────────────────────────────────────────────────
# 加载
# ─────────────────────────────────────────────────────────────────────────────

def load_all_landscapes(log_root: str = 'log') -> Dict[str, dict]:
    """扫描 log/*/landscape.npz，返回 {stage: {alpha_grid, beta_grid, Z, traj_alpha, traj_beta}}。"""
    result = {}
    for stage_dir in sorted(Path(log_root).iterdir()):
        if not stage_dir.is_dir():
            continue
        npz_path = stage_dir / 'landscape.npz'
        if not npz_path.exists():
            continue
        with np.load(npz_path) as d:
            result[stage_dir.name] = {k: d[k] for k in d.files}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 单 stage 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_one_landscape(ax, ld: dict, stage: str):
    """
    在给定 Axes 上绘制 loss landscape + 轨迹。

    坐标系：(0,0) = theta*（训练终点）；等高线范围自动扩展覆盖完整轨迹。

    兼容两种 npz：
      - 新版（save_landscape 自动定范围）：网格已覆盖轨迹
      - 旧版（固定 ±1 范围）：轴范围扩展到轨迹，等高线只绘制网格区域内
    """
    ag = ld['alpha_grid']   # [grid_res]
    bg = ld['beta_grid']    # [grid_res]
    Z  = ld['Z']            # [grid_res, grid_res]
    ta = ld['traj_alpha']   # [N]
    tb = ld['traj_beta']    # [N]

    # ── 等高线（对数压缩让低谷更清晰）──────────────────────────────────────
    Z_plot = np.log1p(Z - Z.min())
    levels = np.linspace(Z_plot.min(), Z_plot.max(), 20)

    B, A = np.meshgrid(bg, ag)
    cf = ax.contourf(B, A, Z_plot, levels=levels, cmap='RdYlGn_r', alpha=0.85)
    ax.contour(B, A, Z_plot, levels=levels[::4], colors='k', linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label='log(1+L−Lmin)', shrink=0.82, pad=0.02)

    # ── 轨迹 ────────────────────────────────────────────────────────────────
    if len(ta) > 0:
        ax.plot(tb, ta, 'o-', color='royalblue', markersize=3,
                linewidth=1.0, label='Trajectory', zorder=3)
        ax.scatter(tb[0], ta[0], color='cyan',  s=50, zorder=4, label='Start')
    ax.scatter([0], [0], color='red', s=80, marker='*', zorder=5, label='θ* (end)')

    # ── 自动轴范围：覆盖网格 + 轨迹，加 5% 边距 ───────────────────────────
    all_b = np.concatenate([bg, tb]) if len(tb) else bg
    all_a = np.concatenate([ag, ta]) if len(ta) else ag
    def _lim(vals, margin=0.05):
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * margin if hi > lo else 1.0
        return lo - pad, hi + pad
    ax.set_xlim(_lim(all_b))
    ax.set_ylim(_lim(all_a))

    # 用虚线框标出实际网格边界（若轨迹超出网格才有意义）
    rect_b = [bg.min(), bg.max(), bg.max(), bg.min(), bg.min()]
    rect_a = [ag.min(), ag.min(), ag.max(), ag.max(), ag.min()]
    ax.plot(rect_b, rect_a, 'k--', linewidth=0.7, alpha=0.4, label='Grid boundary')

    ax.set_xlabel('β (PC2)')
    ax.set_ylabel('α (PC1)')
    ax.set_title(stage)
    ax.legend(fontsize=6, loc='best')


# ─────────────────────────────────────────────────────────────────────────────
# 全局绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_landscapes(landscapes: Dict, out_dir: Path):
    stages = [s for s in STAGE_ORDER if s in landscapes]
    stages += [s for s in landscapes if s not in STAGE_ORDER]
    if not stages:
        print("无 landscape 数据，退出。")
        return

    n  = len(stages)
    nc = min(3, n)          # 每行最多 3 列
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr))
    axes = np.array(axes).reshape(-1)  # 展平，方便索引

    for i, stage in enumerate(stages):
        plot_one_landscape(axes[i], landscapes[stage], stage)

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Loss Landscape（PCA 方向，训练轨迹投影）', fontsize=13)
    fig.tight_layout()

    out = out_dir / 'landscape_all.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  landscape 图已保存: {out}")

    # 同时单独保存每个 stage
    for stage in stages:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        plot_one_landscape(ax2, landscapes[stage], stage)
        fig2.tight_layout()
        single_out = out_dir / f'landscape_{stage}.png'
        fig2.savefig(single_out, dpi=150)
        plt.close(fig2)
        print(f"  landscape 图已保存: {single_out}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='log',  help='log 根目录')
    parser.add_argument('--out', default='',     help='输出目录（默认 log/figures/）')
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.log) / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"扫描 {args.log}/*/landscape.npz ...")
    landscapes = load_all_landscapes(args.log)
    if not landscapes:
        print("未找到 landscape.npz，请先完成训练。")
        return

    print(f"找到 {len(landscapes)} 个 stage 的 landscape。\n")
    plot_all_landscapes(landscapes, out_dir)
    print(f"\n所有图已保存到 {out_dir}")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent)
    main()
