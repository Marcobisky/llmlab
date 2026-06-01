"""
visualize_weight.py — Reads log/*/landscape.npz and generates loss landscape plots.

Each stage gets one subplot: contour = loss landscape, scatter+line = training trajectory,
star (*) = theta* (training endpoint, coordinate origin (0,0)).

All plots are saved to log/<config_name>/fig/ (derived from the --config argument).

Usage:
    python visualize_weight.py --config config/teacher_pretrain.yaml
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

STAGE_ORDER = [
    'teacher_pretrain', 'teacher_sft', 'teacher_grpo', 'teacher_sdpo',
    'student_pretrain', 'student_sft',
    'student_kd', 'student_opd', 'student_grpo',
]


def load_all_landscapes(log_root: str = 'log') -> Dict[str, dict]:
    """Scan log/*/landscape.npz. Returns {stage: {alpha_grid, beta_grid, Z, traj_*}}."""
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


def plot_one_landscape(ax, ld: dict, stage: str):
    """
    Draw loss landscape + trajectory on the given Axes.

    Coordinate system: (0,0) = theta* (training endpoint).
    Grid range is auto-derived from the actual trajectory (new-style npz) or
    fixed +-1 (old-style npz); trajectory points are always visible.
    """
    ag = ld['alpha_grid']   # [grid_res]
    bg = ld['beta_grid']    # [grid_res]
    Z  = ld['Z']            # [grid_res, grid_res]
    ta = ld['traj_alpha']   # [N]
    tb = ld['traj_beta']    # [N]

    # log-compress to make the loss valley more visible
    Z_plot = np.log1p(Z - Z.min())
    levels = np.linspace(Z_plot.min(), Z_plot.max(), 20)

    B, A = np.meshgrid(bg, ag)
    cf = ax.contourf(B, A, Z_plot, levels=levels, cmap='RdYlGn_r', alpha=0.85)
    ax.contour(B, A, Z_plot, levels=levels[::4], colors='k', linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label='log(1+L-Lmin)', shrink=0.82, pad=0.02)

    if len(ta) > 0:
        ax.plot(tb, ta, 'o-', color='royalblue', markersize=3,
                linewidth=1.0, label='Trajectory', zorder=3)
        ax.scatter(tb[0], ta[0], color='cyan',  s=50, zorder=4, label='Start')
    ax.scatter([0], [0], color='red', s=80, marker='*', zorder=5, label='theta* (end)')

    # auto axis limits covering both grid and trajectory
    all_b = np.concatenate([bg, tb]) if len(tb) else bg
    all_a = np.concatenate([ag, ta]) if len(ta) else ag
    def _lim(vals, margin=0.05):
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * margin if hi > lo else 1.0
        return lo - pad, hi + pad
    ax.set_xlim(_lim(all_b))
    ax.set_ylim(_lim(all_a))

    # dashed box marks grid boundary when trajectory extends beyond grid
    rect_b = [bg.min(), bg.max(), bg.max(), bg.min(), bg.min()]
    rect_a = [ag.min(), ag.min(), ag.max(), ag.max(), ag.min()]
    ax.plot(rect_b, rect_a, 'k--', linewidth=0.7, alpha=0.4, label='Grid boundary')

    ax.set_xlabel('beta (PC2)')
    ax.set_ylabel('alpha (PC1)')
    ax.set_title(stage)
    ax.legend(fontsize=6, loc='best')


def _shared_basis_note(landscapes: Dict, log_root: str = 'log') -> str:
    """
    Build a note for the plot title indicating which stages share a PCA basis.
    A stage 'foo' is a base stage if log/foo/pca_basis.npz exists and was produced
    by that stage's own SVD (i.e. it didn't load from somewhere else).
    We detect this by checking for pca_basis.npz in each stage's log dir.
    """
    base_stages = [s for s in landscapes
                   if (Path(log_root) / s / 'pca_basis.npz').exists()]
    if not base_stages:
        return ''
    return f'  |  Shared PCA basis anchored at: {", ".join(base_stages)}'


def plot_all_landscapes(landscapes: Dict, out_dir: Path, log_root: str = 'log'):
    stages = [s for s in STAGE_ORDER if s in landscapes]
    stages += [s for s in landscapes if s not in STAGE_ORDER]
    if not stages:
        print("No landscape data, exiting.")
        return

    n  = len(stages)
    nc = min(3, n)
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr))
    axes = np.array(axes).reshape(-1)

    for i, stage in enumerate(stages):
        plot_one_landscape(axes[i], landscapes[stage], stage)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    basis_note = _shared_basis_note(landscapes, log_root)
    fig.suptitle(
        f'Loss Landscape (PCA Directions, Training Trajectory Projection){basis_note}',
        fontsize=11
    )
    fig.tight_layout()

    out = out_dir / 'landscape_all.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Landscape plot saved: {out}")


def main():
    parser = argparse.ArgumentParser(description='Loss landscape visualizer')
    parser.add_argument('--config', required=True,
                        help='Training config yaml; determines output fig dir (log/<name>/fig/)')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg['output']['log_dir']
    out_dir = Path(log_dir) / 'fig'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning log/*/landscape.npz ...")
    landscapes = load_all_landscapes('log')
    if not landscapes:
        print("No landscape.npz files found. Run training first.")
        return

    print(f"Found {len(landscapes)} stage landscape(s).\n")
    plot_all_landscapes(landscapes, out_dir, log_root='log')
    print(f"\nAll plots saved to {out_dir}")


if __name__ == '__main__':
    main()
