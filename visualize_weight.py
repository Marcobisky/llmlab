"""
visualize_weight.py — Reads log/*/landscape.npz and generates loss landscape plots.

Each stage gets one subplot: contour = loss landscape, scatter+line = training trajectory,
star (*) = theta* (training endpoint, coordinate origin (0,0)).

Single config  → plots only that stage, saved to log/<stage>/fig/
Multiple configs → all specified stages in one figure, saved to log/img/

Usage:
    python visualize_weight.py --config config/teacher_sft.yaml
    python visualize_weight.py --config config/teacher_pretrain.yaml config/teacher_sft.yaml
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

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


def load_landscapes(stages: list, log_root: str = 'log') -> Dict[str, dict]:
    """Load landscape.npz for each stage name. Returns {stage: {alpha_grid, beta_grid, Z, traj_*}}."""
    result = {}
    for stage in stages:
        npz_path = Path(log_root) / stage / 'landscape.npz'
        if not npz_path.exists():
            print(f"  Warning: {npz_path} not found, skipping.")
            continue
        with np.load(npz_path) as d:
            result[stage] = {k: d[k] for k in d.files}
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


def plot_connected_trajectories(landscapes: Dict, stages: List[str], out_dir: Path):
    """
    Single plot showing training trajectories of all stages end-to-end.

    Coordinate system: first stage's theta* = origin (0, 0).
    All stages share the same PCA directions d1, d2 (pretrain anchors the basis;
    downstream stages reuse it), so trajectories live in the same vector space
    and can be stitched together with a coordinate offset.

    Offset derivation in the first stage's final frame:
        traj_i is stored relative to theta*_i:
            traj_i[t] = (theta_i[t] - theta*_i) · (d1, d2)
        If stage i starts from theta*_{i-1}, then its first trajectory point
        should coincide with the previous stage endpoint.  Therefore:
            offset_i = offset_{i-1} - traj_i[0]
        and shifted_traj_i = traj_i + offset_i.

    stages: in chronological order (earliest first).
    """
    # Palette: one distinct color per stage
    PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#bcbd22']

    # ── compute cumulative offsets from first stage final to later stages ─────
    n = len(stages)
    off_a = [0.0] * n   # alpha offset for each stage in the first-stage frame
    off_b = [0.0] * n   # beta offset
    for i in range(1, n):
        ld = landscapes[stages[i]]
        if len(ld['traj_alpha']) == 0:
            off_a[i] = off_a[i - 1]
            off_b[i] = off_b[i - 1]
            continue
        # Align this stage's start with the previous stage endpoint.
        off_a[i] = off_a[i - 1] - float(ld['traj_alpha'][0])
        off_b[i] = off_b[i - 1] - float(ld['traj_beta'][0])

    # ── collect all shifted coordinates for axis scaling ─────────────────────
    all_alpha, all_beta = [0.0], [0.0]   # first stage theta* origin
    for i, stage in enumerate(stages):
        ld = landscapes[stage]
        all_alpha.extend((ld['traj_alpha'] + off_a[i]).tolist())
        all_beta.extend( (ld['traj_beta']  + off_b[i]).tolist())
        all_alpha.append(off_a[i])   # theta*_stage endpoint
        all_beta.append( off_b[i])

    def _lim(vals, margin=0.12):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * margin if hi > lo else 1.0
        return lo - pad, hi + pad

    # ── draw ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, stage in enumerate(stages):
        ld    = landscapes[stage]
        ta    = ld['traj_alpha'] + off_a[i]   # shifted alpha trajectory  [N]
        tb    = ld['traj_beta']  + off_b[i]   # shifted beta  trajectory  [N]
        color = PALETTE[i % len(PALETTE)]

        # trajectory line + dots
        ax.plot(tb, ta, 'o-', color=color, markersize=3, linewidth=1.4,
                label=stage, zorder=3, alpha=0.9)
        # first checkpoint (≈ start of this stage's training)
        ax.scatter(tb[0], ta[0], color=color, s=55, marker='o',
                   edgecolors='white', linewidths=0.8, zorder=5)
        # theta*_stage endpoint (origin of this stage's own coord system)
        ax.scatter(off_b[i], off_a[i], color=color, s=120, marker='*',
                   edgecolors='k', linewidths=0.5, zorder=6,
                   label=f'{stage} θ*')

    # mark the global reference origin = first stage's theta*
    ax.scatter([0], [0], color='red', s=180, marker='*',
               edgecolors='k', linewidths=0.8, zorder=7,
               label=f'{stages[0]} θ* (origin)')
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.25, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.25, linestyle='--')

    ax.set_xlim(_lim(all_beta))
    ax.set_ylim(_lim(all_alpha))
    ax.set_xlabel('beta  (PC2)', fontsize=11)
    ax.set_ylabel('alpha (PC1)', fontsize=11)
    ax.set_title(
        'Training Trajectories — End-to-End\n'
        f'(Shared PCA basis; (0,0) = {stages[0]} θ*)',
        fontsize=11
    )
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = out_dir / 'trajectory_comparison.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Trajectory comparison plot saved: {out}")


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


def sort_stages_chronologically(stages: List[str]) -> List[str]:
    """Sort requested stage names by the canonical training order."""
    known = [s for s in STAGE_ORDER if s in stages]
    unknown = [s for s in stages if s not in STAGE_ORDER]
    return known + unknown


def main():
    parser = argparse.ArgumentParser(description='Loss landscape visualizer')
    parser.add_argument('--config', required=True, nargs='+',
                        help='One or more training config yamls. '
                             'Single: plots that stage to log/<stage>/fig/. '
                             'Multiple: comparison plot to log/img/.')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    # Derive stage names from each config's output.log_dir
    stages = []
    for config_path in args.config:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        stages.append(Path(cfg['output']['log_dir']).name)

    # Output directory: per-stage fig/ for single, shared log/img/ for comparison
    if len(args.config) == 1:
        with open(args.config[0]) as f:
            cfg = yaml.safe_load(f)
        out_dir = Path(cfg['output']['log_dir']) / 'fig'
    else:
        out_dir = Path('log/img')
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = sort_stages_chronologically(stages)

    print(f"Loading landscapes for: {stages}")
    landscapes = load_landscapes(stages)
    if not landscapes:
        print("No landscape.npz files found. Run training first.")
        return

    print(f"Loaded {len(landscapes)} stage landscape(s).\n")
    if len(stages) == 1:
        plot_all_landscapes(landscapes, out_dir, log_root='log')
    else:
        # Multi-stage: connected trajectory in shared PCA coords (no landscape contour)
        loaded_stages = [s for s in stages if s in landscapes]
        plot_connected_trajectories(landscapes, loaded_stages, out_dir)
    print(f"\nAll plots saved to {out_dir}")


if __name__ == '__main__':
    main()
