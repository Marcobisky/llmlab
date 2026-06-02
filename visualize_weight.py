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


def _load_npz_dict(npz_path: Path) -> dict:
    """Load one .npz file into a plain dict."""
    with np.load(npz_path) as d:
        return {k: d[k] for k in d.files}


def load_landscapes(stages: list, log_root: str = 'log') -> Dict[str, dict]:
    """Load landscape.npz and optional run_*/landscape.npz files for each stage."""
    result = {}
    for stage in stages:
        stage_dir = Path(log_root) / stage
        npz_path = stage_dir / 'landscape.npz'
        run_paths = sorted(stage_dir.glob('run_*/landscape.npz'))

        runs = [_load_npz_dict(p) for p in run_paths]
        if npz_path.exists():
            result[stage] = _load_npz_dict(npz_path)
        elif runs:
            # Use the first run for fields like grids/Z; trajectory plotting will
            # use the full runs list.
            result[stage] = dict(runs[0])
        else:
            print(f"  Warning: {npz_path} not found and no run_*/landscape.npz files, skipping.")
            continue

        if runs:
            result[stage]['runs'] = runs
            print(f"  {stage}: loaded {len(runs)} run trajectory file(s)")
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

    runs = _trajectory_runs(ld)
    if runs:
        if len(runs) == 1:
            ta, tb = runs[0]
            ax.plot(tb, ta, 'o-', color='royalblue', markersize=3,
                    linewidth=1.0, label='Trajectory', zorder=3)
            ax.scatter(tb[0], ta[0], color='cyan', s=50, zorder=4, label='Start')
        else:
            for i, (ta_run, tb_run) in enumerate(runs):
                ax.plot(tb_run, ta_run, '-', color='royalblue', linewidth=0.8,
                        alpha=0.18, zorder=2, label='Runs' if i == 0 else None)
            mean_a, mean_b, std_a, std_b = _mean_std_trajectories(runs)
            ax.plot(mean_b, mean_a, 'o-', color='royalblue', markersize=3,
                    linewidth=1.5, label='Mean trajectory', zorder=4)
            every = max(1, len(mean_a) // 12)
            ax.errorbar(mean_b[::every], mean_a[::every],
                        xerr=std_b[::every], yerr=std_a[::every],
                        fmt='none', ecolor='royalblue', elinewidth=0.8,
                        alpha=0.65, capsize=2, zorder=3, label='±1 std')
            ax.scatter(mean_b[0], mean_a[0], color='cyan', s=50,
                       zorder=5, label='Mean start')
    ax.scatter([0], [0], color='red', s=80, marker='*', zorder=5, label='theta* (end)')

    # auto axis limits covering both grid and trajectory
    if runs:
        run_a = np.concatenate([r[0] for r in runs])
        run_b = np.concatenate([r[1] for r in runs])
        all_b = np.concatenate([bg, run_b])
        all_a = np.concatenate([ag, run_a])
    else:
        all_b = bg
        all_a = ag
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


def _trajectory_runs(ld: dict) -> List[tuple]:
    """Return a list of (traj_alpha, traj_beta) arrays for one stage."""
    if ld.get('runs'):
        return [(r['traj_alpha'], r['traj_beta']) for r in ld['runs']
                if len(r['traj_alpha']) and len(r['traj_beta'])]
    if len(ld['traj_alpha']) and len(ld['traj_beta']):
        return [(ld['traj_alpha'], ld['traj_beta'])]
    return []


def _mean_std_trajectories(runs: List[tuple]) -> tuple:
    """
    Stack multiple trajectories to their common prefix length.

    Returns mean_alpha, mean_beta, std_alpha, std_beta, each [T].
    """
    T = min(len(a) for a, _ in runs)
    alpha = np.stack([a[:T] for a, _ in runs], axis=0)
    beta = np.stack([b[:T] for _, b in runs], axis=0)
    return alpha.mean(0), beta.mean(0), alpha.std(0), beta.std(0)


def plot_connected_trajectories(
    landscapes: Dict,
    stages: List[str],
    out_dir: Path,
    tag: str = '',
):
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
        runs = _trajectory_runs(landscapes[stages[i]])
        if not runs:
            off_a[i] = off_a[i - 1]
            off_b[i] = off_b[i - 1]
            continue
        start_a = float(np.mean([a[0] for a, _ in runs]))
        start_b = float(np.mean([b[0] for _, b in runs]))
        # Align this stage's start with the previous stage endpoint.
        off_a[i] = off_a[i - 1] - start_a
        off_b[i] = off_b[i - 1] - start_b

    # ── collect all shifted coordinates for axis scaling ─────────────────────
    all_alpha, all_beta = [0.0], [0.0]   # first stage theta* origin
    for i, stage in enumerate(stages):
        for ta, tb in _trajectory_runs(landscapes[stage]):
            all_alpha.extend((ta + off_a[i]).tolist())
            all_beta.extend((tb + off_b[i]).tolist())
        all_alpha.append(off_a[i])   # theta*_stage endpoint
        all_beta.append( off_b[i])

    def _lim(vals, margin=0.12):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * margin if hi > lo else 1.0
        return lo - pad, hi + pad

    # ── draw ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, stage in enumerate(stages):
        runs = _trajectory_runs(landscapes[stage])
        if not runs:
            continue
        color = PALETTE[i % len(PALETTE)]

        if len(runs) == 1:
            ta = runs[0][0] + off_a[i]   # shifted alpha trajectory [N]
            tb = runs[0][1] + off_b[i]   # shifted beta trajectory [N]
            ax.plot(tb, ta, 'o-', color=color, markersize=3, linewidth=1.4,
                    label=stage, zorder=3, alpha=0.9)
            ax.scatter(tb[0], ta[0], color=color, s=55, marker='o',
                       edgecolors='white', linewidths=0.8, zorder=5)
        else:
            shifted_runs = [(a + off_a[i], b + off_b[i]) for a, b in runs]
            for run_idx, (ta_run, tb_run) in enumerate(shifted_runs):
                ax.plot(tb_run, ta_run, '-', color=color, linewidth=0.8,
                        alpha=0.18, zorder=2,
                        label=f'{stage} runs' if run_idx == 0 else None)

            mean_a, mean_b, std_a, std_b = _mean_std_trajectories(shifted_runs)
            ax.plot(mean_b, mean_a, 'o-', color=color, markersize=3.5,
                    linewidth=1.8, label=f'{stage} mean', zorder=4)
            every = max(1, len(mean_a) // 12)
            ax.errorbar(mean_b[::every], mean_a[::every],
                        xerr=std_b[::every], yerr=std_a[::every],
                        fmt='none', ecolor=color, elinewidth=0.9,
                        alpha=0.65, capsize=2, zorder=3,
                        label=f'{stage} ±1 std')
            ax.scatter(mean_b[0], mean_a[0], color=color, s=55, marker='o',
                       edgecolors='white', linewidths=0.8, zorder=5)

        # theta*_stage endpoint, or mean endpoint when multiple runs are present.
        ax.scatter(off_b[i], off_a[i], color=color, s=120, marker='*',
                   edgecolors='k', linewidths=0.5, zorder=6,
                   label=f'{stage} θ*' if len(runs) == 1 else f'{stage} mean θ*')

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
    out = out_dir / f"trajectory_comparison{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Trajectory comparison plot saved: {out}")

    if len(stages) > 1:
        zoom_stage = stages[-1]
        zoom_runs = _trajectory_runs(landscapes[zoom_stage])
        if zoom_runs:
            color = PALETTE[(len(stages) - 1) % len(PALETTE)]
            shifted_runs = [(a + off_a[-1], b + off_b[-1]) for a, b in zoom_runs]
            fig_z, ax_z = plt.subplots(figsize=(7, 5))
            zoom_alpha, zoom_beta = [], []

            if len(shifted_runs) == 1:
                ta, tb = shifted_runs[0]
                ax_z.plot(tb, ta, 'o-', color=color, markersize=3,
                          linewidth=1.5, label=zoom_stage)
                zoom_alpha.extend(ta.tolist())
                zoom_beta.extend(tb.tolist())
            else:
                for run_idx, (ta_run, tb_run) in enumerate(shifted_runs):
                    ax_z.plot(tb_run, ta_run, '-', color=color, linewidth=0.8,
                              alpha=0.18, label=f'{zoom_stage} runs' if run_idx == 0 else None)
                    zoom_alpha.extend(ta_run.tolist())
                    zoom_beta.extend(tb_run.tolist())
                mean_a, mean_b, std_a, std_b = _mean_std_trajectories(shifted_runs)
                ax_z.plot(mean_b, mean_a, 'o-', color=color, markersize=3.5,
                          linewidth=1.8, label=f'{zoom_stage} mean')
                every = max(1, len(mean_a) // 12)
                ax_z.errorbar(mean_b[::every], mean_a[::every],
                              xerr=std_b[::every], yerr=std_a[::every],
                              fmt='none', ecolor=color, elinewidth=0.9,
                              alpha=0.65, capsize=2, label=f'{zoom_stage} ±1 std')
                zoom_alpha.extend(mean_a.tolist())
                zoom_beta.extend(mean_b.tolist())

            ax_z.scatter([off_b[-1]], [off_a[-1]], color=color, s=120, marker='*',
                         edgecolors='k', linewidths=0.5, label=f'{zoom_stage} θ*')
            zoom_alpha.append(off_a[-1])
            zoom_beta.append(off_b[-1])
            ax_z.set_xlim(_lim(zoom_beta, margin=0.18))
            ax_z.set_ylim(_lim(zoom_alpha, margin=0.18))
            ax_z.set_xlabel('beta  (PC2)', fontsize=11)
            ax_z.set_ylabel('alpha (PC1)', fontsize=11)
            ax_z.set_title(f'{zoom_stage} Trajectory Zoom\n(shared PCA coordinates)', fontsize=11)
            ax_z.legend(fontsize=8, loc='best', framealpha=0.85)
            ax_z.grid(True, alpha=0.25)
            fig_z.tight_layout()
            out_zoom = out_dir / f"trajectory_zoom{tag}.png"
            fig_z.savefig(out_zoom, dpi=150)
            plt.close(fig_z)
            print(f"  Trajectory zoom plot saved: {out_zoom}")


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


def output_tag(stages: List[str]) -> str:
    """Build a unique suffix for multi-config output files."""
    return '' if len(stages) <= 1 else '__' + '__'.join(stages)


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
        if landscapes.get(stages[0], {}).get('runs'):
            plot_connected_trajectories(landscapes, stages, out_dir)
    else:
        # Multi-stage: connected trajectory in shared PCA coords (no landscape contour)
        loaded_stages = [s for s in stages if s in landscapes]
        plot_connected_trajectories(landscapes, loaded_stages, out_dir, output_tag(loaded_stages))
    print(f"\nAll plots saved to {out_dir}")


if __name__ == '__main__':
    main()
