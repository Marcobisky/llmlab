"""
visualize_loss.py — Reads log/*/metrics.jsonl and generates loss/accuracy plots.

All plots are saved to log/<config_name>/fig/ (derived from the --config argument).
The script scans all log/* subdirectories so multi-stage comparison plots include
every available stage.

Plots:
    A: train_loss / val_loss curves (all stages overlaid)
    B: mean_reward curves (GRPO stages only)
    C: task_acc_by_depth bar chart (final checkpoint per stage)
    D: exposure bias (kl_student_prefix - kl_teacher_prefix, KD/OPD stages)

Usage:
    python visualize_loss.py --config config/teacher_pretrain.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

# stage order determines legend ordering and color assignment
STAGE_ORDER = [
    'teacher_pretrain', 'teacher_sft', 'teacher_grpo', 'teacher_sdpo',
    'student_pretrain', 'student_sft',
    'student_kd', 'student_opd', 'student_grpo',
]

COLORS = {
    'teacher_pretrain': '#1f77b4',
    'teacher_sft':      '#aec7e8',
    'teacher_grpo':     '#ff7f0e',
    'teacher_sdpo':     '#ffbb78',
    'student_pretrain': '#2ca02c',
    'student_sft':      '#98df8a',
    'student_kd':       '#d62728',
    'student_opd':      '#9467bd',
    'student_grpo':     '#8c564b',
}


def load_all_metrics(log_root: str = 'log') -> Dict[str, List[Dict]]:
    """
    Scan log/*/metrics.jsonl and group by stage name.
    Returns {stage_name: [row_dict, ...]}.
    """
    data = {}
    log_path = Path(log_root)
    if not log_path.exists():
        return data

    for stage_dir in sorted(log_path.iterdir()):
        if not stage_dir.is_dir():
            continue
        mpath = stage_dir / 'metrics.jsonl'
        if not mpath.exists():
            continue
        rows = []
        with open(mpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            data[stage_dir.name] = rows
    return data


def _sorted_stages(data: Dict) -> List[str]:
    known   = [s for s in STAGE_ORDER if s in data]
    unknown = [s for s in data if s not in STAGE_ORDER]
    return known + unknown


def _color(stage: str) -> str:
    return COLORS.get(stage, '#7f7f7f')


def _col(rows: List[Dict], key: str):
    """Extract a field from row list (skip None values), returns (steps, values) arrays."""
    steps, vals = [], []
    for r in rows:
        v = r.get(key)
        if v is not None:
            steps.append(r['step'])
            vals.append(v)
    return np.array(steps), np.array(vals)


# ─────────────────────────────────────────────────────────────────────────────
# Plot A: Loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(data: Dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_train, ax_val = axes

    for stage in _sorted_stages(data):
        rows  = data[stage]
        color = _color(stage)

        steps_t, train_loss = _col(rows, 'train_loss')
        steps_v, val_loss   = _col(rows, 'val_loss')

        if len(steps_t):
            ax_train.plot(steps_t, train_loss, label=stage, color=color)
        if len(steps_v):
            ax_val.plot(steps_v, val_loss, label=stage, color=color, linestyle='--')

    for ax, title in [(ax_train, 'Train Loss'), (ax_val, 'Val Loss')]:
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / 'A_loss_curves.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot A saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot B: Reward curves (GRPO stages)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_curves(data: Dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))

    has_data = False
    for stage in _sorted_stages(data):
        steps, rewards = _col(data[stage], 'mean_reward')
        if len(steps):
            ax.plot(steps, rewards, label=stage, color=_color(stage))
            has_data = True

    if not has_data:
        plt.close(fig)
        print("  Plot B: no GRPO reward data, skipped.")
        return

    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Verifier Reward (GRPO Stages)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / 'B_reward_curves.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot B saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot C: Task accuracy by depth (final checkpoint bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_acc_by_depth(data: Dict, out_dir: Path):
    stages = _sorted_stages(data)
    n      = len(stages)
    if n == 0:
        print("  Plot C: no data, skipped.")
        return

    depths = list(range(6))
    x      = np.arange(len(depths))
    width  = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, stage in enumerate(stages):
        rows = data[stage]
        accs = None
        for row in reversed(rows):
            v = row.get('task_acc_by_depth')
            if v is not None:
                accs = v
                break
        if accs is None:
            continue

        accs_padded = (list(accs) + [0.0] * max(0, 6 - len(accs)))[:6]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, accs_padded, width=width,
               label=stage, color=_color(stage), alpha=0.85)

    ax.set_xlabel('Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task Accuracy by Depth (Final Checkpoint)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={d}' for d in depths])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    out = out_dir / 'C_acc_by_depth.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot C saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot D: Exposure bias (kl_student_prefix - kl_teacher_prefix over steps)
# ─────────────────────────────────────────────────────────────────────────────

def plot_exposure_bias(data: Dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_abs, ax_bias = axes

    has_data = False
    for stage in _sorted_stages(data):
        rows  = data[stage]
        color = _color(stage)

        steps_s, kl_s = _col(rows, 'kl_student_prefix')
        steps_t, kl_t = _col(rows, 'kl_teacher_prefix')

        if len(steps_s):
            ax_abs.plot(steps_s, kl_s, label=f'{stage} (student prefix)',
                        color=color, linestyle='-')
            has_data = True
        if len(steps_t):
            ax_abs.plot(steps_t, kl_t, label=f'{stage} (teacher prefix)',
                        color=color, linestyle='--')

        common_steps = set(steps_s.tolist()) & set(steps_t.tolist())
        if common_steps:
            s2kl_s = dict(zip(steps_s.tolist(), kl_s.tolist()))
            s2kl_t = dict(zip(steps_t.tolist(), kl_t.tolist()))
            cs = sorted(common_steps)
            bias = [s2kl_s[s] - s2kl_t[s] for s in cs]
            ax_bias.plot(cs, bias, label=stage, color=color)

    if not has_data:
        plt.close(fig)
        print("  Plot D: no KL data (normal for pretrain stages), skipped.")
        return

    ax_abs.set_title('KL on Teacher / Student Prefix')
    ax_abs.set_xlabel('Step')
    ax_abs.set_ylabel('KL Divergence')
    ax_abs.legend(fontsize=6)
    ax_abs.grid(True, alpha=0.3)

    ax_bias.set_title('Exposure Bias (student_prefix_KL - teacher_prefix_KL)')
    ax_bias.set_xlabel('Step')
    ax_bias.set_ylabel('Delta KL')
    ax_bias.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax_bias.legend(fontsize=8)
    ax_bias.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / 'D_exposure_bias.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot D saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Loss / accuracy visualizer')
    parser.add_argument('--config', required=True,
                        help='Training config yaml; determines output fig dir (log/<name>/fig/)')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg['output']['log_dir']
    out_dir = Path(log_dir) / 'fig'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning log/*/metrics.jsonl ...")
    data = load_all_metrics('log')
    if not data:
        print("No metrics.jsonl files found, exiting.")
        return

    print(f"Found {len(data)} stage(s): {list(data.keys())}\n")

    plot_loss_curves(data, out_dir)
    plot_reward_curves(data, out_dir)
    plot_acc_by_depth(data, out_dir)
    plot_exposure_bias(data, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == '__main__':
    main()
