"""
sft.py — Supervised fine-tuning (SFT) for teacher and student models.

Starts from a pretrain checkpoint and fine-tunes on CoT (or stmt) data,
reinforcing chain-of-thought generation ability.

Key differences from pretraining (expressed in the config file):
    data.path             CoT dataset (teacher_sft.jsonl or student_sft.jsonl)
    train.base_model_path warm-start from pretrain checkpoint
    train.lr              10x smaller than pretrain to preserve learned knowledge
    model.dropout         0.0 during fine-tuning

Training logic is fully reused from pretrain.py.

Usage:
    python sft.py --config config/teacher_sft.yaml
    python sft.py --config config/student_sft.yaml
"""
import argparse
import os
import sys
from pathlib import Path


def main(config_path: str):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # sanity check: base model must exist before loading data
    base = cfg.get('train', {}).get('base_model_path', '')
    if base and not Path(base).exists():
        print(f"Error: base_model_path not found: {base}")
        print("  Run pretrain.py first to produce the base checkpoint.")
        sys.exit(1)

    # reuse pretrain.main() — all behaviour is config-driven
    from pretrain import main as pretrain_main
    pretrain_main(config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised fine-tuning')
    parser.add_argument('--config', required=True,
                        help='Path to training config yaml (e.g. config/teacher_sft.yaml)')
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, str(Path(__file__).parent))
    main(args.config)
