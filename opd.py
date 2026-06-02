"""
opd.py - On-policy distillation for the student model.

The current student samples prefixes; the student then matches the frozen
teacher distribution on its own visited prefixes.
"""
import os
from pathlib import Path

from lib.distill import parse_args, run_distillation


if __name__ == '__main__':
    args = parse_args('Student OPD training')
    os.chdir(Path(__file__).parent)
    run_distillation(args.config, mode='opd')
