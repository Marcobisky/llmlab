"""
kd.py - Off-policy knowledge distillation for the student model.

The frozen teacher samples prefixes; the student learns the teacher's full
next-token distribution on those teacher prefixes.
"""
import os
from pathlib import Path

from lib.distill import parse_args, run_distillation


if __name__ == '__main__':
    args = parse_args('Student KD training')
    os.chdir(Path(__file__).parent)
    run_distillation(args.config, mode='kd')
