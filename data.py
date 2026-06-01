"""
data.py — Dataset generator.
Reads a data config yaml and writes the corresponding data/*.jsonl file.

Usage:
    python data.py --config config/expr_500k_depth5.yaml
    python data.py --config config/eval_10k_depth5.yaml

Record schema:
    prompt  : str  — model input prefix (includes [BOS], space-separated tokens)
    target  : str  — tokens to predict (space-separated)
    expr    : str  — raw expression string
    result  : str  — interpreter-computed correct result
    depth   : int  — composition depth = n_unary + (n_operands - 1)
    type    : str  — 'stmt' | 'check' | 'cot'
    split   : str  — 'teacher_train' | 'student_train' | 'eval'
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang import (
    ALL_UNARY, ALL_BINARY,
    interpret, build_cot_trace,
    sample_expr, make_invalid_expr,
    make_wrong_candidate, verify,
    expr_to_tokens, tokens_to_str,
)


# ─────────────────────────────────────────────────────────────────────────────
# Token string builders (written to jsonl prompt / target fields)
# ─────────────────────────────────────────────────────────────────────────────

def _expr_str(expr: str) -> str:
    """Compact expression -> space-separated token string. 'ir23c0' -> 'i r 2 3 c 0'"""
    return tokens_to_str(expr_to_tokens(expr))


def _build_stmt(expr: str, result: str, split: str) -> Dict:
    """
    Statement (stmt).
        prompt = '[BOS] EXPR'
        target = '= RESULT [EOS]'
    """
    expr_t   = _expr_str(expr)
    result_t = _expr_str(result)
    prompt = f"[BOS] {expr_t}"
    target = f"= {result_t} [EOS]"
    return dict(prompt=prompt, target=target, expr=expr,
                result=result, depth=None, type='stmt', split=split)


def _build_check(expr: str, candidate: str, verdict: str,
                 result: Optional[str], split: str) -> Dict:
    """
    Check sentence.
        prompt = '[BOS] EXPR = CANDIDATE ?'
        target = 'VERDICT [EOS]'
    """
    expr_t      = _expr_str(expr)
    candidate_t = _expr_str(candidate) if candidate.isdigit() or candidate == '' else candidate
    prompt = f"[BOS] {expr_t} = {candidate_t} ?"
    target = f"{verdict} [EOS]"
    return dict(prompt=prompt, target=target, expr=expr,
                result=result, depth=None, type='check', split=split)


def _build_cot(expr: str, trace: str, result: str, split: str) -> Dict:
    """
    CoT sentence.
        prompt = '[BOS] EXPR'
        target = '<think> TRACE </think> = RESULT [EOS]'

    trace format (compact): 'ir23c0=ir230=i032=032'
    target format: '<think> i r 2 3 c 0 = i r 2 3 0 = i 0 3 2 = 0 3 2 </think> = 0 3 2 [EOS]'
    """
    expr_t   = _expr_str(expr)
    result_t = _expr_str(result)

    steps = trace.split('=')
    trace_t = ' = '.join(_expr_str(s) for s in steps)

    prompt = f"[BOS] {expr_t}"
    target = f"<think> {trace_t} </think> = {result_t} [EOS]"
    return dict(prompt=prompt, target=target, expr=expr,
                result=result, depth=None, type='cot', split=split)


# ─────────────────────────────────────────────────────────────────────────────
# Single sample generation
# ─────────────────────────────────────────────────────────────────────────────

def _sample_depth_and_operands(
    depth_range: List[int],
    n_operands_range: List[int],
    rng: random.Random,
) -> tuple:
    """
    Sample (total_depth, n_operands, n_unary) satisfying:
        n_unary = total_depth - (n_operands - 1) >= 0
    """
    d_min, d_max = depth_range
    no_min, no_max = n_operands_range

    total_depth = rng.randint(d_min, d_max)

    op_max = min(no_max, total_depth + 1)
    op_min = max(no_min, 1)
    if op_min > op_max:
        op_max = op_min

    n_operands = rng.randint(op_min, op_max)
    n_unary = total_depth - (n_operands - 1)
    return total_depth, n_operands, n_unary


def _token_len(rec: Dict) -> int:
    """Total token count of prompt + target combined (space-separated)."""
    return len((rec['prompt'] + ' ' + rec['target']).split())


def _generate_one(
    cfg: Dict,
    rng: random.Random,
    used_exprs: Set[str],
    split: str,
    sentence_type: str,   # 'stmt' | 'cot' | 'check_vc' | 'check_vw' | 'check_gw'
    context_len: int = 512,
    max_tries: int = 200,
) -> Optional[Dict]:
    """
    Generate one sample record, ensuring expr is not in used_exprs (deduplication).
    Returns None if max_tries is exceeded.
    """
    depth_range       = cfg['depth_range']
    n_operands_range  = cfg['n_operands_range']
    operand_len_range = cfg['operand_len_range']
    ops_enabled       = cfg['ops_enabled']
    vw_corruption     = cfg.get('vw_corruption', 'swap')

    for _ in range(max_tries):
        if sentence_type == 'check_gw':
            expr = make_invalid_expr(ops_enabled, rng)
            if expr in used_exprs:
                continue
            cand_len = rng.randint(1, 4)
            candidate = ''.join(str(rng.randint(0, 9)) for _ in range(cand_len))
            used_exprs.add(expr)
            rec = _build_check(expr, candidate, 'gw', result=None, split=split)
            rec['depth'] = 0
            return rec

        total_depth, n_operands, n_unary = _sample_depth_and_operands(
            depth_range, n_operands_range, rng)
        expr = sample_expr(n_unary, n_operands, tuple(operand_len_range),
                           ops_enabled, rng)
        if expr in used_exprs:
            continue

        result, is_valid = interpret(expr)
        if not is_valid:
            continue

        used_exprs.add(expr)

        if sentence_type == 'stmt':
            rec = _build_stmt(expr, result, split)
            rec['depth'] = total_depth
            if _token_len(rec) > context_len:
                continue
            return rec

        if sentence_type == 'cot':
            trace = build_cot_trace(expr)
            if trace is None:
                continue
            rec = _build_cot(expr, trace, result, split)
            rec['depth'] = total_depth
            if _token_len(rec) > context_len:
                continue
            return rec

        if sentence_type == 'check_vc':
            rec = _build_check(expr, result, 'vc', result, split)
            rec['depth'] = total_depth
            if _token_len(rec) > context_len:
                continue
            return rec

        if sentence_type == 'check_vw':
            wrong = make_wrong_candidate(result, vw_corruption, rng)
            rec = _build_check(expr, wrong, 'vw', result, split)
            rec['depth'] = total_depth
            if _token_len(rec) > context_len:
                continue
            return rec

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_type_schedule(n_samples: int, cfg: Dict, rng: random.Random) -> List[str]:
    """
    Build a shuffled list of sentence type labels based on check_fraction,
    cot_fraction, and verdict_ratio.
    Returns a list of length n_samples with elements in
    {'stmt', 'cot', 'check_vc', 'check_vw', 'check_gw'}.
    """
    check_frac = cfg.get('check_fraction', 0.0)
    cot_frac   = cfg.get('cot_fraction',   0.0)

    n_check = int(n_samples * check_frac)
    n_cot   = int(n_samples * cot_frac)
    n_stmt  = n_samples - n_check - n_cot

    vr = cfg.get('verdict_ratio', [5, 3, 2])
    total_vr = sum(vr)
    n_vc = int(n_check * vr[0] / total_vr)
    n_vw = int(n_check * vr[1] / total_vr)
    n_gw = n_check - n_vc - n_vw

    schedule = (
        ['stmt']      * n_stmt  +
        ['cot']       * n_cot   +
        ['check_vc']  * n_vc    +
        ['check_vw']  * n_vw    +
        ['check_gw']  * n_gw
    )
    rng.shuffle(schedule)
    return schedule


def generate_dataset(
    cfg: Dict,
    split: str,
    used_exprs: Set[str],
    rng: random.Random,
    context_len: int = 512,
) -> List[Dict]:
    """
    Generate the full dataset, returning a list of records.
    used_exprs is updated in-place to ensure cross-dataset deduplication.
    Samples exceeding context_len tokens are discarded.
    """
    n_samples = cfg['n_samples']
    schedule  = _build_type_schedule(n_samples, cfg, rng)

    records = []
    skipped = 0
    for i, stype in enumerate(schedule):
        rec = _generate_one(cfg, rng, used_exprs, split, stype,
                            context_len=context_len)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_samples} generated, {skipped} skipped")

    print(f"  Done: {len(records)} records ({skipped} skipped)")
    return records


def _load_exprs_from_jsonl(path: str) -> Set[str]:
    """Load all 'expr' fields from an existing .jsonl file for deduplication."""
    exprs: Set[str] = set()
    p = Path(path)
    if not p.exists():
        return exprs
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if 'expr' in rec and rec['expr']:
                    exprs.add(rec['expr'])
            except json.JSONDecodeError:
                pass
    return exprs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Dataset generator')
    parser.add_argument('--config', required=True,
                        help='Path to data config yaml (e.g. config/expr_500k_depth5.yaml)')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    print(f"Config: {args.config}")
    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    data_cfg    = full_cfg['data']
    split       = data_cfg['split']
    out_path    = Path(data_cfg['path'])
    seed        = data_cfg.get('seed', 42)
    context_len = full_cfg.get('context_len', 512)

    rng = random.Random(seed)

    used_exprs: Set[str] = set()
    for excl in data_cfg.get('exclude_from', []):
        before = len(used_exprs)
        used_exprs |= _load_exprs_from_jsonl(excl)
        print(f"  Excluded {excl}: added {len(used_exprs)-before} exprs")

    print(f"Generating {data_cfg['n_samples']} [{split}] -> {out_path}  (context_len={context_len})")
    records = generate_dataset(data_cfg, split, used_exprs, rng, context_len=context_len)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Written to {out_path} ({len(records)} records)")

    from collections import Counter
    type_counts  = Counter(r['type']  for r in records)
    depth_counts = Counter(r['depth'] for r in records)
    print(f"  Type distribution:  {dict(type_counts)}")
    print(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")


if __name__ == '__main__':
    main()
