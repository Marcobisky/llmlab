"""
data.py — 数据集生成器。
读入 config/ 中指定的 yaml，生成满足配置的 data/*.jsonl。

用法：
  python data.py                          # 使用文件顶部 CONFIG_YAML 指定的配置
  python data.py config/eval.yaml         # 命令行指定（可覆盖顶部变量）

字段规范（§2.5）：
  prompt  : str  — 模型输入前缀（含 [BOS]，空格分隔 token）
  target  : str  — 需要预测的 token 串（空格分隔）
  expr    : str  — 原始表达式字符串
  result  : str  — 解释器计算的正确结果
  depth   : int  — 组合深度 = n_unary + (n_operands - 1)
  type    : str  — 'stmt' | 'check' | 'cot'
  split   : str  — 'teacher_train' | 'student_train' | 'eval'
"""
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

# ── 用户配置：修改此处选择要生成的数据集 ─────────────────────────────────────
CONFIG_YAML = "config/teacher_pretrain.yaml"
# ─────────────────────────────────────────────────────────────────────────────

# 把 src/ 加入路径，使 lib.lang 可 import
sys.path.insert(0, str(Path(__file__).parent))
from lib.lang import (
    ALL_UNARY, ALL_BINARY,
    interpret, build_cot_trace,
    sample_expr, make_invalid_expr,
    make_wrong_candidate, verify,
    expr_to_tokens, tokens_to_str,
)


# ─────────────────────────────────────────────────────────────────────────────
# Token 串构建（写入 jsonl 的 prompt / target 字段）
# ─────────────────────────────────────────────────────────────────────────────

def _expr_str(expr: str) -> str:
    """紧凑表达式 → 空格分隔 token 串。'ir23c0' → 'i r 2 3 c 0'"""
    return tokens_to_str(expr_to_tokens(expr))


def _build_stmt(expr: str, result: str, split: str) -> Dict:
    """
    陈述句（stmt）。
      pretrain: prompt = '[BOS] EXPR'，target = '= RESULT [EOS]'
      full_text = '[BOS] EXPR = RESULT [EOS]'
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
    检查句（check）。
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
    CoT 句（cot）。
      prompt = '[BOS] EXPR'
      target = '<think> TRACE </think> = RESULT [EOS]'
    TRACE 内部 '=' 已在 build_cot_trace 中生成，这里逐步 token 化。

    trace 格式（紧凑）: 'ir23c0=ir230=i032=032'
    目标格式: '<think> i r 2 3 c 0 = i r 2 3 0 = i 0 3 2 = 0 3 2 </think> = 0 3 2 [EOS]'
    """
    expr_t   = _expr_str(expr)
    result_t = _expr_str(result)

    # 将 trace 中每个步骤 token 化，用 ' = ' 连接
    steps = trace.split('=')
    trace_t = ' = '.join(_expr_str(s) for s in steps)

    prompt = f"[BOS] {expr_t}"
    target = f"<think> {trace_t} </think> = {result_t} [EOS]"
    return dict(prompt=prompt, target=target, expr=expr,
                result=result, depth=None, type='cot', split=split)


# ─────────────────────────────────────────────────────────────────────────────
# 单样本生成
# ─────────────────────────────────────────────────────────────────────────────

def _sample_depth_and_operands(
    depth_range: List[int],
    n_operands_range: List[int],
    rng: random.Random,
) -> tuple:
    """
    采样 (total_depth, n_operands, n_unary)，满足：
      n_unary = total_depth - (n_operands - 1) >= 0
    """
    d_min, d_max = depth_range
    no_min, no_max = n_operands_range

    total_depth = rng.randint(d_min, d_max)

    # 保证 n_operands - 1 <= total_depth
    op_max = min(no_max, total_depth + 1)
    op_min = max(no_min, 1)
    if op_min > op_max:
        op_max = op_min

    n_operands = rng.randint(op_min, op_max)
    n_unary = total_depth - (n_operands - 1)
    return total_depth, n_operands, n_unary


def _token_len(rec: Dict) -> int:
    """计算 prompt + target 合并后的 token 数（空格分隔）。"""
    return len((rec['prompt'] + ' ' + rec['target']).split())


def _generate_one(
    cfg: Dict,
    rng: random.Random,
    used_exprs: Set[str],
    split: str,
    sentence_type: str,   # 'stmt' | 'cot' | 'check_vc' | 'check_vw' | 'check_gw'
    context_len: int = 512,   # 超过此长度的样本被丢弃
    max_tries: int = 200,
) -> Optional[Dict]:
    """
    生成一条样本记录，保证 expr 不在 used_exprs 中（去重）。
    失败时返回 None。
    """
    depth_range       = cfg['depth_range']
    n_operands_range  = cfg['n_operands_range']
    operand_len_range = cfg['operand_len_range']
    ops_enabled       = cfg['ops_enabled']
    vw_corruption     = cfg.get('vw_corruption', 'swap')

    for _ in range(max_tries):
        if sentence_type == 'check_gw':
            # 非法表达式长度很短，不需要长度过滤
            # 非法表达式：纯算子，无操作数
            expr = make_invalid_expr(ops_enabled, rng)
            if expr in used_exprs:
                continue
            # 候选：随机数字串
            cand_len = rng.randint(1, 4)
            candidate = ''.join(str(rng.randint(0, 9)) for _ in range(cand_len))
            used_exprs.add(expr)
            rec = _build_check(expr, candidate, 'gw', result=None, split=split)
            rec['depth'] = 0   # gw 无意义，填 0
            return rec

        # 合法表达式
        total_depth, n_operands, n_unary = _sample_depth_and_operands(
            depth_range, n_operands_range, rng)
        expr = sample_expr(n_unary, n_operands, tuple(operand_len_range),
                           ops_enabled, rng)
        if expr in used_exprs:
            continue

        result, is_valid = interpret(expr)
        if not is_valid:
            continue  # 极少数边界情况（如空串操作数）跳过

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
                continue  # CoT 轨迹过长，重新采样
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

    return None  # 超过重试次数，跳过


# ─────────────────────────────────────────────────────────────────────────────
# 数据集生成主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def _build_type_schedule(n_samples: int, cfg: Dict, rng: random.Random) -> List[str]:
    """
    根据 check_fraction / cot_fraction / verdict_ratio 生成每条样本的句型标签列表。
    返回长度为 n_samples 的列表，元素为 'stmt'|'cot'|'check_vc'|'check_vw'|'check_gw'。
    """
    check_frac = cfg.get('check_fraction', 0.0)
    cot_frac   = cfg.get('cot_fraction',   0.0)
    stmt_frac  = max(0.0, 1.0 - check_frac - cot_frac)

    n_check = int(n_samples * check_frac)
    n_cot   = int(n_samples * cot_frac)
    n_stmt  = n_samples - n_check - n_cot

    # 按 verdict_ratio 分配 check 子类
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
    生成整个数据集，返回记录列表。
    used_exprs 是全局已用 expr 集合（in-place 更新，确保跨数据集去重）。
    context_len：超过此 token 数的样本被丢弃（防止超出模型上下文长度）。
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
            print(f"  {i+1}/{n_samples} 已生成，跳过 {skipped} 条")

    print(f"  完成：{len(records)} 条（跳过 {skipped} 条）")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 从已有 jsonl 加载 expr 集合（eval 去重用）
# ─────────────────────────────────────────────────────────────────────────────

def _load_exprs_from_jsonl(path: str) -> Set[str]:
    """从已存在的 .jsonl 文件中读取所有 expr 字段，返回集合。"""
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
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 命令行可覆盖 CONFIG_YAML
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_YAML

    # 切换工作目录到 src/（使相对路径生效）
    src_dir = Path(__file__).parent
    os.chdir(src_dir)

    print(f"读取配置：{config_path}")
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)

    data_cfg = full_cfg['data']
    split    = data_cfg['split']
    out_path = Path(data_cfg['path'])
    seed     = data_cfg.get('seed', 42)

    rng = random.Random(seed)

    # 若有 exclude_from（eval 去重），先加载已有训练集 expr
    used_exprs: Set[str] = set()
    for excl in data_cfg.get('exclude_from', []):
        before = len(used_exprs)
        used_exprs |= _load_exprs_from_jsonl(excl)
        print(f"  排除 {excl}：加入 {len(used_exprs)-before} 个 expr")

    context_len = full_cfg.get('model', {}).get('context_len', 512)
    print(f"生成 {data_cfg['n_samples']} 条 [{split}] → {out_path}  (context_len={context_len})")
    records = generate_dataset(data_cfg, split, used_exprs, rng, context_len=context_len)

    # 写入 jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"已写入 {out_path}（{len(records)} 条）")

    # 简单统计
    from collections import Counter
    type_counts  = Counter(r['type']  for r in records)
    depth_counts = Counter(r['depth'] for r in records)
    print(f"  句型分布：{dict(type_counts)}")
    print(f"  depth 分布：{dict(sorted(depth_counts.items()))}")


if __name__ == '__main__':
    main()
