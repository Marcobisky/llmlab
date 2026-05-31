"""
lib/lang.py — 语言定义、解释器、表达式采样、CoT 轨迹、Verifier。
唯一真相源，被 data.py / reward.py 等模块 import。
"""
import random
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. 算子定义
# ─────────────────────────────────────────────────────────────────────────────

UNARY = {
    'i': lambda x: x,
    'r': lambda x: x[::-1],
    's': lambda x: ''.join(sorted(x)),
    'S': lambda x: ''.join(sorted(x, reverse=True)),
    'd': lambda x: ''.join(c for i, c in enumerate(x) if i == 0 or c != x[i - 1]),
    'D': lambda x: ''.join(c * 2 for c in x),
    'h': lambda x: x[: len(x) // 2],
    't': lambda x: x[len(x) // 2 :],
    'e': lambda x: x[0::2],
    'o': lambda x: x[1::2],
    'L': lambda x: x[1:] + x[:1] if x else '',
    'R': lambda x: x[-1:] + x[:-1] if x else '',
    'p': lambda x: x + x[::-1],
}

BINARY = {
    'c': lambda a, b: a + b,
    'z': lambda a, b: ''.join(x for pair in zip(a, b) for x in pair)
                      + a[len(b):] + b[len(a):],
}

ALL_UNARY  = list(UNARY.keys())   # 13 算子，顺序固定（Python 3.7+ dict 有序）
ALL_BINARY = list(BINARY.keys())  # 2 算子

# ─────────────────────────────────────────────────────────────────────────────
# 2. 词表（V=32，启用 CoT 时 V=34）
# ─────────────────────────────────────────────────────────────────────────────

VOCAB: List[str] = (
    [str(i) for i in range(10)]  # 0-9        (10)
    + ALL_UNARY                   # 一元算子   (13)
    + ALL_BINARY                  # 二元算子    (2)
    + ['=', '?']                  # 特殊符号    (2)
    + ['vc', 'vw', 'gw']         # 判定        (3)
    + ['[BOS]', '[EOS]']         # 控制        (2)
)  # 共 32 个

VOCAB_COT: List[str] = VOCAB + ['<think>', '</think>']  # 共 34 个

TOKEN2ID = {t: i for i, t in enumerate(VOCAB_COT)}  # 全集（34）

# ─────────────────────────────────────────────────────────────────────────────
# 3. 解释器（对应 README §1.7）
# ─────────────────────────────────────────────────────────────────────────────

def interpret(expr: str) -> Tuple[Optional[str], bool]:
    """
    解析并求值 expr。
    返回 (result_str, is_valid)：不合法时 result_str=None, is_valid=False。
    """
    result, rest = _parse(expr)
    if result is None or rest != '':
        return None, False
    return result, True


def _parse(s: str) -> Tuple[Optional[str], str]:
    """
    递归下降，对应 §1.4 BNF。
    返回 (value, remaining_str)。
    """
    if not s:
        return None, ''
    if s[0] in UNARY:
        inner, rest = _parse(s[1:])
        if inner is None:
            return None, ''
        return UNARY[s[0]](inner), rest
    if s[0].isdigit():
        j = 0
        while j < len(s) and s[j].isdigit():
            j += 1
        left = s[:j]
        if j < len(s) and s[j] in BINARY:
            right, rest = _parse(s[j + 1:])
            if right is None:
                return None, ''
            return BINARY[s[j]](left, right), rest
        return left, s[j:]
    return None, ''

# ─────────────────────────────────────────────────────────────────────────────
# 4. CoT 归约轨迹（对应 README §1.5）
# ─────────────────────────────────────────────────────────────────────────────
# AST 节点类型（用 tuple 表示，避免引入 dataclass）：
#   ('digit',  value_str)
#   ('unary',  op_char, child_node)
#   ('binary', op_char, left_str, right_node)   # left_str 永远是数字串

def _parse_tree(s: str):
    """将 expr 字符串解析为 AST，返回 (node, rest)。"""
    if not s:
        return None, ''
    if s[0] in UNARY:
        child, rest = _parse_tree(s[1:])
        if child is None:
            return None, ''
        return ('unary', s[0], child), rest
    if s[0].isdigit():
        j = 0
        while j < len(s) and s[j].isdigit():
            j += 1
        left = s[:j]
        if j < len(s) and s[j] in BINARY:
            right, rest = _parse_tree(s[j + 1:])
            if right is None:
                return None, ''
            return ('binary', s[j], left, right), rest
        return ('digit', left), s[j:]
    return None, ''


def _node_to_str(node) -> str:
    """AST → 表达式字符串（无空格的紧凑形式）。"""
    if node[0] == 'digit':
        return node[1]
    if node[0] == 'unary':
        return node[1] + _node_to_str(node[2])
    # binary
    return node[2] + node[1] + _node_to_str(node[3])


def _reduce_one(node):
    """
    在 AST 中找最内层可归约节点，归约一步。
    返回 (new_node, changed: bool)。
    策略：深度优先，优先右侧（右侧 = 更内层）。
    """
    if node[0] == 'digit':
        return node, False

    if node[0] == 'unary':
        op, child = node[1], node[2]
        new_child, changed = _reduce_one(child)
        if changed:
            return ('unary', op, new_child), True
        # child 已是 digit，直接求值
        return ('digit', UNARY[op](child[1])), True

    # binary：left 已是 str，right 是子树
    op, left_str, right = node[1], node[2], node[3]
    new_right, changed = _reduce_one(right)
    if changed:
        return ('binary', op, left_str, new_right), True
    # right 已是 digit，直接求值
    return ('digit', BINARY[op](left_str, right[1])), True


def build_cot_trace(expr: str) -> Optional[str]:
    """
    生成 CoT 归约轨迹字符串（紧凑形式，用 '=' 分隔每步）。
    格式：'EXPR=step1=...=result'
    返回 None 若 expr 不合法。

    示例：build_cot_trace('ir23c0') → 'ir23c0=ir230=i032=032'
    """
    tree, rest = _parse_tree(expr)
    if tree is None or rest != '':
        return None

    steps = [expr]
    node = tree
    while node[0] != 'digit':
        node, _ = _reduce_one(node)
        steps.append(_node_to_str(node))

    return '='.join(steps)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 表达式采样（对应 README §2.1）
# ─────────────────────────────────────────────────────────────────────────────

def sample_expr(
    n_unary: int,                        # 前缀一元算子数
    n_operands: int,                     # 操作数个数（→ n_operands-1 个二元算子）
    operand_len_range: Tuple[int, int],  # 每个操作数的位数范围 [min, max]
    ops_enabled: List[str] | str,        # 可用算子列表，'all' 表全部
    rng: random.Random,
) -> str:
    """
    生成一个随机表达式字符串。
    depth_field = n_unary + (n_operands - 1)。

    步骤：
      1. 采样 n_operands 个随机数字串
      2. 用 n_operands-1 个随机二元算子右结合拼接
         例：[A, B, C] + [op1, op2] → 'A op1 B op2 C'（解析为 op1(A, op2(B,C))）
      3. 最外层前缀 n_unary 个随机一元算子
    """
    if ops_enabled == 'all':
        unary_pool  = ALL_UNARY
        binary_pool = ALL_BINARY
    else:
        unary_pool  = [o for o in ops_enabled if o in UNARY]
        binary_pool = [o for o in ops_enabled if o in BINARY]

    # 1. 采样操作数
    lo, hi = operand_len_range
    operands = [
        ''.join(str(rng.randint(0, 9)) for _ in range(rng.randint(lo, hi)))
        for _ in range(n_operands)
    ]

    # 2. 右结合拼接（从右向左）
    expr = operands[-1]
    for i in range(n_operands - 2, -1, -1):
        bop = rng.choice(binary_pool) if binary_pool else 'c'
        expr = operands[i] + bop + expr

    # 3. 前缀一元算子
    for _ in range(n_unary):
        uop = rng.choice(unary_pool)
        expr = uop + expr

    return expr


def make_invalid_expr(ops_enabled: List[str] | str, rng: random.Random) -> str:
    """
    生成一个语法不合法的表达式（纯算子，无操作数）。
    保证 interpret() 返回 is_valid=False。
    """
    pool = ALL_UNARY if ops_enabled == 'all' else [o for o in ops_enabled if o in UNARY]
    n = rng.randint(1, 3)
    return ''.join(rng.choice(pool) for _ in range(n))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Verifier & 错误候选生成（对应 README §1.6）
# ─────────────────────────────────────────────────────────────────────────────

def verify(expr: str, candidate: str) -> str:
    """返回 'vc' | 'vw' | 'gw'。"""
    result, is_valid = interpret(expr)
    if not is_valid:
        return 'gw'
    return 'vc' if candidate == result else 'vw'


def make_wrong_candidate(
    correct: str,
    corruption: str,  # 'swap' | 'replace' | 'truncate' | 'shuffle'
    rng: random.Random,
) -> str:
    """生成一个与 correct 不同的错误候选串（用于生成 vw 检查句）。"""
    for _ in range(20):
        c = _corrupt(correct, corruption, rng)
        if c != correct:
            return c
    # fallback：翻转首位
    return str((int(correct[0]) + 1) % 10) + correct[1:]


def _corrupt(s: str, corruption: str, rng: random.Random) -> str:
    if not s:
        return '0'
    if len(s) == 1:
        return str((int(s) + 1) % 10)

    if corruption == 'swap':
        i, j = rng.sample(range(len(s)), 2)
        lst = list(s)
        lst[i], lst[j] = lst[j], lst[i]
        return ''.join(lst)

    if corruption == 'replace':
        k = rng.randint(1, max(1, len(s) // 2))
        lst = list(s)
        for _ in range(k):
            lst[rng.randrange(len(lst))] = str(rng.randint(0, 9))
        return ''.join(lst)

    if corruption == 'truncate':
        cut = rng.randint(1, len(s) - 1)
        return s[:cut]

    if corruption == 'shuffle':
        lst = list(s)
        rng.shuffle(lst)
        return ''.join(lst)

    return _corrupt(s, 'swap', rng)  # fallback

# ─────────────────────────────────────────────────────────────────────────────
# 7. Token 化工具（供 model.py / data.py 使用）
# ─────────────────────────────────────────────────────────────────────────────

_SPECIAL_TOKENS = ['[BOS]', '[EOS]', '<think>', '</think>', 'vc', 'vw', 'gw']

def tokenize(s: str) -> List[str]:
    """
    将含特殊 token 的字符串拆分为 token 列表。
    多字符特殊 token 优先匹配，其余按字符拆。
    空格视为分隔符，忽略。

    示例：tokenize('[BOS] r 1 2 3 = 3 2 1 [EOS]')
          → ['[BOS]', 'r', '1', '2', '3', '=', '3', '2', '1', '[EOS]']
    """
    tokens = []
    i = 0
    while i < len(s):
        if s[i] == ' ':
            i += 1
            continue
        matched = False
        for sp in _SPECIAL_TOKENS:
            if s[i:i + len(sp)] == sp:
                tokens.append(sp)
                i += len(sp)
                matched = True
                break
        if not matched:
            tokens.append(s[i])
            i += 1
    return tokens


def expr_to_tokens(expr: str) -> List[str]:
    """
    将紧凑表达式字符串（如 'ir23c0'）转为字符 token 列表。
    expr 中仅含单字符 token（算子/数字），直接拆字符。

    示例：expr_to_tokens('ir23c0') → ['i', 'r', '2', '3', 'c', '0']
    """
    return list(expr)


def tokens_to_str(tokens: List[str]) -> str:
    """token 列表 → 空格分隔字符串（写入 jsonl 的格式）。"""
    return ' '.join(tokens)

# ─────────────────────────────────────────────────────────────────────────────
# 8. 单元测试（python src/lib/lang.py 直接运行）
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    print("=== 解释器测试 ===")

    cases = [
        # (expr,        expected_result, expected_valid)
        ('1234',        '1234',          True),
        ('i1234',       '1234',          True),
        ('r1234',       '4321',          True),
        ('s3142',       '1234',          True),
        ('S3142',       '4321',          True),
        ('L1234',       '2341',          True),
        ('R1234',       '4123',          True),
        ('d11223',      '123',           True),
        ('h12345',      '12',            True),
        ('t12345',      '345',           True),
        ('e12345',      '135',           True),
        ('o12345',      '24',            True),
        ('D123',        '112233',        True),
        ('p123',        '123321',        True),
        ('12c34',       '1234',          True),
        ('12z34',       '1324',          True),
        ('12z345',      '13245',         True),
        # README 验证示例
        ('rs18273399',  '99873321',      True),
        ('ir23c0',      '032',           True),
        ('1cr343c98',   '189343',        True),
        # 不合法
        ('rs',          None,            False),
        ('',            None,            False),
    ]

    all_pass = True
    for expr, expected, valid in cases:
        result, is_valid = interpret(expr)
        ok = (result == expected) and (is_valid == valid)
        status = '✓' if ok else '✗'
        if not ok:
            all_pass = False
            print(f"  {status} interpret({expr!r}) = ({result!r}, {is_valid}) "
                  f"expected ({expected!r}, {valid})")
        else:
            print(f"  {status} interpret({expr!r}) = {result!r}")

    print("\n=== CoT 轨迹测试 ===")
    cot_cases = [
        ('ir23c0',   'ir23c0=ir230=i032=032'),
        ('1234',     '1234'),
        ('r1234',    'r1234=4321'),
        ('1cr343c98', '1cr343c98=1cr34398=1c89343=189343'),
    ]
    for expr, expected in cot_cases:
        trace = build_cot_trace(expr)
        ok = trace == expected
        status = '✓' if ok else '✗'
        if not ok:
            all_pass = False
        print(f"  {status} build_cot_trace({expr!r}) = {trace!r}")

    print("\n=== Verifier 测试 ===")
    assert verify('r138', '831') == 'vc'
    assert verify('r138', '813') == 'vw'
    assert verify('rs',   '9201') == 'gw'
    print("  ✓ verify('r138','831')  = 'vc'")
    print("  ✓ verify('r138','813')  = 'vw'")
    print("  ✓ verify('rs','9201')   = 'gw'")

    print()
    print("全部通过 ✓" if all_pass else "有失败项 ✗")


if __name__ == '__main__':
    _run_tests()
