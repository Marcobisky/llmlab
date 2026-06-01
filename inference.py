"""
inference.py — 对话式推理终端。
读 config/inference.yaml 获取模型路径和推理参数。

用法：
    python inference.py
    python inference.py config/inference.yaml
    python inference.py --model model/teacher_pretrain.pt

输入格式：
    rs1234           → 生成模式，计算表达式结果
    rs1234=4321?     → 验证模式，判断候选答案是否正确
    :help            → 帮助
    :cot on/off      → 切换 CoT 轨迹显示
    :temp 0.8        → 设置采样温度（0=贪心）
    :model <path>    → 切换模型权重
    :quit            → 退出
"""
import argparse
import os
import sys
from pathlib import Path

# macOS 上 PyTorch 与 conda OpenMP 可能各自加载 libomp.dylib，
# 设置此变量允许共存（不影响正确性，仅推理脚本设置）
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

CONFIG_YAML = "config/inference.yaml"

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang import TOKEN2ID, interpret
from model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Token 常量
# ─────────────────────────────────────────────────────────────────────────────

ID2TOK   = {v: k for k, v in TOKEN2ID.items()}
BOS_ID   = TOKEN2ID['[BOS]']
EOS_ID   = TOKEN2ID['[EOS]']
EQ_ID    = TOKEN2ID['=']
Q_ID     = TOKEN2ID['?']
THINK_ID     = TOKEN2ID.get('<think>',  32)
THINK_END_ID = TOKEN2ID.get('</think>', 33)

# ANSI 颜色（非 TTY 自动降级）
_USE_COLOR = sys.stdout.isatty()
def _c(code: str) -> str:
    return code if _USE_COLOR else ''

RESET = _c('\033[0m');  BOLD  = _c('\033[1m')
GREEN = _c('\033[92m'); RED   = _c('\033[91m')
CYAN  = _c('\033[96m'); GRAY  = _c('\033[90m')
YELLOW = _c('\033[93m')


# ─────────────────────────────────────────────────────────────────────────────
# 表达式 Tokenize
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_expr(expr: str) -> List[int]:
    """
    表达式字符串（无空格）→ token ID 列表。
    逐字符匹配，先尝试双字符（vc/vw/gw）再单字符。
    """
    ids, i = [], 0
    while i < len(expr):
        two = expr[i:i+2]
        if two in TOKEN2ID:
            ids.append(TOKEN2ID[two])
            i += 2
        else:
            ids.append(TOKEN2ID.get(expr[i], 0))
            i += 1
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# 生成
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    prompt_ids: List[int],
    max_new: int,
    temperature: float,
    top_p: float,
    device: str,
) -> List[int]:
    """
    自回归生成，返回新生成的 token ID 列表（不含 prompt）。
    temperature=0 → 贪心；temperature>0 → 采样（可选 top-p nucleus）。
    """
    model.eval()
    context_len = model.pos_emb.num_embeddings   # 96
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    # ids: [1, L_prompt]
    generated = []

    for _ in range(max_new):
        if ids.shape[1] >= context_len:
            break
        logits = model(ids)             # [1, T, V]
        nxt = logits[0, -1]             # [V]  最后位置的 logits

        if temperature == 0.0:
            next_id = int(nxt.argmax())
        else:
            nxt = nxt / temperature
            probs = F.softmax(nxt, dim=-1)
            if top_p < 1.0:
                # Nucleus sampling（top-p）
                sp, si = probs.sort(descending=True)
                cumsum = sp.cumsum(0)
                sp[cumsum - sp > top_p] = 0.0
                sp /= sp.sum().clamp(min=1e-9)
                probs = torch.zeros_like(probs).scatter_(0, si, sp)
            next_id = int(torch.multinomial(probs, 1))

        generated.append(next_id)
        if next_id == EOS_ID:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# 输出解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_generated(gen_ids: List[int]) -> Tuple[str, Optional[List[str]], Optional[str]]:
    """
    解析生成的 token ID 列表，返回 (result, cot_steps, verdict)。

    支持三种格式：
      直接格式   : '= RESULT [EOS]'
      CoT 格式   : '<think> TRACE </think> = RESULT [EOS]'
      验证格式   : 'vc/vw/gw [EOS]'

    cot_steps: List[str] 或 None（每个 step 是一个紧凑表达式字符串）
    verdict  : 'vc'|'vw'|'gw' 或 None
    """
    toks = [ID2TOK.get(i, '?') for i in gen_ids]

    # 验证模式输出
    if toks and toks[0] in ('vc', 'vw', 'gw'):
        return '', None, toks[0]

    # CoT trace 提取
    cot_steps = None
    if '<think>' in toks and '</think>' in toks:
        try:
            s = toks.index('<think>') + 1
            e = toks.index('</think>')
            trace = ''.join(toks[s:e])       # 紧凑字符串，如 'ir23c0=ir230=i032=032'
            cot_steps = [st for st in trace.split('=') if st]
        except ValueError:
            pass

    # 最后一个 '=' 之后的内容为最终结果（兼容直接和 CoT 格式）
    last_eq = max((i for i, t in enumerate(toks) if t == '='), default=-1)
    result = ''
    if last_eq >= 0:
        parts = []
        for t in toks[last_eq + 1:]:
            if t == '[EOS]':
                break
            parts.append(t)
        result = ''.join(parts)

    return result, cot_steps, None


# ─────────────────────────────────────────────────────────────────────────────
# 用户输入解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_input(raw: str) -> Tuple[Optional[str], str, str]:
    """
    解析用户输入，返回 (mode, expr, candidate)。
    mode: 'generate' | 'verify' | None（无效）
    """
    s = raw.strip().replace(' ', '')
    if not s:
        return None, '', ''

    if s.endswith('?') and '=' in s:
        body = s[:-1]
        idx  = body.rfind('=')
        expr, cand = body[:idx], body[idx+1:]
        if expr and cand:
            return 'verify', expr, cand
        return None, '', ''

    return 'generate', s, ''


# ─────────────────────────────────────────────────────────────────────────────
# 显示
# ─────────────────────────────────────────────────────────────────────────────

def display_generate(
    expr: str,
    result: str,
    cot_steps,
    show_cot: bool,
    show_raw: bool = False,
    prompt_ids: List[int] = None,
    gen_ids: List[int] = None,
):
    correct, valid = interpret(expr)

    # ── Raw token sequence ──────────────────────────────────────────────────
    if show_raw and prompt_ids is not None and gen_ids is not None:
        all_ids = prompt_ids + gen_ids
        raw_str = ' '.join(ID2TOK.get(i, f'?{i}') for i in all_ids)
        print(f"\n  {CYAN}Raw tokens{RESET} : {GRAY}{raw_str}{RESET}")

    # ── CoT trace / result ──────────────────────────────────────────────────
    if show_cot and cot_steps:
        print(f"\n  {CYAN}Expression{RESET} : {BOLD}{expr}{RESET}")
        print(f"  {CYAN}CoT Trace{RESET}  :")
        for i, step in enumerate(cot_steps):
            label = "   start" if i == 0 else f"  step {i:2d}"
            print(f"  {GRAY}{label}{RESET} : {step}")
        print(f"  {CYAN}Result    {RESET} : {BOLD}{result}{RESET}")
    else:
        if not show_raw:
            print()
        print(f"  {BOLD}{expr}{RESET}  =  {BOLD}{result}{RESET}")

    # 正确性校验（解释器）
    if valid:
        if result == correct:
            print(f"  {GREEN}✓ Correct{RESET}")
        else:
            print(f"  {RED}✗ Wrong{RESET}  (correct: {BOLD}{correct}{RESET})")
    else:
        print(f"  {YELLOW}⚠ Expression syntax invalid{RESET}")
    print()


def display_verify(expr: str, candidate: str, verdict: str):
    v_style = {
        'vc': (GREEN, '✓ correct'),
        'vw': (RED,   '✗ wrong'),
        'gw': (RED,   '⚠ invalid expression'),
    }
    color, text = v_style.get(verdict, (RESET, verdict))

    print(f"\n  {CYAN}Expression{RESET} : {BOLD}{expr}{RESET}")
    print(f"  {CYAN}Candidate{RESET}  : {candidate}")
    print(f"  {CYAN}Verdict{RESET}    : {color}{BOLD}{verdict}{RESET}  {color}({text}){RESET}")

    if verdict == 'vw':
        correct, ok = interpret(expr)
        if ok:
            print(f"  {CYAN}Correct{RESET}    : {BOLD}{correct}{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, model_cfg: dict, device: str):
    if not Path(model_path).exists():
        print(f"{RED}⚠ 模型文件不存在: {model_path}{RESET}")
        sys.exit(1)
    model = build_model(model_cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    # 推理不用 torch.compile：
    #   1. 交互式单条查询，编译开销远大于收益
    #   2. macOS 上 compile 会二次加载 libomp.dylib，与 conda OpenMP 冲突 → SIGABRT
    return model


def print_help(show_cot: bool, show_raw: bool, temperature: float):
    print(f"""
  {BOLD}Commands{RESET}
    :cot on/off        CoT 轨迹显示（当前: {'on' if show_cot else 'off'}）
    :raw on/off        完整 token 序列显示，含 [BOS]/[EOS]/<think>（当前: {'on' if show_raw else 'off'}）
    :temp <0-2>        采样温度（0=贪心，当前: {temperature:.1f}）
    :model <path>      切换模型权重文件
    :quit              退出

  {BOLD}Input formats{RESET}
    rs1234             → 生成：计算表达式结果
    rs1234=4321?       → 验证：判断候选答案是否正确

  {BOLD}Operators{RESET}
    Unary  : i r s S d D h t e o L R p
    Binary : c (concat)  z (zip/interleave)

  {BOLD}Examples{RESET}
    > rs1234           reverse then sort
    > Dp123            palindrome then double
    > 12c34=1234?      verify concat result
""")


def main():
    parser = argparse.ArgumentParser(description='LLMlab Inference')
    parser.add_argument('config', nargs='?', default=CONFIG_YAML,
                        help='yaml 配置文件路径')
    parser.add_argument('--model', default=None,
                        help='模型权重路径（覆盖 yaml 中的 model_path）')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path  = args.model or cfg.get('model_path', 'model/teacher_pretrain.pt')
    model_cfg   = cfg['model']
    infer_cfg   = cfg.get('inference', {})
    temperature = float(infer_cfg.get('temperature',   0.0))
    top_p       = float(infer_cfg.get('top_p',         1.0))
    max_new     = int(infer_cfg.get('max_new_tokens',  96))
    show_cot    = bool(infer_cfg.get('show_cot',       True))
    show_raw    = bool(infer_cfg.get('show_raw',       False))
    device      = infer_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # ── 加载模型 ────────────────────────────────────────────────────────────
    print(f"加载模型: {model_path}  device={device}")
    model = load_model(model_path, model_cfg, device)
    ctx = model_cfg['context_len']

    # ── 欢迎界面 ────────────────────────────────────────────────────────────
    print()
    print('╔' + '═'*58 + '╗')
    print(f'║  {BOLD}LLMlab Inference{RESET}{" "*42}║')
    print(f'║  model : {Path(model_path).name:<48}║')
    print(f'║  ctx   : {ctx:<3}   device: {device:<43}║')
    print('╚' + '═'*58 + '╝')
    print(f'  输入表达式计算，:help 查看帮助，:quit 退出\n')

    # ── REPL 主循环 ─────────────────────────────────────────────────────────
    def prompt_str():
        parts = []
        if temperature > 0:
            parts.append(f'T={temperature:.1f}')
        if show_cot:
            parts.append('CoT')
        if show_raw:
            parts.append('Raw')
        label = f'[{" ".join(parts)}] ' if parts else ''
        return f'{GRAY}{label}{RESET}> '

    while True:
        try:
            raw = input(prompt_str())
        except (EOFError, KeyboardInterrupt):
            print(f'\n{GRAY}再见！{RESET}')
            break

        raw = raw.strip()
        if not raw:
            continue

        # ── 命令 ────────────────────────────────────────────────────────────
        if raw.startswith(':'):
            parts = raw[1:].strip().lower().split()
            cmd   = parts[0] if parts else ''

            if cmd in ('quit', 'exit', 'q'):
                print(f'{GRAY}再见！{RESET}')
                break

            elif cmd == 'help':
                print_help(show_cot, show_raw, temperature)

            elif cmd == 'cot':
                if len(parts) > 1:
                    show_cot = parts[1] == 'on'
                else:
                    show_cot = not show_cot
                print(f'  CoT: {"on" if show_cot else "off"}')

            elif cmd == 'raw':
                if len(parts) > 1:
                    show_raw = parts[1] == 'on'
                else:
                    show_raw = not show_raw
                print(f'  Raw tokens: {"on" if show_raw else "off"}')

            elif cmd == 'temp':
                if len(parts) > 1:
                    try:
                        temperature = max(0.0, float(parts[1]))
                        print(f'  温度: {temperature:.2f}')
                    except ValueError:
                        print(f'  {RED}⚠ 请输入数字，如 :temp 0.8{RESET}')
                else:
                    print(f'  当前温度: {temperature:.2f}')

            elif cmd == 'model':
                if len(parts) > 1:
                    new_path = parts[1]
                    try:
                        state    = torch.load(new_path, map_location=device, weights_only=True)
                        raw_model = getattr(model, '_orig_mod', model)
                        raw_model.load_state_dict(state)
                        model_path = new_path
                        print(f'  {GREEN}模型已切换: {new_path}{RESET}')
                    except Exception as e:
                        print(f'  {RED}⚠ 加载失败: {e}{RESET}')
                else:
                    print(f'  当前模型: {model_path}')

            else:
                print(f'  {YELLOW}⚠ 未知命令 :{cmd}，输入 :help 查看帮助{RESET}')
            continue

        # ── 表达式处理 ──────────────────────────────────────────────────────
        mode, expr, candidate = parse_input(raw)

        if mode is None:
            print(f'  {YELLOW}⚠ 无法解析输入，请检查格式（:help 查看示例）{RESET}\n')
            continue

        if mode == 'generate':
            prompt = [BOS_ID] + tokenize_expr(expr)
            gen    = generate(model, prompt, max_new, temperature, top_p, device)
            result, cot_steps, _ = parse_generated(gen)
            display_generate(expr, result, cot_steps, show_cot,
                             show_raw=show_raw, prompt_ids=prompt, gen_ids=gen)

        elif mode == 'verify':
            prompt = ([BOS_ID] + tokenize_expr(expr)
                      + [EQ_ID] + tokenize_expr(candidate) + [Q_ID])
            gen    = generate(model, prompt, max_new=8, temperature=0.0,
                              top_p=1.0, device=device)
            _, _, verdict = parse_generated(gen)
            if verdict:
                display_verify(expr, candidate, verdict)
            else:
                raw_out = ''.join(ID2TOK.get(i, '?') for i in gen)
                print(f'  {YELLOW}⚠ 模型未输出有效判定 (got: "{raw_out}"){RESET}\n')


if __name__ == '__main__':
    main()
