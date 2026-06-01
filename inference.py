"""
inference.py — Interactive inference REPL.
Reads model architecture and inference parameters from a training config yaml.

Usage:
    python inference.py --config config/teacher_pretrain.yaml
    python inference.py --config config/teacher_sft.yaml --model path/to/override.pt

Input formats:
    rs1234           -> generate: compute expression result
    rs1234=4321?     -> verify: judge whether candidate answer is correct
    :help            -> show help
    :cot on/off      -> toggle CoT trace display
    :temp 0.8        -> set sampling temperature (0 = greedy)
    :model <path>    -> switch model weights at runtime
    :quit            -> exit
"""
import argparse
import os
import sys
from pathlib import Path

# Prevent OpenMP conflict on macOS (PyTorch + conda may load libomp.dylib twice)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lib.lang  import TOKEN2ID, interpret
from lib.model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Token constants
# ─────────────────────────────────────────────────────────────────────────────

ID2TOK   = {v: k for k, v in TOKEN2ID.items()}
BOS_ID   = TOKEN2ID['[BOS]']
EOS_ID   = TOKEN2ID['[EOS]']
EQ_ID    = TOKEN2ID['=']
Q_ID     = TOKEN2ID['?']
THINK_ID     = TOKEN2ID.get('<think>',  32)
THINK_END_ID = TOKEN2ID.get('</think>', 33)

# ANSI colors (auto-disabled for non-TTY output)
_USE_COLOR = sys.stdout.isatty()
def _c(code: str) -> str:
    return code if _USE_COLOR else ''

RESET = _c('\033[0m');  BOLD  = _c('\033[1m')
GREEN = _c('\033[92m'); RED   = _c('\033[91m')
CYAN  = _c('\033[96m'); GRAY  = _c('\033[90m')
YELLOW = _c('\033[93m')


# ─────────────────────────────────────────────────────────────────────────────
# Expression tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_expr(expr: str) -> List[int]:
    """
    Expression string (no spaces) -> list of token IDs.
    Tries two-char tokens (vc/vw/gw) before single-char.
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
# Generation
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
    Autoregressive generation. Returns newly generated token IDs (excluding prompt).
    temperature=0 -> greedy; temperature>0 -> sampling with optional top-p nucleus.
    """
    model.eval()
    context_len = model.pos_emb.num_embeddings
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    # ids: [1, L_prompt]
    generated = []

    for _ in range(max_new):
        if ids.shape[1] >= context_len:
            break
        logits = model(ids)             # [1, T, V]
        nxt = logits[0, -1]             # [V]

        if temperature == 0.0:
            next_id = int(nxt.argmax())
        else:
            nxt = nxt / temperature
            probs = F.softmax(nxt, dim=-1)
            if top_p < 1.0:
                # nucleus (top-p) sampling
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
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_generated(gen_ids: List[int]) -> Tuple[str, Optional[List[str]], Optional[str]]:
    """
    Parse generated token IDs into (result, cot_steps, verdict).

    Supported formats:
      Direct : '= RESULT [EOS]'
      CoT    : '<think> TRACE </think> = RESULT [EOS]'
      Verify : 'vc/vw/gw [EOS]'

    cot_steps: List[str] or None (each step is a compact expression string)
    verdict  : 'vc' | 'vw' | 'gw' or None
    """
    toks = [ID2TOK.get(i, '?') for i in gen_ids]

    # verify mode output
    if toks and toks[0] in ('vc', 'vw', 'gw'):
        return '', None, toks[0]

    # CoT trace extraction
    cot_steps = None
    if '<think>' in toks and '</think>' in toks:
        try:
            s = toks.index('<think>') + 1
            e = toks.index('</think>')
            trace = ''.join(toks[s:e])
            cot_steps = [st for st in trace.split('=') if st]
        except ValueError:
            pass

    # last '=' in sequence marks start of final result
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
# Input parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_input(raw: str) -> Tuple[Optional[str], str, str]:
    """
    Parse user input. Returns (mode, expr, candidate).
    mode: 'generate' | 'verify' | None (invalid)
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
# Display
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

    if show_raw and prompt_ids is not None and gen_ids is not None:
        all_ids = prompt_ids + gen_ids
        raw_str = ' '.join(ID2TOK.get(i, f'?{i}') for i in all_ids)
        print(f"\n  {CYAN}Raw tokens{RESET} : {GRAY}{raw_str}{RESET}")

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

    if valid:
        if result == correct:
            print(f"  {GREEN}Correct{RESET}")
        else:
            print(f"  {RED}Wrong{RESET}  (correct: {BOLD}{correct}{RESET})")
    else:
        print(f"  {YELLOW}Warning: expression syntax invalid{RESET}")
    print()


def display_verify(expr: str, candidate: str, verdict: str):
    v_style = {
        'vc': (GREEN, 'correct'),
        'vw': (RED,   'wrong'),
        'gw': (RED,   'invalid expression'),
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
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, model_cfg: dict, device: str):
    if not Path(model_path).exists():
        print(f"{RED}Error: model file not found: {model_path}{RESET}")
        sys.exit(1)
    model = build_model(model_cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    # Note: torch.compile is intentionally skipped for inference:
    #   1. Single-query latency; compile overhead far exceeds savings.
    #   2. On macOS, compile triggers duplicate libomp.dylib load -> SIGABRT.
    return model


def print_help(show_cot: bool, show_raw: bool, temperature: float):
    print(f"""
  {BOLD}Commands{RESET}
    :cot on/off        CoT trace display (current: {'on' if show_cot else 'off'})
    :raw on/off        Full token sequence display, incl. [BOS]/[EOS]/<think> (current: {'on' if show_raw else 'off'})
    :temp <0-2>        Sampling temperature (0=greedy, current: {temperature:.1f})
    :model <path>      Switch model weights file
    :quit              Exit

  {BOLD}Input formats{RESET}
    rs1234             -> generate: compute expression result
    rs1234=4321?       -> verify: judge whether candidate is correct

  {BOLD}Operators{RESET}
    Unary  : i r s S d D h t e o L R p
    Binary : c (concat)  z (zip/interleave)

  {BOLD}Examples{RESET}
    > rs1234           reverse then sort
    > Dp123            palindrome then double
    > 12c34=1234?      verify concat result
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LLMlab Inference REPL')
    parser.add_argument('--config', required=True,
                        help='Training config yaml (e.g. config/teacher_pretrain.yaml)')
    parser.add_argument('--model', default=None,
                        help='Override model weights path from config')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path  = args.model or cfg['output']['model_path']
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

    print(f"Loading model: {model_path}  device={device}")
    model = load_model(model_path, model_cfg, device)
    ctx = model_cfg['context_len']

    print()
    print('=' * 60)
    print(f'  LLMlab Inference')
    print(f'  model : {Path(model_path).name}')
    print(f'  ctx   : {ctx}   device: {device}')
    print('=' * 60)
    print('  Type expression to compute, :help for commands\n')

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
            print(f'\n{GRAY}Goodbye!{RESET}')
            break

        raw = raw.strip()
        if not raw:
            continue

        if raw.startswith(':'):
            parts = raw[1:].strip().lower().split()
            cmd   = parts[0] if parts else ''

            if cmd in ('quit', 'exit', 'q'):
                print(f'{GRAY}Goodbye!{RESET}')
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
                        print(f'  Temperature: {temperature:.2f}')
                    except ValueError:
                        print(f'  {RED}Error: expected a number, e.g. :temp 0.8{RESET}')
                else:
                    print(f'  Current temperature: {temperature:.2f}')

            elif cmd == 'model':
                if len(parts) > 1:
                    new_path = parts[1]
                    try:
                        state    = torch.load(new_path, map_location=device, weights_only=True)
                        raw_model = getattr(model, '_orig_mod', model)
                        raw_model.load_state_dict(state)
                        model_path = new_path
                        print(f'  {GREEN}Switched to: {new_path}{RESET}')
                    except Exception as e:
                        print(f'  {RED}Failed to load: {e}{RESET}')
                else:
                    print(f'  Current model: {model_path}')

            else:
                print(f'  {YELLOW}Unknown command :{cmd}  — type :help{RESET}')
            continue

        mode, expr, candidate = parse_input(raw)

        if mode is None:
            print(f'  {YELLOW}Cannot parse input (see :help for examples){RESET}\n')
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
                print(f'  {YELLOW}No valid verdict from model (got: "{raw_out}"){RESET}\n')


if __name__ == '__main__':
    main()
