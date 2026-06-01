"""
lib/model.py — Decoder-only Transformer shared by teacher and student.
Hyper-parameters (d_model, n_layers, etc.) are injected from the 'model' block of a config yaml.

Usage:
    from lib.model import build_model
    model = build_model(cfg['model'])   # cfg = yaml.safe_load(...)

Parameter counts (bias=False for all Linear):
    teacher (d=192, L=6, H=6, dh=32, ffn=768, V=34)  ≈ 2.67M
    student (d=64,  L=3, H=4, dh=16, ffn=256, V=34)  ≈ 0.15M
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention (called after pre-LN).
    All Linear layers have no bias, consistent with GPT-style practice.
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float):
        super().__init__()
        self.n_heads    = n_heads
        self.d_head     = d_head
        self.dropout    = dropout       # passed to scaled_dot_product_attention
        d_inner = n_heads * d_head

        self.qkv        = nn.Linear(d_model, 3 * d_inner, bias=False)
        self.proj       = nn.Linear(d_inner, d_model,      bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        H, dh   = self.n_heads, self.d_head

        qkv = self.qkv(x)                          # [B, T, 3*H*dh]
        q, k, v = qkv.split(H * dh, dim=-1)

        q = q.view(B, T, H, dh).transpose(1, 2)   # [B, H, T, dh]
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)

        # Flash Attention (PyTorch >= 2.0 uses FlashAttn kernel on Ampere+ GPUs):
        #   - no explicit [T,T] causal mask (avoids O(T^2) memory allocation)
        #   - IO-aware block computation with much better bandwidth utilization
        #   - is_causal=True equivalent to upper-triangular mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )                                           # [B, H, T, dh]

        out = out.transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.resid_drop(self.proj(out))


class FFN(nn.Module):
    """
    Two-layer MLP: Linear -> Activation -> Linear -> Dropout.
    bias=False; activation supports 'gelu' | 'relu'.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float, activation: str = 'gelu'):
        super().__init__()
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),  # [B,T,D] -> [B,T,d_ffn]
            act,
            nn.Linear(d_ffn, d_model, bias=False),  # [B,T,d_ffn] -> [B,T,D]
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    Pre-LN Transformer block:
        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 d_ffn: int, dropout: float, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, d_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, d_ffn, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.attn(self.norm1(x))  # residual + pre-LN attention
        x = x + self.ffn(self.norm2(x))   # residual + pre-LN FFN
        return x                           # [B, T, D]


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Decoder-only Transformer (GPT-style).
    - Learned positional embeddings
    - Pre-LN
    - Causal self-attention
    - GELU FFN
    - Output head shares weights with input embedding (tie_embedding=true)

    forward(idx)           -> logits [B, T, V]
    forward(idx, targets)  -> logits [B, T, V], loss (scalar)
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        V   = cfg['vocab_size']    # vocabulary size, 32 or 34 (with CoT tokens)
        C   = cfg['context_len']   # maximum sequence length, e.g. 96
        D   = cfg['d_model']       # model dimension
        L   = cfg['n_layers']      # number of Transformer blocks
        H   = cfg['n_heads']       # number of attention heads
        dh  = cfg['d_head']        # dimension per head
        dff = cfg['d_ffn']         # FFN hidden dimension
        p   = cfg.get('dropout', 0.0)
        act = cfg.get('activation', 'gelu')

        self.tok_emb = nn.Embedding(V, D)   # [V, D]
        self.pos_emb = nn.Embedding(C, D)   # [C, D], learned
        self.drop    = nn.Dropout(p)

        self.blocks  = nn.ModuleList(
            [Block(D, H, dh, dff, p, act) for _ in range(L)]
        )
        self.norm = nn.LayerNorm(D)              # final pre-LN
        self.head = nn.Linear(D, V, bias=False)  # [D, V], unembedding

        if cfg.get('tie_embedding', True):
            self.head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        """std=0.02 init; residual projections scaled by 1/sqrt(L) (GPT-2 practice)."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)
        # residual projection scaling
        n_layers = len(self.blocks)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(block.ffn.net[-2].weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(
        self,
        idx: torch.Tensor,                     # [B, T]  token IDs
        targets: Optional[torch.Tensor] = None, # [B, T]  labels, -100 = ignore
    ):
        B, T = idx.shape
        assert T <= self.pos_emb.num_embeddings, \
            f"Sequence length {T} exceeds context_len {self.pos_emb.num_embeddings}"

        pos = torch.arange(T, device=idx.device)               # [T]
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # [B, T, D]

        for block in self.blocks:
            x = block(x)                                        # [B, T, D]

        x = self.norm(x)                                        # [B, T, D]
        logits = self.head(x)                                   # [B, T, V]

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),               # [B*T, V]
                targets.view(-1),                               # [B*T]
                ignore_index=-100,
            )
            return logits, loss

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Dict) -> Transformer:
    """Build model from yaml 'model' block and print parameter count."""
    model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  ({n_params/1e6:.3f}M)")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
