"""
model.py — Decoder-only Transformer，teacher 和 student 共用同一套代码。
具体超参（d_model、n_layers 等）由 config yaml 的 model 块注入。

调用：
    from model import build_model
    model = build_model(cfg['model'])       # cfg = yaml.safe_load(...)

参数量参考（bias=False for all Linear）：
    teacher (d=192, L=6, H=6, dh=32, ffn=768, V=34)  ≈ 2.67M
    student (d=64,  L=3, H=4, dh=16, ffn=256, V=34)  ≈ 0.15M
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 基础模块
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力（pre-LN 之后调用）。
    所有 Linear 不用 bias，与 GPT-style 实践一致。
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads          # H
        self.d_head  = d_head           # d_h
        d_inner = n_heads * d_head      # H * d_h

        self.qkv  = nn.Linear(d_model, 3 * d_inner, bias=False)  # → [B,T, 3*H*d_h]
        self.proj = nn.Linear(d_inner, d_model,      bias=False)  # → [B,T, D]
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B=batch, T=seq_len, D=d_model]
        B, T, D = x.shape
        H, dh   = self.n_heads, self.d_head

        qkv = self.qkv(x)                       # [B, T, 3*H*dh]
        q, k, v = qkv.split(H * dh, dim=-1)     # each [B, T, H*dh]

        # reshape to [B, H, T, dh] for multi-head attention
        q = q.view(B, T, H, dh).transpose(1, 2)  # [B, H, T, dh]
        k = k.view(B, T, H, dh).transpose(1, 2)  # [B, H, T, dh]
        v = v.view(B, T, H, dh).transpose(1, 2)  # [B, H, T, dh]

        # Scaled dot-product attention with causal mask
        scale = dh ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale   # [B, H, T, T]
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                              # [B, H, T, dh]
        out = out.transpose(1, 2).contiguous().view(B, T, H * dh)  # [B, T, H*dh]
        return self.resid_drop(self.proj(out))      # [B, T, D]


class FFN(nn.Module):
    """
    两层 MLP：Linear → Activation → Linear → Dropout。
    bias=False；activation 支持 'gelu' | 'relu'。
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float, activation: str = 'gelu'):
        super().__init__()
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),  # [B,T,D] → [B,T,d_ffn]
            act,
            nn.Linear(d_ffn, d_model, bias=False),  # [B,T,d_ffn] → [B,T,D]
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    Pre-LN Transformer block：
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
# 完整模型
# ─────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Decoder-only Transformer（GPT 风格）。
    - 学习式位置编码（learned）
    - Pre-LN
    - 因果自注意力
    - GELU FFN
    - 输出层与输入 embedding 权重共享（tie_embedding=true）

    forward(idx)           → logits [B, T, V]
    forward(idx, targets)  → logits [B, T, V], loss (scalar)
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        V   = cfg['vocab_size']    # 词表大小，32 或 34（含 CoT token）
        C   = cfg['context_len']   # 最大序列长度，48
        D   = cfg['d_model']       # 模型维度
        L   = cfg['n_layers']      # Transformer block 数
        H   = cfg['n_heads']       # 注意力头数
        dh  = cfg['d_head']        # 每头维度
        dff = cfg['d_ffn']         # FFN 隐层维度
        p   = cfg.get('dropout', 0.0)
        act = cfg.get('activation', 'gelu')

        self.tok_emb = nn.Embedding(V, D)   # [V, D]
        self.pos_emb = nn.Embedding(C, D)   # [C, D]，learned
        self.drop    = nn.Dropout(p)

        self.blocks  = nn.ModuleList(
            [Block(D, H, dh, dff, p, act) for _ in range(L)]
        )
        self.norm = nn.LayerNorm(D)          # final pre-LN
        self.head = nn.Linear(D, V, bias=False)  # [D, V]，unembedding

        if cfg.get('tie_embedding', True):
            self.head.weight = self.tok_emb.weight  # 共享权重

        self._init_weights()

    def _init_weights(self):
        """Std=0.02 初始化；残差投影使用 1/√L 缩放（GPT-2 做法）。"""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)
        # 残差投影缩放
        n_layers = len(self.blocks)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(block.ffn.net[-2].weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(
        self,
        idx: torch.Tensor,                     # [B, T]  token IDs
        targets: Optional[torch.Tensor] = None, # [B, T]  labels，-100 表示忽略
    ):
        B, T = idx.shape
        assert T <= self.pos_emb.num_embeddings, \
            f"序列长度 {T} 超过 context_len {self.pos_emb.num_embeddings}"

        pos = torch.arange(T, device=idx.device)           # [T]
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # [B, T, D]

        for block in self.blocks:
            x = block(x)                                    # [B, T, D]

        x = self.norm(x)                                    # [B, T, D]
        logits = self.head(x)                               # [B, T, V]

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),           # [B*T, V]
                targets.view(-1),                           # [B*T]
                ignore_index=-100,
            )
            return logits, loss

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Dict) -> Transformer:
    """从 yaml model 块构建模型，打印参数量。"""
    model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}  ({n_params/1e6:.3f}M)")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
