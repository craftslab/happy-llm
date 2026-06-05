"""Encoder module: LayerNorm, MLP, EncoderLayer, and Encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from multihead_attention import ModelArgs, MultiHeadAttention


class LayerNorm(nn.Module):
    """Layer Norm 层"""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    """Encoder层"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out


class Encoder(nn.Module):
    """Encoder 块"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList(
            [EncoderLayer(args) for _ in range(args.n_layer)]
        )
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分别通过 N 层 Encoder Layer
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

