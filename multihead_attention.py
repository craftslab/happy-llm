"""Multi-head attention module."""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import _tensor_summary

logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    """Configuration for MultiHeadAttention."""

    dim: int
    n_heads: int
    n_embd: int | None = None
    n_layer: int = 1
    dropout: float = 0.0
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        if self.n_embd is None:
            self.n_embd = self.dim


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, args: ModelArgs, is_causal: bool = False):
        super().__init__()
        assert args.dim % args.n_heads == 0

        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        logger.debug("multihead_attention start")
        logger.debug(_tensor_summary("q", q))
        logger.debug(_tensor_summary("k", k))
        logger.debug(_tensor_summary("v", v))
        logger.debug(
            "config: n_heads=%d, head_dim=%d, is_causal=%s, training=%s",
            self.n_heads,
            self.head_dim,
            self.is_causal,
            self.training,
        )

        bsz, seqlen, _ = q.shape
        seqlen_k = k.size(1)
        scale = math.sqrt(self.head_dim)
        logger.debug(
            "batch=%d, seqlen_q=%d, seqlen_k=%d, scale=sqrt(head_dim)=%.6f",
            bsz,
            seqlen,
            seqlen_k,
            scale,
        )

        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
        logger.debug(_tensor_summary("xq (after Wq)", xq))
        logger.debug(_tensor_summary("xk (after Wk)", xk))
        logger.debug(_tensor_summary("xv (after Wv)", xv))

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen_k, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen_k, self.n_heads, self.head_dim).transpose(1, 2)
        logger.debug(_tensor_summary("xq (multi-head)", xq))
        logger.debug(_tensor_summary("xk (multi-head)", xk))
        logger.debug(_tensor_summary("xv (multi-head)", xv))

        raw_scores = torch.matmul(xq, xk.transpose(2, 3))
        logger.debug(_tensor_summary("raw_scores (Q @ K^T)", raw_scores))

        scores = raw_scores / scale
        logger.debug(_tensor_summary("scaled_scores (Q @ K^T / sqrt(head_dim))", scores))

        if self.is_causal:
            assert hasattr(self, "mask")
            causal_slice = self.mask[:, :, :seqlen, :seqlen_k]
            logger.debug(
                "applying causal mask: slice shape=%s",
                tuple(causal_slice.shape),
            )
            scores = scores + causal_slice
            logger.debug(_tensor_summary("scores (after causal mask)", scores))
        else:
            logger.debug("causal mask not applied")

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        logger.debug(_tensor_summary("attention_weights (softmax)", attn_weights))

        row_sums = attn_weights.sum(dim=-1)
        logger.debug(
            "attention row sums: min=%.6f, max=%.6f",
            row_sums.min().item(),
            row_sums.max().item(),
        )

        logger.debug(
            "applying attn_dropout: training=%s, p=%s",
            self.attn_dropout.training,
            self.attn_dropout.p,
        )
        attn_weights_before = attn_weights
        attn_weights = self.attn_dropout(attn_weights)
        logger.debug(_tensor_summary("attention_weights (after attn_dropout)", attn_weights))
        logger.debug(
            "attn_dropout modified weights: %s",
            not torch.equal(attn_weights_before, attn_weights),
        )

        head_output = torch.matmul(attn_weights, xv)
        logger.debug(_tensor_summary("head_output (weights @ V)", head_output))

        output = head_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        logger.debug(_tensor_summary("output (concat heads)", output))

        output = self.wo(output)
        logger.debug(_tensor_summary("output (after Wo)", output))

        logger.debug(
            "applying resid_dropout: training=%s, p=%s",
            self.resid_dropout.training,
            self.resid_dropout.p,
        )
        output_before = output
        output = self.resid_dropout(output)
        logger.debug(_tensor_summary("output (after resid_dropout)", output))
        logger.debug(
            "resid_dropout modified output: %s",
            not torch.equal(output_before, output),
        )
        logger.debug("multihead_attention done")

        return output
