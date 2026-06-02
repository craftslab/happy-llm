"""Scaled dot-product attention."""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    """Format tensor shape and basic statistics for logging."""
    with torch.no_grad():
        return (
            f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}, min={tensor.min().item():.6f}, "
            f"max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
        )


def attention(query, key, value, dropout=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: Query tensor, shape (..., seq_q, d_k).
        key: Key tensor, shape (..., seq_k, d_k).
        value: Value tensor, shape (..., seq_k, d_v).
        dropout: Optional dropout module applied to attention weights.

    Returns:
        output: Weighted sum of values, shape (..., seq_q, d_v).
        p_attn: Attention weights after softmax, shape (..., seq_q, seq_k).
    """
    logger.debug("attention start")
    logger.debug(_tensor_summary("query", query))
    logger.debug(_tensor_summary("key", key))
    logger.debug(_tensor_summary("value", value))

    d_k = query.size(-1)
    scale = math.sqrt(d_k)
    logger.debug("d_k=%d, scale=sqrt(d_k)=%.6f", d_k, scale)

    raw_scores = torch.matmul(query, key.transpose(-2, -1))
    logger.debug(_tensor_summary("raw_scores (Q @ K^T)", raw_scores))

    scores = raw_scores / scale
    logger.debug(_tensor_summary("scaled_scores (Q @ K^T / sqrt(d_k))", scores))

    p_attn = scores.softmax(dim=-1)
    logger.debug(_tensor_summary("attention_weights (softmax)", p_attn))

    row_sums = p_attn.sum(dim=-1)
    logger.debug(
        "attention row sums: min=%.6f, max=%.6f",
        row_sums.min().item(),
        row_sums.max().item(),
    )

    if dropout is not None:
        training = getattr(dropout, "training", None)
        p = getattr(dropout, "p", None)
        logger.debug(
            "applying dropout: training=%s, p=%s",
            training,
            p,
        )
        p_attn_before = p_attn
        p_attn = dropout(p_attn)
        logger.debug(_tensor_summary("attention_weights (after dropout)", p_attn))
        changed = not torch.equal(p_attn_before, p_attn)
        logger.debug("dropout modified weights: %s", changed)
    else:
        logger.debug("dropout not applied")

    output = torch.matmul(p_attn, value)
    logger.debug(_tensor_summary("output (weights @ value)", output))
    logger.debug("attention done")

    return output, p_attn
