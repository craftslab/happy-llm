import pytest
import torch

from multihead_attention import ModelArgs, MultiHeadAttention


def _make_module(
    dim: int = 64,
    n_heads: int = 8,
    dropout: float = 0.0,
    is_causal: bool = False,
) -> MultiHeadAttention:
    args = ModelArgs(dim=dim, n_heads=n_heads, dropout=dropout, max_seq_len=32)
    module = MultiHeadAttention(args, is_causal=is_causal)
    module.eval()
    return module


def test_output_shape():
    batch, seq, dim = 2, 8, 64
    mha = _make_module(dim=dim, n_heads=8)
    x = torch.randn(batch, seq, dim)

    out = mha(x, x, x)

    assert out.shape == (batch, seq, dim)


def test_invalid_head_count():
    with pytest.raises(AssertionError):
        MultiHeadAttention(ModelArgs(dim=30, n_heads=4))


def test_causal_mask_blocks_future_tokens():
    dim = 32
    mha = _make_module(dim=dim, n_heads=4, is_causal=True)
    x1 = torch.randn(1, 6, dim)
    x2 = x1.clone()
    x2[:, 3:, :] = torch.randn(1, 3, dim)

    out1 = mha(x1, x1, x1)
    out2 = mha(x2, x2, x2)

    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6)


def test_non_causal_uses_future_tokens():
    dim = 32
    mha = _make_module(dim=dim, n_heads=4, is_causal=False)
    x1 = torch.randn(1, 6, dim)
    x2 = x1.clone()
    x2[:, 3:, :] = torch.randn(1, 3, dim)

    out1 = mha(x1, x1, x1)
    out2 = mha(x2, x2, x2)

    assert not torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6)


def test_cross_attention_accepts_different_qkv():
    dim = 32
    mha = _make_module(dim=dim, n_heads=4)
    q = torch.randn(2, 5, dim)
    k = torch.randn(2, 7, dim)
    v = torch.randn(2, 7, dim)

    out = mha(q, k, v)

    assert out.shape == (2, 5, dim)


def test_gradient_flow():
    dim = 32
    mha = _make_module(dim=dim, n_heads=4)
    x = torch.randn(2, 4, dim, requires_grad=True)

    out = mha(x, x, x)
    out.sum().backward()

    assert x.grad is not None
    assert mha.wq.weight.grad is not None
