import math

import torch
import torch.nn as nn

from attention import attention


def test_output_shapes():
    batch, seq_q, seq_k, d_k, d_v = 2, 3, 4, 8, 6
    q = torch.randn(batch, seq_q, d_k)
    k = torch.randn(batch, seq_k, d_k)
    v = torch.randn(batch, seq_k, d_v)

    out, weights = attention(q, k, v)

    assert out.shape == (batch, seq_q, d_v)
    assert weights.shape == (batch, seq_q, seq_k)


def test_attention_weights_sum_to_one():
    q = torch.randn(1, 5, 4)
    k = torch.randn(1, 7, 4)
    v = torch.randn(1, 7, 4)

    _, weights = attention(q, k, v)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 5), atol=1e-6)


def test_manual_computation():
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])

    out, weights = attention(q, k, v)

    expected_scores = torch.tensor([[[1.0 / math.sqrt(2), 0.0]]])
    expected_weights = expected_scores.softmax(dim=-1)
    expected_out = torch.matmul(expected_weights, v)

    assert torch.allclose(weights, expected_weights, atol=1e-6)
    assert torch.allclose(out, expected_out, atol=1e-6)


def test_output_is_weighted_sum_of_values():
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 10)

    out, weights = attention(q, k, v)

    expected = torch.matmul(weights, v)
    assert torch.allclose(out, expected, atol=1e-6)


def test_with_dropout():
    torch.manual_seed(0)
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)

    dropout = nn.Dropout(p=0.5)
    dropout.train()

    out_with, weights_with = attention(q, k, v, dropout=dropout)
    _, weights_without = attention(q, k, v)

    assert out_with.shape == weights_without.shape
    assert weights_with.shape == weights_without.shape
    assert not torch.allclose(weights_with, weights_without)
