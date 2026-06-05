import pytest
import torch

from encoder import Encoder, EncoderLayer, LayerNorm, MLP
from multihead_attention import ModelArgs


def _make_args(
    dim: int = 64,
    n_heads: int = 8,
    dropout: float = 0.0,
    n_layer: int = 1,
) -> ModelArgs:
    return ModelArgs(
        dim=dim,
        n_heads=n_heads,
        dropout=dropout,
        max_seq_len=32,
        n_layer=n_layer,
    )


def _make_layer(
    dim: int = 64,
    n_heads: int = 8,
    dropout: float = 0.0,
) -> EncoderLayer:
    args = _make_args(dim=dim, n_heads=n_heads, dropout=dropout)
    layer = EncoderLayer(args)
    layer.eval()
    return layer


def _make_encoder(
    dim: int = 64,
    n_heads: int = 8,
    dropout: float = 0.0,
    n_layer: int = 2,
) -> Encoder:
    args = _make_args(dim=dim, n_heads=n_heads, dropout=dropout, n_layer=n_layer)
    encoder = Encoder(args)
    encoder.eval()
    return encoder


def test_output_shape():
    batch, seq, dim = 2, 8, 64
    layer = _make_layer(dim=dim)
    x = torch.randn(batch, seq, dim)

    out = layer(x)

    assert out.shape == (batch, seq, dim)


def test_attention_is_non_causal():
    """Encoder 使用双向注意力，修改未来 token 会影响过去位置的输出。"""
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    x1 = torch.randn(1, 6, dim)
    x2 = x1.clone()
    x2[:, 3:, :] = torch.randn(1, 3, dim)

    out1 = layer(x1)
    out2 = layer(x2)

    assert not torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6)


def test_has_pre_layer_norm_submodules():
    layer = _make_layer(dim=32, n_heads=4)

    assert hasattr(layer, "attention_norm")
    assert hasattr(layer, "fnn_norm")
    assert hasattr(layer, "attention")
    assert hasattr(layer, "feed_forward")
    assert layer.attention.is_causal is False


def test_residual_connection_changes_output():
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    x = torch.randn(1, 4, dim)

    out = layer(x)

    assert not torch.allclose(out, x)


def test_gradient_flow():
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    x = torch.randn(2, 4, dim, requires_grad=True)

    out = layer(x)
    out.sum().backward()

    assert x.grad is not None
    assert layer.attention.wq.weight.grad is not None
    assert layer.feed_forward.w1.weight.grad is not None
    assert layer.attention_norm.a_2.grad is not None
    assert layer.fnn_norm.a_2.grad is not None


def test_layer_norm_normalizes_last_dimension():
    norm = LayerNorm(4)
    x = torch.tensor([[[1.0, 3.0, 5.0, 7.0]]])

    out = norm(x)

    assert torch.allclose(out.mean(dim=-1), torch.zeros(1, 1), atol=1e-5)


def test_mlp_output_shape():
    mlp = MLP(dim=16, hidden_dim=32, dropout=0.0)
    mlp.eval()
    x = torch.randn(2, 5, 16)

    out = mlp(x)

    assert out.shape == (2, 5, 16)


def test_mlp_dropout_changes_output_in_train_mode():
    torch.manual_seed(0)
    mlp = MLP(dim=8, hidden_dim=8, dropout=0.5)
    mlp.train()
    x = torch.randn(1, 3, 8)

    out1 = mlp(x)
    out2 = mlp(x)

    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2)


def test_invalid_head_count():
    with pytest.raises(AssertionError):
        EncoderLayer(ModelArgs(dim=30, n_heads=4))


def test_encoder_output_shape():
    batch, seq, dim = 2, 8, 64
    encoder = _make_encoder(dim=dim, n_layer=3)
    x = torch.randn(batch, seq, dim)

    out = encoder(x)

    assert out.shape == (batch, seq, dim)


def test_encoder_has_n_layers():
    encoder = _make_encoder(n_layer=4)

    assert len(encoder.layers) == 4
    assert hasattr(encoder, "norm")


def test_encoder_applies_final_layer_norm():
    encoder = _make_encoder(dim=32, n_layer=2)
    x = torch.randn(1, 5, 32)

    out = encoder(x)

    assert torch.allclose(out.mean(dim=-1), torch.zeros(1, 5), atol=1e-5)


def test_encoder_stacked_layers_differ_from_single_layer():
    dim = 32
    args = _make_args(dim=dim, n_heads=4, n_layer=1)
    single = EncoderLayer(args)
    stacked = Encoder(_make_args(dim=dim, n_heads=4, n_layer=2))
    single.eval()
    stacked.eval()

    x = torch.randn(1, 4, dim)
    out_single = single(x)
    out_stacked = stacked(x)

    assert not torch.allclose(out_single, out_stacked)


def test_encoder_gradient_flow():
    encoder = _make_encoder(dim=32, n_heads=4, n_layer=2)
    x = torch.randn(2, 4, 32, requires_grad=True)

    out = encoder(x)
    out.sum().backward()

    assert x.grad is not None
    assert encoder.layers[0].attention.wq.weight.grad is not None
    assert encoder.layers[1].feed_forward.w1.weight.grad is not None
    assert encoder.norm.a_2.grad is not None
