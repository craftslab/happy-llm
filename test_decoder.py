import pytest
import torch

from decoder import Decoder, DecoderLayer
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
) -> DecoderLayer:
    layer = DecoderLayer(_make_args(dim=dim, n_heads=n_heads, dropout=dropout))
    layer.eval()
    return layer


def _make_decoder(
    dim: int = 64,
    n_heads: int = 8,
    dropout: float = 0.0,
    n_layer: int = 2,
) -> Decoder:
    decoder = Decoder(_make_args(dim=dim, n_heads=n_heads, dropout=dropout, n_layer=n_layer))
    decoder.eval()
    return decoder


def test_decoder_layer_output_shape():
    batch, dec_seq, enc_seq, dim = 2, 8, 12, 64
    layer = _make_layer(dim=dim)
    x = torch.randn(batch, dec_seq, dim)
    enc_out = torch.randn(batch, enc_seq, dim)

    out = layer(x, enc_out)

    assert out.shape == (batch, dec_seq, dim)


def test_decoder_layer_has_expected_submodules():
    layer = _make_layer(dim=32, n_heads=4)

    assert hasattr(layer, "attention_norm_1")
    assert hasattr(layer, "attention_norm_2")
    assert hasattr(layer, "ffn_norm")
    assert hasattr(layer, "mask_attention")
    assert hasattr(layer, "attention")
    assert hasattr(layer, "feed_forward")
    assert layer.mask_attention.is_causal is True
    assert layer.attention.is_causal is False


def test_mask_attention_blocks_future_tokens():
    """掩码自注意力下，修改未来 token 不影响过去位置输出。"""
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    enc_out = torch.randn(1, 6, dim)
    x1 = torch.randn(1, 6, dim)
    x2 = x1.clone()
    x2[:, 3:, :] = torch.randn(1, 3, dim)

    out1 = layer(x1, enc_out)
    out2 = layer(x2, enc_out)

    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6)


def test_cross_attention_uses_encoder_output():
    """修改 encoder 输出会影响 decoder 层输出。"""
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    x = torch.randn(1, 5, dim)
    enc_out1 = torch.randn(1, 7, dim)
    enc_out2 = enc_out1.clone()
    enc_out2[:, 4:, :] = torch.randn(1, 3, dim)

    out1 = layer(x, enc_out1)
    out2 = layer(x, enc_out2)

    assert not torch.allclose(out1, out2, atol=1e-6)


def test_cross_attention_accepts_different_seq_lengths():
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    x = torch.randn(2, 5, dim)
    enc_out = torch.randn(2, 9, dim)

    out = layer(x, enc_out)

    assert out.shape == (2, 5, dim)


def test_decoder_layer_residual_changes_output():
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    x = torch.randn(1, 4, dim)
    enc_out = torch.randn(1, 4, dim)

    out = layer(x, enc_out)

    assert not torch.allclose(out, x)


def test_decoder_layer_gradient_flow():
    dim = 32
    layer = _make_layer(dim=dim, n_heads=4)
    x = torch.randn(2, 4, dim, requires_grad=True)
    enc_out = torch.randn(2, 6, dim, requires_grad=True)

    out = layer(x, enc_out)
    out.sum().backward()

    assert x.grad is not None
    assert enc_out.grad is not None
    assert layer.mask_attention.wq.weight.grad is not None
    assert layer.attention.wq.weight.grad is not None
    assert layer.feed_forward.w1.weight.grad is not None


def test_invalid_head_count():
    with pytest.raises(AssertionError):
        DecoderLayer(ModelArgs(dim=30, n_heads=4))


def test_decoder_output_shape():
    batch, dec_seq, enc_seq, dim = 2, 8, 10, 64
    decoder = _make_decoder(dim=dim, n_layer=3)
    x = torch.randn(batch, dec_seq, dim)
    enc_out = torch.randn(batch, enc_seq, dim)

    out = decoder(x, enc_out)

    assert out.shape == (batch, dec_seq, dim)


def test_decoder_has_n_layers():
    decoder = _make_decoder(n_layer=4)

    assert len(decoder.layers) == 4
    assert hasattr(decoder, "norm")


def test_decoder_applies_final_layer_norm():
    decoder = _make_decoder(dim=32, n_layer=2)
    x = torch.randn(1, 5, 32)
    enc_out = torch.randn(1, 7, 32)

    out = decoder(x, enc_out)

    assert torch.allclose(out.mean(dim=-1), torch.zeros(1, 5), atol=1e-5)


def test_decoder_stacked_layers_differ_from_single_layer():
    dim = 32
    single = DecoderLayer(_make_args(dim=dim, n_heads=4, n_layer=1))
    stacked = Decoder(_make_args(dim=dim, n_heads=4, n_layer=2))
    single.eval()
    stacked.eval()

    x = torch.randn(1, 4, dim)
    enc_out = torch.randn(1, 6, dim)
    out_single = single(x, enc_out)
    out_stacked = stacked(x, enc_out)

    assert not torch.allclose(out_single, out_stacked)


def test_decoder_gradient_flow():
    decoder = _make_decoder(dim=32, n_heads=4, n_layer=2)
    x = torch.randn(2, 4, 32, requires_grad=True)
    enc_out = torch.randn(2, 6, 32, requires_grad=True)

    out = decoder(x, enc_out)
    out.sum().backward()

    assert x.grad is not None
    assert enc_out.grad is not None
    assert decoder.layers[0].mask_attention.wq.weight.grad is not None
    assert decoder.layers[1].feed_forward.w1.weight.grad is not None
    assert decoder.norm.a_2.grad is not None
