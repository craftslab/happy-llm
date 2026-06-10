import pytest
import torch

from multihead_attention import ModelArgs
from transformer import PositionalEncoding, Transformer


def _make_args(
    dim: int = 64,
    n_heads: int = 4,
    n_layer: int = 1,
    vocab_size: int = 128,
    block_size: int = 32,
    dropout: float = 0.0,
) -> ModelArgs:
    return ModelArgs(
        dim=dim,
        n_heads=n_heads,
        n_layer=n_layer,
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=dropout,
        max_seq_len=block_size,
    )


def _make_model(**kwargs) -> Transformer:
    model = Transformer(_make_args(**kwargs))
    model.eval()
    return model


def test_transformer_inference_logits_shape():
    batch, seq, vocab_size = 2, 8, 128
    model = _make_model(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (batch, seq))

    logits, loss = model(idx)

    assert logits.shape == (batch, 1, vocab_size)
    assert loss is None


def test_transformer_training_logits_and_loss():
    batch, seq, vocab_size = 2, 8, 128
    model = _make_model(vocab_size=vocab_size)
    model.train()
    idx = torch.randint(0, vocab_size, (batch, seq))
    targets = torch.randint(0, vocab_size, (batch, seq))

    logits, loss = model(idx, targets)

    assert logits.shape == (batch, seq, vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_transformer_exceeds_block_size_raises():
    model = _make_model(block_size=8)
    idx = torch.randint(0, 128, (1, 9))

    with pytest.raises(AssertionError):
        model(idx)


def test_transformer_requires_vocab_size():
    with pytest.raises(AssertionError):
        Transformer(
            ModelArgs(dim=64, n_heads=4, block_size=32, max_seq_len=32)
        )


def test_transformer_requires_block_size():
    with pytest.raises(AssertionError):
        Transformer(
            ModelArgs(dim=64, n_heads=4, vocab_size=128, max_seq_len=32)
        )


def test_get_num_params():
    model = _make_model(dim=32, n_heads=4, n_layer=1, vocab_size=64)

    total = model.get_num_params()
    non_embedding = model.get_num_params(non_embedding=True)

    assert total > 0
    assert non_embedding < total
    assert non_embedding == total - model.transformer.wte.weight.numel()


def test_positional_encoding_preserves_shape():
    args = _make_args(dim=32, block_size=16)
    pe = PositionalEncoding(args)
    x = torch.randn(2, 10, 32)

    out = pe(x)

    assert out.shape == x.shape


def test_positional_encoding_changes_values():
    args = _make_args(dim=32, block_size=16)
    pe = PositionalEncoding(args)
    x = torch.zeros(1, 4, 32)

    out = pe(x)

    assert not torch.allclose(out, x)


def test_loss_ignores_masked_targets():
    model = _make_model(vocab_size=32)
    model.train()
    idx = torch.randint(0, 32, (1, 4))
    targets_all = torch.randint(0, 32, (1, 4))
    targets_masked = targets_all.clone()
    targets_masked[:, 2:] = -1

    _, loss_all = model(idx, targets_all)
    _, loss_masked = model(idx, targets_masked)

    assert torch.isfinite(loss_all)
    assert torch.isfinite(loss_masked)
    assert not torch.allclose(loss_all, loss_masked)


def test_transformer_gradient_flow():
    model = _make_model(dim=32, n_heads=4, vocab_size=64)
    model.train()
    idx = torch.randint(0, 64, (2, 6))
    targets = torch.randint(0, 64, (2, 6))

    logits, loss = model(idx, targets)
    loss.backward()

    assert model.transformer.wte.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.transformer.encoder.layers[0].attention.wq.weight.grad is not None
