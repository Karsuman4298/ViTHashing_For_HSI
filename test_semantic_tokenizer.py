import pytest
import torch

from semantic_tokenizer import HSISemanticTokenizer


@pytest.mark.parametrize("bands", [20, 48, 144, 176])
def test_tokenizer_shapes_and_gradients(bands):
    model = HSISemanticTokenizer(
        num_tokens=4,
        token_dim=32,
        feature_channels=8,
        input_channels=1,
        output_dim=64,
        spectral_pooling="attention",
    )
    model.train()
    x = torch.randn(1, bands, 13, 13, requires_grad=True)

    tokens = model(x)
    assert tokens.shape == (1, 5, 64)

    loss = tokens.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_tokenizer_accepts_train_and_eval_modes(mode):
    model = HSISemanticTokenizer(
        num_tokens=4,
        token_dim=32,
        feature_channels=8,
        output_dim=64,
        spectral_pooling="mean",
    )
    if mode == "train":
        model.train()
    else:
        model.eval()

    x = torch.randn(1, 48, 13, 13)
    tokens = model(x)
    assert tokens.shape == (1, 5, 64)


@pytest.mark.parametrize("spectral_pooling", ["mean", "attention"])
def test_attention_maps_are_normalized(spectral_pooling):
    model = HSISemanticTokenizer(
        num_tokens=3,
        token_dim=16,
        feature_channels=6,
        output_dim=32,
        spectral_pooling=spectral_pooling,
    )
    model.eval()
    x = torch.randn(1, 48, 17, 17)

    tokens, attention = model(x, return_attention=True)
    assert tokens.shape == (1, 4, 32)
    assert attention.shape == (1, 3, 17, 17)
    sums = attention.sum(dim=(2, 3))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
