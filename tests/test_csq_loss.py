import torch

from losses.csq import CSQLoss


def test_csq_penalizes_collapsed_hashes():
    loss_fn = CSQLoss(bit_length=16, num_classes=7, bit_balance_weight=1.0)
    logits = torch.full((8, 16), 5.0)
    targets = torch.arange(8) % 7

    loss = float(loss_fn(logits, targets))

    assert loss > 1.5, f"Expected collapsed hashes to be penalized strongly, got {loss:.4f}"
