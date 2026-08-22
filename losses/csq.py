from __future__ import annotations
import torch
from torch import nn

class CSQLoss(nn.Module):
    """
    CSQ-style loss that keeps class centers separated and penalizes bit collapse.

    The original implementation only optimized cosine similarity to the target center,
    which allowed the model to collapse to a near-constant sign pattern. We add a
    bit-balance regularizer so the hash bits remain near 50/50 and do not drift to a
    trivial all-positive or all-negative representation.
    """

    def __init__(
        self,
        bit_length: int,
        num_classes: int,
        scale: float = 1.0,
        bit_balance_weight: float = 0.5,
        diversity_weight: float = 0.5,
    ):
        super().__init__()
        self.bit_length = int(bit_length)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.bit_balance_weight = float(bit_balance_weight)
        self.diversity_weight = float(diversity_weight)

        centers = torch.randint(0, 2, (num_classes, bit_length), dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("hash_centers", centers)

    def forward(self, hash_codes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if hash_codes.dim() != 2:
            raise ValueError(f"Expected [B, bits] hash codes, got {tuple(hash_codes.shape)}")
        if hash_codes.size(-1) != self.bit_length:
            raise ValueError(f"Expected {self.bit_length} bits, but got {hash_codes.size(-1)}")

        target_centers = self.hash_centers.to(hash_codes.device)[targets.to(torch.long)]

        cosine = torch.sum(hash_codes * target_centers, dim=1)
        hash_norm = torch.norm(hash_codes, p=2, dim=1) + 1e-8
        center_norm = torch.norm(target_centers, p=2, dim=1) + 1e-8
        cosine_sim = cosine / (hash_norm * center_norm)

        class_loss = 1.0 - cosine_sim

        # Use tanh as a differentiable approximation of the sign function.
        # torch.sign() has zero gradient, preventing these regularizers from working.
        relaxed_codes = torch.tanh(hash_codes)
        bit_mean = relaxed_codes.mean(dim=0)
        bit_balance = bit_mean.pow(2).mean()

        # Encourage diversity by discouraging a near-constant sign pattern across the batch.
        # A fully collapsed hash space has bit_mean close to ±1, which makes this term large.
        diversity_penalty = torch.mean((bit_mean.abs() - 0.0).pow(2))

        total = class_loss.mean() * self.scale + self.bit_balance_weight * bit_balance + self.diversity_weight * diversity_penalty
        return total

__all__ = ["CSQLoss"]
