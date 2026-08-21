from __future__ import annotations
import torch
from torch import nn

class CSQLoss(nn.Module):
    """
    A corrected, high-performance CSQ-style loss using orthogonal-leaning 
    class-specific hash centers.
    """
    def __init__(self, bit_length: int, num_classes: int, scale: float = 1.0):
        super().__init__()
        self.bit_length = int(bit_length)
        self.num_classes = int(num_classes)
        self.scale = scale

        # Fix 1: Generate discrete, balanced binary centers (-1 or 1)
        # For small bit lengths, consider a Hadamard matrix to ensure strict orthogonality
        centers = torch.randint(0, 2, (num_classes, bit_length)).float() * 2.0 - 1.0
        self.register_buffer("hash_centers", centers)

    def forward(self, hash_codes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if hash_codes.dim() != 2:
            raise ValueError(f"Expected [B, bits] hash codes, got {tuple(hash_codes.shape)}")
        if hash_codes.size(-1) != self.bit_length:
            raise ValueError(f"Expected {self.bit_length} bits, but got {hash_codes.size(-1)}")

        target_centers = self.hash_centers.to(hash_codes.device)[targets.to(torch.long)]

        # Fix 2: Remove tanh optimization bottlenecks. Use continuous values 
        # directly for stable norm-based cosine similarity, or switch to BCE loss 
        # on the dot product (standard for Central Similarity Quantization).
        cosine = torch.sum(hash_codes * target_centers, dim=1)
        hash_norm = torch.norm(hash_codes, p=2, dim=1) + 1e-8
        center_norm = torch.norm(target_centers, p=2, dim=1) + 1e-8
        
        cosine_sim = cosine / (hash_norm * center_norm)
        
        # Scale and average the distance
        loss = 1.0 - cosine_sim
        return loss.mean() * self.scale

__all__ = ["CSQLoss"]
