from __future__ import annotations
import torch
from torch import nn

class DPNLoss(nn.Module):
    """
    A corrected DPN-style polarization loss that pushes continuous features 
    towards discrete binary states (-1 or +1) without representation collapse.
    """
    def __init__(self, bit_length: int, scale: float = 1.0):
        super().__init__()
        self.bit_length = int(bit_length)
        self.scale = scale

    def forward(self, hash_codes: torch.Tensor) -> torch.Tensor:
        if hash_codes.dim() != 2:
            raise ValueError(f"Expected [B, bits], got {tuple(hash_codes.shape)}")
        if hash_codes.size(-1) != self.bit_length:
            raise ValueError(
                f"Expected {self.bit_length} bits, but got {hash_codes.size(-1)}"
            )
        
        # Continuous approximation of binary values
        h = torch.tanh(hash_codes)
        
        # Fix: Force values to polarise toward EITHER -1 OR +1.
        # This penalizes values near 0 and reaches a minimum at h = -1 and h = 1.
        loss = (h.pow(2) - 1.0).pow(2).mean()
        
        return loss * self.scale

__all__ = ["DPNLoss"]
