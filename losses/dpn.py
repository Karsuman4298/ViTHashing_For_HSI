from __future__ import annotations
import torch
from torch import nn

class DPNLoss(nn.Module):
    """
    A corrected DPN-style polarization loss that pushes continuous features 
    towards discrete binary states (-1 or +1) without representation collapse,
    and includes semantic classification loss.
    """
    def __init__(self, bit_length: int, num_classes: int, scale: float = 1.0, pol_weight: float = 1.0):
        super().__init__()
        self.bit_length = int(bit_length)
        self.scale = scale
        self.pol_weight = pol_weight
        self.num_classes = int(num_classes)
        
        # Center-similarity formulation to provide semantic meaning
        centers = torch.randint(0, 2, (num_classes, bit_length), dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("hash_centers", centers)

    def forward(self, hash_codes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if hash_codes.dim() != 2:
            raise ValueError(f"Expected [B, bits], got {tuple(hash_codes.shape)}")
        if hash_codes.size(-1) != self.bit_length:
            raise ValueError(
                f"Expected {self.bit_length} bits, but got {hash_codes.size(-1)}"
            )
        
        # Classification loss (semantic similarity to centers)
        target_centers = self.hash_centers.to(hash_codes.device)[targets.to(torch.long)]
        cosine = torch.sum(hash_codes * target_centers, dim=1)
        hash_norm = torch.norm(hash_codes, p=2, dim=1) + 1e-8
        center_norm = torch.norm(target_centers, p=2, dim=1) + 1e-8
        cosine_sim = cosine / (hash_norm * center_norm)
        class_loss = 1.0 - cosine_sim
        
        # Continuous approximation of binary values
        h = torch.tanh(hash_codes)
        
        # Polarization loss: Force values to polarise toward EITHER -1 OR +1.
        pol_loss = (h.pow(2) - 1.0).pow(2).mean()
        
        loss = class_loss.mean() * self.scale + pol_loss * self.pol_weight
        return loss

__all__ = ["DPNLoss"]
