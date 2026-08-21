"""Standalone semantic tokenizer for hyperspectral image (HSI) cubes."""

from __future__ import annotations

import torch
from torch import nn


class HSISemanticTokenizer(nn.Module):
    """Convert an HSI cube into a fixed number of semantic tokens.

    The input can be either ``[B, 1, bands, height, width]`` or
    ``[B, bands, height, width]``. The output is ``[B, num_tokens + 1, output_dim]``
    after applying the CLS token and positional embeddings.
    """

    def __init__(
        self,
        num_tokens=4,
        token_dim=64,
        feature_channels=8,
        input_channels=1,
        output_dim=768,
        spectral_pooling="attention",
        dropout=0.1,
    ):
        super().__init__()
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        if token_dim < 1 or feature_channels < 1:
            raise ValueError("token_dim and feature_channels must be positive")
        if spectral_pooling not in {"mean", "attention"}:
            raise ValueError("spectral_pooling must be 'mean' or 'attention'")

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.spectral_pooling = spectral_pooling

        self.spectral_spatial_features = nn.Sequential(
            nn.Conv3d(
                input_channels,
                feature_channels,
                kernel_size=(3, 3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
        )

        self.spectral_attention = None
        if self.spectral_pooling == "attention":
            self.spectral_attention = nn.Conv3d(feature_channels, 1, kernel_size=1, bias=False)

        self.spatial_features = nn.Sequential(
            nn.Conv2d(feature_channels, token_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True),
        )

        self.attention = nn.Conv2d(token_dim, num_tokens, kernel_size=1, bias=False)
        self.output_projection = nn.Linear(token_dim, output_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_tokens + 1, output_dim))
        self.dropout = nn.Dropout(dropout)

    def _as_5d(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        elif x.ndim != 5:
            raise ValueError(
                "Expected HSI input with shape [B, bands, H, W] or "
                "[B, channels, bands, H, W]"
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channel(s), got {x.shape[1]}"
            )
        return x

    def _spectral_pool(self, x):
        if self.spectral_pooling == "mean":
            return x.mean(dim=2)
        if self.spectral_attention is None:
            raise RuntimeError("spectral_attention must be initialized for attention pooling")

        weights = self.spectral_attention(x).squeeze(1)
        weights = torch.softmax(weights, dim=1)
        x = x.permute(0, 2, 1, 3, 4)
        x = x * weights.unsqueeze(2)
        x = x.sum(dim=1)
        return x

    def forward(self, x, return_attention=False):
        x = self._as_5d(x)
        batch_size, _, _, height, width = x.shape

        x = self.spectral_spatial_features(x)
        x = self._spectral_pool(x)
        x = self.spatial_features(x)

        attention = self.attention(x).flatten(2).softmax(dim=-1)
        values = x.flatten(2)
        tokens = torch.einsum("bln,bcn->blc", attention, values)
        tokens = self.output_projection(tokens)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.position_embeddings
        tokens = self.dropout(tokens)

        if return_attention:
            attention_map = attention.view(batch_size, self.num_tokens, height, width)
            return tokens, attention_map
        return tokens


__all__ = ["HSISemanticTokenizer"]
