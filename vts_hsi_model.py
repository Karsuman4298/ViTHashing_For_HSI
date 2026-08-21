from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from TransformerModel.modeling import Encoder, VIT_CONFIGS
from semantic_tokenizer import HSISemanticTokenizer


@dataclass
class VTSHSIConfig:
    num_tokens: int = 4
    token_dim: int = 64
    feature_channels: int = 8
    output_dim: int = 768
    hidden_size: int = 768
    spectral_pooling: str = "attention"
    use_all_tokens: bool = True
    hash_bit_length: int = 32
    vit_model_name: str = "ViT-B_16"
    input_channels: int = 1
    dropout: float = 0.1


class VTSHSIModel(nn.Module):
    def __init__(
        self,
        num_tokens: int = 4,
        token_dim: int = 64,
        feature_channels: int = 8,
        output_dim: int = 768,
        spectral_pooling: str = "attention",
        use_all_tokens: bool = True,
        hash_bit_length: int = 32,
        vit_model_name: str = "ViT-B_16",
        input_channels: int = 1,
        img_size: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = VIT_CONFIGS[vit_model_name]
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.use_all_tokens = use_all_tokens
        self.hash_bit_length = hash_bit_length

        self.tokenizer = HSISemanticTokenizer(
            num_tokens=num_tokens,
            token_dim=token_dim,
            feature_channels=feature_channels,
            input_channels=input_channels,
            output_dim=output_dim,
            spectral_pooling=spectral_pooling,
            dropout=dropout,
        )
        self.encoder = Encoder(self.config, vis=False)

        in_features = (num_tokens + 1) * output_dim if use_all_tokens else output_dim
        self.hash_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit_length),
        )

    def forward(self, x):
        token_sequence = self.tokenizer(x)
        encoded, _ = self.encoder(token_sequence)
        if self.use_all_tokens:
            pooled = encoded.reshape(encoded.size(0), -1)
        else:
            pooled = encoded[:, 0]
        return self.hash_head(pooled)


__all__ = ["VTSHSIConfig", "VTSHSIModel"]
