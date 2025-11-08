from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class BookEncoderConfig(PretrainedConfig):
    model_type = "book-encoder"

    def __init__(
        self, input_dim: int = 1200, hidden_dim: int = 512, output_dim: int = 256, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


class BookEncoderModel(PreTrainedModel):
    config_class = BookEncoderConfig

    def __init__(self, config: BookEncoderConfig) -> None:
        super().__init__(config)
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return F.normalize(self.net(x), p=2, dim=-1)
