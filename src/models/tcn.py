"""Temporal Convolutional Network (TCN) for 3-class participation classification.

Causal dilated convolutions consume the same ``[Batch, Timesteps, Features]``
tensor as the LSTM so comparisons use identical data/splits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    """Causal conv + residual + norm + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        if self.padding:
            out = out[..., : -self.padding]  # keep causality
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + self.residual(x)


class ParticipationTCN(nn.Module):
    """Causal dilated TCN classifier, equivalent input contract to LSTM."""

    def __init__(
        self,
        input_size: int = 77,
        hidden_size: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.dropout_p = dropout

        self.input_proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
                for level in range(num_levels)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        out = x.transpose(1, 2)
        out = self.input_proj(out)
        for block in self.blocks:
            out = block(out)
        avg_pool = out.mean(dim=-1)
        last_step = out[..., -1]
        combined = torch.cat([avg_pool, last_step], dim=-1)
        return self.head(combined)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)
