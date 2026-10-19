"""Customized LSTM model for 3-class observable learning participation classification.

Implements Task T22:
Consumes [Batch, Timesteps=40, Features=77] sequence tensor and outputs logits/probabilities
for:
0: HIGH_OBSERVED_PARTICIPATION
1: MODERATE_OBSERVED_PARTICIPATION
2: LOW_OBSERVED_PARTICIPATION
according to configs/baseline.yaml.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from src.contracts.labels import CANONICAL_TRAINING_LABELS

logger = logging.getLogger(__name__)


class ParticipationLSTM(nn.Module):
    """Bidirectional / Unidirectional LSTM Classifier for Observable Learning Participation."""

    def __init__(
        self,
        input_size: int = 77,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.30,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # Classification head with layer norm for stability
        fc_in = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, timesteps, input_size].

        Returns:
            Logits of shape [batch_size, num_classes].
        """
        # lstm_out: [batch_size, timesteps, hidden_size * num_directions]
        lstm_out, (hn, cn) = self.lstm(x)

        # Use global average pooling over time + final hidden state
        avg_pool = torch.mean(lstm_out, dim=1)
        last_step = lstm_out[:, -1, :]
        combined = 0.5 * (avg_pool + last_step)

        dropped = self.dropout(combined)
        logits = self.head(dropped)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities via softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/baseline.yaml",
        input_size: int = 77,
    ) -> ParticipationLSTM:
        """Instantiate LSTM model from baseline YAML config."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_cfg = cfg.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            dropout=float(model_cfg.get("dropout", 0.30)),
        )
