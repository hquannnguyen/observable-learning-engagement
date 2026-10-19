"""PyTorch Dataset for temporal sequence windows."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowFeatureDataset(Dataset):
    """Dataset for temporal feature windows and participation labels."""

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Args:

        features: [N, T, F] numpy array.
        labels: [N] numpy array of class indices (0..num_classes-1).
        """
        # Exclude unlabeled / invalid (-1) windows
        if labels is not None:
            valid_mask = labels >= 0
            self.features = features[valid_mask].astype(np.float32)
            self.labels = labels[valid_mask].astype(np.int64)
        else:
            self.features = features.astype(np.float32)
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = torch.from_numpy(self.features[idx])
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return (x,)

    def compute_class_weights(self, num_classes: int = 3) -> torch.Tensor:
        """Compute inverse-frequency class weights for cross-entropy loss."""
        if self.labels is None or len(self.labels) == 0:
            return torch.ones(num_classes, dtype=torch.float32)

        counts = np.bincount(self.labels, minlength=num_classes)
        total = len(self.labels)
        weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
        return torch.from_numpy(weights.astype(np.float32))
