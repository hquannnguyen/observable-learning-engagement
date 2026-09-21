"""Validation helpers for temporal feature tensors."""

from __future__ import annotations

import numpy as np


def validate_temporal_features(
    features,
    *,
    expected_timesteps: int | None = None,
    min_feature_dim: int = 1,
) -> np.ndarray:
    """Validate the [T, F] tensor consumed by LSTM/TCN models."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Temporal features must have shape [T,F], got {array.shape}")
    if array.shape[0] <= 0:
        raise ValueError("Temporal features require at least one timestep")
    if array.shape[1] < min_feature_dim:
        raise ValueError(
            f"Feature dimension must be >= {min_feature_dim}, got {array.shape[1]}"
        )
    if expected_timesteps is not None and array.shape[0] != expected_timesteps:
        raise ValueError(
            f"Expected {expected_timesteps} timesteps, got {array.shape[0]}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Model-ready temporal features must not contain NaN/Inf")
    return array
