"""Quality + model-confidence uncertainty gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.contracts.labels import CANONICAL_TRAINING_LABELS, UNCERTAIN_LABEL


@dataclass(frozen=True)
class UncertaintyThresholds:
    min_visible_ratio: float = 0.60
    min_mean_keypoint_confidence: float = 0.30
    min_probability: float = 0.60
    upper_body_min_visible_ratio: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("min_visible_ratio", self.min_visible_ratio),
            ("min_mean_keypoint_confidence", self.min_mean_keypoint_confidence),
            ("min_probability", self.min_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0,1], got {value}")
        if self.upper_body_min_visible_ratio is not None and not 0 <= self.upper_body_min_visible_ratio <= 1:
            raise ValueError("upper_body_min_visible_ratio must be in [0,1]")


def explain_uncertainty_gate(
    probabilities: Sequence[float],
    *,
    visible_ratio: float,
    mean_keypoint_confidence: float,
    upper_body_visible_ratio: float | None = None,
    thresholds: UncertaintyThresholds = UncertaintyThresholds(),
    class_labels: Sequence[str] = CANONICAL_TRAINING_LABELS,
) -> tuple[str, str]:
    """Return ``(label, reason)`` for the quality + model-confidence gate.

    ``reason`` is one of ``LOW_VISIBILITY``, ``LOW_KEYPOINT_CONF``,
    ``LOW_MODEL_CONF`` or ``ACCEPTED``. The label is always canonical (never a
    short display label).
    """
    if tuple(class_labels) != CANONICAL_TRAINING_LABELS:
        raise ValueError("class_labels must match the canonical training-label order")
    if not 0.0 <= visible_ratio <= 1.0:
        raise ValueError("visible_ratio must be in [0,1]")
    if not 0.0 <= mean_keypoint_confidence <= 1.0:
        raise ValueError("mean_keypoint_confidence must be in [0,1]")

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size != len(class_labels):
        raise ValueError(
            f"Expected {len(class_labels)} probabilities, got shape {probs.shape}"
        )
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be finite values in [0,1]")
    if not np.isclose(probs.sum(), 1.0, atol=1e-5):
        raise ValueError(f"probabilities must sum to 1, got {probs.sum():.6f}")

    upper_body_ok = (
        thresholds.upper_body_min_visible_ratio is not None and upper_body_visible_ratio is not None
        and upper_body_visible_ratio >= thresholds.upper_body_min_visible_ratio
    )
    if visible_ratio < thresholds.min_visible_ratio and not upper_body_ok:
        return UNCERTAIN_LABEL, "LOW_VISIBILITY"
    if mean_keypoint_confidence < thresholds.min_mean_keypoint_confidence:
        return UNCERTAIN_LABEL, "LOW_KEYPOINT_CONF"

    best_index = int(np.argmax(probs))
    if float(probs[best_index]) < thresholds.min_probability:
        return UNCERTAIN_LABEL, "LOW_MODEL_CONF"
    return str(class_labels[best_index]), "ACCEPTED"


def apply_uncertainty_gate(
    probabilities: Sequence[float],
    *,
    visible_ratio: float,
    mean_keypoint_confidence: float,
    thresholds: UncertaintyThresholds = UncertaintyThresholds(),
    class_labels: Sequence[str] = CANONICAL_TRAINING_LABELS,
) -> str:
    """Return a canonical class label or UNCERTAIN.

    The function is intentionally model-agnostic and is used by tests and the
    future application layer. It never returns short display labels.
    Probability indices must follow the canonical training-label order.
    """
    label, _reason = explain_uncertainty_gate(
        probabilities,
        visible_ratio=visible_ratio,
        mean_keypoint_confidence=mean_keypoint_confidence,
        thresholds=thresholds,
        class_labels=class_labels,
    )
    return label
