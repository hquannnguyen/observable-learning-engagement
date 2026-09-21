import pytest

from src.contracts.labels import HIGH, UNCERTAIN_LABEL
from src.uncertainty.gate import UncertaintyThresholds, apply_uncertainty_gate

THRESHOLDS = UncertaintyThresholds(
    min_visible_ratio=0.60,
    min_mean_keypoint_confidence=0.30,
    min_probability=0.60,
)


def test_low_input_visibility_becomes_uncertain():
    out = apply_uncertainty_gate(
        [0.90, 0.05, 0.05], visible_ratio=0.40, mean_keypoint_confidence=0.90,
        thresholds=THRESHOLDS,
    )
    assert out == UNCERTAIN_LABEL


def test_low_keypoint_confidence_becomes_uncertain():
    out = apply_uncertainty_gate(
        [0.90, 0.05, 0.05], visible_ratio=0.90, mean_keypoint_confidence=0.20,
        thresholds=THRESHOLDS,
    )
    assert out == UNCERTAIN_LABEL


def test_low_model_confidence_becomes_uncertain():
    out = apply_uncertainty_gate(
        [0.40, 0.35, 0.25], visible_ratio=0.90, mean_keypoint_confidence=0.80,
        thresholds=THRESHOLDS,
    )
    assert out == UNCERTAIN_LABEL


def test_good_quality_returns_canonical_class():
    out = apply_uncertainty_gate(
        [0.80, 0.10, 0.10], visible_ratio=0.90, mean_keypoint_confidence=0.80,
        thresholds=THRESHOLDS,
    )
    assert out == HIGH


def test_invalid_probability_vector_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        apply_uncertainty_gate(
            [0.80, 0.20, 0.20], visible_ratio=0.90, mean_keypoint_confidence=0.80,
            thresholds=THRESHOLDS,
        )


@pytest.mark.parametrize("labels", [
    ["HIGH", "MODERATE", "LOW"],
    [HIGH, HIGH, HIGH],
    [HIGH, UNCERTAIN_LABEL, "LOW_OBSERVED_PARTICIPATION"],
    ["LOW_OBSERVED_PARTICIPATION", "MODERATE_OBSERVED_PARTICIPATION", HIGH],
    [HIGH],
])
def test_noncanonical_class_order_rejected(labels):
    with pytest.raises(ValueError, match="canonical training-label order"):
        apply_uncertainty_gate(
            [0.8, 0.1, 0.1], visible_ratio=0.9,
            mean_keypoint_confidence=0.9, class_labels=labels,
        )
