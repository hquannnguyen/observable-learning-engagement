from pathlib import Path

import yaml

from src.contracts.labels import CANONICAL_TRAINING_LABELS, DISPLAY_LABEL_BY_CANONICAL

ROOT = Path(__file__).resolve().parents[1]


def load_baseline():
    with (ROOT / "configs" / "baseline.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_baseline_has_canonical_three_training_classes():
    config = load_baseline()
    classes = tuple(config["model"]["classes"])
    assert config["model"]["num_classes"] == 3
    assert classes == CANONICAL_TRAINING_LABELS
    assert "UNCERTAIN" not in classes


def test_uncertainty_is_system_output():
    config = load_baseline()
    assert config["uncertainty"]["output_label"] == "UNCERTAIN"


def test_display_mapping_matches_code_contract():
    config = load_baseline()
    assert config["reporting"]["display_label_map"] == DISPLAY_LABEL_BY_CANONICAL


def test_versions_exist_in_baseline():
    config = load_baseline()
    for key in ("dataset_version", "split_version", "label_schema_version", "feature_schema_version"):
        assert isinstance(config["data"][key], str)
        assert config["data"][key].strip()


def test_quality_thresholds_are_probabilities():
    config = load_baseline()
    for key in ("min_visible_ratio", "min_mean_keypoint_confidence"):
        value = config["quality_gate"][key]
        assert 0.0 <= value <= 1.0
    assert 0.0 <= config["uncertainty"]["min_probability"] <= 1.0
