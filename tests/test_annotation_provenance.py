"""Tests for honest annotation provenance and strict human QA (remediation A2/A6)."""

import pandas as pd
import pytest

from scripts.dataset_qa import (
    compute_cohen_kappa,
    compute_double_annotation_agreement,
)
from src.contracts.labels import HIGH, LOW, MODERATE


def make_annotations(labels, sources):
    rows = []
    for i, (label, source) in enumerate(zip(labels, sources)):
        rows.append(
            {
                "session_id": "s1",
                "camera_id": "cam_A",
                "track_id": f"track_{i:03d}",
                "t_start_ms": i * 2000,
                "t_end_ms": i * 2000 + 4000,
                "participation_label": label,
                "label_source": source,
            }
        )
    return pd.DataFrame(rows)


def test_identical_files_are_flagged():
    df = make_annotations([HIGH, MODERATE, LOW], ["HUMAN"] * 3)
    result = compute_double_annotation_agreement(df.copy(), df.copy())
    assert result["identical_files"] is True
    assert result["cohen_kappa"] == 1.0


def test_strict_human_rejects_weak_labels():
    a = make_annotations([HIGH, MODERATE], ["HUMAN", "WEAK_RULE"])
    b = make_annotations([HIGH, MODERATE], ["HUMAN", "HUMAN"])
    with pytest.raises(ValueError, match="non-human labels"):
        compute_double_annotation_agreement(a, b, strict_human=True)


def test_strict_human_accepts_disagreement():
    a = make_annotations([HIGH, MODERATE, LOW], ["HUMAN"] * 3)
    b = make_annotations([HIGH, LOW, LOW], ["ADJUDICATED"] * 3)
    result = compute_double_annotation_agreement(a, b, strict_human=True)
    assert result["identical_files"] is False
    assert result["valid_pairs_count"] == 3
    assert result["agree_count"] == 2
    assert result["cohen_kappa"] < 1.0


def test_cohen_kappa_perfect_and_chance():
    labels = [HIGH, MODERATE, LOW]
    assert compute_cohen_kappa(labels, labels, labels)[0] == 1.0
