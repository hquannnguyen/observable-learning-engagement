"""Tests for window-id based evaluation joins (remediation A4)."""

import pandas as pd
import pytest

from scripts.evaluate import load_test_metadata, metrics_or_none


def make_meta(**overrides):
    rows = []
    for i in range(3):
        row = {
            "window_id": f"w{i}",
            "split": "test",
            "session_id": "s1",
            "track_id": 1,
            "participation_label": "HIGH_OBSERVED_PARTICIPATION",
            "mean_visible_ratio": 0.9,
            "mean_pose_confidence": 0.8,
        }
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def test_valid_metadata_is_indexed_by_window_id(tmp_path):
    path = tmp_path / "windows.csv"
    make_meta().to_csv(path, index=False)
    meta = load_test_metadata(path)
    assert list(meta.index) == ["w0", "w1", "w2"]


def test_duplicate_window_id_is_rejected(tmp_path):
    df = make_meta()
    df.loc[1, "window_id"] = "w0"
    path = tmp_path / "windows.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate window_id"):
        load_test_metadata(path)


def test_missing_required_column_is_rejected(tmp_path):
    df = make_meta().drop(columns=["mean_visible_ratio"])
    path = tmp_path / "windows.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_test_metadata(path)


def test_no_test_rows_is_rejected(tmp_path):
    df = make_meta(split="train")
    path = tmp_path / "windows.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="No rows with split"):
        load_test_metadata(path)


def test_metrics_or_none_handles_empty():
    assert metrics_or_none([], []) is None
