"""Contract tests for frozen session splits (T18 / remediation A3)."""

import hashlib

import pytest

from scripts.build_dataset import (
    load_frozen_splits,
    split_sessions,
    validate_splits,
)


def _write_split_files(tmp_path, train, val, test):
    for name, sessions in (("train", train), ("val", val), ("test", test)):
        (tmp_path / f"{name}_sessions.txt").write_text(
            "\n".join(sessions) + "\n", encoding="utf-8"
        )


def test_frozen_split_roundtrip_and_hashes(tmp_path):
    _write_split_files(tmp_path, ["s1", "s2"], ["s3"], ["s4"])
    train, val, test, hashes = load_frozen_splits(tmp_path)
    assert train == ["s1", "s2"]
    assert val == ["s3"]
    assert test == ["s4"]
    assert set(hashes) == {"train", "val", "test"}
    assert hashes["train"] == hashlib.sha256(
        (tmp_path / "train_sessions.txt").read_bytes()
    ).hexdigest()


def test_incomplete_split_is_rejected(tmp_path):
    (tmp_path / "train_sessions.txt").write_text("s1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete frozen split"):
        load_frozen_splits(tmp_path)


def test_missing_frozen_split_returns_none(tmp_path):
    assert load_frozen_splits(tmp_path) is None


def test_overlap_between_splits_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(["s1", "s2"], ["s2"], ["s3"], ["s1", "s2", "s3"])


def test_session_missing_from_split_is_rejected():
    with pytest.raises(ValueError, match="missing from the frozen split"):
        validate_splits(["s1"], ["s2"], ["s3"], ["s1", "s2", "s3", "s4"])


def test_split_referencing_unknown_session_is_rejected():
    with pytest.raises(ValueError, match="no pose data"):
        validate_splits(["s1"], ["s2"], ["s9"], ["s1", "s2", "s3"])


def test_fallback_split_is_disjoint():
    sessions = [f"s{i:02d}" for i in range(11)]
    train, val, test = split_sessions(sessions, seed=42)
    assert len(train) + len(val) + len(test) == len(sessions)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert set(train) | set(val) | set(test) == set(sessions)


def test_fallback_split_needs_three_sessions():
    with pytest.raises(ValueError, match="at least 3 sessions"):
        split_sessions(["s1", "s2"])
