import csv
import json
from pathlib import Path

from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    validate_annotation_label,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def read_csv(name):
    with (SAMPLE / name).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_metadata_example_has_required_fields_and_valid_phases():
    data = json.loads((SAMPLE / "metadata.example.json").read_text(encoding="utf-8"))
    required = {
        "session_id", "date", "camera_id", "fps", "resolution",
        "camera_view", "participants", "class_phase", "lighting_condition",
        "consent_confirmed", "notes",
    }
    assert required.issubset(data.keys())
    assert isinstance(data["consent_confirmed"], bool)
    assert data["fps"] > 0
    assert data["participants"] >= 0
    for phase in data["class_phase"]:
        assert int(phase["start_ms"]) >= 0
        assert int(phase["end_ms"]) > int(phase["start_ms"])


def test_annotation_example_semantics():
    rows = read_csv("annotation.example.csv")
    required = {
        "session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms",
        "class_phase", "atomic_behavior", "participation_label",
        "annotation_state", "label_source", "label_rule_version",
        "annotation_guide_version", "annotation_confidence", "occlusion_level",
        "zone_id", "annotator_id", "notes",
    }
    assert rows
    assert required.issubset(rows[0].keys())

    for row in rows:
        start = int(row["t_start_ms"])
        end = int(row["t_end_ms"])
        assert start >= 0
        assert end > start
        assert row["annotation_state"] in ANNOTATION_STATES
        confidence = float(row["annotation_confidence"])
        assert 0.0 <= confidence <= 1.0
        assert row["annotation_guide_version"].strip()
        validate_annotation_label(
            participation_label=row["participation_label"],
            annotation_state=row["annotation_state"],
            label_source=row["label_source"],
            label_rule_version=row["label_rule_version"],
        )


def test_window_example_semantics_and_versions():
    rows = read_csv("participation_windows.example.csv")
    required = {
        "window_id", "session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms",
        "class_phase", "zone_id", "feature_schema_version", "dataset_version",
        "split_version", "label_schema_version", "quality_state",
        "mean_visible_ratio", "mean_pose_confidence", "participation_label",
        "label_source", "label_rule_version", "annotation_guide_version", "split",
    }
    assert rows
    assert required.issubset(rows[0].keys())

    for row in rows:
        assert int(row["t_start_ms"]) >= 0
        assert int(row["t_end_ms"]) > int(row["t_start_ms"])
        assert row["split"] in {"train", "val", "test"}
        for version_key in (
            "feature_schema_version", "dataset_version", "split_version",
            "label_schema_version", "annotation_guide_version",
        ):
            assert row[version_key].strip()
        assert 0.0 <= float(row["mean_visible_ratio"]) <= 1.0
        assert 0.0 <= float(row["mean_pose_confidence"]) <= 1.0
        validate_annotation_label(
            participation_label=row["participation_label"],
            annotation_state=row["quality_state"],
            label_source=row["label_source"],
            label_rule_version=row["label_rule_version"],
        )


def test_sample_contains_human_and_weak_label_examples():
    rows = read_csv("annotation.example.csv")
    valid_rows = [r for r in rows if r["annotation_state"] == "VALID"]
    assert {r["label_source"] for r in valid_rows} >= {"HUMAN", "WEAK_RULE"}
    assert all(
        not r["participation_label"] or r["participation_label"] in CANONICAL_TRAINING_LABELS
        for r in rows
    )



def test_samples_match_metadata_timeline_and_each_other():
    metadata = json.loads((SAMPLE / "metadata.example.json").read_text(encoding="utf-8"))
    annotations = read_csv("annotation.example.csv")
    windows = read_csv("participation_windows.example.csv")
    for row in annotations + windows:
        assert row["session_id"] == metadata["session_id"]
        assert row["camera_id"] == metadata["camera_id"]
        start, end = int(row["t_start_ms"]), int(row["t_end_ms"])
        assert any(
            phase["name"] == row["class_phase"]
            and phase["start_ms"] <= start < end <= phase["end_ms"]
            for phase in metadata["class_phase"]
        ), row
    key_fields = ("session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms")
    indexed = {tuple(row[k] for k in key_fields): row for row in annotations}
    for window in windows:
        annotation = indexed[tuple(window[k] for k in key_fields)]
        for field in ("class_phase", "participation_label", "label_source", "label_rule_version"):
            assert window[field] == annotation[field]
