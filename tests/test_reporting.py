"""Tests for anonymized reporting and aggregation modules (Task T32, T37, T38)."""

from pathlib import Path
import pandas as pd
import pytest

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
)
from src.models.baseline_heuristic import WindowAssessment
from src.reporting.aggregator import (
    aggregate_by_time,
    aggregate_by_zone,
    build_results_dataframe,
    build_timeline_dataframe,
    build_zone_summary_dataframe,
)
from src.reporting.exporter import export_bundle


@pytest.fixture
def sample_assessments():
    return [
        WindowAssessment(
            track_id=1,
            t_start_ms=0,
            t_end_ms=4000,
            canonical_label=HIGH,
            probabilities=(0.85, 0.10, 0.05),
            visible_ratio=0.92,
            mean_keypoint_conf=0.88,
            atomic_behaviors=["BOARD_OR_INSTRUCTOR_FACING"],
            zone_id="F-C",
        ),
        WindowAssessment(
            track_id=1,
            t_start_ms=2000,
            t_end_ms=6000,
            canonical_label=HIGH,
            probabilities=(0.80, 0.15, 0.05),
            visible_ratio=0.90,
            mean_keypoint_conf=0.85,
            atomic_behaviors=["BOARD_OR_INSTRUCTOR_FACING"],
            zone_id="F-C",
        ),
        WindowAssessment(
            track_id=2,
            t_start_ms=0,
            t_end_ms=4000,
            canonical_label=MODERATE,
            probabilities=(0.10, 0.80, 0.10),
            visible_ratio=0.80,
            mean_keypoint_conf=0.75,
            atomic_behaviors=["READING_WRITING"],
            zone_id="M-C",
        ),
        WindowAssessment(
            track_id=3,
            t_start_ms=0,
            t_end_ms=4000,
            canonical_label=UNCERTAIN_LABEL,
            probabilities=(0.33, 0.34, 0.33),
            visible_ratio=0.35,
            mean_keypoint_conf=0.25,
            atomic_behaviors=["OCCLUDED"],
            zone_id="B-L",
        ),
    ]


def test_aggregate_by_time_anonymization(sample_assessments):
    results_df = build_results_dataframe(sample_assessments)
    timeline_df = aggregate_by_time(results_df, bucket_sec=5, anonymize=True)

    assert not timeline_df.empty
    assert "time_sec" in timeline_df.columns
    assert "HIGH_pct" in timeline_df.columns
    assert "MODERATE_pct" in timeline_df.columns
    assert "LOW_pct" in timeline_df.columns
    assert "UNCERTAIN_pct" in timeline_df.columns

    # Check anonymization
    assert "track_id" not in timeline_df.columns
    assert "bbox_x1" not in timeline_df.columns


def test_aggregate_by_zone_anonymization(sample_assessments):
    results_df = build_results_dataframe(sample_assessments)
    zone_df = aggregate_by_zone(results_df, anonymize=True)

    assert len(zone_df) == 9  # 3x3 classroom zones
    assert "sample_count_N" in zone_df.columns
    assert "track_id" not in zone_df.columns

    fc_row = zone_df[zone_df["zone_id"] == "F-C"].iloc[0]
    assert fc_row["sample_count_N"] == 2
    assert fc_row["HIGH_pct"] == 100.0


def test_export_bundle_creates_artifacts(tmp_path):
    predictions = pd.DataFrame([
        {
            "window_id": "session_001_cam_A_trk001_w0000",
            "session_id": "session_001",
            "t_start_ms": 0,
            "minute_index": 0,
            "zone_id": "M-C",
            "pred_label": HIGH,
            "gated_label": HIGH,
            "true_label": HIGH,
            "gate_reason": "accepted",
            "mean_visible_ratio": 0.85,
            "mean_pose_confidence": 0.80,
            "prob_high": 0.85,
            "prob_moderate": 0.10,
            "prob_low": 0.05,
        },
        {
            "window_id": "session_001_cam_A_trk002_w0000",
            "session_id": "session_001",
            "t_start_ms": 2000,
            "minute_index": 0,
            "zone_id": "B-L",
            "pred_label": LOW,
            "gated_label": UNCERTAIN_LABEL,
            "true_label": LOW,
            "gate_reason": "low_visibility",
            "mean_visible_ratio": 0.40,
            "mean_pose_confidence": 0.35,
            "prob_high": 0.05,
            "prob_moderate": 0.20,
            "prob_low": 0.75,
        },
    ])

    out_dir = tmp_path / "export_test"
    paths = export_bundle(predictions, out_dir, experiment_id="exp_test")

    assert (out_dir / "results.csv").is_file()
    assert (out_dir / "timeline.csv").is_file()
    assert (out_dir / "zone_summary.csv").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "summary_report.md").is_file()
