"""Tests for aggregation and export bundle (remediation C2)."""

import json
from pathlib import Path

import pandas as pd

from src.contracts.labels import HIGH, LOW, MODERATE, UNCERTAIN_LABEL
from src.reporting.aggregation import (
    aggregate_timeline,
    aggregate_zones,
    build_summary,
)
from src.reporting.exporter import export_bundle


def make_predictions():
    rows = [
        (0, HIGH, HIGH, "A"),
        (0, MODERATE, MODERATE, "A"),
        (0, LOW, UNCERTAIN_LABEL, "B"),
        (60000, HIGH, HIGH, "B"),
        (60000, LOW, LOW, "B"),
        (60000, MODERATE, MODERATE, "A"),
    ]
    return pd.DataFrame(
        [
            {
                "window_id": f"w{i}",
                "session_id": "s1",
                "track_id": 1,
                "t_start_ms": t,
                "true_label": true,
                "pred_label": gated,
                "gated_label": gated,
                "label_source": "WEAK_RULE",
                "mean_visible_ratio": 0.9,
                "mean_pose_confidence": 0.8,
                "zone_id": zone,
            }
            for i, (t, true, gated, zone) in enumerate(rows)
        ]
    )


def test_timeline_sums_to_total():
    df = make_predictions()
    timeline = aggregate_timeline(df)
    assert len(timeline) == 2
    assert timeline["n_windows"].sum() == len(df)
    assert timeline["n_uncertain"].sum() == 1


def test_zone_summary_has_n_per_zone():
    zones = aggregate_zones(make_predictions())
    assert set(zones["zone_id"]) == {"A", "B"}
    assert zones["n_windows"].sum() == 6


def test_timeline_does_not_merge_sessions():
    df = make_predictions()
    df["session_id"] = ["s1", "s1", "s2", "s1", "s1", "s2"]
    timeline = aggregate_timeline(df)
    assert len(timeline) == 4  # 2 sessions x 2 minutes
    assert timeline["n_windows"].sum() == 6
    assert {"s1", "s2"} == set(timeline["session_id"])


def test_zone_summary_empty_without_real_zones():
    df = make_predictions()
    df["zone_id"] = None
    zones = aggregate_zones(df)
    assert zones.empty


def test_summary_reports_coverage_and_warning():
    summary = build_summary(make_predictions(), experiment_id="exp-test")
    assert summary["total_windows"] == 6
    assert summary["uncertain_windows"] == 1
    assert summary["coverage"] == round(5 / 6, 4)
    assert "NOT human ground-truth" in summary["metric_warning"]


def test_export_bundle_writes_all_artifacts(tmp_path):
    paths = export_bundle(make_predictions(), tmp_path, experiment_id="exp-test")
    for path in paths.values():
        assert Path(path).exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_id"] == "exp-test"
    report = (tmp_path / "summary_report.md").read_text(encoding="utf-8")
    assert "Limitations" in report
    assert "Observable" in report
