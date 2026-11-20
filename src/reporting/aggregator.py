"""Aggregation and reporting module for CVF25PRJ10.

Implements Task T32 (Zone/bias aggregation), Task T37 (Timeline aggregation),
and Task T38 (Export result/report).
Strictly adheres to reporting sample count N alongside zone metrics.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
    to_display_label,
)
from src.models.baseline_heuristic import WindowAssessment

ALL_ZONES = [
    "F-L", "F-C", "F-R",
    "M-L", "M-C", "M-R",
    "B-L", "B-C", "B-R",
]


def build_results_dataframe(
    assessments: List[WindowAssessment],
    session_id: str = "session_001",
    camera_id: str = "A",
    class_phase: str = "lecture",
) -> pd.DataFrame:
    """Build standardized tabular results conforming to docs/dataset_schema.md."""
    rows = []
    for idx, w in enumerate(assessments):
        display_lbl = to_display_label(w.canonical_label)
        rows.append({
            "window_idx": idx + 1,
            "session_id": session_id,
            "camera_id": camera_id,
            "track_id": w.track_id,
            "t_start_ms": w.t_start_ms,
            "t_end_ms": w.t_end_ms,
            "class_phase": class_phase,
            "zone_id": w.zone_id,
            "canonical_label": w.canonical_label,
            "display_label": display_lbl,
            "prob_high": round(w.probabilities[0], 4),
            "prob_moderate": round(w.probabilities[1], 4),
            "prob_low": round(w.probabilities[2], 4),
            "visible_ratio": round(w.visible_ratio, 4),
            "mean_keypoint_conf": round(w.mean_keypoint_conf, 4),
            "atomic_behaviors": ";".join(w.atomic_behaviors),
        })

    return pd.DataFrame(rows)


def build_timeline_dataframe(results_df: pd.DataFrame, bucket_sec: int = 5) -> pd.DataFrame:
    """Aggregate participation distribution over time windows."""
    if results_df.empty:
        return pd.DataFrame(columns=[
            "time_sec", "active_tracks", "HIGH_pct", "MODERATE_pct", "LOW_pct", "UNCERTAIN_pct"
        ])

    df = results_df.copy()
    df["time_sec"] = (df["t_start_ms"] / 1000.0 // bucket_sec) * bucket_sec

    timeline_rows = []
    for t_sec, group in df.groupby("time_sec"):
        total = len(group)
        counts = group["display_label"].value_counts().to_dict()

        timeline_rows.append({
            "time_sec": int(t_sec),
            "active_tracks": total,
            "HIGH_pct": round(counts.get("HIGH", 0) / total * 100, 1),
            "MODERATE_pct": round(counts.get("MODERATE", 0) / total * 100, 1),
            "LOW_pct": round(counts.get("LOW", 0) / total * 100, 1),
            "UNCERTAIN_pct": round(counts.get("UNCERTAIN", 0) / total * 100, 1),
        })

    return pd.DataFrame(timeline_rows).sort_values("time_sec").reset_index(drop=True)


def build_zone_summary_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across 3x3 classroom zones, always reporting sample count N."""
    summary_rows = []

    for zone in ALL_ZONES:
        group = results_df[results_df["zone_id"] == zone] if not results_df.empty else pd.DataFrame()
        n_samples = len(group)

        if n_samples > 0:
            counts = group["display_label"].value_counts().to_dict()
            high_pct = round(counts.get("HIGH", 0) / n_samples * 100, 1)
            mod_pct = round(counts.get("MODERATE", 0) / n_samples * 100, 1)
            low_pct = round(counts.get("LOW", 0) / n_samples * 100, 1)
            unc_pct = round(counts.get("UNCERTAIN", 0) / n_samples * 100, 1)
            mean_vis = round(float(group["visible_ratio"].mean()), 3)
            mean_conf = round(float(group["mean_keypoint_conf"].mean()), 3)
        else:
            high_pct = mod_pct = low_pct = unc_pct = 0.0
            mean_vis = mean_conf = 0.0

        summary_rows.append({
            "zone_id": zone,
            "sample_count_N": n_samples,
            "mean_visible_ratio": mean_vis,
            "mean_keypoint_conf": mean_conf,
            "HIGH_pct": high_pct,
            "MODERATE_pct": mod_pct,
            "LOW_pct": low_pct,
            "UNCERTAIN_pct": unc_pct,
        })

    return pd.DataFrame(summary_rows)
