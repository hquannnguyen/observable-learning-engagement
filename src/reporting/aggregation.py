"""Aggregation helpers for timeline and classroom-zone reporting.

All outputs describe observable statistics. They must never contain individual
student judgments or identity information.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.contracts.labels import (
    CANONICAL_TRAINING_LABELS,
    UNCERTAIN_LABEL,
)

COUNT_COLUMNS = [
    "n_windows",
    "n_high",
    "n_moderate",
    "n_low",
    "n_uncertain",
    "pct_high",
    "pct_moderate",
    "pct_low",
    "pct_uncertain",
    "mean_visible_ratio",
    "mean_pose_confidence",
]


def _counts(labels: Iterable[str]) -> Dict[str, int]:
    values = list(labels)
    return {
        "n_windows": len(values),
        "n_high": values.count(CANONICAL_TRAINING_LABELS[0]),
        "n_moderate": values.count(CANONICAL_TRAINING_LABELS[1]),
        "n_low": values.count(CANONICAL_TRAINING_LABELS[2]),
        "n_uncertain": values.count(UNCERTAIN_LABEL),
    }


def _percentages(counts: Dict[str, int]) -> Dict[str, float]:
    total = max(counts["n_windows"], 1)
    return {
        "pct_high": round(counts["n_high"] / total, 4),
        "pct_moderate": round(counts["n_moderate"] / total, 4),
        "pct_low": round(counts["n_low"] / total, 4),
        "pct_uncertain": round(counts["n_uncertain"] / total, 4),
    }


def aggregate_timeline(
    predictions: pd.DataFrame,
    bucket_seconds: int = 60,
) -> pd.DataFrame:
    """Aggregate gated labels per session into fixed time buckets (default 1 minute).

    Sessions are never merged: ``t_start_ms`` is relative to each recording, so
    grouping only by minute would collapse several sessions into one bucket.
    """
    empty_columns = ["session_id", "minute_index", "minute_start_ms"] + COUNT_COLUMNS
    if predictions.empty:
        return pd.DataFrame(columns=empty_columns)

    required = {"t_start_ms", "gated_label"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    df = predictions.copy()
    df["t_start_ms"] = pd.to_numeric(df["t_start_ms"], errors="coerce")
    df = df[df["t_start_ms"].notna()]
    bucket_ms = int(bucket_seconds * 1000)
    df["minute_index"] = (df["t_start_ms"] // bucket_ms).astype(int)
    if "session_id" not in df.columns:
        df["session_id"] = "UNKNOWN"

    rows: List[Dict[str, Any]] = []
    for (session_id, minute_index), group in df.groupby(
        ["session_id", "minute_index"], sort=True
    ):
        counts = _counts(group["gated_label"])
        row: Dict[str, Any] = {
            "session_id": str(session_id),
            "minute_index": int(minute_index),
            "minute_start_ms": int(minute_index * bucket_ms),
            **counts,
            **_percentages(counts),
            "mean_visible_ratio": round(float(group["mean_visible_ratio"].mean()), 4)
            if "mean_visible_ratio" in group
            else None,
            "mean_pose_confidence": round(float(group["mean_pose_confidence"].mean()), 4)
            if "mean_pose_confidence" in group
            else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_zones(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by zone. Returns an empty frame when no real zone data exists."""
    if predictions.empty or "zone_id" not in predictions.columns:
        return pd.DataFrame(columns=["zone_id"] + COUNT_COLUMNS)

    df = predictions.copy()
    df["zone_id"] = df["zone_id"].fillna("UNKNOWN").astype(str).str.strip()
    df.loc[df["zone_id"].isin(["", "nan", "None"]), "zone_id"] = "UNKNOWN"
    if (df["zone_id"] == "UNKNOWN").all():
        return pd.DataFrame(columns=["zone_id"] + COUNT_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for zone_id, group in df.groupby("zone_id", sort=True):
        counts = _counts(group["gated_label"])
        row = {
            "zone_id": zone_id,
            **counts,
            **_percentages(counts),
            "mean_visible_ratio": round(float(group["mean_visible_ratio"].mean()), 4)
            if "mean_visible_ratio" in group
            else None,
            "mean_pose_confidence": round(float(group["mean_pose_confidence"].mean()), 4)
            if "mean_pose_confidence" in group
            else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(
    predictions: pd.DataFrame,
    *,
    experiment_id: Optional[str] = None,
    metric_scope: str = "weak_rule_labels",
    limitations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable summary with explicit caveats."""
    total = int(len(predictions))
    uncertain = int((predictions["gated_label"] == UNCERTAIN_LABEL).sum()) if total else 0
    accepted = total - uncertain
    counts = _counts(predictions["gated_label"]) if total else _counts([])

    accepted_accuracy = None
    has_ground_truth = (
        "true_label" in predictions.columns
        and predictions["true_label"].notna().any()
    )
    if accepted and has_ground_truth:
        accepted_df = predictions[predictions["gated_label"] != UNCERTAIN_LABEL]
        valid_truth = accepted_df["true_label"].notna()
        if valid_truth.any():
            matches = (
                accepted_df.loc[valid_truth, "gated_label"]
                == accepted_df.loc[valid_truth, "true_label"]
            ).sum()
            accepted_accuracy = round(float(matches / int(valid_truth.sum())), 4)

    summary: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "metric_scope": metric_scope,
        "has_ground_truth": bool(has_ground_truth),
        "total_windows": total,
        "accepted_windows": accepted,
        "uncertain_windows": uncertain,
        "coverage": round(accepted / total, 4) if total else None,
        "uncertain_rate": round(uncertain / total, 4) if total else None,
        "accepted_accuracy_vs_labeled_source": accepted_accuracy,
        "class_counts": counts,
        "sessions": sorted(predictions["session_id"].dropna().astype(str).unique().tolist())
        if "session_id" in predictions
        else [],
        "label_sources": {
            str(k): int(v)
            for k, v in predictions["label_source"].fillna("NONE").astype(str).value_counts().items()
        }
        if "label_source" in predictions
        else {},
        "metric_warning": (
            "No ground-truth labels in this inference run; only observable class "
            "distribution and coverage are reported."
            if not has_ground_truth
            else "Metrics compare model output against the labels recorded in the input "
            f"({metric_scope}). Weak-rule metrics are NOT human ground-truth accuracy."
        ),
        "limitations": limitations
        or [
            "Observable participation only; no emotion/intention inference.",
            "2D pose features; head_orientation_proxy is not gaze.",
            "No identity, no face recognition, anonymous track ids only.",
        ],
    }
    return summary
