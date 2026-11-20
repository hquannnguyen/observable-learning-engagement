"""Position-stratified evidence, with independent human and weak scopes."""
from __future__ import annotations

import pandas as pd
from src.contracts.labels import CANONICAL_TRAINING_LABELS, UNCERTAIN_LABEL
from src.training.trainer import compute_multiclass_metrics


def _f1(frame: pd.DataFrame, prediction_column: str):
    if frame.empty:
        return None
    labels = list(CANONICAL_TRAINING_LABELS)
    valid = frame[frame["true_label"].isin(labels) & frame[prediction_column].isin(labels)]
    if valid.empty:
        return None
    return compute_multiclass_metrics(valid["true_label"].map(labels.index).to_numpy(),
                                     valid[prediction_column].map(labels.index).to_numpy())["macro_f1"]


def analyze_positions(predictions: pd.DataFrame, min_samples: int = 20):
    """Never report weak agreement under a human-performance column."""
    df = predictions.copy()
    for key, default in (("session_id", "UNKNOWN"), ("zone_id", "UNKNOWN"),
                         ("label_source", "NONE"), ("true_label", None)):
        if key not in df:
            df[key] = default
        df[key] = df[key].fillna(default if default is not None else "")
    rows = []
    for (session, zone), group in df.groupby(["session_id", "zone_id"], dropna=False):
        accepted = group["gated_label"] != UNCERTAIN_LABEL
        human = group[group["label_source"].isin(["HUMAN", "ADJUDICATED"]) & group["true_label"].isin(CANONICAL_TRAINING_LABELS)]
        weak = group[(group["label_source"] == "WEAK_RULE") & group["true_label"].isin(CANONICAL_TRAINING_LABELS)]
        rows.append({
            "session_id": str(session), "zone_id": str(zone), "n_windows": len(group),
            "n_human": len(human), "n_weak": len(weak),
            "coverage": round(float(accepted.mean()), 4),
            "uncertain_rate": round(float((~accepted).mean()), 4),
            "mean_visible_ratio": float(group["mean_visible_ratio"].mean()) if "mean_visible_ratio" in group else None,
            "mean_pose_confidence": float(group["mean_pose_confidence"].mean()) if "mean_pose_confidence" in group else None,
            "human_raw_macro_f1": _f1(human, "pred_label"),
            "human_accepted_macro_f1": _f1(human, "gated_label"),
            "weak_agreement_macro_f1": _f1(weak, "pred_label"),
            "human_sample_warning": "INSUFFICIENT_HUMAN_SAMPLES" if len(human) < min_samples else "READ_WITH_POSE_QUALITY_AND_OCCLUSION",
        })
    table = pd.DataFrame(rows)
    report = {
        "status": "HUMAN_REVIEW_REQUIRED" if table.empty or table["n_human"].sum() == 0 else "DESCRIPTIVE_STRATIFIED_METRICS",
        "min_human_samples_per_group": min_samples,
        "groups": table.where(pd.notna(table), None).to_dict(orient="records"),
        "limitations": ["Zone is geometric, not a verified seat map.",
                        "Overlapping windows are not independent people.",
                        "No causal or fairness conclusion follows from a heatmap or small groups.",
                        "Human annotation and viewpoint/occlusion review are required before interpreting positional differences."],
    }
    # DataFrame float columns otherwise retain NaN after where(None).
    for row in report["groups"]:
        for key, value in row.items():
            if isinstance(value, float) and pd.isna(value):
                row[key] = None
    return table, report
