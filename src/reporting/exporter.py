"""Export anonymized reporting artifacts from per-window predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.reporting.aggregation import (
    aggregate_timeline,
    aggregate_zones,
    build_summary,
)


REQUIRED_PREDICTION_COLUMNS = {
    "window_id",
    "session_id",
    "gated_label",
    "mean_visible_ratio",
    "mean_pose_confidence",
}


def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")


def _markdown_report(
    summary: Dict[str, Any],
    timeline: pd.DataFrame,
    zones: pd.DataFrame,
) -> str:
    lines = [
        "# Observable Learning Participation — Summary",
        "",
        "> This report describes observable pose/motion statistics only. It does not",
        "> measure emotion, motivation, intention or internal attention.",
        "",
        f"- Experiment: `{summary.get('experiment_id')}`",
        f"- Metric scope: `{summary.get('metric_scope')}`",
        f"- Windows: {summary['total_windows']} total, "
        f"{summary['accepted_windows']} accepted, {summary['uncertain_windows']} uncertain",
        f"- Coverage: {summary['coverage']}",
        f"- Sessions: {', '.join(summary.get('sessions', [])) or 'N/A'}",
        "",
        "## Class distribution",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    counts = summary.get("class_counts", {})
    lines.append(f"| HIGH | {counts.get('n_high', 0)} |")
    lines.append(f"| MODERATE | {counts.get('n_moderate', 0)} |")
    lines.append(f"| LOW | {counts.get('n_low', 0)} |")
    lines.append(f"| UNCERTAIN | {counts.get('n_uncertain', 0)} |")

    lines += [
        "",
        "## Timeline",
        "",
        "| Session | Minute | Windows | HIGH | MODERATE | LOW | UNCERTAIN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timeline.itertuples(index=False):
        lines.append(
            f"| {row.session_id} | {row.minute_index} | {row.n_windows} | {row.n_high} | "
            f"{row.n_moderate} | {row.n_low} | {row.n_uncertain} |"
        )

    if zones.empty:
        lines += [
            "",
            "## Zone summary",
            "",
            "N/A — no real zone assignment in this dataset. Do not interpret a heatmap.",
        ]
    else:
        lines += [
            "",
            "## Zone summary",
            "",
            "| Zone | N | HIGH | MODERATE | LOW | UNCERTAIN |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in zones.itertuples(index=False):
            lines.append(
                f"| {row.zone_id} | {row.n_windows} | {row.n_high} | "
                f"{row.n_moderate} | {row.n_low} | {row.n_uncertain} |"
            )

    lines += ["", "## Limitations", ""]
    for item in summary.get("limitations", []):
        lines.append(f"- {item}")
    lines.append(f"- {summary.get('metric_warning', '')}")
    lines.append("")
    return "\n".join(lines)


def export_bundle(
    predictions: pd.DataFrame,
    out_dir: str | Path,
    *,
    experiment_id: Optional[str] = None,
    metric_scope: str = "weak_rule_labels",
    bucket_seconds: int = 60,
) -> Dict[str, str]:
    """Write results.csv, timeline.csv, zone_summary.csv, summary.json, summary_report.md."""
    _validate_predictions(predictions)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    timeline = aggregate_timeline(predictions, bucket_seconds=bucket_seconds)
    zones = aggregate_zones(predictions)
    summary = build_summary(
        predictions,
        experiment_id=experiment_id,
        metric_scope=metric_scope,
    )

    paths = {
        "results": out / "results.csv",
        "timeline": out / "timeline.csv",
        "zone_summary": out / "zone_summary.csv",
        "summary": out / "summary.json",
        "summary_report": out / "summary_report.md",
    }

    predictions.to_csv(paths["results"], index=False)
    timeline.to_csv(paths["timeline"], index=False)
    zones.to_csv(paths["zone_summary"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary_report"].write_text(
        _markdown_report(summary, timeline, zones), encoding="utf-8"
    )

    return {key: str(path) for key, path in paths.items()}
