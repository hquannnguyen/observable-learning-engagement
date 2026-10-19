"""Pose Quality Gate module.

Evaluates pose quality according to configs/baseline.yaml:
- minimum visible ratio (default 0.60)
- minimum mean keypoint confidence (default 0.30)
- minimum track duration (default 2.0s)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml

from src.pose.tracker import compute_track_statistics

logger = logging.getLogger(__name__)


def load_quality_gate_config(
    config_path: Union[str, Path, Dict[str, Any]] = "configs/baseline.yaml",
) -> Dict[str, float]:
    """Load quality gate thresholds from YAML or dictionary.

    Returns:
        Dict with keys: min_visible_ratio, min_mean_keypoint_confidence, min_track_seconds.
    """
    if isinstance(config_path, dict):
        raw_cfg = config_path
    else:
        path = Path(config_path)
        if not path.is_file():
            raw_cfg = {}
        else:
            with path.open("r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}

    qg_cfg = raw_cfg.get("quality_gate", {})
    tr_cfg = raw_cfg.get("tracking", {})

    return {
        "min_visible_ratio": float(qg_cfg.get("min_visible_ratio", 0.60)),
        "min_mean_keypoint_confidence": float(
            qg_cfg.get("min_mean_keypoint_confidence", 0.30)
        ),
        "min_track_seconds": float(tr_cfg.get("min_track_seconds", 2.0)),
    }


def evaluate_pose_quality(
    df: pd.DataFrame,
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    min_visible_ratio: Optional[float] = None,
    min_mean_keypoint_confidence: Optional[float] = None,
    min_track_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """Evaluate quality state for each detection row.

    Adds columns:
    - quality_state: 'VALID', 'OCCLUDED', or 'INVALID'
    - quality_reason: 'ok', 'track_too_short', 'low_visible_ratio', 'low_confidence', etc.

    Args:
        df: DataFrame containing pose detections.
        config: Path to baseline config or config dict.
        min_visible_ratio: Optional override for visible ratio threshold.
        min_mean_keypoint_confidence: Optional override for confidence threshold.
        min_track_seconds: Optional override for track duration threshold.

    Returns:
        DataFrame with added quality evaluation columns.
    """
    if df.empty:
        res = df.copy()
        res["quality_state"] = []
        res["quality_reason"] = []
        return res

    cfg = load_quality_gate_config(config or "configs/baseline.yaml")
    min_vis = (
        min_visible_ratio
        if min_visible_ratio is not None
        else cfg["min_visible_ratio"]
    )
    min_conf = (
        min_mean_keypoint_confidence
        if min_mean_keypoint_confidence is not None
        else cfg["min_mean_keypoint_confidence"]
    )
    min_dur = (
        min_track_seconds
        if min_track_seconds is not None
        else cfg["min_track_seconds"]
    )

    out = df.copy()

    # Determine track duration validity
    if "track_id" in out.columns and out["track_id"].notna().any():
        track_stats = compute_track_statistics(out, min_track_seconds=min_dur)
        short_track_ids = set(
            track_stats[~track_stats["is_valid_duration"]]["track_id"]
        )
    else:
        short_track_ids = set()

    quality_states = []
    quality_reasons = []

    for _, row in out.iterrows():
        t_id = row.get("track_id")
        vis = float(row.get("visible_ratio", 0.0))
        mean_c = float(row.get("mean_keypoint_conf", 0.0))

        if pd.notna(t_id) and int(t_id) in short_track_ids:
            quality_states.append("INVALID")
            quality_reasons.append("track_too_short")
        elif mean_c < min_conf:
            quality_states.append("INVALID")
            quality_reasons.append("low_confidence")
        elif vis < min_vis:
            quality_states.append("OCCLUDED")
            quality_reasons.append("low_visible_ratio")
        else:
            quality_states.append("VALID")
            quality_reasons.append("ok")

    out["quality_state"] = quality_states
    out["quality_reason"] = quality_reasons
    return out


def filter_quality_poses(
    df: pd.DataFrame,
    allowed_states: tuple = ("VALID",),
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Filter poses keeping only rows with acceptable quality state.

    Args:
        df: Input pose DataFrame.
        allowed_states: Tuple of allowed states (default ('VALID',)).
        config: Config path or dict.

    Returns:
        Filtered DataFrame.
    """
    evaluated = evaluate_pose_quality(df, config=config)
    return evaluated[evaluated["quality_state"].isin(allowed_states)].copy()
