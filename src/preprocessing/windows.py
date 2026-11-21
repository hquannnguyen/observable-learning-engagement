"""Sliding window builder for temporal sequence modeling.

Implements Task T17:
Generates fixed 4-second windows (40 timesteps @ 10 FPS) with 2-second stride (20 timesteps)
according to configs/baseline.yaml and docs/dataset_schema.md Section 6.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.feature_contract import validate_temporal_features
from src.preprocessing.features import (
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V2,
    compute_track_features,
)

logger = logging.getLogger(__name__)


def build_track_windows(
    track_df: pd.DataFrame,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    sample_fps: float = 10.0,
    dataset_version: str = "v0.1",
    split_version: str = "split-v0.1",
    label_schema_version: str = "labels-v1",
    annotation_guide_version: str = "v0.2",
    split: str = "train",
    include_mask: bool = False,
    quality_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Slice a continuous pose track into sliding temporal feature windows.

    Args:
        track_df: Cleaned pose DataFrame for one track.
        window_seconds: Window duration in seconds (default 4.0).
        stride_seconds: Stride in seconds (default 2.0).
        sample_fps: Frame rate (default 10.0).
        dataset_version: Version tag.
        split_version: Split version tag.
        label_schema_version: Label schema tag.
        annotation_guide_version: Guide version tag.
        split: Split partition ('train', 'val', or 'test').
        include_mask: Append 17 keypoint visibility masks (posefeat-v2, 94 dims).
        quality_thresholds: Optional dict with ``min_visible_ratio`` and
            ``min_mean_keypoint_confidence``; defaults come from baseline.yaml.

    Returns:
        (list of window metadata dicts, 3D tensor of shape [N_windows, 40, F])
    """
    expected_dim = FEATURE_DIM_V2 if include_mask else FEATURE_DIM_V1
    schema_version = (
        FEATURE_SCHEMA_VERSION_V2 if include_mask else FEATURE_SCHEMA_VERSION
    )
    min_vis = float((quality_thresholds or {}).get("min_visible_ratio", 0.60))
    min_conf = float(
        (quality_thresholds or {}).get("min_mean_keypoint_confidence", 0.30)
    )
    upper_min = (quality_thresholds or {}).get("upper_body_min_visible_ratio", 0.72) if (
        quality_thresholds or {}
    ).get("allow_upper_body_visibility") else None

    W = int(round(window_seconds * sample_fps))
    S = int(round(stride_seconds * sample_fps))
    if W <= 0 or S <= 0 or sample_fps <= 0:
        raise ValueError("Window, stride and sample_fps must be positive")
    if track_df.empty:
        return [], np.empty((0, W, expected_dim), dtype=np.float32)
    for key in ("session_id", "camera_id", "track_id"):
        if track_df[key].nunique(dropna=False) != 1:
            raise ValueError(f"A window track must have exactly one {key}")

    track_df = track_df.sort_values("frame_idx").copy()
    features, _ = compute_track_features(
        track_df, sample_fps=sample_fps, include_mask=include_mask
    )

    W = int(round(window_seconds * sample_fps))  # 40 steps
    S = int(round(stride_seconds * sample_fps))  # 20 steps
    total_steps = len(features)

    if total_steps < W:
        return [], np.empty((0, W, features.shape[1]), dtype=np.float32)

    window_records: List[Dict[str, Any]] = []
    window_tensors: List[np.ndarray] = []

    session_id = str(track_df["session_id"].iloc[0])
    camera_id = str(track_df["camera_id"].iloc[0])
    track_id = int(track_df["track_id"].iloc[0])
    for window_idx, start_idx in enumerate(range(0, total_steps - W + 1, S)):
        end_idx = start_idx + W
        w_tensor = features[start_idx:end_idx]

        # Enforce contract
        validated = validate_temporal_features(w_tensor, expected_timesteps=W)

        w_rows = track_df.iloc[start_idx:end_idx]
        # Never compress a lost track interval into a nominal four-second window.
        if (w_rows["frame_idx"].diff().dropna() != 1).any():
            continue
        gaps = w_rows["timestamp_ms"].diff().dropna()
        if ((gaps <= 0) | (gaps > 1.5 * 1000.0 / sample_fps)).any():
            continue
        phases = w_rows.get("class_phase", pd.Series("UNKNOWN", index=w_rows.index)).fillna("UNKNOWN").unique()
        class_phase = str(phases[0]) if len(phases) == 1 else "TRANSITION"
        zone_values = w_rows["zone_id"].dropna().mode() if "zone_id" in w_rows else pd.Series(dtype=str)
        zone_id = zone_values.iloc[0] if len(zone_values) else None
        t_start_ms = int(w_rows["timestamp_ms"].min())
        t_end_ms = int(w_rows["timestamp_ms"].max())

        mean_vis = (
            float(w_rows["visible_ratio"].mean())
            if "visible_ratio" in w_rows.columns
            else 1.0
        )
        mean_conf = (
            float(w_rows["mean_keypoint_conf"].mean())
            if "mean_keypoint_conf" in w_rows.columns
            else 1.0
        )

        upper_visible = (
            float(w_rows["upper_body_visible_ratio"].mean())
            if "upper_body_visible_ratio" in w_rows else None
        )
        if mean_vis < min_vis and not (
            upper_min is not None and upper_visible is not None and upper_visible >= float(upper_min)
        ):
            quality_state = "INVALID"
            quality_reason = "LOW_VISIBILITY"
        elif mean_conf < min_conf:
            quality_state = "INVALID"
            quality_reason = "LOW_KEYPOINT_CONF"
        else:
            quality_state = "VALID"
            quality_reason = None
        if quality_state == "VALID" and class_phase == "TRANSITION":
            quality_state, quality_reason = "TRANSITION", "PHASE_TRANSITION"

        win_id = f"{session_id}_{camera_id}_trk{track_id:03d}_w{window_idx:04d}"

        record: Dict[str, Any] = {
            "window_id": win_id,
            "session_id": session_id,
            "camera_id": camera_id,
            "track_id": track_id,
            "t_start_ms": t_start_ms,
            "t_end_ms": t_end_ms,
            "class_phase": class_phase,
            "declared_class_phase": ";".join(sorted(w_rows["declared_class_phase"].dropna().astype(str).unique())) if "declared_class_phase" in w_rows else class_phase,
            "phase_source": ";".join(sorted(w_rows["phase_source"].unique())) if "phase_source" in w_rows else "POSE_COLUMN",
            "phase_verified": bool(w_rows["phase_verified"].all()) if "phase_verified" in w_rows else False,
            "zone_id": zone_id,
            "feature_schema_version": schema_version,
            "dataset_version": dataset_version,
            "split_version": split_version,
            "label_schema_version": label_schema_version,
            "quality_state": quality_state,
            "quality_reason": quality_reason,
            "mean_visible_ratio": round(mean_vis, 4),
            "mean_upper_body_visible_ratio": round(upper_visible, 4) if upper_visible is not None else None,
            "mean_pose_confidence": round(mean_conf, 4),
            "max_interpolated_gap_frames": int(w_rows["max_interpolated_gap_frames"].max()) if "max_interpolated_gap_frames" in w_rows else 0,
            "participation_label": None,
            "label_source": None,
            "label_rule_version": None,
            "annotation_guide_version": annotation_guide_version,
            "split": split,
        }

        window_records.append(record)
        window_tensors.append(validated)

    tensor_stack = np.stack(window_tensors, axis=0) if window_tensors else np.empty((0, W, features.shape[1]), dtype=np.float32)
    return window_records, tensor_stack
