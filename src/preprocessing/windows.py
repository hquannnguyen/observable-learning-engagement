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
from src.preprocessing.features import FEATURE_SCHEMA_VERSION, compute_track_features

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

    Returns:
        (list of window metadata dicts, 3D tensor of shape [N_windows, 40, F])
    """
    if track_df.empty:
        return [], np.empty((0, 40, 77), dtype=np.float32)

    track_df = track_df.sort_values("frame_idx").copy()
    features, _ = compute_track_features(track_df, sample_fps=sample_fps)

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
    zone_id = track_df["zone_id"].iloc[0] if "zone_id" in track_df.columns else None

    # Determine class_phase from track or metadata
    class_phase = (
        track_df["class_phase"].iloc[0]
        if "class_phase" in track_df.columns and pd.notna(track_df["class_phase"].iloc[0])
        else "lecture"
    )

    window_idx = 0
    for start_idx in range(0, total_steps - W + 1, S):
        end_idx = start_idx + W
        w_tensor = features[start_idx:end_idx]

        # Enforce contract
        validated = validate_temporal_features(w_tensor, expected_timesteps=W)

        w_rows = track_df.iloc[start_idx:end_idx]
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

        quality_state = "VALID" if mean_vis >= 0.60 and mean_conf >= 0.30 else "INVALID"

        win_id = f"{session_id}_{camera_id}_trk{track_id:03d}_w{window_idx:04d}"

        record: Dict[str, Any] = {
            "window_id": win_id,
            "session_id": session_id,
            "camera_id": camera_id,
            "track_id": track_id,
            "t_start_ms": t_start_ms,
            "t_end_ms": t_end_ms,
            "class_phase": class_phase,
            "zone_id": zone_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_version": dataset_version,
            "split_version": split_version,
            "label_schema_version": label_schema_version,
            "quality_state": quality_state,
            "mean_visible_ratio": round(mean_vis, 4),
            "mean_pose_confidence": round(mean_conf, 4),
            "participation_label": None,
            "label_source": None,
            "label_rule_version": None,
            "annotation_guide_version": annotation_guide_version,
            "split": split,
        }

        window_records.append(record)
        window_tensors.append(validated)
        window_idx += 1

    tensor_stack = np.stack(window_tensors, axis=0) if window_tensors else np.empty((0, W, features.shape[1]), dtype=np.float32)
    return window_records, tensor_stack
