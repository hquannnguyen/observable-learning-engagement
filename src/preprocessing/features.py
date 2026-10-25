"""Feature extraction module for pose and motion features.

Implements Task T15:
Extracts normalized spatial coordinates, temporal velocities, geometric angles,
and posture stability indicators according to configs/baseline.yaml
(feature_schema_version: posefeat-v1).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Anatomical landmark indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

FEATURE_SCHEMA_VERSION = "posefeat-v1"
FEATURE_SCHEMA_VERSION_V2 = "posefeat-v2"
FEATURE_DIM_V1 = 77
FEATURE_DIM_V2 = 94  # 77 + 17 keypoint visibility masks


def compute_track_features(
    track_df: pd.DataFrame,
    sample_fps: float = 10.0,
    include_mask: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-frame feature vectors for a single continuous track.

    Args:
        track_df: DataFrame of pose detections for a single track, sorted by frame_idx.
        sample_fps: Frame rate for computing velocity (default 10.0 FPS).
        include_mask: If True, append 17 keypoint-validity mask channels
            (``posefeat-v2``, 94 dims). Missing keypoints still get 0.0
            coordinates for model compatibility, but the mask makes the
            imputation explicit.

    Returns:
        (feature_matrix [T, F], feature_names [F])
    """
    expected_dim = FEATURE_DIM_V2 if include_mask else FEATURE_DIM_V1
    if track_df.empty:
        return np.empty((0, expected_dim), dtype=np.float32), []

    track_df = track_df.sort_values("frame_idx").copy()
    T = len(track_df)

    # 1. Normalized keypoints (34 features)
    norm_kpt_cols = []
    for i in range(17):
        norm_kpt_cols.extend([f"norm_kpt_{i:02d}_x", f"norm_kpt_{i:02d}_y"])

    # Fallback to raw kpt centered if norm_kpt not yet computed
    if not all(col in track_df.columns for col in norm_kpt_cols):
        from src.preprocessing.cleaner import normalize_keypoints

        track_df = normalize_keypoints(track_df, method="torso")

    raw_kpt_coords = track_df[norm_kpt_cols].to_numpy(dtype=np.float32)
    # Mask is computed BEFORE imputation, so a missing joint is never presented
    # as a real (0, 0) observation without a flag.
    kpt_mask = (
        np.isfinite(raw_kpt_coords).reshape(T, 17, 2).all(axis=2).astype(np.float32)
    )
    kpt_coords = np.nan_to_num(raw_kpt_coords, nan=0.0)

    # 2. Keypoint velocities (34 features: delta / dt)
    dt = 1.0 / sample_fps
    velocities = np.zeros_like(kpt_coords)
    if T > 1:
        velocities[1:] = (kpt_coords[1:] - kpt_coords[:-1]) / dt
        velocities[0] = velocities[1]  # forward pad first step

    # 3. Geometric features (6 features)
    # 3.1 Shoulder slope (roll angle)
    ls_x = kpt_coords[:, LEFT_SHOULDER * 2]
    ls_y = kpt_coords[:, LEFT_SHOULDER * 2 + 1]
    rs_x = kpt_coords[:, RIGHT_SHOULDER * 2]
    rs_y = kpt_coords[:, RIGHT_SHOULDER * 2 + 1]
    shoulder_slope = np.arctan2(rs_y - ls_y, rs_x - ls_x + 1e-6)

    # 3.2 Head orientation proxy (2D offset from mid-shoulder to nose)
    nose_x = kpt_coords[:, NOSE * 2]
    nose_y = kpt_coords[:, NOSE * 2 + 1]
    mid_shoulder_x = (ls_x + rs_x) / 2.0
    mid_shoulder_y = (ls_y + rs_y) / 2.0
    head_proxy_x = nose_x - mid_shoulder_x
    head_proxy_y = nose_y - mid_shoulder_y

    # 3.3 Hand raise indicators (positive when wrist is higher than shoulder in image coords, y points down)
    lw_y = kpt_coords[:, LEFT_WRIST * 2 + 1]
    rw_y = kpt_coords[:, RIGHT_WRIST * 2 + 1]
    left_hand_raised = np.maximum(0.0, ls_y - lw_y)
    right_hand_raised = np.maximum(0.0, rs_y - rw_y)

    # 3.4 Head-down indicator (larger when nose drops significantly below shoulder line)
    head_down = np.maximum(0.0, nose_y - mid_shoulder_y)

    # 4. Motion energy (1 feature: overall RMS velocity)
    motion_energy = np.sqrt(np.mean(velocities**2, axis=1, keepdims=True))

    # 5. Quality indicators (2 features)
    vis_ratio = (
        track_df["visible_ratio"].to_numpy(dtype=np.float32)[:, None]
        if "visible_ratio" in track_df.columns
        else np.ones((T, 1), dtype=np.float32)
    )
    mean_conf = (
        track_df["mean_keypoint_conf"].to_numpy(dtype=np.float32)[:, None]
        if "mean_keypoint_conf" in track_df.columns
        else np.ones((T, 1), dtype=np.float32)
    )

    geom_features = np.column_stack(
        [
            shoulder_slope,
            head_proxy_x,
            head_proxy_y,
            left_hand_raised,
            right_hand_raised,
            head_down,
        ]
    )

    blocks = [
        kpt_coords,      # 34
        velocities,      # 34
        geom_features,   # 6
        motion_energy,   # 1
        vis_ratio,       # 1
        mean_conf,       # 1
    ]
    if include_mask:
        blocks.append(kpt_mask)  # 17

    feature_matrix = np.hstack(blocks).astype(np.float32)

    # Feature names list
    feature_names = (
        norm_kpt_cols
        + [f"vel_{col}" for col in norm_kpt_cols]
        + [
            "shoulder_slope",
            "head_proxy_x",
            "head_proxy_y",
            "left_hand_raised",
            "right_hand_raised",
            "head_down",
            "motion_energy",
            "visible_ratio",
            "mean_keypoint_conf",
        ]
    )
    if include_mask:
        feature_names += [f"kpt_mask_{i:02d}" for i in range(17)]

    # Enforce strictly finite (no NaN or Inf)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    return feature_matrix, feature_names
