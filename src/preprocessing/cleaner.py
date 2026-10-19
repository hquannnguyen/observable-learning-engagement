"""Keypoint cleaning, interpolation, smoothing, and normalization module.

Implements Task T14:
- Linearly interpolates short missing keypoint gaps per track (retaining quality masks).
- Applies temporal smoothing (Gaussian / moving average) to suppress jitter.
- Performs torso/body-scale invariant keypoint normalization for sitting students.
- Adheres to the contract: never encode missing points as (0, 0).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# COCO keypoint indices for anatomical reference
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


def _interpolate_bounded_series(series: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate series only for NaN gaps that do not exceed max_gap."""
    is_na = series.isna()
    if not is_na.any():
        return series.copy()

    # Label contiguous NaN runs
    groups = (~is_na).cumsum()
    counts = groups[is_na].map(groups[is_na].value_counts())

    # NaNs belonging to gaps larger than max_gap must remain NaN
    large_gap_mask = pd.Series(False, index=series.index)
    large_gap_mask[is_na] = counts > max_gap

    interpolated = series.interpolate(method="linear", limit_direction="both")
    interpolated[large_gap_mask] = np.nan
    return interpolated


def interpolate_missing_keypoints(
    df: pd.DataFrame,
    max_gap_frames: int = 3,
) -> pd.DataFrame:
    """Interpolate missing (NaN) keypoint coordinates for short temporal gaps.

    Args:
        df: Input pose DataFrame sorted by [track_id, frame_idx].
        max_gap_frames: Maximum number of consecutive missing frames to interpolate.
            Gaps longer than this remain NaN entirely.

    Returns:
        DataFrame with short gaps interpolated, retaining confidence information.
    """
    if df.empty or "track_id" not in df.columns:
        return df.copy()

    out = df.copy()

    # Identify all (x, y) column pairs
    kpt_x_cols = [f"kpt_{i:02d}_x" for i in range(17) if f"kpt_{i:02d}_x" in out.columns]
    kpt_y_cols = [f"kpt_{i:02d}_y" for i in range(17) if f"kpt_{i:02d}_y" in out.columns]

    # Process each track independently
    cleaned_groups = []
    for _, group in out.groupby("track_id", sort=False):
        group = group.sort_values("frame_idx").copy()

        for x_col, y_col in zip(kpt_x_cols, kpt_y_cols):
            group[x_col] = _interpolate_bounded_series(group[x_col], max_gap=max_gap_frames)
            group[y_col] = _interpolate_bounded_series(group[y_col], max_gap=max_gap_frames)

        cleaned_groups.append(group)

    result = pd.concat(cleaned_groups, axis=0).sort_index()
    return result


def smooth_keypoints(
    df: pd.DataFrame,
    window_size: int = 3,
    method: str = "moving_average",
) -> pd.DataFrame:
    """Apply temporal smoothing to keypoint coordinates to reduce detector jitter.

    Args:
        df: Input pose DataFrame.
        window_size: Odd integer temporal filter size (default: 3).
        method: Smoothing method: 'moving_average' or 'gaussian'.

    Returns:
        DataFrame with smoothed keypoint coordinates.
    """
    if df.empty or "track_id" not in df.columns or window_size <= 1:
        return df.copy()

    out = df.copy()
    coord_cols = [
        col
        for col in out.columns
        if (col.startswith("kpt_") and (col.endswith("_x") or col.endswith("_y")))
    ]

    smoothed_groups = []
    for _, group in out.groupby("track_id", sort=False):
        group = group.sort_values("frame_idx").copy()

        if len(group) >= window_size:
            if method == "gaussian":
                # Gaussian weights for 3-tap window: [0.25, 0.50, 0.25]
                # for 5-tap: approx binomial
                group[coord_cols] = (
                    group[coord_cols]
                    .rolling(
                        window=window_size,
                        center=True,
                        min_periods=1,
                        win_type="gaussian",
                    )
                    .mean(std=1.0)
                )
            else:
                group[coord_cols] = (
                    group[coord_cols]
                    .rolling(window=window_size, center=True, min_periods=1)
                    .mean()
                )

        smoothed_groups.append(group)

    result = pd.concat(smoothed_groups, axis=0).sort_index()
    return result


def normalize_keypoints(
    df: pd.DataFrame,
    method: str = "torso",
) -> pd.DataFrame:
    """Normalize keypoints relative to body center and scale.

    Produces columns `norm_kpt_{i:02d}_x` and `norm_kpt_{i:02d}_y` in addition
    to original coordinates.

    Args:
        df: Input pose DataFrame.
        method: Normalization reference:
            - 'torso': Centered at mid-shoulder, scaled by torso length (or fallback).
            - 'bbox': Normalized to bounding box [0, 1] range.

    Returns:
        DataFrame containing added normalized keypoint columns:
            norm_kpt_00_x .. norm_kpt_16_y.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    norm_x_cols = [f"norm_kpt_{i:02d}_x" for i in range(17)]
    norm_y_cols = [f"norm_kpt_{i:02d}_y" for i in range(17)]

    # Pre-allocate normalized columns
    for col in norm_x_cols + norm_y_cols:
        out[col] = np.nan

    for idx, row in out.iterrows():
        # Get shoulder keypoints
        ls_x = row.get("kpt_05_x")
        ls_y = row.get("kpt_05_y")
        rs_x = row.get("kpt_06_x")
        rs_y = row.get("kpt_06_y")

        # Get hip keypoints
        lh_x = row.get("kpt_11_x")
        lh_y = row.get("kpt_11_y")
        rh_x = row.get("kpt_12_x")
        rh_y = row.get("kpt_12_y")

        b_x1 = row.get("bbox_x1", 0.0)
        b_y1 = row.get("bbox_y1", 0.0)
        b_x2 = row.get("bbox_x2", 1.0)
        b_y2 = row.get("bbox_y2", 1.0)
        bbox_w = max(b_x2 - b_x1, 1.0)
        bbox_h = max(b_y2 - b_y1, 1.0)

        if method == "bbox":
            center_x = b_x1
            center_y = b_y1
            scale_x = bbox_w
            scale_y = bbox_h
        else:  # 'torso' method
            # Center: mid-shoulder if available, else bbox center
            if pd.notna(ls_x) and pd.notna(rs_x):
                center_x = (ls_x + rs_x) / 2.0
                center_y = (ls_y + rs_y) / 2.0
                shoulder_dist = np.hypot(rs_x - ls_x, rs_y - ls_y)
            else:
                center_x = (b_x1 + b_x2) / 2.0
                center_y = (b_y1 + b_y2) / 2.0
                shoulder_dist = bbox_w * 0.5

            # Scale: torso distance (mid-shoulder to mid-hip)
            has_hips = (
                pd.notna(lh_x) and pd.notna(rh_x) and pd.notna(lh_y) and pd.notna(rh_y)
            )
            if has_hips:
                mid_hip_x = (lh_x + rh_x) / 2.0
                mid_hip_y = (lh_y + rh_y) / 2.0
                torso_len = np.hypot(mid_hip_x - center_x, mid_hip_y - center_y)
                scale = max(torso_len, 20.0)
            else:
                # Classroom setting fallback: hips occluded by desk
                scale = max(shoulder_dist * 1.5, bbox_h * 0.5, 20.0)

            scale_x = scale
            scale_y = scale

        # Calculate normalized coordinates for all 17 keypoints
        for i in range(17):
            kx = row.get(f"kpt_{i:02d}_x")
            ky = row.get(f"kpt_{i:02d}_y")

            if pd.notna(kx) and pd.notna(ky):
                out.at[idx, f"norm_kpt_{i:02d}_x"] = round(
                    float((kx - center_x) / scale_x), 4
                )
                out.at[idx, f"norm_kpt_{i:02d}_y"] = round(
                    float((ky - center_y) / scale_y), 4
                )

    return out


def clean_and_normalize_poses(
    df: pd.DataFrame,
    min_track_seconds: float = 2.0,
    max_gap_frames: int = 3,
    smooth_window: int = 3,
    normalization_method: str = "torso",
    fill_remaining_missing: bool = False,
) -> pd.DataFrame:
    """End-to-end cleaning and normalization pipeline for pose detections.

    1. Filters tracks shorter than min_track_seconds.
    2. Interpolates short missing keypoint gaps (up to max_gap_frames).
    3. Smooths coordinates over time per track.
    4. Computes scale- and position-normalized keypoints.
    5. Optionally fills remaining missing points (forward/backward per track).

    Args:
        df: Raw pose detections DataFrame.
        min_track_seconds: Minimum track duration threshold.
        max_gap_frames: Max frames to interpolate.
        smooth_window: Window size for smoothing filter.
        normalization_method: 'torso' or 'bbox'.
        fill_remaining_missing: If True, forward/backward fill remaining NaNs within track.

    Returns:
        Cleaned, smoothed, and normalized poses DataFrame.
    """
    from src.pose.tracker import filter_short_tracks

    if df.empty:
        return df.copy()

    logger.info("Starting pose cleaning pipeline on %d rows...", len(df))

    # Step 1: Filter short tracks
    step1 = filter_short_tracks(df, min_track_seconds=min_track_seconds)
    if step1.empty:
        logger.warning("No tracks satisfied min_track_seconds=%.1f", min_track_seconds)
        return step1

    # Step 2: Interpolate short gaps
    step2 = interpolate_missing_keypoints(step1, max_gap_frames=max_gap_frames)

    # Step 3: Smooth keypoints temporally
    step3 = smooth_keypoints(step2, window_size=smooth_window)

    # Step 4: Normalize keypoints
    step4 = normalize_keypoints(step3, method=normalization_method)

    # Step 5: Optional remaining fill
    if fill_remaining_missing:
        coord_cols = [
            c
            for c in step4.columns
            if c.startswith("kpt_") or c.startswith("norm_kpt_")
        ]
        filled_groups = []
        for _, grp in step4.groupby("track_id", sort=False):
            grp[coord_cols] = grp[coord_cols].ffill().bfill()
            filled_groups.append(grp)
        step4 = pd.concat(filled_groups, axis=0).sort_index()

    logger.info("Pose cleaning complete. Final rows: %d", len(step4))
    return step4
