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
    """Vectorized body-center/scale normalization, retaining missing pairs.

    Uses the same torso/shoulder/bbox fallbacks as the original row-wise code.
    Neither the raw coordinates nor confidence columns are changed.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    n = len(out)
    columns = [f"kpt_{i:02d}_{axis}" for i in range(17) for axis in ("x", "y")]
    coords = out.reindex(columns=columns).to_numpy(dtype=float).reshape(n, 17, 2)
    def field(name, default):
        return out[name].to_numpy(dtype=float) if name in out else np.full(n, default, dtype=float)
    x1, y1 = field("bbox_x1", 0), field("bbox_y1", 0)
    x2, y2 = field("bbox_x2", 1), field("bbox_y2", 1)
    width, height = np.maximum(x2 - x1, 1), np.maximum(y2 - y1, 1)
    if method == "bbox":
        center = np.column_stack([x1, y1])
        scale = np.column_stack([width, height])
    else:
        ls, rs, lh, rh = coords[:, 5], coords[:, 6], coords[:, 11], coords[:, 12]
        shoulders = ~np.isnan(ls[:, 0]) & ~np.isnan(rs[:, 0])
        center = np.where(shoulders[:, None], (ls + rs) / 2,
                          np.column_stack([(x1 + x2) / 2, (y1 + y2) / 2]))
        shoulder_distance = np.where(shoulders, np.hypot(*(rs - ls).T), width * 0.5)
        hips = ~np.isnan(lh).any(axis=1) & ~np.isnan(rh).any(axis=1)
        torso = np.hypot(*((lh + rh) / 2 - center).T)
        body_scale = np.where(hips, np.maximum(torso, 20),
                              np.maximum(np.maximum(shoulder_distance * 1.5, height * 0.5), 20))
        scale = np.column_stack([body_scale, body_scale])
    normalized = (coords - center[:, None, :]) / scale[:, None, :]
    normalized[np.isnan(coords).any(axis=2)] = np.nan
    normalized = np.round(normalized, 4)
    out[[f"norm_kpt_{i:02d}_x" for i in range(17)]] = normalized[:, :, 0]
    out[[f"norm_kpt_{i:02d}_y" for i in range(17)]] = normalized[:, :, 1]
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
