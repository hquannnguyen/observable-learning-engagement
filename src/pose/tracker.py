"""Tracking module for human pose tracks (ByteTrack baseline).

Provides track duration analysis, track filtering, and track continuity checks
according to configs/baseline.yaml and docs/dataset_schema.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_track_statistics(
    df: pd.DataFrame,
    min_track_seconds: float = 2.0,
) -> pd.DataFrame:
    """Compute summary statistics for each track in a poses DataFrame.

    Args:
        df: DataFrame containing pose detections with columns:
            track_id, timestamp_ms, frame_idx, visible_ratio, bbox_conf.
        min_track_seconds: Minimum duration threshold for a track to be considered valid.

    Returns:
        DataFrame indexed by track_id with columns:
            track_id, total_frames, start_ms, end_ms, duration_s,
            mean_visible_ratio, mean_bbox_conf, is_valid_duration.
    """
    if df.empty or "track_id" not in df.columns:
        return pd.DataFrame(
            columns=[
                "track_id",
                "total_frames",
                "start_ms",
                "end_ms",
                "duration_s",
                "mean_visible_ratio",
                "mean_bbox_conf",
                "is_valid_duration",
            ]
        )

    # Exclude rows without a valid track_id
    valid_df = df[df["track_id"].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame()

    stats: List[Dict[str, Any]] = []
    for track_id, group in valid_df.groupby("track_id"):
        total_frames = len(group)
        start_ms = int(group["timestamp_ms"].min())
        end_ms = int(group["timestamp_ms"].max())
        duration_s = round((end_ms - start_ms) / 1000.0, 3)

        mean_vis = (
            round(float(group["visible_ratio"].mean()), 4)
            if "visible_ratio" in group.columns
            else 0.0
        )
        mean_conf = (
            round(float(group["bbox_conf"].mean()), 4)
            if "bbox_conf" in group.columns
            else 0.0
        )

        is_valid = duration_s >= min_track_seconds

        stats.append(
            {
                "track_id": int(track_id),
                "total_frames": total_frames,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_s": duration_s,
                "mean_visible_ratio": mean_vis,
                "mean_bbox_conf": mean_conf,
                "is_valid_duration": is_valid,
            }
        )

    return pd.DataFrame(stats)


def filter_short_tracks(
    df: pd.DataFrame,
    min_track_seconds: float = 2.0,
) -> pd.DataFrame:
    """Filter out detections from tracks shorter than min_track_seconds.

    Args:
        df: DataFrame containing pose detections with 'track_id' and 'timestamp_ms'.
        min_track_seconds: Duration threshold in seconds.

    Returns:
        Filtered DataFrame keeping only detections belonging to sufficiently long tracks.
    """
    if df.empty or "track_id" not in df.columns:
        return df.copy()

    stats = compute_track_statistics(df, min_track_seconds=min_track_seconds)
    if stats.empty:
        return df.iloc[0:0].copy()

    valid_track_ids = set(stats[stats["is_valid_duration"]]["track_id"])
    filtered = df[df["track_id"].isin(valid_track_ids)].copy()

    num_removed = len(df) - len(filtered)
    logger.info(
        "Filtered short tracks (<%.1fs): kept %d/%d tracks (%d detections removed)",
        min_track_seconds,
        len(valid_track_ids),
        len(stats),
        num_removed,
    )
    return filtered
