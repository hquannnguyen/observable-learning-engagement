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


def merge_fragmented_tracks(
    df: pd.DataFrame,
    max_gap_seconds: float = 1.5,
    max_center_distance: float = 80.0,
) -> pd.DataFrame:
    """Merge fragmented track IDs when a track momentarily disappears and reappears nearby.

    Looks for pairs of tracks (A, B) where track A terminates at t_end, track B starts at t_start,
    0 < (t_start - t_end) <= max_gap_seconds * 1000, and Euclidean distance between A's last
    bbox center and B's first bbox center is within max_center_distance.

    Args:
        df: Input DataFrame with pose detections (must have 'track_id', 'timestamp_ms', 'bbox_x1'..'bbox_y2').
        max_gap_seconds: Maximum time gap between track end and start (default: 1.5s).
        max_center_distance: Maximum pixel distance between bbox centers (default: 80.0px).

    Returns:
        DataFrame with merged track_id values.
    """
    if df.empty or "track_id" not in df.columns or "timestamp_ms" not in df.columns:
        return df.copy()

    bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    if not all(c in df.columns for c in bbox_cols):
        return df.copy()

    valid_df = df[df["track_id"].notna()].copy()
    if valid_df.empty:
        return df.copy()

    track_bounds = {}
    for tid, grp in valid_df.groupby("track_id"):
        grp_sorted = grp.sort_values("timestamp_ms")
        first_row = grp_sorted.iloc[0]
        last_row = grp_sorted.iloc[-1]
        c_start = (
            (float(first_row["bbox_x1"]) + float(first_row["bbox_x2"])) / 2.0,
            (float(first_row["bbox_y1"]) + float(first_row["bbox_y2"])) / 2.0,
        )
        c_end = (
            (float(last_row["bbox_x1"]) + float(last_row["bbox_x2"])) / 2.0,
            (float(last_row["bbox_y1"]) + float(last_row["bbox_y2"])) / 2.0,
        )
        track_bounds[tid] = {
            "start_ms": float(first_row["timestamp_ms"]),
            "end_ms": float(last_row["timestamp_ms"]),
            "center_start": c_start,
            "center_end": c_end,
        }

    id_map: Dict[Any, Any] = {}
    max_gap_ms = max_gap_seconds * 1000.0
    sorted_tracks = sorted(track_bounds.keys(), key=lambda t: track_bounds[t]["start_ms"])

    for i in range(len(sorted_tracks)):
        t_b = sorted_tracks[i]
        b_info = track_bounds[t_b]

        best_cand = None
        best_dist = float("inf")

        for j in range(i):
            t_a = sorted_tracks[j]
            root_a = t_a
            while root_a in id_map:
                root_a = id_map[root_a]

            a_info = track_bounds[root_a]
            gap = b_info["start_ms"] - a_info["end_ms"]

            if 0 < gap <= max_gap_ms:
                dist = (
                    (b_info["center_start"][0] - a_info["center_end"][0]) ** 2
                    + (b_info["center_start"][1] - a_info["center_end"][1]) ** 2
                ) ** 0.5
                if dist <= max_center_distance and dist < best_dist:
                    best_cand = root_a
                    best_dist = dist

        if best_cand is not None:
            id_map[t_b] = best_cand
            track_bounds[best_cand]["end_ms"] = max(track_bounds[best_cand]["end_ms"], b_info["end_ms"])
            track_bounds[best_cand]["center_end"] = b_info["center_end"]

    if not id_map:
        return df.copy()

    out = df.copy()

    def _resolve(tid: Any) -> Any:
        curr = tid
        while curr in id_map:
            curr = id_map[curr]
        return curr

    out["track_id"] = out["track_id"].map(lambda x: _resolve(x) if pd.notna(x) else x)
    logger.info("Merged %d fragmented track IDs", len(id_map))
    return out
