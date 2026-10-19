#!/usr/bin/env python3
"""CLI script to clean, smooth, and normalize extracted pose keypoints.

Usage examples:
    # Clean a single extracted pose CSV:
    python scripts/clean_poses.py --input data/poses/class_1_cam_A_poses_tracked.csv \
                                 --output data/poses/class_1_cam_A_poses_cleaned.csv

    # Run full cleaning with quality evaluation and report:
    python scripts/clean_poses.py --input data/poses/class_1_cam_A_poses_tracked.csv \
                                 --output data/poses/class_1_cam_A_poses_cleaned.csv \
                                 --report reports/pose_quality_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.pose.tracker import compute_track_statistics
from src.preprocessing.cleaner import clean_and_normalize_poses
from src.preprocessing.quality_gate import (
    evaluate_pose_quality,
    load_quality_gate_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clean_poses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, interpolate, smooth, and normalize pose keypoints per track."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw/tracked pose CSV or Parquet file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output cleaned pose file (CSV or Parquet).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline config (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--min-track-seconds",
        type=float,
        default=None,
        help="Minimum duration threshold for tracks (default: from baseline.yaml, typically 2.0s).",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=3,
        help="Maximum gap size (frames) for linear interpolation (default: 3).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Window size for temporal smoothing (default: 3).",
    )
    parser.add_argument(
        "--norm-method",
        choices=["torso", "bbox"],
        default="torso",
        help="Coordinate normalization method (default: torso).",
    )
    parser.add_argument(
        "--fill-remaining",
        action="store_true",
        help="Forward/backward fill remaining missing coordinates per track.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional path to save track quality summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_file():
        logger.error("Input file does not exist: %s", in_path)
        sys.exit(1)

    cfg = load_quality_gate_config(args.config)
    eff_min_dur = (
        args.min_track_seconds
        if args.min_track_seconds is not None
        else cfg["min_track_seconds"]
    )

    logger.info("Loading pose detections from %s...", in_path)
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    logger.info("Loaded %d rows with %d columns.", len(df), len(df.columns))

    # Evaluate track statistics
    stats = compute_track_statistics(df, min_track_seconds=eff_min_dur)
    logger.info(
        "Detected %d tracks total (%d valid >= %.1fs, %d short).",
        len(stats),
        int(stats["is_valid_duration"].sum()) if not stats.empty else 0,
        eff_min_dur,
        int((~stats["is_valid_duration"]).sum()) if not stats.empty else 0,
    )

    if args.report and not stats.empty:
        rep_path = Path(args.report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(rep_path, index=False)
        logger.info("Saved track quality summary report to: %s", rep_path)

    # Run cleaning and normalization pipeline
    cleaned_df = clean_and_normalize_poses(
        df=df,
        min_track_seconds=eff_min_dur,
        max_gap_frames=args.max_gap_frames,
        smooth_window=args.smooth_window,
        normalization_method=args.norm_method,
        fill_remaining_missing=args.fill_remaining,
    )

    # Evaluate quality states
    final_df = evaluate_pose_quality(
        cleaned_df,
        config=args.config,
        min_track_seconds=eff_min_dur,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        final_df.to_parquet(out_path, index=False)
    else:
        final_df.to_csv(out_path, index=False)

    logger.info("Successfully cleaned poses saved to: %s (%d rows)", out_path, len(final_df))


if __name__ == "__main__":
    main()
