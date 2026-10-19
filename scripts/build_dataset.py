#!/usr/bin/env python3
"""Build temporal window dataset and feature tensors from pose detections.

Implements Task T17, T18, T19:
1. Reads cleaned/tracked pose files from data/poses/.
2. Builds 4s sliding windows (40 timesteps @ 10 FPS, 2s stride).
3. Assigns labels (human annotations if available, otherwise weak rules).
4. Splits by session to prevent frame leakage.
5. Saves participation_windows.parquet and window_features.npz.

Usage:
    python scripts/build_dataset.py --poses-dir data/poses --output-dir data/features
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.labels import CANONICAL_TRAINING_LABELS
from src.preprocessing.cleaner import clean_and_normalize_poses
from src.preprocessing.weak_labels import assign_weak_labels_to_windows
from src.preprocessing.windows import build_track_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_dataset")

LABEL_TO_INT = {lb: i for i, lb in enumerate(CANONICAL_TRAINING_LABELS)}


def split_sessions(
    session_ids: List[str],
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Partition unique sessions into train/val/test without leakage."""
    sorted_sessions = sorted(list(set(session_ids)))
    n = len(sorted_sessions)

    if n == 1:
        # Single session fallback: train=1, val=1, test=1 with warning
        logger.warning("Only 1 session available. Using for all splits.")
        return sorted_sessions, sorted_sessions, sorted_sessions
    elif n == 2:
        return [sorted_sessions[0]], [sorted_sessions[1]], [sorted_sessions[1]]
    elif n <= 4:
        return sorted_sessions[:-2], [sorted_sessions[-2]], [sorted_sessions[-1]]
    else:
        # e.g. 6 sessions: 4 train, 1 val, 1 test
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        n_train = n - n_val - n_test
        return sorted_sessions[:n_train], sorted_sessions[n_train:n_train + n_val], sorted_sessions[n_train + n_val:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build window dataset for temporal modeling.")
    parser.add_argument("--poses-dir", type=str, default="data/poses", help="Directory containing pose CSV/Parquet files.")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory for features.")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory to save train/val/test splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--window-seconds", type=float, default=4.0, help="Window duration in seconds (default: 4.0s).")
    parser.add_argument("--stride-seconds", type=float, default=2.0, help="Stride duration in seconds (default: 2.0s).")
    args = parser.parse_args()

    poses_dir = Path(args.poses_dir)
    out_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(
        list(poses_dir.glob("*_poses.parquet"))
        + list(poses_dir.glob("*_poses.csv"))
        + list(poses_dir.glob("*_poses_cleaned.csv"))
        + list(poses_dir.glob("*_poses_tracked.csv"))
    )

    if not pose_files:
        logger.error("No pose files found in %s", poses_dir)
        sys.exit(1)

    logger.info("Found %d pose file(s) in %s.", len(pose_files), poses_dir)

    all_dfs = []
    seen_files = set()
    for pf in pose_files:
        # Avoid duplicate basenames
        if pf.stem in seen_files:
            continue
        seen_files.add(pf.stem)

        logger.info("Loading %s...", pf.name)
        if pf.suffix == ".parquet":
            df = pd.read_parquet(pf)
        else:
            df = pd.read_csv(pf)

        if not df.empty and "track_id" in df.columns:
            # Clean and normalize if not already done
            if "norm_kpt_00_x" not in df.columns:
                df = clean_and_normalize_poses(df, min_track_seconds=1.0)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        logger.error("No valid pose tracks extracted.")
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    session_ids = combined_df["session_id"].unique().tolist()
    train_sess, val_sess, test_sess = split_sessions(session_ids, seed=args.seed)

    logger.info("Session Splits:")
    logger.info("  Train (%d): %s", len(train_sess), train_sess)
    logger.info("  Val   (%d): %s", len(val_sess), val_sess)
    logger.info("  Test  (%d): %s", len(test_sess), test_sess)

    # Save split text files
    (splits_dir / "train_sessions.txt").write_text("\n".join(train_sess) + "\n", encoding="utf-8")
    (splits_dir / "val_sessions.txt").write_text("\n".join(val_sess) + "\n", encoding="utf-8")
    (splits_dir / "test_sessions.txt").write_text("\n".join(test_sess) + "\n", encoding="utf-8")

    split_meta = {
        "dataset_version": "v0.1",
        "split_version": "split-v0.1",
        "seed": args.seed,
        "created_at": datetime.now().isoformat(),
        "train_sessions": train_sess,
        "val_sessions": val_sess,
        "test_sessions": test_sess,
    }
    (splits_dir / "split_metadata.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    # Build windows per track
    all_window_records: List[Dict[str, Any]] = []
    all_window_tensors: List[np.ndarray] = []

    for (s_id, t_id), track_group in combined_df.groupby(["session_id", "track_id"]):
        split_name = "train" if s_id in train_sess else ("val" if s_id in val_sess else "test")

        records, tensors = build_track_windows(
            track_df=track_group,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            split=split_name,
        )

        if records and len(tensors) > 0:
            # Assign weak labels adhering to label contract
            labeled = assign_weak_labels_to_windows(records, tensors)
            all_window_records.extend(labeled)
            all_window_tensors.append(tensors)

    if not all_window_records:
        logger.error("No windows could be built (tracks may be shorter than %.1fs).", args.window_seconds)
        sys.exit(1)

    all_tensors = np.concatenate(all_window_tensors, axis=0)
    windows_df = pd.DataFrame(all_window_records)

    logger.info("Total generated windows: %d across shape %s", len(windows_df), all_tensors.shape)

    # Class balance summary
    class_counts = windows_df["participation_label"].value_counts(dropna=False).to_dict()
    logger.info("Class distribution: %s", class_counts)

    # Save windows dataframe
    out_csv = out_dir / "participation_windows.csv"
    out_parquet = out_dir / "participation_windows.parquet"
    windows_df.to_csv(out_csv, index=False)
    try:
        windows_df.to_parquet(out_parquet, index=False)
        logger.info("Saved windows metadata to: %s & %s", out_csv, out_parquet)
    except Exception as e:
        logger.warning("Could not save parquet (%s); CSV saved at %s", e, out_csv)

    # Split arrays for train/val/test
    train_mask = (windows_df["split"] == "train").to_numpy()
    val_mask = (windows_df["split"] == "val").to_numpy()
    test_mask = (windows_df["split"] == "test").to_numpy()

    # Integer labels (-1 for unlabeled / invalid)
    labels_int = np.array(
        [LABEL_TO_INT.get(lbl, -1) for lbl in windows_df["participation_label"]],
        dtype=np.int64,
    )

    tensors_npz = out_dir / "window_features.npz"
    np.savez_compressed(
        tensors_npz,
        X_train=all_tensors[train_mask],
        y_train=labels_int[train_mask],
        X_val=all_tensors[val_mask],
        y_val=labels_int[val_mask],
        X_test=all_tensors[test_mask],
        y_test=labels_int[test_mask],
        all_features=all_tensors,
        all_labels=labels_int,
    )
    logger.info("Saved feature tensors to: %s", tensors_npz)
    logger.info(
        "Splits summary: Train=%d, Val=%d, Test=%d windows",
        train_mask.sum(),
        val_mask.sum(),
        test_mask.sum(),
    )


if __name__ == "__main__":
    main()
