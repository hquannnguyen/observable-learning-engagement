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
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.labels import CANONICAL_TRAINING_LABELS
from src.preprocessing.cleaner import clean_and_normalize_poses
from src.preprocessing.context import attach_session_context
from src.preprocessing.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V2,
)
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
    """Partition unique sessions into disjoint train/val/test groups.

    This is only a fallback for a brand-new dataset. Once
    ``data/splits/{train,val,test}_sessions.txt`` exist, the builder must read
    them instead of calling this function (see ``load_frozen_splits``).
    """
    if seed != 42:
        logger.warning("Fallback split is deterministic; seed=%s has no effect.", seed)

    sorted_sessions = sorted(set(session_ids))
    n = len(sorted_sessions)
    if n < 3:
        raise ValueError(
            "Need at least 3 sessions to create disjoint train/val/test splits; "
            f"got {n}: {sorted_sessions}"
        )

    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = 1
        n_test = n - 2

    return (
        sorted_sessions[:n_train],
        sorted_sessions[n_train:n_train + n_val],
        sorted_sessions[n_train + n_val:],
    )


def read_session_file(path: Path) -> List[str]:
    """Read a split file: one session id per line, blank lines/comments ignored."""
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    sessions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not sessions:
        raise ValueError(f"Split file is empty: {path}")
    return sessions


def load_frozen_splits(
    splits_dir: Path,
) -> Optional[Tuple[List[str], List[str], List[str], Dict[str, str]]]:
    """Load an existing frozen split, or return None when it is incomplete.

    Returns:
        (train, val, test, sha256_by_split) or None.
    """
    names = ("train", "val", "test")
    paths = {name: splits_dir / f"{name}_sessions.txt" for name in names}
    present = [name for name, path in paths.items() if path.is_file()]
    if not present:
        return None
    if len(present) != 3:
        raise ValueError(
            "Incomplete frozen split: found "
            + ", ".join(sorted(present))
            + "; expected all of train/val/test. Delete or complete the split files."
        )

    splits = {name: read_session_file(paths[name]) for name in names}
    hashes = {
        name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
        for name in names
    }
    return splits["train"], splits["val"], splits["test"], hashes


def validate_splits(
    train: List[str],
    val: List[str],
    test: List[str],
    all_sessions: List[str],
) -> None:
    """Fail loudly on overlap, duplicate membership, or unknown sessions."""
    all_existing = set(all_sessions)
    train_set, val_set, test_set = set(train), set(val), set(test)

    for name, values in (("train", train), ("val", val), ("test", test)):
        if len(values) != len(set(values)):
            raise ValueError(f"Split '{name}' contains duplicate sessions: {values}")

    overlaps = {
        "train/val": train_set & val_set,
        "train/test": train_set & test_set,
        "val/test": val_set & test_set,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"Session leakage between splits: {bad}")

    unknown = sorted((train_set | val_set | test_set) - all_existing)
    if unknown:
        raise ValueError(f"Split files reference sessions with no pose data: {unknown}")

    missing = sorted(all_existing - (train_set | val_set | test_set))
    if missing:
        raise ValueError(
            "Pose data contains sessions missing from the frozen split: "
            f"{missing}. Update the split files deliberately, do not auto-assign."
        )



def load_human_annotations(human_dir: Path) -> pd.DataFrame:
    """Load real human labels from a directory of annotator CSVs.

    Only rows with a canonical label and ``label_source`` in
    ``HUMAN``/``ADJUDICATED`` are used. ``ADJUDICATED`` wins over ``HUMAN`` for
    the same ``window_id``. This function never invents labels.
    """
    if not human_dir.is_dir():
        return pd.DataFrame(
            columns=["window_id", "participation_label", "label_source", "label_rule_version"]
        )

    frames: List[pd.DataFrame] = []
    for path in sorted(human_dir.glob("*.csv")):
        if path.name == "annotation_template.csv":
            continue
        df = pd.read_csv(path)
        if "window_id" not in df.columns or "participation_label" not in df.columns:
            logger.warning("Skipping %s: missing window_id/participation_label", path.name)
            continue
        if "label_source" not in df.columns:
            logger.warning("Skipping %s: missing label_source", path.name)
            continue
        keep = df[
            df["participation_label"].isin(CANONICAL_TRAINING_LABELS)
            & df["label_source"].isin(["HUMAN", "ADJUDICATED"])
        ].copy()
        if keep.empty:
            continue
        if "label_rule_version" not in keep.columns:
            keep["label_rule_version"] = ""
        frames.append(
            keep[["window_id", "participation_label", "label_source", "label_rule_version"]]
        )
        logger.info("Loaded %d human labels from %s", len(keep), path.name)

    if not frames:
        return pd.DataFrame(
            columns=["window_id", "participation_label", "label_source", "label_rule_version"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["_priority"] = combined["label_source"].map({"ADJUDICATED": 0, "HUMAN": 1}).fillna(2)
    combined = (
        combined.sort_values("_priority")
        .drop_duplicates("window_id", keep="first")
        .drop(columns="_priority")
    )
    return combined


def apply_human_labels(
    windows_df: pd.DataFrame,
    human_df: pd.DataFrame,
    mode: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Override/augment weak labels with human labels for ``human``/``hybrid`` modes."""
    out = windows_df.copy()
    stats = {"human_matched": 0, "human_unmatched": 0, "dropped_unlabelled": 0}

    if human_df.empty:
        if mode == "human":
            raise ValueError(
                "labels=human but no HUMAN/ADJUDICATED rows were found. "
                "Run scripts/sample_annotation_subset.py and collect real labels first."
            )
        logger.warning("labels=hybrid but no human labels found; keeping weak labels only.")
        return out, stats

    label_map = human_df.set_index("window_id")
    matched = out["window_id"].isin(label_map.index)
    stats["human_matched"] = int(matched.sum())
    stats["human_unmatched"] = int((~matched).sum())

    for col in ("participation_label", "label_source", "label_rule_version"):
        out.loc[matched, col] = out.loc[matched, "window_id"].map(label_map[col])

    if mode == "human":
        before = len(out)
        out = out[matched].reset_index(drop=True)
        stats["dropped_unlabelled"] = before - len(out)
        if out.empty:
            raise ValueError("labels=human produced zero labelled windows.")

    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build window dataset for temporal modeling.")
    parser.add_argument("--poses-dir", type=str, default="data/poses", help="Directory containing pose CSV/Parquet files.")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory for features.")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory to save train/val/test splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--allow-unverified-phases", action="store_true", help="Exploratory weak labels only; legacy generated phase metadata is not verified.")
    parser.add_argument(
        "--labels",
        choices=("weak", "human", "hybrid"),
        default="weak",
        help="weak = heuristic labels only; human = real human labels only; hybrid = human where available.",
    )
    parser.add_argument(
        "--human-annotations-dir",
        type=str,
        default="data/annotations/human",
        help="Directory containing real annotator CSVs (used by --labels human|hybrid).",
    )
    parser.add_argument(
        "--feature-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 = 77 dims (legacy); v2 = 94 dims with explicit keypoint masks.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Config used for quality thresholds (min_visible_ratio, min_mean_keypoint_confidence).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    quality_thresholds = cfg.get("quality_gate", {})
    sample_fps = args.sample_fps or float(cfg.get("video", {}).get("sample_fps", 10))
    window_seconds = args.window_seconds or float(cfg.get("sequence", {}).get("window_seconds", 4))
    stride_seconds = args.stride_seconds or float(cfg.get("sequence", {}).get("stride_seconds", 2))
    dataset_version = args.dataset_version or cfg.get("data", {}).get("dataset_version", "v0.1")
    split_version = cfg.get("data", {}).get("split_version", "split-v0.1")

    human_stats: Optional[Dict[str, int]] = None
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
                df = pd.concat([clean_and_normalize_poses(g, min_track_seconds=float(cfg.get("tracking", {}).get("min_track_seconds", 2))) for _, g in df.groupby(["session_id", "camera_id"])], ignore_index=True)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        logger.error("No valid pose tracks extracted.")
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    keys = ["session_id", "camera_id", "track_id", "frame_idx"]
    if combined_df.duplicated(keys).any():
        raise ValueError("Duplicate pose observations; keep one canonical pose file per recording/camera")
    combined_df, context_audit = attach_session_context(combined_df, Path(args.raw_dir), allow_unverified=args.allow_unverified_phases)
    session_ids = combined_df["session_id"].unique().tolist()

    frozen = load_frozen_splits(splits_dir)
    if frozen is None:
        logger.warning(
            "No frozen split found in %s; creating a deterministic fallback split. "
            "Freeze it before official experiments.",
            splits_dir,
        )
        train_sess, val_sess, test_sess = split_sessions(session_ids, seed=args.seed)
        (splits_dir / "train_sessions.txt").write_text("\n".join(train_sess) + "\n", encoding="utf-8")
        (splits_dir / "val_sessions.txt").write_text("\n".join(val_sess) + "\n", encoding="utf-8")
        (splits_dir / "test_sessions.txt").write_text("\n".join(test_sess) + "\n", encoding="utf-8")
        split_hashes = {
            name: hashlib.sha256((splits_dir / f"{name}_sessions.txt").read_bytes()).hexdigest()
            for name in ("train", "val", "test")
        }
        split_source = "auto_generated"
    else:
        train_sess, val_sess, test_sess, split_hashes = frozen
        split_source = "frozen_files"

    validate_splits(train_sess, val_sess, test_sess, session_ids)

    logger.info("Session Splits (%s):", split_source)
    logger.info("  Train (%d): %s", len(train_sess), train_sess)
    logger.info("  Val   (%d): %s", len(val_sess), val_sess)
    logger.info("  Test  (%d): %s", len(test_sess), test_sess)

    split_meta = {
        "dataset_version": dataset_version,
        "split_version": split_version,
        "seed": args.seed,
        "created_at": datetime.now().isoformat(),
        "split_source": split_source,
        "split_files_sha256": split_hashes,
        "label_mode": args.labels,
        "train_sessions": train_sess,
        "val_sessions": val_sess,
        "test_sessions": test_sess,
    }

    # Build windows per track
    all_window_records: List[Dict[str, Any]] = []
    all_window_tensors: List[np.ndarray] = []

    for (s_id, camera_id, t_id), track_group in combined_df.groupby(["session_id", "camera_id", "track_id"]):
        split_name = "train" if s_id in train_sess else ("val" if s_id in val_sess else "test")

        records, tensors = build_track_windows(
            track_df=track_group,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            sample_fps=sample_fps,
            dataset_version=dataset_version,
            split_version=split_version,
            split=split_name,
            include_mask=(args.feature_version == "v2"),
            quality_thresholds=quality_thresholds,
        )

        if records and len(tensors) > 0:
            all_window_records.extend(records)
            all_window_tensors.append(tensors)

    if not all_window_records:
        logger.error("No windows could be built (tracks may be shorter than %.1fs).", window_seconds)
        sys.exit(1)

    all_tensors = np.concatenate(all_window_tensors, axis=0)

    if args.labels == "human":
        records = all_window_records
    else:
        records = assign_weak_labels_to_windows(all_window_records, all_tensors)

    windows_df = pd.DataFrame(records)
    windows_df["_row_idx"] = np.arange(len(windows_df))

    if args.labels in ("human", "hybrid"):
        human_df = load_human_annotations(Path(args.human_annotations_dir))
        windows_df, human_stats = apply_human_labels(windows_df, human_df, args.labels)
        all_tensors = all_tensors[windows_df["_row_idx"].to_numpy()]
        logger.info("Human label join: %s", human_stats)
        split_meta["label_mode"] = args.labels
        split_meta["human_label_stats"] = human_stats

    windows_df = windows_df.drop(columns="_row_idx")

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
    window_ids = windows_df["window_id"].astype(str).to_numpy(dtype=str)

    if train_mask.sum() + val_mask.sum() + test_mask.sum() != len(windows_df):
        raise AssertionError("Every window must belong to exactly one split.")

    tensors_npz = out_dir / "window_features.npz"
    feature_schema_version = (
        FEATURE_SCHEMA_VERSION_V2
        if args.feature_version == "v2"
        else FEATURE_SCHEMA_VERSION
    )
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
        window_ids=window_ids,
        window_ids_train=window_ids[train_mask],
        window_ids_val=window_ids[val_mask],
        window_ids_test=window_ids[test_mask],
        feature_schema_version=np.array(feature_schema_version),
        dataset_version=np.array(dataset_version),
        split_version=np.array(split_version),
        sample_fps=np.array(sample_fps),
    )
    logger.info("Saved feature tensors to: %s", tensors_npz)

    # Split report: auditable counts, no reliance on stale prose reports.
    split_report: Dict[str, Any] = {
        "dataset_version": dataset_version,
        "split_version": split_version,
        "sample_fps": sample_fps,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "allow_unverified_phases": args.allow_unverified_phases,
        "context_audit": context_audit,
        "phase_counts": windows_df["class_phase"].value_counts().to_dict(),
        "declared_phase_counts": windows_df["declared_class_phase"].value_counts().to_dict(),
        "split_source": split_source,
        "split_files_sha256": split_hashes,
        "label_mode": args.labels,
        "human_label_stats": human_stats,
        "feature_schema_version": feature_schema_version,
        "total_windows": int(len(windows_df)),
        "total_tensor_shape": list(all_tensors.shape),
        "splits": {},
    }
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        part = windows_df.loc[mask]
        split_report["splits"][name] = {
            "sessions": sorted(part["session_id"].unique().tolist()),
            "windows": int(len(part)),
            "class_counts": {
                str(k): int(v)
                for k, v in part["participation_label"].value_counts(dropna=False).items()
            },
            "label_source_counts": {
                str(k): int(v)
                for k, v in part["label_source"].value_counts(dropna=False).items()
            },
            "quality_state_counts": {
                str(k): int(v)
                for k, v in part["quality_state"].value_counts(dropna=False).items()
            },
        }
    split_report_path = out_dir / "split_report.json"
    split_report_path.write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    logger.info("Saved split report to: %s", split_report_path)

    split_meta["label_mode"] = args.labels
    split_meta["human_label_stats"] = human_stats
    (out_dir / "build_metadata.json").write_text(
        json.dumps(split_meta, indent=2), encoding="utf-8"
    )
    logger.info(
        "Splits summary: Train=%d, Val=%d, Test=%d windows",
        train_mask.sum(),
        val_mask.sum(),
        test_mask.sum(),
    )


if __name__ == "__main__":
    main()
