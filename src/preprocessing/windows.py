"""Sliding-window dataset builder conforming to docs/dataset_schema.md Section 6.

Implements Task T17 (Sliding-window builder).
Produces window metadata CSV and model-ready [N, T=40, F=64] numpy feature archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.contracts.labels import LABEL_TO_INDEX
from src.preprocessing.features import FEATURE_SCHEMA_VERSION, build_window_features


def build_window_dataset(
    poses_path: Path,
    annotations_path: Path,
    output_dir: Path,
    session_id: str = "session_001",
    target_timesteps: int = 40,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract temporal feature tensors [N, 40, 64] and metadata for each window."""
    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    print(f"[INFO] Loading poses from {poses_path} ...")
    df_poses = pd.read_parquet(poses_path) if poses_path.suffix == ".parquet" else pd.read_csv(poses_path)

    print(f"[INFO] Loading annotations from {annotations_path} ...")
    df_annos = pd.read_csv(annotations_path)

    # Keypoint column names
    kpt_x_cols = [f"kpt_{i:02d}_x" for i in range(17)]
    kpt_y_cols = [f"kpt_{i:02d}_y" for i in range(17)]
    kpt_conf_cols = [f"kpt_{i:02d}_conf" for i in range(17)]

    # Pre-index poses by (track_id, timestamp_ms)
    df_poses = df_poses.sort_values(["track_id", "timestamp_ms"])

    feature_tensors = []
    labels_list = []
    metadata_rows = []

    print(f"[INFO] Building features for {len(df_annos)} windows...")

    for idx, row in df_annos.iterrows():
        track_id = int(row["track_id"])
        t_start = int(row["t_start_ms"])
        t_end = int(row["t_end_ms"])

        # Filter poses for this window
        mask = (
            (df_poses["track_id"] == track_id)
            & (df_poses["timestamp_ms"] >= t_start)
            & (df_poses["timestamp_ms"] <= t_end)
        )
        w_poses = df_poses[mask]

        if len(w_poses) < 3:
            # Skip or pad if too few poses
            continue

        # Extract (T_obs, 17, 3) keypoints sequence
        kpts_seq = np.zeros((len(w_poses), 17, 3), dtype=np.float32)
        for i in range(17):
            kpts_seq[:, i, 0] = w_poses[kpt_x_cols[i]].values
            kpts_seq[:, i, 1] = w_poses[kpt_y_cols[i]].values
            kpts_seq[:, i, 2] = w_poses[kpt_conf_cols[i]].values

        # Build [40, 64] tensor
        feat_tensor = build_window_features(kpts_seq, target_timesteps=target_timesteps)

        window_id = f"{session_id}_trk{track_id:03d}_{t_start:07d}_{t_end:07d}"
        part_label = str(row.get("participation_label", "")).strip()
        state = str(row.get("annotation_state", "VALID")).strip()

        label_idx = LABEL_TO_INDEX.get(part_label, -1) if state == "VALID" else -1

        feature_tensors.append(feat_tensor)
        labels_list.append(label_idx)

        metadata_rows.append({
            "window_id": window_id,
            "session_id": session_id,
            "camera_id": row.get("camera_id", "cam_A"),
            "track_id": track_id,
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "class_phase": row.get("class_phase", "lecture"),
            "zone_id": row.get("zone_id", "M-C"),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_version": "v0.1",
            "split_version": "split-v0.1",
            "label_schema_version": "labels-v1",
            "quality_state": state,
            "mean_visible_ratio": round(float(w_poses["visible_ratio"].mean()), 4),
            "mean_pose_confidence": round(float(w_poses["mean_keypoint_conf"].mean()), 4),
            "participation_label": part_label,
            "label_source": row.get("label_source", ""),
            "label_rule_version": row.get("label_rule_version", ""),
            "annotation_guide_version": row.get("annotation_guide_version", "v0.2"),
            "split": "train",  # Default; will be assigned by group split script
        })

    X = np.array(feature_tensors, dtype=np.float32)  # [N, 40, 64]
    y = np.array(labels_list, dtype=np.int64)        # [N,]
    meta_df = pd.DataFrame(metadata_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_file = output_dir / f"{session_id}_features.npz"
    csv_file = output_dir / f"{session_id}_windows.csv"

    np.savez_compressed(
        npz_file,
        X=X,
        y=y,
        window_ids=meta_df["window_id"].to_numpy(dtype=str),
        track_ids=meta_df["track_id"].to_numpy(dtype=str),
    )
    meta_df.to_csv(csv_file, index=False, encoding="utf-8")

    print(f"\n[OK] Built dataset: {X.shape[0]} windows with shape {X.shape[1:]}")
    print(f"[OK] Saved features NPZ to: {npz_file}")
    print(f"[OK] Saved windows CSV to: {csv_file}")

    return X, y, meta_df
