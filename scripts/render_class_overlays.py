#!/usr/bin/env python3
"""Render pose skeletons from YOLO11 HD parquet data onto videos and snapshot images for class_1..class_7.

Usage:
    python scripts/render_class_overlays.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from src.pose.visualization import (
    draw_hud_header,
    draw_skeleton_on_frame,
    render_pose_overlay_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger = logging.getLogger("render_class_overlays")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
POSES_DIR = ROOT / "data" / "poses"
SNAPSHOT_DIR = ROOT / "reports" / "class_overlays"


def extract_snapshot_images(
    video_path: Path,
    poses_df: pd.DataFrame,
    class_name: str,
    output_dir: Path,
    num_snapshots: int = 3,
) -> List[Path]:
    """Save high-resolution snapshot images with drawn skeletons for quick inspection."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video %s for snapshots", video_path)
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_images: List[Path] = []

    # Group poses by frame_idx
    poses_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for _, row in poses_df.iterrows():
        f_idx = int(row["frame_idx"])
        if f_idx not in poses_by_frame:
            poses_by_frame[f_idx] = []
        poses_by_frame[f_idx].append(row.to_dict())

    available_frame_indices = sorted(poses_by_frame.keys())
    if not available_frame_indices:
        cap.release()
        return []

    # Pick evenly spaced frame indices (e.g. 25%, 50%, 75%)
    step = max(1, len(available_frame_indices) // (num_snapshots + 1))
    selected_indices = [
        available_frame_indices[min(len(available_frame_indices) - 1, step * (i + 1))]
        for i in range(num_snapshots)
    ]

    current_native = 0
    sampled_idx = 0
    eff_sample_fps = 10.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_ms = (current_native / native_fps) * 1000.0
        # Check if this native frame corresponds to a sampled frame
        is_sampled = (current_native % max(1, int(round(native_fps / eff_sample_fps)))) == 0

        if is_sampled:
            if sampled_idx in selected_indices:
                frame_poses = poses_by_frame.get(sampled_idx, [])
                annotated = frame.copy()

                for p_dict in frame_poses:
                    kpts = np.zeros((17, 3), dtype=np.float32)
                    for k in range(17):
                        kpts[k, 0] = p_dict.get(f"kpt_{k:02d}_x", np.nan)
                        kpts[k, 1] = p_dict.get(f"kpt_{k:02d}_y", np.nan)
                        kpts[k, 2] = p_dict.get(f"kpt_{k:02d}_conf", 0.0)

                    bbox = (
                        float(p_dict.get("bbox_x1", 0)),
                        float(p_dict.get("bbox_y1", 0)),
                        float(p_dict.get("bbox_x2", 0)),
                        float(p_dict.get("bbox_y2", 0)),
                    )
                    t_id = p_dict.get("track_id")
                    track_id = int(t_id) if pd.notna(t_id) else None

                    draw_skeleton_on_frame(
                        frame=annotated,
                        keypoints=kpts,
                        bbox=bbox,
                        track_id=track_id,
                        conf_threshold=0.30,
                        blur_face=False,
                        show_bbox=True,
                    )

                draw_hud_header(
                    frame=annotated,
                    session_title=f"{class_name} (YOLO11m-Pose HD)",
                    frame_idx=sampled_idx,
                    timestamp_ms=int(round(current_time_ms)),
                    active_tracks=len(frame_poses),
                    pipeline_label="YOLO11m-Pose + ByteTrack (1280px)",
                )

                out_img = output_dir / f"{class_name}_frame_{sampled_idx:04d}.jpg"
                cv2.imwrite(str(out_img), annotated)
                saved_images.append(out_img)
                logger.info("Saved skeleton snapshot image: %s (%d tracks)", out_img.name, len(frame_poses))

            sampled_idx += 1

        current_native += 1

    cap.release()
    return saved_images


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    all_videos = [f"class_{i}" for i in range(1, 8)]

    logger.info("=" * 65)
    logger.info("RENDERING SKELETON OVERLAYS FOR 7 CLASS VIDEOS (YOLO11)")
    logger.info("=" * 65)

    results = []

    for idx, c_name in enumerate(all_videos, start=1):
        vid_file = RAW_DIR / f"{c_name}.mp4"
        poses_file = POSES_DIR / f"{c_name}_cam_A_poses.parquet"

        if not vid_file.is_file():
            logger.warning("[%d/7] Video not found: %s", idx, vid_file)
            continue
        if not poses_file.is_file():
            logger.warning("[%d/7] Pose parquet not found: %s", idx, poses_file)
            continue

        logger.info("[%d/7] Processing %s...", idx, c_name)
        df = pd.read_parquet(poses_file)
        logger.info("  Loaded %d detections across %d frames.", len(df), df["frame_idx"].nunique())

        # 1. Render Video Overlay (MP4)
        out_vid = POSES_DIR / f"{c_name}_yolo11m_hd_overlay.mp4"
        render_pose_overlay_video(
            video_path=vid_file,
            output_path=out_vid,
            poses_df=df,
            sample_fps=10.0,
            conf_threshold=0.30,
            blur_face=False,
            pipeline_label="YOLO11m-Pose HD (1280px) + ByteTrack",
            session_title=f"{c_name}_cam_A",
        )
        logger.info("  -> Rendered video: %s (%.2f MB)", out_vid.name, out_vid.stat().st_size / (1024 * 1024))

        # 2. Extract 2 high-res snapshot JPG images per class
        snapshots = extract_snapshot_images(
            video_path=vid_file,
            poses_df=df,
            class_name=c_name,
            output_dir=SNAPSHOT_DIR,
            num_snapshots=2,
        )

        results.append({
            "class": c_name,
            "detections": len(df),
            "video_overlay": out_vid,
            "snapshots": snapshots,
        })

    logger.info("=" * 65)
    logger.info("ALL 7 CLASS VIDEOS POSE SKELETONS RENDERED SUCCESSFULLY!")
    logger.info("Output Videos Saved To : %s", POSES_DIR)
    logger.info("Snapshot Images Saved To: %s", SNAPSHOT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
