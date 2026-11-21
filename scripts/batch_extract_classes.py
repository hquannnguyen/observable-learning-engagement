#!/usr/bin/env python3
"""Batch extract poses for all class_*.mp4 videos using YOLO11-Pose and ByteTrack.

Usage:
    # Fast extraction (640px, ~10-12 mins on CPU):
    python scripts/batch_extract_classes.py --imgsz 640

    # High-accuracy HD extraction (1280px, ~50-60 mins on CPU):
    python scripts/batch_extract_classes.py --imgsz 1280

    # Custom config:
    python scripts/batch_extract_classes.py --config configs/experiments/pose_yolo11m_hd.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.pose.extractor import (
    extract_poses_from_video,
    load_pose_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("batch_extract_yolo11")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract 17 pose keypoints from all class_*.mp4 videos using YOLO11."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing class_*.mp4 videos (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/poses",
        help="Directory to save pose files (default: data/poses).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/pose_yolo11m_hd.yaml",
        help="Config file (default: configs/experiments/pose_yolo11m_hd.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model path (default: from config, yolo11m-pose.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference resolution: 640 (fast) or 1280 (high-accuracy HD).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cpu', 'cuda').",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames per video (useful for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all class_*.mp4 videos
    class_videos = sorted(raw_path.glob("class_*.mp4"))
    if not class_videos:
        logger.error("No class_*.mp4 videos found in %s!", raw_path)
        sys.exit(1)

    logger.info("Found %d class videos to process: %s", len(class_videos), [v.name for v in class_videos])

    # Load configuration
    cfg = load_pose_config(args.config)
    model_name = args.model or cfg.get("model", "yolo11m-pose.pt")
    eff_imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 1280)
    eff_conf = cfg.get("confidence_threshold", 0.35)
    eff_fps = cfg.get("sample_fps", 10.0)

    logger.info("Model: %s | Resolution: %dpx | Confidence: %.2f | Sample FPS: %.1f",
                model_name, eff_imgsz, eff_conf, eff_fps)

    # Preload YOLO11 model once
    logger.info("Preloading YOLO11 model into memory: %s ...", model_name)
    model = YOLO(model_name)

    total_start = time.perf_counter()
    summary = {}

    for idx, video_file in enumerate(class_videos, start=1):
        sess = video_file.stem  # e.g. "class_1"
        out_csv = out_path / f"{sess}_cam_A_poses.csv"
        out_parquet = out_path / f"{sess}_cam_A_poses.parquet"

        logger.info("[%d/%d] Extracting poses for %s -> %s & .parquet",
                    idx, len(class_videos), video_file.name, out_csv.name)

        vid_start = time.perf_counter()
        df, meta = extract_poses_from_video(
            video_path=video_file,
            output_path=out_csv,
            session_id=sess,
            camera_id="cam_A",
            config_path=args.config,
            sample_fps=eff_fps,
            conf_threshold=eff_conf,
            track=True,
            model=model,
            imgsz=eff_imgsz,
            device=args.device,
            max_frames=args.max_frames,
        )

        # Save parquet format as well
        df.to_parquet(out_parquet, index=False)
        vid_elapsed = time.perf_counter() - vid_start

        meta["elapsed_sec"] = round(vid_elapsed, 2)
        summary[sess] = meta
        logger.info("[%d/%d] Completed %s: %d frames sampled, %d detections in %.1fs (%.2f FPS)",
                    idx, len(class_videos), video_file.name,
                    meta["sampled_frames"], meta["total_detections"],
                    vid_elapsed, float(meta["sampled_frames"]) / max(vid_elapsed, 1e-6))

    total_elapsed = time.perf_counter() - total_start
    total_frames = sum(m["sampled_frames"] for m in summary.values())
    total_detections = sum(m["total_detections"] for m in summary.values())

    logger.info("=" * 60)
    logger.info("BATCH EXTRACTION COMPLETED SUCCESSFULLY!")
    logger.info("Total Videos      : %d", len(class_videos))
    logger.info("Total Frames      : %d", total_frames)
    logger.info("Total Detections  : %d", total_detections)
    logger.info("Total Time        : %.1fs (%.2f minutes)", total_elapsed, total_elapsed / 60.0)
    logger.info("Saved to          : %s", out_path.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
