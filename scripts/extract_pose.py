#!/usr/bin/env python3
"""CLI script to extract 17 pose keypoints using YOLOv8-Pose.

Usage examples:
    # Process a single video from data/raw:
    python scripts/extract_pose.py --video data/raw/class_1.mp4 --output data/poses/class_1_poses.csv

    # Process all videos in data/raw:
    python scripts/extract_pose.py --video-dir data/raw --output data/poses --format csv

    # With tracking enabled and custom sample FPS:
    python scripts/extract_pose.py --video data/raw/class_1.mp4 --track --sample-fps 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose.extractor import (
    extract_poses_from_directory,
    extract_poses_from_video,
    load_pose_config,
)
from src.pose.visualization import render_pose_overlay_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_pose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 17 pose keypoints per frame from raw video using YOLOv8-Pose."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--video",
        type=str,
        help="Path to a single video file (e.g. data/raw/class_1.mp4).",
    )
    group.add_argument(
        "--video-dir",
        type=str,
        default="data/raw",
        help="Directory with raw video files to batch process (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/poses",
        help="Path to output file (if single video) or output directory (default: data/poses).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session identifier (default: video stem, e.g. 'class_1').",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="cam_A",
        help="Camera identifier (default: cam_A).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline YAML config (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Sampling FPS override (default: from baseline.yaml, typically 10).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Keypoint confidence threshold override (default: from baseline.yaml, typically 0.30).",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable ByteTrack tracking to preserve consistent track IDs.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="Output format (default: csv).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cpu', 'cuda', or specific device id).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pose model weights path/name (e.g. 'yolo11m-pose.pt', 'yolov8m-pose.pt').",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image resolution (default 640, recommend 1280 for classroom accuracy).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum sampled frames to process (useful for smoke test).",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Also render and save an overlay MP4 video with skeleton keypoints and track IDs.",
    )
    parser.add_argument(
        "--blur-face",
        action="store_true",
        help="Apply privacy blurring to faces/heads when saving visualization video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load baseline config
    cfg = load_pose_config(args.config)
    eff_sample_fps = args.sample_fps if args.sample_fps is not None else cfg["sample_fps"]
    eff_conf = args.conf if args.conf is not None else cfg["confidence_threshold"]

    logger.info("Configuration: model=%s, sample_fps=%.1f, conf_threshold=%.2f, tracker=%s",
                cfg["model"], eff_sample_fps, eff_conf, cfg["tracker"])

    if args.video:
        video_file = Path(args.video)
        if not video_file.is_file():
            logger.error("Video file does not exist: %s", video_file)
            sys.exit(1)

        out_path = Path(args.output)
        if out_path.is_dir() or (not out_path.suffix and not out_path.exists()):
            out_path.mkdir(parents=True, exist_ok=True)
            sess = args.session or video_file.stem
            ext = ".parquet" if args.format == "parquet" else ".csv"
            out_file = out_path / f"{sess}_{args.camera}_poses{ext}"
        else:
            out_file = out_path

        df, meta = extract_poses_from_video(
            video_path=video_file,
            output_path=out_file,
            session_id=args.session,
            camera_id=args.camera,
            config_path=args.config,
            sample_fps=args.sample_fps,
            conf_threshold=args.conf,
            track=args.track,
            model_name=args.model,
            imgsz=args.imgsz,
            device=args.device,
            max_frames=args.max_frames,
        )

        if args.format == "both":
            parquet_file = out_file.with_suffix(".parquet")
            try:
                df.to_parquet(parquet_file, index=False)
                logger.info("Also saved Parquet to: %s", parquet_file)
            except (ImportError, ModuleNotFoundError) as err:
                logger.warning("Could not save Parquet (engine missing: %s); CSV already saved.", err)

        logger.info(
            "Extraction complete for %s! Processed %d sampled frames (%d detections saved to %s).",
            video_file.name,
            meta["sampled_frames"],
            meta["total_detections"],
            out_file,
        )

        if args.save_viz:
            viz_out = out_file.parent / f"{sess}_{args.camera}_pose_overlay.mp4"
            logger.info("Rendering pose overlay video to %s...", viz_out)
            render_pose_overlay_video(
                video_path=video_file,
                output_path=viz_out,
                poses_df=df,
                sample_fps=eff_sample_fps,
                conf_threshold=eff_conf,
                blur_face=args.blur_face,
                max_frames=args.max_frames,
            )
            logger.info("Visualization video saved: %s", viz_out)

    else:
        video_dir = Path(args.video_dir)
        if not video_dir.is_dir():
            logger.error("Video directory does not exist: %s", video_dir)
            sys.exit(1)

        logger.info("Batch extracting poses from directory: %s -> %s", video_dir, args.output)
        results = extract_poses_from_directory(
            raw_dir=video_dir,
            output_dir=args.output,
            camera_id=args.camera,
            config_path=args.config,
            track=args.track,
            output_format=args.format,
            device=args.device,
            max_frames_per_video=args.max_frames,
        )

        total_videos = len(results)
        total_detections = sum(r["total_detections"] for r in results.values())
        logger.info(
            "Batch extraction finished! Processed %d videos, total detections: %d.",
            total_videos,
            total_detections,
        )


if __name__ == "__main__":
    main()
