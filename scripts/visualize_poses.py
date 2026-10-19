#!/usr/bin/env python3
"""CLI script to render pose skeleton wireframes and track IDs on videos.

Usage examples:
    # 1. Render overlay for a video using pre-extracted pose CSV:
    python scripts/visualize_poses.py --video data/raw/class_1.mp4 \
                                      --poses data/poses/class_1_cam_A_poses.csv \
                                      --output data/poses/class_1_pose_overlay.mp4

    # 2. Extract pose and render overlay live in one pass:
    python scripts/visualize_poses.py --video data/raw/class_1.mp4 \
                                      --output data/poses/class_1_pose_overlay.mp4 \
                                      --track

    # 3. Batch render all sessions with face blur for privacy:
    python scripts/visualize_poses.py --all --blur-face
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose.extractor import extract_poses_from_video
from src.pose.visualization import render_pose_overlay_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visualize_poses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 17 COCO skeleton keypoints, limbs, and track IDs onto video."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to raw input video (e.g. data/raw/class_1.mp4).",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default=None,
        help="Path to pre-extracted pose CSV/Parquet file (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target output video path (e.g. data/poses/class_1_overlay.mp4).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Batch process all videos in data/raw that have poses in data/poses.",
    )
    parser.add_argument(
        "--blur-face",
        action="store_true",
        help="Apply Gaussian blur to faces/heads for privacy anonymization.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for keypoints (default: 0.30).",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=10.0,
        help="Sample FPS to match pose extraction (default: 10.0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max sampled frames to render (useful for quick previews).",
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
        "--track",
        action="store_true",
        help="Run ByteTrack tracking if extracting poses live.",
    )
    return parser.parse_args()


def process_single(
    video_path: Path,
    poses_path: Path | None,
    out_path: Path,
    conf_threshold: float,
    sample_fps: float,
    blur_face: bool,
    max_frames: int | None,
    track: bool,
    model_name: str | None = None,
    imgsz: int | None = None,
) -> None:
    logger.info("Processing: %s", video_path.name)
    df = None

    if poses_path is not None and poses_path.is_file() and model_name is None:
        logger.info("Loading pre-extracted poses from %s", poses_path)
        if poses_path.suffix == ".parquet":
            df = pd.read_parquet(poses_path)
        else:
            df = pd.read_csv(poses_path)
    else:
        logger.info("Running pose extraction on %s (model=%s, imgsz=%s)...", video_path.name, model_name or "default", imgsz or "default")
        df, _ = extract_poses_from_video(
            video_path=video_path,
            sample_fps=sample_fps,
            conf_threshold=conf_threshold,
            track=track,
            model_name=model_name,
            imgsz=imgsz,
            max_frames=max_frames,
        )

    render_pose_overlay_video(
        video_path=video_path,
        output_path=out_path,
        poses_df=df,
        sample_fps=sample_fps,
        conf_threshold=conf_threshold,
        blur_face=blur_face,
        max_frames=max_frames,
    )
    logger.info("Output video ready: %s", out_path)


def main() -> None:
    args = parse_args()

    if args.all:
        raw_dir = ROOT / "data" / "raw"
        poses_dir = ROOT / "data" / "poses"
        out_dir = ROOT / "data" / "poses" / "overlays"
        out_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(list(raw_dir.glob("*.mp4")))
        if not videos:
            logger.error("No MP4 videos found in %s", raw_dir)
            sys.exit(1)

        for vid in videos:
            # Check for existing pose csv
            pose_csv = poses_dir / f"{vid.stem}_cam_A_poses.csv"
            if not pose_csv.is_file():
                pose_csv = poses_dir / f"{vid.stem}_poses.csv"
            if not pose_csv.is_file():
                pose_csv = None

            out_vid = out_dir / f"{vid.stem}_pose_overlay.mp4"
            process_single(
                video_path=vid,
                poses_path=pose_csv,
                out_path=out_vid,
                conf_threshold=args.conf,
                sample_fps=args.sample_fps,
                blur_face=args.blur_face,
                max_frames=args.max_frames,
                track=args.track,
                model_name=args.model,
                imgsz=args.imgsz,
            )
        return

    if not args.video:
        logger.error("Must specify either --video or --all. Run with -h for help.")
        sys.exit(1)

    vid_path = Path(args.video)
    if not vid_path.is_file():
        logger.error("Video file does not exist: %s", vid_path)
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ROOT / "data" / "poses" / f"{vid_path.stem}_pose_overlay.mp4"

    poses_path = Path(args.poses) if args.poses else None
    if poses_path is None:
        # Check default location
        candidate = ROOT / "data" / "poses" / f"{vid_path.stem}_cam_A_poses.csv"
        if candidate.is_file():
            poses_path = candidate

    process_single(
        video_path=vid_path,
        poses_path=poses_path,
        out_path=out_path,
        conf_threshold=args.conf,
        sample_fps=args.sample_fps,
        blur_face=args.blur_face,
        max_frames=args.max_frames,
        track=args.track,
        model_name=args.model,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
