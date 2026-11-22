#!/usr/bin/env python3
"""Batch extract poses for all session_001 to session_006 videos using YOLO11-Pose and ByteTrack.

Usage:
    # Run full batch with YOLO11 HD (1280px):
    python scripts/batch_extract_sessions.py --imgsz 1280

    # Run on Google Colab GPU saving directly to Google Drive with auto-resume:
    python scripts/batch_extract_sessions.py \
        --raw-dir "/content/drive/MyDrive/ThiGiacMaytinh" \
        --output-dir "/content/drive/MyDrive/ThiGiacMaytinh/poses" \
        --resume \
        --imgsz 640 \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import shutil
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
logger = logging.getLogger("batch_extract_sessions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract 17 pose keypoints from session_001..006 using YOLO11."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Root raw directory containing session_* folders (default: data/raw).",
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
        help="Inference resolution: 640 (faster) or 1280 (HD high-accuracy).",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Specific sessions to process (e.g. --sessions session_001 session_002). Default: all session_001..session_006.",
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
        help="Max frames per video (useful for smoke testing).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up existing session pose CSV files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sessions whose pose parquet files already exist in output-dir.",
    )
    return parser.parse_args()


def backup_existing_poses(out_path: Path, session_names: list[str]) -> None:
    backup_dir = out_path / "_backup_yolov8n_sessions"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for sess in session_names:
        old_csv = out_path / f"{sess}_cam_A_poses.csv"
        old_parquet = out_path / f"{sess}_cam_A_poses.parquet"
        if old_csv.is_file():
            dest = backup_dir / old_csv.name
            if not dest.exists():
                shutil.copy2(old_csv, dest)
                logger.info("Backed up existing %s -> %s", old_csv.name, backup_dir.name)
        if old_parquet.is_file():
            dest = backup_dir / old_parquet.name
            if not dest.exists():
                shutil.copy2(old_parquet, dest)
                logger.info("Backed up existing %s -> %s", old_parquet.name, backup_dir.name)


def resolve_session_video(raw_path: Path, sess: str) -> Path | None:
    """Find video path for a session supporting multiple folder layouts."""
    candidates = [
        raw_path / sess / "cam_A.mp4",
        raw_path / sess / "cam_A.MOV",
        raw_path / sess / "cam_A.avi",
        raw_path / f"{sess}.mp4",
        raw_path / f"{sess}.MOV",
        raw_path / "data" / "raw" / sess / "cam_A.mp4",
        raw_path / "data" / "raw" / sess / "cam_A.MOV",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Check any video inside session directory
    sess_dir = raw_path / sess
    if sess_dir.is_dir():
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            matches = list(sess_dir.glob(f"*{ext}")) + list(sess_dir.glob(f"*{ext.upper()}"))
            if matches:
                return matches[0]

    # Recursive check
    for ext in [".mp4", ".mov", ".avi", ".mkv"]:
        rec = list(raw_path.glob(f"**/{sess}/cam_A{ext}")) + list(raw_path.glob(f"**/{sess}{ext}"))
        if rec:
            return rec[0]

    return None


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine target sessions
    if args.sessions:
        target_sessions = sorted(args.sessions)
    else:
        target_sessions = [f"session_00{i}" for i in range(1, 7)]

    # Validate video files exist
    valid_sessions = []
    for sess in target_sessions:
        v_file = resolve_session_video(raw_path, sess)
        if v_file and v_file.is_file():
            valid_sessions.append((sess, v_file))
        else:
            logger.warning("Video not found for %s under %s, skipping.", sess, raw_path)

    if not valid_sessions:
        logger.error("No valid session videos found to process in %s!", raw_path)
        logger.error("Directory contents of %s:", raw_path)
        if raw_path.is_dir():
            for child in sorted(raw_path.iterdir()):
                logger.error(" - %s (%s)", child.name, "dir" if child.is_dir() else f"{child.stat().st_size} bytes")
        sys.exit(1)

    logger.info("Found %d session videos to process:", len(valid_sessions))
    for s, vf in valid_sessions:
        logger.info("  • %s -> %s", s, vf)

    # Backup existing poses unless disabled
    if not args.no_backup:
        backup_existing_poses(out_path, [s for s, _ in valid_sessions])

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

    for idx, (sess, video_file) in enumerate(valid_sessions, start=1):
        out_csv = out_path / f"{sess}_cam_A_poses.csv"
        out_parquet = out_path / f"{sess}_cam_A_poses.parquet"

        # Auto-resume check: skip if parquet already exists and has valid size
        if args.resume and out_parquet.is_file() and out_parquet.stat().st_size > 1024:
            logger.info("=" * 60)
            logger.info("[%d/%d] RESUME MODE: Skipping %s (Found %s, size=%.1f KB)",
                        idx, len(valid_sessions), sess, out_parquet.name, out_parquet.stat().st_size / 1024.0)
            logger.info("=" * 60)
            continue

        logger.info("=" * 60)
        logger.info("[%d/%d] STARTING POSE EXTRACTION: %s (%s)",
                    idx, len(valid_sessions), sess, video_file.name)
        logger.info("Output targets: %s & %s", out_csv.name, out_parquet.name)
        logger.info("=" * 60)

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
        fps_speed = float(meta["sampled_frames"]) / max(vid_elapsed, 1e-6)
        logger.info("[%d/%d] FINISHED %s: %d frames sampled, %d detections in %.1fs (%.2f FPS)",
                    idx, len(valid_sessions), sess,
                    meta["sampled_frames"], meta["total_detections"],
                    vid_elapsed, fps_speed)

    total_elapsed = time.perf_counter() - total_start
    total_frames = sum(m["sampled_frames"] for m in summary.values()) if summary else 0
    total_detections = sum(m["total_detections"] for m in summary.values()) if summary else 0

    logger.info("=" * 60)
    logger.info("POSE EXTRACTION BATCH COMPLETED!")
    logger.info("Processed Sessions: %d", len(summary))
    logger.info("Total Frames      : %d", total_frames)
    logger.info("Total Detections  : %d", total_detections)
    logger.info("Total Time        : %.1fs (%.2f hours)", total_elapsed, total_elapsed / 3600.0)
    logger.info("Saved to          : %s", out_path.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
