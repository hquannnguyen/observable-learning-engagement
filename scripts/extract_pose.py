"""CLI script for YOLOv8-Pose extraction and tracking (Task T06 / T07).

Usage:
    python scripts/extract_pose.py --video data/raw/session_001/cam_A.mp4 --sample-fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.pose.extractor import PoseExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YOLOv8 poses with ByteTrack tracking.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default="data/poses", help="Output directory for poses")
    parser.add_argument("--sample-fps", type=int, default=10, help="Sampling FPS (default: 10)")
    parser.add_argument("--max-seconds", type=float, default=None, help="Max duration in seconds to process")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start offset in seconds")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="YOLOv8 pose model name")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--camera-id", type=str, default="cam_A", help="Camera identifier (default: cam_A)")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{video_path.stem}_poses.parquet"

    print(f"Loading pose extractor with model {args.model}...")
    extractor = PoseExtractor(model_name=args.model, confidence_threshold=args.conf)

    print(f"Processing video: {video_path}")
    print(f"Sampling: {args.sample_fps} FPS | Max duration: {args.max_seconds}s | Camera: {args.camera_id}")

    all_rows = []
    start_time = time.time()
    frame_count = 0

    for sampled_idx, timestamp_ms, frame, poses in extractor.process_video(
        video_path,
        sample_fps=args.sample_fps,
        max_duration_seconds=args.max_seconds,
        start_seconds=args.start_seconds,
    ):
        frame_count += 1
        for p in poses:
            all_rows.append(p.to_dict(session_id=video_path.parent.name, camera_id=args.camera_id))

        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Processed {frame_count} sampled frames ({timestamp_ms/1000:.1f}s) - {fps:.1f} FPS")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s ({frame_count} frames, {len(all_rows)} pose detections)")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(out_file, index=False)
        print(f"Saved pose parquet to: {out_file}")
    else:
        print("Warning: No poses were detected in the processed interval.")


if __name__ == "__main__":
    main()
