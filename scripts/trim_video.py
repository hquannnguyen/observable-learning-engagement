"""Utility script to trim and compress large video files into pilot segments (Task T05).

Usage examples:
    # Cut first 10 minutes from a 5GB video:
    python scripts/trim_video.py --input "path/to/IMG_3354.MOV" --output "data/raw/session_001/cam_A_pilot.mp4" --duration-min 10

    # Cut 5 minutes starting from minute 10:
    python scripts/trim_video.py --input "path/to/IMG_3354.MOV" --output "data/raw/session_001/cam_A_pilot.mp4" --start-min 10 --duration-min 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2


def trim_video(
    input_path: str | Path,
    output_path: str | Path,
    start_sec: float = 0.0,
    duration_sec: float = 600.0,
    target_width: int | None = 1920,
    target_height: int | None = 1080,
) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_file}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {input_file}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / native_fps if native_fps > 0 else 0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Input Video: {input_file.name}")
    print(f"   Resolution: {orig_w}x{orig_h} | FPS: {native_fps:.1f} | Total duration: {total_sec/60:.1f} mins ({total_frames} frames)")

    start_frame = int(start_sec * native_fps)
    num_frames_to_cut = int(duration_sec * native_fps)
    end_frame = min(total_frames, start_frame + num_frames_to_cut)

    if start_frame >= total_frames:
        raise ValueError(f"Start time {start_sec}s exceeds video duration {total_sec:.1f}s")

    out_w = target_width if target_width and target_width < orig_w else orig_w
    out_h = target_height if target_height and target_height < orig_h else orig_h
    # Preserve aspect ratio if resizing
    if target_width and target_width < orig_w:
        out_h = int(orig_h * (out_w / orig_w))
        # Ensure dimensions are even numbers for video codecs
        out_w = out_w - (out_w % 2)
        out_h = out_h - (out_h % 2)

    print(f"[TRIM] Trimming from {start_sec/60:.2f}m to {end_frame/native_fps/60:.2f}m ({(end_frame - start_frame)/native_fps/60:.1f} mins)")
    print(f"[OUTPUT] Resolution: {out_w}x{out_h} -> {output_file}")

    # Set reader start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, native_fps, (out_w, out_h))

    current_frame = start_frame
    frames_written = 0
    start_time = time.time()

    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (out_w, out_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            frames_written += 1
            current_frame += 1

            if frames_written % 150 == 0:
                pct = (current_frame - start_frame) / max(1, end_frame - start_frame) * 100
                elapsed = time.time() - start_time
                speed = frames_written / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\r[PROGRESS] {pct:.1f}% ({frames_written}/{end_frame - start_frame} frames, {speed:.1f} FPS)...")
                sys.stdout.flush()

        print(f"\n[OK] Finished! Exported {frames_written} frames in {time.time() - start_time:.1f}s.")
        print(f"[INFO] Output file size: {output_file.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"[OK] Saved to: {output_file}")

    finally:
        cap.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim large video files into pilot segments.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to raw video file")
    parser.add_argument("--output", "-o", type=str, default="data/raw/session_001/cam_A_pilot.mp4", help="Output path")
    parser.add_argument("--start-min", type=float, default=0.0, help="Start offset in minutes (default: 0)")
    parser.add_argument("--duration-min", type=float, default=10.0, help="Duration to cut in minutes (default: 10)")
    parser.add_argument("--width", type=int, default=1920, help="Max output width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Max output height (default: 1080)")

    args = parser.parse_args()

    trim_video(
        input_path=args.input,
        output_path=args.output,
        start_sec=args.start_min * 60.0,
        duration_sec=args.duration_min * 60.0,
        target_width=args.width,
        target_height=args.height,
    )


if __name__ == "__main__":
    main()
