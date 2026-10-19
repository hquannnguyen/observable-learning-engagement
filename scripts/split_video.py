#!/usr/bin/env python3
"""CLI utility to split a raw video into session directories with metadata templates.

Usage examples:
    # 1. Split into 15-minute sessions automatically:
    python scripts/split_video.py --video data/raw/IMG_3354.MOV --duration 15m --start-session 1

    # 2. Cut a specific pilot segment (e.g., first 12 minutes):
    python scripts/split_video.py --video data/raw/IMG_3354.MOV --cuts "00:00-12:00" --session-ids "session_001"

    # 3. Cut with custom time intervals:
    python scripts/split_video.py --video data/raw/IMG_3354.MOV \
        --cuts "00:00-25:00,25:00-50:00,50:00-01:19:10" \
        --phases "lecture,individual_work,discussion"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("split_video")


def parse_time_str(time_str: str) -> float:
    """Parse time string formatted as [[HH:]MM:]SS[.ms] or with suffix 'm'/'s' to seconds."""
    s = time_str.strip()
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    if s.endswith("s"):
        return float(s[:-1])

    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return float(h) * 3600.0 + float(m) * 60.0 + float(sec)
    elif len(parts) == 2:
        m, sec = parts
        return float(m) * 60.0 + float(sec)
    elif len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"Unrecognized time format: '{time_str}'")


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Retrieve video properties using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        fps = 30.0

    duration = frames / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration,
        "resolution": f"{width}x{height}",
    }


def create_metadata_template(
    session_id: str,
    camera_id: str = "A",
    fps: int = 30,
    resolution: str = "1920x1080",
    duration_s: float = 0.0,
    phase_name: str = "lecture",
    participants: int = 8,
    date_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Create metadata dictionary strictly matching docs/dataset_schema.md."""
    dur_ms = int(round(duration_s * 1000.0))
    return {
        "session_id": session_id,
        "date": date_str or str(date.today()),
        "camera_id": camera_id,
        "fps": int(round(fps)),
        "resolution": resolution,
        "camera_view": "front_oblique",
        "participants": participants,
        "class_phase": [
            {
                "name": phase_name,
                "start_ms": 0,
                "end_ms": dur_ms,
            }
        ],
        "lighting_condition": "normal",
        "consent_confirmed": True,
        "notes": f"Generated from raw recording for {session_id}.",
    }


def cut_video_segment(
    input_file: Path,
    output_file: Path,
    start_s: float,
    duration_s: float,
    transcode_h264: bool = True,
) -> bool:
    """Cut video segment using ffmpeg.

    If transcode_h264=True, transcode to standard H.264 + faststart for CVAT/Web compatibility.
    Otherwise, use stream copy (-c copy).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_s),
        "-i", str(input_file),
        "-t", str(duration_s),
    ]

    if transcode_h264:
        # High quality, fast encoding with hardware acceleration if available
        cmd.extend([
            "-c:v", "h264_mf",
            "-b:v", "5M",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
        ])
    else:
        cmd.extend(["-c", "copy"])

    cmd.append(str(output_file))

    logger.info("Executing ffmpeg: start=%.1fs (%s), duration=%.1fs (%s) -> %s",
                start_s, format_seconds(start_s), duration_s, format_seconds(duration_s), output_file.name)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 and transcode_h264:
        # Fallback to software libx264
        logger.warning("Hardware encoder failed, falling back to libx264: %s", proc.stderr[-200:])
        fallback_cmd = [
            "ffmpeg", "-y", "-ss", str(start_s), "-i", str(input_file), "-t", str(duration_s),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output_file)
        ]
        proc = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("FFmpeg error on fallback: %s", proc.stderr[-500:])
            return False
    elif proc.returncode != 0:
        logger.error("FFmpeg error: %s", proc.stderr[-500:])
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw video file into session subdirectories with metadata.json."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/IMG_3354.MOV",
        help="Path to source raw video (default: data/raw/IMG_3354.MOV).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory where session folders will be created (default: data/raw).",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        help="Fixed duration per segment (e.g. '15m' or '900s').",
    )
    parser.add_argument(
        "--cuts",
        type=str,
        default=None,
        help="Comma-separated time cuts, e.g. '00:00-15:00,15:00-30:00' or '0:00-12:00'.",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help="Comma-separated activity phase names matching each cut (e.g. 'lecture,individual_work,discussion').",
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default="session_",
        help="Prefix for session folder names (default: 'session_').",
    )
    parser.add_argument(
        "--start-session",
        type=int,
        default=1,
        help="Starting session index number (default: 1).",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="A",
        help="Camera identifier (default: 'A').",
    )
    parser.add_argument(
        "--participants",
        type=int,
        default=8,
        help="Estimated number of students in frame (default: 8).",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Use ffmpeg stream copy (-c copy) instead of H.264 transcoding (faster, but HEVC MOV might fail in CVAT/browser).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    info = get_video_info(video_path)
    total_duration = info["duration"]
    logger.info("Loaded '%s': Resolution=%s, FPS=%.2f, Total Duration=%.1fs (%s)",
                video_path.name, info["resolution"], info["fps"], total_duration, format_seconds(total_duration))

    segments: List[Tuple[float, float, str, str]] = []  # (start_s, dur_s, session_id, phase_name)

    if args.cuts:
        cut_list = [c.strip() for c in args.cuts.split(",") if c.strip()]
        phase_list = (
            [p.strip() for p in args.phases.split(",")]
            if args.phases
            else ["lecture"] * len(cut_list)
        )
        if len(phase_list) < len(cut_list):
            phase_list.extend(["lecture"] * (len(cut_list) - len(phase_list)))

        for i, cut in enumerate(cut_list):
            if "-" not in cut:
                raise ValueError(f"Invalid cut interval format '{cut}'. Must be 'start-end'.")
            c_start_s, c_end_s = cut.split("-", 1)
            t_start = parse_time_str(c_start_s)
            t_end = min(parse_time_str(c_end_s), total_duration)
            t_dur = t_end - t_start
            if t_dur <= 0:
                logger.warning("Skipping cut '%s' with non-positive duration.", cut)
                continue
            sess_id = f"{args.session_prefix}{args.start_session + i:03d}"
            segments.append((t_start, t_dur, sess_id, phase_list[i]))

    elif args.duration:
        seg_dur = parse_time_str(args.duration)
        num_segs = int(total_duration // seg_dur)
        if total_duration % seg_dur > 60.0:  # remaining chunk > 1 minute
            num_segs += 1

        for i in range(num_segs):
            t_start = i * seg_dur
            t_dur = min(seg_dur, total_duration - t_start)
            if t_dur < 10.0:
                break
            sess_id = f"{args.session_prefix}{args.start_session + i:03d}"
            # Default alternating phases
            phase_name = "lecture" if i % 3 == 0 else ("individual_work" if i % 3 == 1 else "discussion")
            segments.append((t_start, t_dur, sess_id, phase_name))
    else:
        logger.error("Must specify either --duration (e.g. 15m) or --cuts (e.g. 00:00-15:00,15:00-30:00).")
        sys.exit(1)

    out_base = Path(args.output_dir)
    logger.info("Planning to generate %d session(s) in %s:", len(segments), out_base)

    for start_s, dur_s, sess_id, phase in segments:
        logger.info("  * [%s] Time: %s -> %s (Duration: %s) | Phase: %s",
                    sess_id, format_seconds(start_s), format_seconds(start_s + dur_s), format_seconds(dur_s), phase)

    created_sessions = []
    for start_s, dur_s, sess_id, phase in segments:
        sess_dir = out_base / sess_id
        sess_dir.mkdir(parents=True, exist_ok=True)
        video_out = sess_dir / f"cam_{args.camera_id}.mp4"

        success = cut_video_segment(
            input_file=video_path,
            output_file=video_out,
            start_s=start_s,
            duration_s=dur_s,
            transcode_h264=not args.copy_only,
        )

        if success:
            meta = create_metadata_template(
                session_id=sess_id,
                camera_id=args.camera_id,
                fps=int(round(info["fps"])),
                resolution=info["resolution"],
                duration_s=dur_s,
                phase_name=phase,
                participants=args.participants,
            )
            meta_path = sess_dir / "metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Created session '%s': %s & metadata.json", sess_id, video_out)
            created_sessions.append(sess_id)

    logger.info("Done! Successfully generated %d session(s): %s", len(created_sessions), ", ".join(created_sessions))


if __name__ == "__main__":
    main()
