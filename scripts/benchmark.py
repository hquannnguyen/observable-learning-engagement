#!/usr/bin/env python3
"""Benchmark pose/tracking and temporal-model inference.

All numbers are measured on the current machine; nothing is estimated.
Tracker ID-switch/fragmentation metrics require ground-truth tracks and are
reported as N/A, not invented.

Usage:
    python scripts/benchmark.py --video data/raw/class_1.mp4 --max-frames 60
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import model_from_checkpoint
from src.pose.extractor import extract_poses_from_video
from src.pose.tracker import compute_track_statistics


def benchmark_pose(video: Path, max_frames: int, trackers: List[str]) -> List[Dict[str, Any]]:
    import cv2
    from ultralytics import YOLO

    # Load and warm up once so tracker timings exclude model loading.
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(str(video))
    ret, warm_frame = cap.read()
    cap.release()
    if ret:
        model.predict(warm_frame, verbose=False)

    results: List[Dict[str, Any]] = []
    for tracker in trackers:
        model.predictor = None  # reset tracker state between configurations
        tracemalloc.start()
        start = time.perf_counter()
        df, meta = extract_poses_from_video(
            video_path=video,
            track=True,
            tracker=tracker,
            model=model,
            max_frames=max_frames,
        )
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        stats = compute_track_statistics(df, min_track_seconds=2.0)
        durations = stats["duration_s"].tolist() if not stats.empty else []
        results.append(
            {
                "tracker": tracker,
                "sampled_frames": meta["sampled_frames"],
                "detections": meta["total_detections"],
                "elapsed_s": round(elapsed, 3),
                "fps": round(meta["sampled_frames"] / max(elapsed, 1e-9), 2),
                "tracks": int(len(stats)),
                "mean_track_duration_s": round(float(np.mean(durations)), 3)
                if durations
                else None,
                "median_track_duration_s": round(float(np.median(durations)), 3)
                if durations
                else None,
                "id_switches_per_min": "N/A (no ground-truth tracks)",
                "fragmentation_per_person": "N/A (no ground-truth tracks)",
                "peak_python_ram_mb": round(peak / (1024 * 1024), 1),
            }
        )
    return results


def benchmark_models(experiments: List[Path]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for exp in experiments:
        meta_file = exp / "metadata.json"
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        ckpt_path = Path(meta.get("checkpoint_path", ""))
        if not ckpt_path.is_file():
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = model_from_checkpoint(checkpoint).eval()
        input_size = int(checkpoint.get("input_size", 77))
        x = torch.randn(122, 40, input_size)
        params = sum(p.numel() for p in model.parameters())
        trials = []
        with torch.no_grad():
            model(x)  # warm-up
            for _ in range(3):
                start = time.perf_counter()
                reps = 100
                for _ in range(reps):
                    model(x)
                trials.append((time.perf_counter() - start) / reps)
        elapsed = float(min(trials))  # best-case CPU latency; less noisy than mean
        results.append(
            {
                "experiment_id": meta.get("experiment_id"),
                "model_type": checkpoint.get("model_type", "lstm").upper(),
                "params": params,
                "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
                "ms_per_122_windows_cpu": round(elapsed * 1000, 2),
                "ms_per_window_cpu": round(elapsed * 1000 / 122, 3),
            }
        )
    return results


def benchmark_end_to_end(
    video: Path,
    checkpoint: Path,
    max_frames: int,
) -> Dict[str, Any]:
    """Run predict_video.py as a subprocess and read its measured inference_meta.json."""
    out_dir = ROOT / "reports" / "inference" / f"benchmark_{video.stem}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "predict_video.py"),
        "--video", str(video),
        "--checkpoint", str(checkpoint),
        "--out-dir", str(out_dir),
        "--max-frames", str(max_frames),
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            "predict_video failed:\n" + (proc.stderr[-2000:] or proc.stdout[-2000:])
        )
    meta_path = out_dir / "inference_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["subprocess_wall_s"] = round(wall, 3)
    return meta


def write_markdown(payload: Dict[str, Any], out: Path) -> None:
    lines = [
        "# Benchmark (measured)",
        "",
        f"- Machine: {payload['machine']}",
        f"- Python: {payload['python']}",
        f"- PyTorch: {payload['torch']}",
        f"- Video: `{payload.get('video')}`",
        f"- Pose benchmark frames: {payload.get('pose_frames', payload.get('max_frames'))}",
        f"- End-to-end max frames: {payload.get('max_frames')}"
        + (
            f" (actual sampled: {payload['end_to_end']['sampled_frames']})"
            if payload.get("end_to_end")
            else ""
        ),
        "",
        "## Pose / tracking",
        "",
        "| Tracker | Frames | Detections | FPS | Tracks | Median track (s) | Peak Python RAM (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("pose", []):
        lines.append(
            f"| {row['tracker']} | {row['sampled_frames']} | {row['detections']} | "
            f"{row['fps']} | {row['tracks']} | {row['median_track_duration_s']} | "
            f"{row['peak_python_ram_mb']} |"
        )
    lines += [
        "",
        "ID switches/minute and fragmentation/person: **N/A** — no ground-truth track annotations.",
        "",
        "## Temporal models",
        "",
        "| Experiment | Model | Params | Checkpoint (MB) | ms / 122 windows | ms / window |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("models", []):
        lines.append(
            f"| {row['experiment_id']} | {row['model_type']} | {row['params']} | "
            f"{row['checkpoint_size_mb']} | {row['ms_per_122_windows_cpu']} | "
            f"{row['ms_per_window_cpu']} |"
        )
    lines += [
        "",
        "> These are single-run CPU measurements. They are not a scientific comparison "
        "of tracker quality; run with the same hardware and multiple repetitions before quoting.",
        "",
    ]

    e2e = payload.get("end_to_end")
    if e2e:
        lines += [
            "## End-to-end (pose → features → model → export)",
            "",
            "| Video | Sampled frames | Windows | Accepted | Elapsed (s) | E2E FPS | s per video minute |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| {Path(e2e['video']).name} | {e2e['sampled_frames']} | {e2e['windows']} | "
            f"{e2e['accepted']} | {e2e['elapsed_s']} | {e2e['end_to_end_fps']} | "
            f"{e2e['seconds_per_video_minute']} |",
            "",
        ]

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pose/tracking/model performance.")
    parser.add_argument("--video", default="data/raw/class_1.mp4")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--pose-frames", type=int, default=60, help="Frames for pose/tracker benchmark.")
    parser.add_argument("--trackers", default="bytetrack.yaml")
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--experiments", default="reports/experiments")
    parser.add_argument("--out", default="reports/benchmark.md")
    parser.add_argument("--end-to-end", action="store_true", help="Also run predict_video.py end-to-end.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for --end-to-end.")
    args = parser.parse_args()

    video = Path(args.video)
    trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]

    payload: Dict[str, Any] = {
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "video": str(video),
        "max_frames": args.max_frames,
        "pose_frames": args.pose_frames,
        "pose": [],
        "models": [],
        "end_to_end": None,
    }

    if not args.skip_pose:
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        payload["pose"] = benchmark_pose(video, args.pose_frames, trackers)

    exp_root = Path(args.experiments)
    experiments = sorted(
        [
            p
            for p in exp_root.iterdir()
            if p.is_dir() and (p / "metadata.json").is_file()
        ]
    )
    payload["models"] = benchmark_models(experiments)

    if args.end_to_end:
        if args.checkpoint:
            checkpoint = Path(args.checkpoint)
        else:
            candidates = [
                Path(row["checkpoint_path"])
                for row in payload["models"]
                if row.get("model_type") == "LSTM"
            ]
            if not candidates:
                raise ValueError("No LSTM checkpoint found for --end-to-end.")
            checkpoint = candidates[0]
        print(f"End-to-end checkpoint: {checkpoint}", flush=True)
        payload["end_to_end"] = benchmark_end_to_end(
            video, checkpoint, args.max_frames
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, out)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {json_out} and {out}")


if __name__ == "__main__":
    main()
