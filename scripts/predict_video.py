#!/usr/bin/env python3
"""Run pose → windows → temporal model → gate on a single new video.

This is the inference counterpart of build_dataset.py. It does not require a
train/test split and produces no ground-truth metrics; it exports observable
class distribution and coverage only.

Usage:
    python scripts/predict_video.py --video data/raw/class_1.mp4 \
        --checkpoint models/checkpoints/<experiment>.pt \
        --out-dir reports/inference/class_1
"""

from __future__ import annotations

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.assign_zones import assign as assign_zones_df
from scripts.assign_zones import load_grid
from src.contracts.labels import (
    CANONICAL_TRAINING_LABELS,
    INDEX_TO_LABEL,
    UNCERTAIN_LABEL,
)
from src.models.factory import model_from_checkpoint
from src.pose.extractor import extract_poses_from_video
from src.pose.visualization import render_pose_overlay_video
from src.preprocessing.cleaner import clean_and_normalize_poses
from src.preprocessing.windows import build_track_windows
from src.reporting.exporter import export_bundle
from src.reporting.bias_analysis import analyze_positions
from src.uncertainty.gate import UncertaintyThresholds, explain_uncertainty_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict_video")


def predict_dataframe(
    poses: pd.DataFrame,
    model: Any,
    *,
    input_size: int,
    thresholds: UncertaintyThresholds,
    quality_thresholds: Dict[str, float],
    sample_fps: float = 10.0,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Build windows per track, run the model, apply the uncertainty gate."""
    records: List[Dict[str, Any]] = []
    tensors: List[np.ndarray] = []
    include_mask = input_size == 94

    for (session_id, camera_id, track_id), group in poses.groupby(["session_id", "camera_id", "track_id"]):
        win_records, win_tensors = build_track_windows(
            track_df=group,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            sample_fps=sample_fps,
            split="inference",
            include_mask=include_mask,
            quality_thresholds=quality_thresholds,
        )
        if win_records:
            records.extend(win_records)
            tensors.append(win_tensors)

    if not records:
        raise ValueError(
            "No window could be built. Track may be shorter than the window "
            f"({window_seconds}s) or no person was detected."
        )

    X = np.concatenate(tensors, axis=0)
    if X.shape[-1] != input_size:
        raise ValueError(
            f"Feature dimension {X.shape[-1]} does not match checkpoint input_size {input_size}."
        )

    model.eval()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    with torch.no_grad():
        probs = np.concatenate([model.predict_proba(torch.from_numpy(X[i:i + batch_size]).float().to(device)).cpu().numpy() for i in range(0, len(X), batch_size)])

    rows: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        pred_idx = int(np.argmax(probs[i]))
        gated, reason = explain_uncertainty_gate(
            probs[i],
            visible_ratio=float(record["mean_visible_ratio"]),
            mean_keypoint_confidence=float(record["mean_pose_confidence"]),
            thresholds=thresholds,
            class_labels=CANONICAL_TRAINING_LABELS,
        )
        rows.append(
            {
                "window_id": record["window_id"],
                "session_id": record["session_id"],
                "camera_id": record["camera_id"],
                "track_id": record["track_id"],
                "t_start_ms": record["t_start_ms"],
                "t_end_ms": record["t_end_ms"],
                "class_phase": record.get("class_phase"),
                "zone_id": record.get("zone_id"),
                "true_label": None,
                "pred_label": INDEX_TO_LABEL[pred_idx],
                "gated_label": gated,
                "p_max": round(float(probs[i][pred_idx]), 6),
                "gate_reason": reason,
                "label_source": None,
                "mean_visible_ratio": record["mean_visible_ratio"],
                "mean_pose_confidence": record["mean_pose_confidence"],
                "quality_state": record["quality_state"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full inference flow on a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-zone", action="store_true")
    parser.add_argument("--zone-config", default="configs/zones/grid_3x3.yaml")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-preview", action="store_true", help="Skip head-blurred preview; never export an unblurred preview implicitly.")
    args = parser.parse_args()

    video = Path(args.video)
    checkpoint_path = Path(args.checkpoint)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    session_id = args.session_id or video.stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports/inference") / session_id
    started = time.perf_counter()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    qg = cfg.get("quality_gate", {})
    unc = cfg.get("uncertainty", {})
    thresholds = UncertaintyThresholds(
        min_visible_ratio=float(qg.get("min_visible_ratio", 0.60)),
        min_mean_keypoint_confidence=float(qg.get("min_mean_keypoint_confidence", 0.30)),
        min_probability=float(unc.get("min_probability", 0.60)),
    )

    logger.info("Extracting pose for %s ...", video)
    poses, meta = extract_poses_from_video(
        video_path=video,
        session_id=session_id,
        track=True,
        max_frames=args.max_frames,
        device=args.device,
        config_path=args.config,
    )
    if poses.empty:
        raise ValueError("No pose detections; cannot infer.")

    if not args.no_zone:
        zone_cfg_path = Path(args.zone_config)
        if zone_cfg_path.is_file():
            poses = assign_zones_df(poses, load_grid(zone_cfg_path))

    raw_poses = poses.copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_poses.to_csv(out_dir / "poses.csv", index=False)
    poses = clean_and_normalize_poses(poses, min_track_seconds=float(cfg.get("tracking", {}).get("min_track_seconds", 2.0)))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = model_from_checkpoint(checkpoint)
    model.to(torch.device(args.device or "cpu"))
    input_size = int(checkpoint.get("input_size", 77))

    predictions = predict_dataframe(
        poses,
        model,
        input_size=input_size,
        thresholds=thresholds,
        quality_thresholds=qg,
        sample_fps=float(meta["sample_fps"]),
        window_seconds=float(cfg.get("sequence", {}).get("window_seconds", 4.0)),
        stride_seconds=float(cfg.get("sequence", {}).get("stride_seconds", 2.0)),
        batch_size=args.batch_size,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)
    positions, position_report = analyze_positions(predictions)
    positions.to_csv(out_dir / "position_metrics.csv", index=False)
    (out_dir / "position_report.json").write_text(json.dumps(position_report, indent=2, allow_nan=False), encoding="utf-8")
    preview_path = None
    if not args.skip_preview:
        preview_path = out_dir / "head_blurred_preview.mp4"
        render_pose_overlay_video(video_path=video, output_path=preview_path, poses_df=raw_poses,
                                  sample_fps=float(meta["sample_fps"]), blur_face=True,
                                  conf_threshold=float(meta["confidence_threshold"]), max_frames=args.max_frames,
                                  pipeline_label=f"{meta['pose_model']} + {meta.get('tracker', 'tracking')}", session_title=session_id)

    paths = export_bundle(
        predictions,
        out_dir / "export",
        experiment_id=checkpoint_path.stem,
        metric_scope="inference_no_ground_truth",
    )
    elapsed = time.perf_counter() - started
    video_seconds = float(meta["sampled_frames"]) / max(float(meta["sample_fps"]), 1e-9)
    inference_meta = {
        "video": str(video),
        "session_id": session_id,
        "checkpoint": str(checkpoint_path),
        "sampled_frames": int(meta["sampled_frames"]),
        "raw_frames": int(meta["raw_frames"]),
        "sample_fps": float(meta["sample_fps"]),
        "resolution": meta.get("resolution"),
        "pose_model": meta.get("pose_model"),
        "imgsz": meta.get("imgsz"),
        "tracker": meta.get("tracker"),
        "config": str(Path(args.config).resolve()),
        "temporal_device": str(next(model.parameters()).device),
        "preview": str(preview_path) if preview_path else None,
        "poses": str(out_dir / "poses.csv"),
        "privacy_mode": "head_blur_detected_people_requires_visual_review" if preview_path else "no_preview",
        "ground_truth_metrics_available": False,
        "detections": int(meta["total_detections"]),
        "windows": int(len(predictions)),
        "accepted": int((predictions["gated_label"] != UNCERTAIN_LABEL).sum()),
        "uncertain": int((predictions["gated_label"] == UNCERTAIN_LABEL).sum()),
        "elapsed_s": round(elapsed, 3),
        "end_to_end_fps": round(float(meta["sampled_frames"]) / max(elapsed, 1e-9), 2),
        "seconds_per_video_minute": round(elapsed / max(video_seconds / 60.0, 1e-9), 2),
    }
    (out_dir / "inference_meta.json").write_text(
        json.dumps(inference_meta, indent=2), encoding="utf-8"
    )

    logger.info("Video: %s | frames sampled: %s", video.name, meta["sampled_frames"])
    logger.info(
        "Windows: %d | accepted: %d | uncertain: %d",
        len(predictions),
        int((predictions["gated_label"] != UNCERTAIN_LABEL).sum()),
        int((predictions["gated_label"] == UNCERTAIN_LABEL).sum()),
    )
    logger.info(
        "End-to-end: %.2f FPS | %.2f s per video minute | elapsed %.1fs",
        inference_meta["end_to_end_fps"],
        inference_meta["seconds_per_video_minute"],
        elapsed,
    )
    for name, path in paths.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
