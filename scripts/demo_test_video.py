#!/usr/bin/env python3
"""Run fast test-set video demo inference + anonymized prediction overlay rendering.

Reuses pre-extracted test poses (``data/poses/session_005_cam_A_poses.parquet`` or
``session_006_cam_A_poses.parquet``) and the trained temporal checkpoint to infer
observable participation, compare against ``Dataset v1`` ground truth when present,
and render an H.264 browser-playable MP4 with colored class boxes and head blur.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.assign_zones import assign as assign_zones_df
from scripts.assign_zones import load_grid
from scripts.predict_video import predict_dataframe
from src.contracts.labels import CANONICAL_TRAINING_LABELS, UNCERTAIN_LABEL
from src.models.factory import model_from_checkpoint
from src.pose.visualization import render_pose_overlay_video
from src.preprocessing.cleaner import clean_and_normalize_poses
from src.reporting.exporter import export_bundle
from src.training.trainer import compute_multiclass_metrics
from src.uncertainty.gate import UncertaintyThresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_test_video")

LABEL_TO_INT = {lb: i for i, lb in enumerate(CANONICAL_TRAINING_LABELS)}


def _load_poses_for_session(session_id: str) -> pd.DataFrame:
    pq = ROOT / "data" / "poses" / f"{session_id}_cam_A_poses.parquet"
    csv = ROOT / "data" / "poses" / f"{session_id}_cam_A_poses.csv"
    if pq.is_file():
        return pd.read_parquet(pq)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Pose file not found for {session_id}: {pq} or {csv}")


def run_test_video_demo(
    *,
    session_id: str,
    checkpoint_path: Path,
    out_dir: Path,
    config_path: Path = ROOT / "configs" / "experiments" / "adjudicated_v03_600.yaml",
    zone_config_path: Path = ROOT / "configs" / "zones" / "grid_3x3.yaml",
    annotations_path: Path = ROOT / "data" / "features" / "dataset_v1" / "annotations_v1.csv",
    start_s: float = 0.0,
    duration_s: float = 60.0,
    blur_face: bool = True,
    min_probability: Optional[float] = None,
    resize_width: int = 1280,
    render_video: bool = True,
) -> Dict[str, Any]:
    """Run inference on a test video segment and render anonymized prediction overlay."""
    started = time.perf_counter()
    video_path = ROOT / "data" / "raw" / session_id / "cam_A.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Test video not found: {video_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    qg = cfg.get("quality_gate", {})
    unc = cfg.get("uncertainty", {})
    sample_fps = float(cfg.get("video", {}).get("sample_fps", 10.0))
    window_seconds = float(cfg.get("sequence", {}).get("window_seconds", 4.0))
    stride_seconds = float(cfg.get("sequence", {}).get("stride_seconds", 2.0))
    tracking = cfg.get("tracking", {})

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    calibration = ckpt.get("calibration") or {}
    temperature = float(calibration.get("temperature", 1.0))
    eff_min_prob = float(
        min_probability
        if min_probability is not None
        else calibration.get("min_probability", unc.get("min_probability", 0.60))
    )
    thresholds = UncertaintyThresholds(
        min_visible_ratio=float(qg.get("min_visible_ratio", 0.60)),
        min_mean_keypoint_confidence=float(qg.get("min_mean_keypoint_confidence", 0.30)),
        min_probability=eff_min_prob,
        upper_body_min_visible_ratio=float(qg.get("upper_body_min_visible_ratio", 0.72))
        if qg.get("allow_upper_body_visibility")
        else None,
    )

    start_ms = max(0.0, float(start_s) * 1000.0)
    end_ms = start_ms + max(4.0, float(duration_s)) * 1000.0
    buf_ms = window_seconds * 1000.0

    raw_poses = _load_poses_for_session(session_id)
    in_window = raw_poses[raw_poses["timestamp_ms"].between(start_ms, end_ms)]
    if in_window.empty:
        raise ValueError(f"Không có dữ liệu pose trong khoảng {start_s:.1f}–{start_s+duration_s:.1f}s cho {session_id}.")
    active_tracks = set(in_window["track_id"].dropna().unique().tolist())
    # Keep full history up to end_ms + buf_ms for active tracks so window_id & timestamps match Dataset v1 exactly
    slice_poses = raw_poses[
        raw_poses["track_id"].isin(active_tracks) & (raw_poses["timestamp_ms"] <= end_ms + buf_ms)
    ].copy()

    if zone_config_path.is_file():
        slice_poses = assign_zones_df(slice_poses, load_grid(zone_config_path))

    cleaned = pd.concat(
        [
            clean_and_normalize_poses(
                group,
                min_track_seconds=float(tracking.get("min_track_seconds", 2.0)),
                merge_max_gap_seconds=tracking.get("merge_max_gap_seconds"),
            )
            for _, group in slice_poses.groupby(["session_id", "camera_id"], sort=False)
        ],
        ignore_index=True,
    )

    model = model_from_checkpoint(ckpt)
    model.to(torch.device("cpu"))
    input_size = int(ckpt.get("input_size", 94))

    preds = predict_dataframe(
        cleaned,
        model,
        input_size=input_size,
        thresholds=thresholds,
        quality_thresholds=qg,
        sample_fps=sample_fps,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        temperature=temperature,
    )
    preds = preds[(preds["t_start_ms"] >= start_ms) & (preds["t_start_ms"] < end_ms)].copy().reset_index(drop=True)
    if preds.empty:
        raise ValueError("Không dựng được cửa sổ dự đoán nào trong khoảng thời gian đã chọn.")

    # Join Dataset v1 ground truth by (session_id, track_id, t_start_ms, t_end_ms) when present
    gt_stats: Dict[str, Any] = {"matched_gt_windows": 0}
    if annotations_path.is_file():
        ann = pd.read_csv(annotations_path, dtype=object, keep_default_na=False)
        ann_valid = ann[(ann["session_id"] == session_id) & (ann["annotation_state"] == "VALID")].copy()
        if not ann_valid.empty:
            ann_valid["track_id_int"] = pd.to_numeric(ann_valid["track_id"], errors="coerce")
            ann_valid["t0_int"] = pd.to_numeric(ann_valid["t_start_ms"], errors="coerce")
            ann_valid["t1_int"] = pd.to_numeric(ann_valid["t_end_ms"], errors="coerce")
            gt_map = {
                (int(r["track_id_int"]), int(r["t0_int"]), int(r["t1_int"])): (
                    str(r["participation_label"]),
                    str(r.get("label_source", "")),
                )
                for _, r in ann_valid.dropna(subset=["track_id_int", "t0_int", "t1_int"]).iterrows()
            }
            true_labels = []
            gt_sources = []
            matched_flags = []
            for _, row in preds.iterrows():
                key = (int(row["track_id"]), int(round(float(row["t_start_ms"]))), int(round(float(row["t_end_ms"]))))
                hit = gt_map.get(key)
                if hit:
                    true_labels.append(hit[0])
                    gt_sources.append(hit[1])
                    matched_flags.append(row["gated_label"] == hit[0] if row["gated_label"] != UNCERTAIN_LABEL else None)
                else:
                    true_labels.append(None)
                    gt_sources.append(None)
                    matched_flags.append(None)
            preds["true_label"] = true_labels
            preds["label_source"] = gt_sources
            preds["is_correct"] = matched_flags

            gt_sub = preds[preds["true_label"].isin(CANONICAL_TRAINING_LABELS)]
            if not gt_sub.empty:
                y_true = np.array([LABEL_TO_INT[x] for x in gt_sub["true_label"]], dtype=np.int64)
                y_raw = np.array([LABEL_TO_INT[x] for x in gt_sub["pred_label"]], dtype=np.int64)
                raw_m = compute_multiclass_metrics(y_true, y_raw, num_classes=3)
                acc_sub = gt_sub[gt_sub["gated_label"] != UNCERTAIN_LABEL]
                acc_m = None
                if not acc_sub.empty:
                    acc_m = compute_multiclass_metrics(
                        np.array([LABEL_TO_INT[x] for x in acc_sub["true_label"]], dtype=np.int64),
                        np.array([LABEL_TO_INT[x] for x in acc_sub["gated_label"]], dtype=np.int64),
                        num_classes=3,
                    )
                gt_stats = {
                    "matched_gt_windows": int(len(gt_sub)),
                    "gt_accepted_windows": int(len(acc_sub)),
                    "gt_raw_macro_f1": raw_m["macro_f1"],
                    "gt_raw_accuracy": raw_m["accuracy"],
                    "gt_accepted_macro_f1": acc_m["macro_f1"] if acc_m else None,
                    "gt_accepted_accuracy": acc_m["accuracy"] if acc_m else None,
                }

    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    overlay_path: Optional[Path] = None
    if render_video:
        render_poses = raw_poses[raw_poses["timestamp_ms"].between(max(0.0, start_ms - 100.0), end_ms + 100.0)].copy()
        overlay_path = out_dir / "demo_overlay.mp4"
        max_frames = max(1, int(round(float(duration_s) * sample_fps)))
        render_pose_overlay_video(
            video_path=video_path,
            output_path=overlay_path,
            poses_df=render_poses,
            sample_fps=sample_fps,
            conf_threshold=float(qg.get("min_mean_keypoint_confidence", 0.30)),
            blur_face=blur_face,
            max_frames=max_frames,
            transcode_h264=True,
            pipeline_label=f"{ckpt.get('model_type', 'MODEL').upper()} + Gate",
            session_title=f"{session_id} ({int(start_s)}-{int(start_s + duration_s)}s)",
            predictions_df=preds,
            start_ms=start_ms,
            resize_width=resize_width,
        )

    has_gt = gt_stats["matched_gt_windows"] > 0
    export_bundle(
        preds,
        out_dir / "export",
        experiment_id=checkpoint_path.stem,
        metric_scope="human_adjudicated_partial" if has_gt else "inference_no_ground_truth",
        anonymize=True,
    )

    counts = preds["gated_label"].value_counts().to_dict()
    total_w = int(len(preds))
    acc_w = int((preds["gated_label"] != UNCERTAIN_LABEL).sum())
    summary = {
        "session_id": session_id,
        "checkpoint": str(checkpoint_path),
        "model_type": str(ckpt.get("model_type", "tcn")).upper(),
        "start_s": float(start_s),
        "duration_s": float(duration_s),
        "blur_face": bool(blur_face),
        "min_probability": eff_min_prob,
        "temperature": temperature,
        "total_windows": total_w,
        "accepted_windows": acc_w,
        "uncertain_windows": total_w - acc_w,
        "coverage": round(acc_w / max(total_w, 1), 4),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "ground_truth_eval": gt_stats,
        "video_file": str(overlay_path) if overlay_path else None,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    (out_dir / "demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast test-set video demo & render overlay.")
    parser.add_argument("--session", default="session_005", choices=("session_005", "session_006"))
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/20261126-110817-tcn-adjudicated-v03-600.pt",
    )
    parser.add_argument("--config", default="configs/experiments/adjudicated_v03_600.yaml")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "reports" / "demo_test" / f"{args.session}_{int(args.start_s)}s_{int(args.duration_s)}s"
    )
    res = run_test_video_demo(
        session_id=args.session,
        checkpoint_path=Path(args.checkpoint),
        out_dir=out_dir,
        config_path=Path(args.config),
        start_s=args.start_s,
        duration_s=args.duration_s,
        blur_face=not args.no_blur,
        min_probability=args.min_probability,
        render_video=not args.skip_video,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
