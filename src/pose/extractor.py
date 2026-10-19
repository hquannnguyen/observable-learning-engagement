"""Pose extractor module using YOLOv8-Pose.

Implements per-frame keypoint extraction for video sequences according to
docs/dataset_schema.md and configs/baseline.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# COCO 17 human pose keypoint names in order
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Canonical columns defined in docs/dataset_schema.md Section 5
POSE_BASE_COLUMNS = [
    "session_id",
    "camera_id",
    "frame_idx",
    "timestamp_ms",
    "track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_conf",
]

POSE_KEYPOINT_COLUMNS: List[str] = []
for i in range(17):
    POSE_KEYPOINT_COLUMNS.extend([f"kpt_{i:02d}_x", f"kpt_{i:02d}_y", f"kpt_{i:02d}_conf"])

POSE_EXTRA_COLUMNS = ["visible_ratio", "mean_keypoint_conf", "zone_id"]

POSE_COLUMNS: List[str] = POSE_BASE_COLUMNS + POSE_KEYPOINT_COLUMNS + POSE_EXTRA_COLUMNS


def load_pose_config(
    config_path: Union[str, Path, Dict[str, Any]] = "configs/baseline.yaml",
) -> Dict[str, Any]:
    """Load pose extraction settings from YAML or dictionary.

    Args:
        config_path: Path to YAML config or pre-loaded config dict.

    Returns:
        Dict with keys: model, confidence_threshold, sample_fps, tracker.
    """
    if isinstance(config_path, dict):
        raw_cfg = config_path
    else:
        path = Path(config_path)
        if not path.is_file():
            logger.warning("Config file %s not found. Using defaults.", config_path)
            raw_cfg = {}
        else:
            with path.open("r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}

    pose_cfg = raw_cfg.get("pose", {})
    video_cfg = raw_cfg.get("video", {})
    tracking_cfg = raw_cfg.get("tracking", {})

    return {
        "model": pose_cfg.get("model", "yolov8n-pose.pt"),
        "confidence_threshold": float(pose_cfg.get("confidence_threshold", 0.30)),
        "sample_fps": float(video_cfg.get("sample_fps", 10.0)),
        "tracker": tracking_cfg.get("tracker", "bytetrack.yaml"),
        "imgsz": pose_cfg.get("imgsz", None),
    }


def extract_frame_poses(
    frame: np.ndarray,
    model: Any,
    session_id: str,
    camera_id: str,
    frame_idx: int,
    timestamp_ms: int,
    conf_threshold: float = 0.30,
    track: bool = False,
    tracker: str = "bytetrack.yaml",
    zone_id: Optional[str] = None,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract pose detections from a single frame using YOLOv8-Pose.

    Args:
        frame: BGR image (OpenCV format).
        model: Loaded YOLO model instance.
        session_id: Identifier of the session.
        camera_id: Identifier of the camera.
        frame_idx: Index of the sampled frame.
        timestamp_ms: Timestamp in milliseconds from video start.
        conf_threshold: Confidence threshold for keypoint visibility.
        track: Whether to enable tracking (ByteTrack) to get track_id.
        tracker: Tracker config name/path.
        zone_id: Optional spatial zone ID.
        device: Device to run inference on (e.g. 'cuda', 'cpu').
        imgsz: Inference image size (e.g. 1280 for higher precision).

    Returns:
        List of dicts, each representing one detected person/track.
    """
    kwargs: Dict[str, Any] = {"verbose": False, "conf": conf_threshold}
    if imgsz is not None:
        kwargs["imgsz"] = int(imgsz)
    if device is not None:
        kwargs["device"] = device

    if track:
        results = model.track(frame, persist=True, tracker=tracker, **kwargs)[0]
    else:
        results = model.predict(frame, **kwargs)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    num_detections = len(results.boxes)
    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    boxes_conf = results.boxes.conf.cpu().numpy()
    boxes_id = (
        results.boxes.id.cpu().numpy()
        if results.boxes.id is not None
        else None
    )

    has_keypoints = results.keypoints is not None and results.keypoints.data is not None
    if has_keypoints:
        kpt_tensor = results.keypoints.data.cpu().numpy()  # shape (N, 17, 3)
    else:
        kpt_tensor = np.zeros((num_detections, 17, 3), dtype=np.float32)

    rows: List[Dict[str, Any]] = []

    for p in range(num_detections):
        row: Dict[str, Any] = {
            "session_id": str(session_id),
            "camera_id": str(camera_id),
            "frame_idx": int(frame_idx),
            "timestamp_ms": int(timestamp_ms),
            "track_id": int(boxes_id[p]) if boxes_id is not None else None,
            "bbox_x1": round(float(boxes_xyxy[p, 0]), 2),
            "bbox_y1": round(float(boxes_xyxy[p, 1]), 2),
            "bbox_x2": round(float(boxes_xyxy[p, 2]), 2),
            "bbox_y2": round(float(boxes_xyxy[p, 3]), 2),
            "bbox_conf": round(float(boxes_conf[p]), 4),
        }

        p_kpt = kpt_tensor[p]  # shape (17, 3): [x, y, conf]
        valid_kpt_count = 0
        conf_sum = 0.0

        for i in range(17):
            x_val = float(p_kpt[i, 0])
            y_val = float(p_kpt[i, 1])
            c_val = float(p_kpt[i, 2])
            conf_sum += c_val

            # Contract rule: do not encode missing keypoint as valid (0, 0)
            if c_val < conf_threshold or (x_val == 0.0 and y_val == 0.0):
                row[f"kpt_{i:02d}_x"] = np.nan
                row[f"kpt_{i:02d}_y"] = np.nan
            else:
                row[f"kpt_{i:02d}_x"] = round(x_val, 2)
                row[f"kpt_{i:02d}_y"] = round(y_val, 2)
                valid_kpt_count += 1

            row[f"kpt_{i:02d}_conf"] = round(c_val, 4)

        row["visible_ratio"] = round(float(valid_kpt_count / 17.0), 4)
        row["mean_keypoint_conf"] = round(float(conf_sum / 17.0), 4)
        row["zone_id"] = zone_id

        rows.append(row)

    return rows


def extract_poses_from_video(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    session_id: Optional[str] = None,
    camera_id: str = "cam_A",
    config_path: Union[str, Path, Dict[str, Any]] = "configs/baseline.yaml",
    sample_fps: Optional[float] = None,
    conf_threshold: Optional[float] = None,
    track: bool = False,
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    zone_id: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract pose keypoints from a video file and return a DataFrame.

    Args:
        video_path: Path to raw video file.
        output_path: Optional path to save CSV or Parquet file.
        session_id: Session identifier. If None, derived from video filename.
        camera_id: Camera identifier (default 'cam_A').
        config_path: Path to config YAML or preloaded dict.
        sample_fps: Sampling FPS override. If None, loaded from config.
        conf_threshold: Confidence threshold override. If None, loaded from config.
        track: Whether to enable tracking (ByteTrack).
        model: Optional preloaded YOLO model. If None, loaded using config.
        model_name: Optional model file or name override (e.g. 'yolo11m-pose.pt').
        imgsz: Optional inference image size override (e.g. 1280).
        device: Device to use for inference ('cuda', 'cpu').
        zone_id: Optional zone identifier.
        max_frames: Optional cap on sampled frames (useful for testing).

    Returns:
        Tuple of (pd.DataFrame matching schema, metadata dictionary).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Resolve config
    cfg = load_pose_config(config_path)
    eff_sample_fps = float(sample_fps if sample_fps is not None else cfg["sample_fps"])
    eff_conf_thresh = float(
        conf_threshold if conf_threshold is not None else cfg["confidence_threshold"]
    )
    eff_model_name = model_name or cfg["model"]
    eff_imgsz = imgsz if imgsz is not None else cfg.get("imgsz", None)
    tracker_cfg = cfg["tracker"]

    # Resolve session_id
    if session_id is None:
        session_id = video_path.stem

    # Load YOLO model if not provided
    if model is None:
        from ultralytics import YOLO

        logger.info("Loading YOLO pose model: %s (imgsz=%s)", eff_model_name, eff_imgsz)
        model = YOLO(eff_model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    orig_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps <= 0.0:
        orig_fps = 30.0
        logger.warning("Video reported invalid FPS; fallback to 30.0")

    time_step_ms = 1000.0 / eff_sample_fps
    next_sample_time_ms = 0.0

    raw_frame_idx = 0
    sampled_frame_idx = 0
    all_rows: List[Dict[str, Any]] = []

    logger.info(
        "Processing %s (fps=%.1f, frames=%d, sample_fps=%.1f)",
        video_path.name,
        orig_fps,
        total_raw_frames,
        eff_sample_fps,
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_ms = raw_frame_idx * (1000.0 / orig_fps)

            # Sample according to desired sample_fps
            if current_time_ms + 1e-4 >= next_sample_time_ms:
                timestamp_ms = int(round(current_time_ms))

                frame_rows = extract_frame_poses(
                    frame=frame,
                    model=model,
                    session_id=session_id,
                    camera_id=camera_id,
                    frame_idx=sampled_frame_idx,
                    timestamp_ms=timestamp_ms,
                    conf_threshold=eff_conf_thresh,
                    track=track,
                    tracker=tracker_cfg,
                    zone_id=zone_id,
                    device=device,
                    imgsz=eff_imgsz,
                )
                all_rows.extend(frame_rows)

                sampled_frame_idx += 1
                next_sample_time_ms += time_step_ms

                if sampled_frame_idx % 500 == 0:
                    pct = (raw_frame_idx / total_raw_frames * 100.0) if total_raw_frames > 0 else 0.0
                    logger.info("Pose extraction progress: %d sampled frames (%.1f%% of video)", sampled_frame_idx, pct)

                if max_frames is not None and sampled_frame_idx >= max_frames:
                    break

            raw_frame_idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(all_rows, columns=POSE_COLUMNS)

    metadata = {
        "video_path": str(video_path.resolve()),
        "session_id": session_id,
        "camera_id": camera_id,
        "original_fps": orig_fps,
        "sample_fps": eff_sample_fps,
        "resolution": f"{width}x{height}",
        "raw_frames": raw_frame_idx,
        "sampled_frames": sampled_frame_idx,
        "total_detections": len(df),
        "confidence_threshold": eff_conf_thresh,
        "track_enabled": track,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".parquet":
            try:
                df.to_parquet(out, index=False)
                logger.info("Saved pose detections to Parquet: %s", out)
            except (ImportError, ModuleNotFoundError) as err:
                csv_fallback = out.with_suffix(".csv")
                df.to_csv(csv_fallback, index=False)
                logger.warning("Parquet engine not available (%s); saved to CSV instead: %s", err, csv_fallback)
        else:
            df.to_csv(out, index=False)
            logger.info("Saved pose detections to CSV: %s", out)

    return df, metadata


def extract_poses_from_directory(
    raw_dir: Union[str, Path] = "data/raw",
    output_dir: Union[str, Path] = "data/poses",
    camera_id: str = "cam_A",
    config_path: Union[str, Path, Dict[str, Any]] = "configs/baseline.yaml",
    track: bool = False,
    output_format: str = "csv",
    device: Optional[str] = None,
    max_frames_per_video: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Batch extract poses from all video files in a directory.

    Args:
        raw_dir: Directory containing raw video files.
        output_dir: Destination directory for extracted pose CSV/Parquet files.
        camera_id: Camera identifier.
        config_path: Path to baseline config.
        track: Whether to enable tracking.
        output_format: 'csv', 'parquet', or 'both'.
        device: Device to use for inference.
        max_frames_per_video: Optional max sampled frames per video.

    Returns:
        Dict mapping video stem name to its extraction metadata.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = sorted(
        [p for p in raw_path.glob("*") if p.suffix.lower() in video_extensions]
    )

    if not video_files:
        logger.warning("No video files found in %s", raw_dir)
        return {}

    # Preload YOLO model once for the whole batch
    from ultralytics import YOLO

    cfg = load_pose_config(config_path)
    model = YOLO(cfg["model"])

    results_summary: Dict[str, Dict[str, Any]] = {}

    for v_file in video_files:
        session_id = v_file.stem
        out_base = out_path / f"{session_id}_{camera_id}_poses"

        out_target = (
            out_base.with_suffix(".parquet")
            if output_format == "parquet"
            else out_base.with_suffix(".csv")
        )

        df, meta = extract_poses_from_video(
            video_path=v_file,
            output_path=out_target,
            session_id=session_id,
            camera_id=camera_id,
            config_path=config_path,
            track=track,
            model=model,
            device=device,
            max_frames=max_frames_per_video,
        )

        if output_format == "both":
            parquet_target = out_base.with_suffix(".parquet")
            df.to_parquet(parquet_target, index=False)

        results_summary[session_id] = meta

    return results_summary
