"""Pose and skeleton visualization module.

Provides functions to render 17 COCO keypoints, skeleton limbs,
bounding boxes, track IDs, and privacy face-blur onto video frames.
Can write annotated overlay videos to MP4.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# COCO 17 Keypoint Limb Connections (pairs of keypoint indices)
COCO_LIMBS: List[Tuple[int, int]] = [
    (0, 1), (0, 2),   # Nose to Left/Right Eye
    (1, 3), (2, 4),   # Eyes to Left/Right Ear
    (5, 6),           # Left Shoulder to Right Shoulder
    (5, 7), (7, 9),   # Left Shoulder -> Left Elbow -> Left Wrist
    (6, 8), (8, 10),  # Right Shoulder -> Right Elbow -> Right Wrist
    (5, 11), (6, 12), # Left/Right Shoulder to Hip (torso sides)
    (11, 12),         # Left Hip to Right Hip
    (11, 13), (13, 15), # Left Hip -> Left Knee -> Left Ankle
    (12, 14), (14, 16), # Right Hip -> Right Knee -> Right Ankle
]

# Color palette in BGR
COLOR_LEFT = (235, 130, 20)      # Electric Blue / Cyan
COLOR_RIGHT = (30, 160, 245)     # Warm Amber / Orange
COLOR_CENTER = (50, 205, 50)     # Emerald Green
COLOR_BBOX = (180, 105, 255)     # Soft Purple
COLOR_BOX_BG = (40, 40, 40)      # Dark Slate for tags
COLOR_TEXT = (255, 255, 255)     # White

PARTICIPATION_DISPLAY_SHORT = {
    "HIGH_OBSERVED_PARTICIPATION": "HIGH",
    "MODERATE_OBSERVED_PARTICIPATION": "MODERATE",
    "LOW_OBSERVED_PARTICIPATION": "LOW",
    "UNCERTAIN": "UNCERTAIN",
    "HIGH": "HIGH",
    "MODERATE": "MODERATE",
    "LOW": "LOW",
}
PARTICIPATION_BGR_COLORS = {
    "HIGH": (60, 195, 50),        # Green
    "MODERATE": (0, 195, 255),    # Amber/Yellow
    "LOW": (55, 55, 235),         # Red
    "UNCERTAIN": (145, 145, 145), # Slate Gray
}


def get_limb_color(pt1_idx: int, pt2_idx: int) -> Tuple[int, int, int]:
    """Return distinct color based on left/right/center body anatomy."""
    # Left side is odd keypoints (1, 3, 5, 7, 9, 11, 13, 15)
    # Right side is even keypoints (2, 4, 6, 8, 10, 12, 14, 16)
    if pt1_idx % 2 == 1 and pt2_idx % 2 == 1:
        return COLOR_LEFT
    elif pt1_idx % 2 == 0 and pt2_idx % 2 == 0 and pt1_idx > 0 and pt2_idx > 0:
        return COLOR_RIGHT
    else:
        return COLOR_CENTER


def apply_face_blur(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    keypoints: Optional[np.ndarray] = None,
    conf_threshold: float = 0.30,
) -> None:
    """Anonymize detected person's face/head using Gaussian blur.

    Args:
        frame: BGR image to modify in place.
        bbox: (x1, y1, x2, y2) bounding box.
        keypoints: Optional (17, 3) keypoints array [x, y, conf].
        conf_threshold: Confidence threshold for facial keypoints.
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Estimate head box from face keypoints (0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear)
    head_pts: List[Tuple[float, float]] = []
    if keypoints is not None:
        for i in range(5):
            if keypoints[i, 2] >= conf_threshold and keypoints[i, 0] > 0 and keypoints[i, 1] > 0:
                head_pts.append((keypoints[i, 0], keypoints[i, 1]))

    if len(head_pts) >= 2:
        hx_coords = [p[0] for p in head_pts]
        hy_coords = [p[1] for p in head_pts]
        face_width = max(hx_coords) - min(hx_coords)
        margin_x = max(25.0, face_width * 0.75, (x2 - x1) * 0.12)
        # Eye/nose keypoints alone form a narrow strip. Include the complete
        # head/face rather than blurring only the eye line.
        margin_y = max(40.0, face_width * 0.85, (y2 - y1) * 0.15)
        hx1 = int(max(0, min(hx_coords) - margin_x))
        hy1 = int(max(0, min(hy_coords) - margin_y))
        hx2 = int(min(w_img, max(hx_coords) + margin_x))
        hy2 = int(min(h_img, max(hy_coords) + margin_y))
    else:
        # Fallback: top 25% of the person's bounding box
        box_w = x2 - x1
        box_h = y2 - y1
        hx1 = int(max(0, x1))
        hy1 = int(max(0, y1))
        hx2 = int(min(w_img, x2))
        hy2 = int(min(h_img, y1 + box_h * 0.28))

    if hx2 > hx1 + 4 and hy2 > hy1 + 4:
        head_roi = frame[hy1:hy2, hx1:hx2]
        # Kernel size must be positive odd integer
        small = cv2.resize(head_roi, (max(1, (hx2 - hx1) // 16), max(1, (hy2 - hy1) // 16)), interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(cv2.GaussianBlur(small, (15, 15), 0), (hx2 - hx1, hy2 - hy1), interpolation=cv2.INTER_LINEAR)
        frame[hy1:hy2, hx1:hx2] = blurred


def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    track_id: Optional[int] = None,
    conf_threshold: float = 0.30,
    blur_face: bool = False,
    show_bbox: bool = True,
    participation_label: Optional[str] = None,
    confidence: Optional[float] = None,
) -> None:
    """Draw 17 COCO keypoints, connecting limbs, bounding box, and track/label tag."""
    # 1. Privacy Anonymization if requested
    if blur_face and bbox is not None:
        apply_face_blur(frame, bbox, keypoints=keypoints, conf_threshold=conf_threshold)

    # 2. Draw Skeleton Limbs
    for pt1_idx, pt2_idx in COCO_LIMBS:
        x1, y1, c1 = keypoints[pt1_idx]
        x2, y2, c2 = keypoints[pt2_idx]

        if (
            not np.isnan(x1) and not np.isnan(x2)
            and c1 >= conf_threshold and c2 >= conf_threshold
            and not (x1 == 0 and y1 == 0) and not (x2 == 0 and y2 == 0)
        ):
            # Sanity check: prevent aberrant limb stretching across the whole frame
            if bbox is not None:
                box_w = abs(bbox[2] - bbox[0])
                box_h = abs(bbox[3] - bbox[1])
                max_allowed_len = max(box_w, box_h) * 1.6 + 30.0
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist > max_allowed_len:
                    continue

            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            color = get_limb_color(pt1_idx, pt2_idx)
            cv2.line(frame, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

    # 3. Draw Keypoint Dots
    for i in range(17):
        x, y, c = keypoints[i]
        if not np.isnan(x) and not np.isnan(y) and c >= conf_threshold and not (x == 0 and y == 0):
            pt = (int(round(x)), int(round(y)))
            # Outer dark ring for contrast, inner bright dot
            color = COLOR_LEFT if i % 2 == 1 else (COLOR_RIGHT if i > 0 else COLOR_CENTER)
            cv2.circle(frame, pt, radius=4, color=(20, 20, 20), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # 4. Draw Bounding Box & Track ID / Participation Label Tag
    if show_bbox and bbox is not None:
        bx1, by1, bx2, by2 = [int(round(v)) for v in bbox]
        short_lbl = PARTICIPATION_DISPLAY_SHORT.get(str(participation_label).strip(), "") if participation_label else ""
        box_color = PARTICIPATION_BGR_COLORS.get(short_lbl, COLOR_BBOX)
        box_thick = 2 if short_lbl else 1
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, thickness=box_thick, lineType=cv2.LINE_AA)

        # Label tag banner
        base_id = f"#{track_id}" if track_id is not None else "Person"
        if short_lbl:
            prob_part = f" ({confidence:.2f})" if confidence is not None else ""
            label_text = f"{base_id} {short_lbl}{prob_part}"
        else:
            label_text = f"ID: {track_id}" if track_id is not None else "Person"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48 if short_lbl else 0.45
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        tag_y1 = max(0, by1 - txt_h - 6)
        tag_y2 = by1
        tag_x1 = bx1
        tag_x2 = min(frame.shape[1], bx1 + txt_w + 10)

        tag_bg = box_color if short_lbl else COLOR_BOX_BG
        txt_col = (15, 15, 15) if short_lbl in ("HIGH", "MODERATE") else COLOR_TEXT
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), tag_bg, thickness=-1)
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), box_color, thickness=1)
        cv2.putText(
            frame,
            label_text,
            (tag_x1 + 5, tag_y2 - 4),
            font,
            font_scale,
            txt_col,
            thickness,
            lineType=cv2.LINE_AA,
        )


def draw_hud_header(
    frame: np.ndarray,
    session_title: str,
    frame_idx: int,
    timestamp_ms: int,
    active_tracks: int,
    pipeline_label: str = "YOLO Pose + tracking",
    label_summary: Optional[str] = None,
) -> None:
    """Draw top HUD information bar for professional presentation."""
    h_img, w_img = frame.shape[:2]
    header_h = 36

    # Semi-transparent overlay top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, header_h), (25, 25, 30), thickness=-1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Bottom border line
    cv2.line(frame, (0, header_h), (w_img, header_h), (0, 200, 120), thickness=1)

    time_sec = timestamp_ms / 1000.0
    time_str = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}.{int((time_sec % 1) * 10):01d}"
    hud_text = (
        f"Session: {session_title}  |  Time: {time_str}  |  "
        f"Frame: #{frame_idx}  |  Tracks: {active_tracks}  |  {pipeline_label}"
    )
    if label_summary:
        hud_text += f"  |  {label_summary}"

    cv2.putText(
        frame,
        hud_text,
        (16, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
        lineType=cv2.LINE_AA,
    )


def _build_prediction_lookup(
    predictions_df: Optional[pd.DataFrame],
) -> Dict[int, List[Tuple[float, float, str, Optional[float]]]]:
    """Index per-window model predictions by track_id for fast timestamp lookup."""
    lookup: Dict[int, List[Tuple[float, float, str, Optional[float]]]] = {}
    if predictions_df is None or predictions_df.empty:
        return lookup
    label_col = "gated_label" if "gated_label" in predictions_df.columns else "pred_label"
    for _, row in predictions_df.iterrows():
        try:
            tid = int(row["track_id"])
            t0 = float(row["t_start_ms"])
            t1 = float(row["t_end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        lbl = str(row.get(label_col, "") or "")
        prob_raw = row.get("p_max")
        prob = float(prob_raw) if prob_raw is not None and not pd.isna(prob_raw) else None
        lookup.setdefault(tid, []).append((t0, t1, lbl, prob))
    return lookup


def _lookup_track_prediction(
    lookup: Dict[int, List[Tuple[float, float, str, Optional[float]]]],
    track_id: Optional[int],
    timestamp_ms: float,
) -> Tuple[Optional[str], Optional[float]]:
    """Pick the prediction window containing timestamp_ms (preferring window center proximity)."""
    if track_id is None or track_id not in lookup:
        return None, None
    best: Optional[Tuple[float, str, Optional[float]]] = None
    for t0, t1, lbl, prob in lookup[track_id]:
        if t0 <= timestamp_ms <= t1:
            center_dist = abs(timestamp_ms - 0.5 * (t0 + t1))
            if best is None or center_dist < best[0]:
                best = (center_dist, lbl, prob)
    if best is None:
        return None, None
    return best[1], best[2]


def render_pose_overlay_video(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    poses_df: Optional[pd.DataFrame] = None,
    sample_fps: Optional[float] = None,
    conf_threshold: float = 0.30,
    blur_face: bool = False,
    max_frames: Optional[int] = None,
    transcode_h264: bool = True,
    pipeline_label: str = "YOLO Pose + tracking",
    session_title: Optional[str] = None,
    predictions_df: Optional[pd.DataFrame] = None,
    start_ms: float = 0.0,
    resize_width: Optional[int] = None,
) -> Path:
    """Render a video with skeleton keypoints, limbs, track IDs, and optional predictions."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Infer sample_fps from pose data timestamps if not explicitly given
    inferred_fps = None
    if poses_df is not None and not poses_df.empty and "timestamp_ms" in poses_df.columns:
        unique_ts = np.sort(poses_df["timestamp_ms"].unique())
        if len(unique_ts) >= 2:
            med_diff = float(np.median(np.diff(unique_ts)))
            if med_diff > 0:
                inferred_fps = round(1000.0 / med_diff, 1)

    eff_sample_fps = sample_fps if sample_fps is not None else (inferred_fps or 10.0)
    frame_step = max(1, int(round(native_fps / eff_sample_fps)))
    out_fps = float(eff_sample_fps)

    out_w, out_h = width, height
    if resize_width and resize_width > 0 and width > resize_width:
        out_w = int(resize_width)
        out_h = max(2, int(round(height * out_w / max(width, 1))) // 2 * 2)

    logger.info(
        "Rendering pose video: %s -> %s (Resolution: %dx%d -> %dx%d, Native FPS: %.2f, Sample FPS: %.2f, Step: %d)",
        video_path.name,
        output_path.name,
        width,
        height,
        out_w,
        out_h,
        native_fps,
        out_fps,
        frame_step,
    )

    poses_by_frame_idx: Dict[int, List[Dict[str, Any]]] = {}
    poses_by_ts: Dict[int, List[Dict[str, Any]]] = {}
    if poses_df is not None and not poses_df.empty:
        for row in poses_df.to_dict(orient="records"):
            f_idx = int(row["frame_idx"])
            ts = int(row.get("timestamp_ms", 0))
            poses_by_frame_idx.setdefault(f_idx, []).append(row)
            poses_by_ts.setdefault(ts, []).append(row)

    pred_lookup = _build_prediction_lookup(predictions_df)

    temp_out = output_path.with_name(f"_raw_{output_path.stem}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_out), fourcc, out_fps, (out_w, out_h))

    if not writer.isOpened():
        raise IOError(f"Failed to initialize VideoWriter for {temp_out}")

    start_native_idx = max(0, int(round((max(0.0, float(start_ms)) / 1000.0) * native_fps)))
    if start_native_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_native_idx)

    current_native_idx = start_native_idx
    sampled_frame_count = 0
    session_title = session_title or video_path.stem
    max_target = max_frames
    next_sample_time_ms = max(0.0, float(start_ms))

    try:
        while True:
            current_time_ms = (current_native_idx / native_fps) * 1000.0
            if current_time_ms + 1e-4 < next_sample_time_ms:
                if not cap.grab():
                    break
                current_native_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            global_sample_idx = int(round((current_time_ms / 1000.0) * eff_sample_fps))
            frame_poses = poses_by_frame_idx.get(global_sample_idx, [])
            if not frame_poses and poses_by_ts:
                closest_ts = min(poses_by_ts.keys(), key=lambda t: abs(t - current_time_ms))
                if abs(closest_ts - current_time_ms) <= 80:
                    frame_poses = poses_by_ts[closest_ts]

            counts_hud = {"HIGH": 0, "MODERATE": 0, "LOW": 0, "UNCERTAIN": 0}
            for p_row in frame_poses:
                bbox = (
                    float(p_row.get("bbox_x1", 0)),
                    float(p_row.get("bbox_y1", 0)),
                    float(p_row.get("bbox_x2", 0)),
                    float(p_row.get("bbox_y2", 0)),
                )
                t_id = p_row.get("track_id")
                t_id = None if pd.isna(t_id) else int(t_id)

                kpts = np.zeros((17, 3), dtype=np.float32)
                for k in range(17):
                    kpts[k, 0] = float(p_row.get(f"kpt_{k:02d}_x", np.nan))
                    kpts[k, 1] = float(p_row.get(f"kpt_{k:02d}_y", np.nan))
                    kpts[k, 2] = float(p_row.get(f"kpt_{k:02d}_conf", 0.0))

                part_lbl, pred_prob = _lookup_track_prediction(pred_lookup, t_id, current_time_ms)
                short_lbl = PARTICIPATION_DISPLAY_SHORT.get(part_lbl or "", "")
                if short_lbl in counts_hud:
                    counts_hud[short_lbl] += 1

                draw_skeleton_on_frame(
                    frame=frame,
                    keypoints=kpts,
                    bbox=bbox,
                    track_id=t_id,
                    conf_threshold=conf_threshold,
                    blur_face=blur_face,
                    participation_label=part_lbl,
                    confidence=pred_prob,
                )

            label_summary = (
                f"H:{counts_hud['HIGH']} M:{counts_hud['MODERATE']} L:{counts_hud['LOW']} U:{counts_hud['UNCERTAIN']}"
                if pred_lookup
                else None
            )
            draw_hud_header(
                frame=frame,
                session_title=session_title,
                frame_idx=global_sample_idx,
                timestamp_ms=int(round(current_time_ms)),
                active_tracks=len(frame_poses),
                pipeline_label=pipeline_label,
                label_summary=label_summary,
            )

            out_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA) if (out_w != width or out_h != height) else frame
            writer.write(out_frame)
            sampled_frame_count += 1
            next_sample_time_ms += 1000.0 / eff_sample_fps

            if max_target is not None and sampled_frame_count >= max_target:
                logger.info("Reached target frame limit (%d)", max_target)
                break

            current_native_idx += 1

    finally:
        cap.release()
        writer.release()

    # Transcode to standard H.264 using ffmpeg for wide playback compatibility
    if transcode_h264:
        try:
            from src.pose.video_codec import ffmpeg_executable
            cmd = [
                ffmpeg_executable(), "-y",
                "-i", str(temp_out),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-preset", "fast",
                "-crf", "22",
                str(output_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if output_path.exists():
                temp_out.unlink(missing_ok=True)
                logger.info("Successfully rendered and transcoded H.264 video: %s", output_path)
                return output_path
        except Exception as e:
            logger.warning("ffmpeg transcode failed or skipped: %s. Keeping mp4v file.", e)
            temp_out.rename(output_path)
            return output_path
    else:
        temp_out.rename(output_path)

    return output_path
