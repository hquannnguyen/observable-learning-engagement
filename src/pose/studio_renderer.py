"""Studio rendering engine for fast classroom video annotation.

Extracts frames via OpenCV (no external ffmpeg required), overlays bounding boxes
and skeletons, and generates 5-keyframe temporal progression strips.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    ANNOTATION_STATES,
)
from src.pose.visualization import (
    COCO_LIMBS,
    COLOR_CENTER,
    COLOR_LEFT,
    COLOR_RIGHT,
    apply_face_blur,
    get_limb_color,
)

logger = logging.getLogger(__name__)

# Colors in BGR
COLOR_LABEL_HIGH = (60, 200, 60)       # Green
COLOR_LABEL_MODERATE = (30, 200, 245)   # Amber / Yellow
COLOR_LABEL_LOW = (50, 50, 235)         # Red
COLOR_LABEL_OCCLUDED = (160, 160, 160)  # Gray
COLOR_LABEL_DEFAULT = (200, 140, 60)    # Soft Blue
COLOR_SELECTED_BORDER = (255, 255, 255) # White border for selected
COLOR_SELECTED_BOX = (0, 255, 255)      # Cyan / Bright Yellow


def get_label_color(label: str, state: str = "VALID") -> Tuple[int, int, int]:
    """Return BGR color corresponding to the label and state."""
    if state == "OCCLUDED" or label == "OCCLUDED":
        return COLOR_LABEL_OCCLUDED
    if label in (HIGH, "HIGH", "CAO"):
        return COLOR_LABEL_HIGH
    if label in (MODERATE, "MODERATE", "VỪA"):
        return COLOR_LABEL_MODERATE
    if label in (LOW, "LOW", "THẤP"):
        return COLOR_LABEL_LOW
    return COLOR_LABEL_DEFAULT


def extract_frame_at_ms(video_path: Path, timestamp_ms: float) -> Optional[np.ndarray]:
    """Extract a single frame from video at given millisecond using OpenCV.
    
    Returns RGB image as numpy array, or None if failed.
    """
    if not video_path.is_file():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = max(0, int(round(timestamp_ms / 1000.0 * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback to pos_msec
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_ms))
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def render_classroom_scene(
    frame_rgb: np.ndarray,
    detections: pd.DataFrame,
    selected_track_id: Optional[str] = None,
    labels_by_track: Optional[Dict[str, Tuple[str, str]]] = None,
    show_skeleton: bool = True,
    blur_face: bool = False,
    conf_threshold: float = 0.30,
    *,
    show_all_skeletons: bool = True,
) -> np.ndarray:
    """Render full classroom scene with colored bounding boxes and skeleton overlays.
    
    Args:
        frame_rgb: RGB frame image.
        detections: DataFrame of detections near current timestamp with track_id, bbox, kpts.
        selected_track_id: ID of the student currently selected.
        labels_by_track: Map of track_id -> (participation_label, annotation_state).
        show_skeleton: Whether to draw skeleton bones and keypoints.
        show_all_skeletons: Also draw skeletons for non-selected tracks when enabled.
        blur_face: Whether to apply privacy blurring to heads.
        conf_threshold: Minimum confidence for drawing keypoints.
        
    Returns:
        RGB image with overlays.
    """
    if labels_by_track is None:
        labels_by_track = {}

    # Work in BGR for OpenCV drawing
    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h_img, w_img = canvas.shape[:2]

    # First pass: Draw non-selected students
    # Second pass: Draw selected student on top
    selected_row = None

    for _, row in detections.iterrows():
        t_id = str(row["track_id"])
        if selected_track_id is not None and str(selected_track_id) == t_id:
            selected_row = row
            continue

        label, state = labels_by_track.get(t_id, ("", "VALID"))
        color = get_label_color(label, state)
        _draw_student_box_and_skeleton(
            canvas, row, color, is_selected=False, show_skeleton=show_skeleton and show_all_skeletons,
            blur_face=blur_face, conf_threshold=conf_threshold, label=label,
        )

    # Draw selected student on top with highlight
    if selected_row is not None:
        t_id = str(selected_row["track_id"])
        label, state = labels_by_track.get(t_id, ("", "VALID"))
        color = get_label_color(label, state)
        _draw_student_box_and_skeleton(
            canvas, selected_row, color, is_selected=True, show_skeleton=show_skeleton,
            blur_face=blur_face, conf_threshold=conf_threshold, label=label,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _draw_student_box_and_skeleton(
    canvas: np.ndarray,
    row: pd.Series,
    color: Tuple[int, int, int],
    is_selected: bool,
    show_skeleton: bool,
    blur_face: bool,
    conf_threshold: float,
    label: str,
):
    """Draw a single student's box and skeleton on canvas."""
    x1, y1 = float(row.get("bbox_x1", 0)), float(row.get("bbox_y1", 0))
    x2, y2 = float(row.get("bbox_x2", 0)), float(row.get("bbox_y2", 0))
    bx1, by1 = int(round(x1)), int(round(y1))
    bx2, by2 = int(round(x2)), int(round(y2))

    # Keypoints array
    kpts = np.zeros((17, 3), dtype=np.float32)
    has_kpts = False
    for i in range(17):
        kx = float(row.get(f"kpt_{i:02d}_x", 0.0))
        ky = float(row.get(f"kpt_{i:02d}_y", 0.0))
        kc = float(row.get(f"kpt_{i:02d}_conf", 0.0))
        kpts[i] = [kx, ky, kc]
        if kc >= conf_threshold and (kx > 0 or ky > 0):
            has_kpts = True

    # Face blur
    if blur_face:
        apply_face_blur(canvas, (x1, y1, x2, y2), keypoints=kpts, conf_threshold=conf_threshold)

    # Skeleton
    if show_skeleton and has_kpts:
        box_w = abs(x2 - x1)
        box_h = abs(y2 - y1)
        max_allowed_len = max(box_w, box_h) * 1.6 + 30.0

        for pt1_idx, pt2_idx in COCO_LIMBS:
            kx1, ky1, c1 = kpts[pt1_idx]
            kx2, ky2, c2 = kpts[pt2_idx]
            if c1 >= conf_threshold and c2 >= conf_threshold and (kx1 > 0 or ky1 > 0) and (kx2 > 0 or ky2 > 0):
                dist = np.hypot(kx2 - kx1, ky2 - ky1)
                if dist <= max_allowed_len:
                    p1 = (int(round(kx1)), int(round(ky1)))
                    p2 = (int(round(kx2)), int(round(ky2)))
                    limb_col = (0, 255, 255) if is_selected else get_limb_color(pt1_idx, pt2_idx)
                    cv2.line(canvas, p1, p2, limb_col, thickness=2 if is_selected else 1, lineType=cv2.LINE_AA)

        for i in range(17):
            kx, ky, kc = kpts[i]
            if kc >= conf_threshold and (kx > 0 or ky > 0):
                pt = (int(round(kx)), int(round(ky)))
                dot_col = (0, 255, 255) if is_selected else (COLOR_LEFT if i % 2 == 1 else COLOR_RIGHT)
                cv2.circle(canvas, pt, radius=4 if is_selected else 3, color=(20, 20, 20), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, pt, radius=3 if is_selected else 2, color=dot_col, thickness=-1, lineType=cv2.LINE_AA)

    # Bounding Box
    if is_selected:
        # Thick outer white border + inner colored border
        cv2.rectangle(canvas, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2), COLOR_SELECTED_BORDER, thickness=3, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness=2, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness=1, lineType=cv2.LINE_AA)

    # Tag Banner
    track_id = str(row["track_id"])
    tag_prefix = f">> #{track_id}" if is_selected else f"#{track_id}"
    short_lbl = ""
    if label in (HIGH, "HIGH"):
        short_lbl = " CAO"
    elif label in (MODERATE, "MODERATE"):
        short_lbl = " VỪA"
    elif label in (LOW, "LOW"):
        short_lbl = " THẤP"
    elif label == "OCCLUDED":
        short_lbl = " CHE"

    tag_text = f"{tag_prefix}{short_lbl}"
    if is_selected:
        tag_text += " <<"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 if not is_selected else 0.55
    thickness = 1 if not is_selected else 2
    (txt_w, txt_h), baseline = cv2.getTextSize(tag_text, font, font_scale, thickness)

    tag_y1 = max(0, by1 - txt_h - 6)
    tag_y2 = by1
    tag_x1 = bx1
    tag_x2 = min(canvas.shape[1], bx1 + txt_w + 10)

    # Tag Background
    cv2.rectangle(canvas, (tag_x1, tag_y1), (tag_x2, tag_y2), (25, 25, 25), thickness=-1)
    tag_border = COLOR_SELECTED_BORDER if is_selected else color
    cv2.rectangle(canvas, (tag_x1, tag_y1), (tag_x2, tag_y2), tag_border, thickness=1)
    text_color = (255, 255, 255) if not is_selected else (0, 255, 255)
    cv2.putText(canvas, tag_text, (tag_x1 + 4, tag_y2 - 4), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def render_progression_strip(
    video_path: Path,
    pose_df: pd.DataFrame,
    track_id: str,
    t_start_ms: float,
    t_end_ms: float,
    show_skeleton: bool = True,
    blur_face: bool = False,
    conf_threshold: float = 0.30,
) -> List[Tuple[str, np.ndarray]]:
    """Render 5 cropped temporal progression keyframes for target student (+0s, +1s, +2s, +3s, +4s).
    
    Returns list of (label, RGB_image).
    """
    offsets = [0.0, 1.0, 2.0, 3.0, 4.0]
    total_duration_s = max(0.1, (t_end_ms - t_start_ms) / 1000.0)
    
    # Scale offsets if window is not 4.0s
    scaled_offsets = [off * (total_duration_s / 4.0) for off in offsets]
    timepoints = [t_start_ms + off * 1000.0 for off in scaled_offsets]

    results = []

    # Filter pose dataframe for target track
    track_poses = pose_df[pose_df["track_id"].astype(str) == str(track_id)]

    for idx, t_ms in enumerate(timepoints):
        label_text = f"+{scaled_offsets[idx]:.1f}s ({t_ms / 1000.0:.1f}s)"

        # Find closest detection within 400ms
        detection = None
        if not track_poses.empty:
            diffs = (track_poses["timestamp_ms"] - t_ms).abs()
            min_diff_idx = diffs.idxmin()
            if diffs.loc[min_diff_idx] <= 400.0:
                detection = track_poses.loc[min_diff_idx]

        # Extract frame
        full_frame_rgb = extract_frame_at_ms(video_path, t_ms)

        if full_frame_rgb is None:
            placeholder = _create_placeholder(f"{label_text}\nKhông mở được video")
            results.append((label_text, placeholder))
            continue

        if detection is None:
            placeholder = _create_placeholder(f"{label_text}\nBỊ CHE / KHÔNG THẤY")
            results.append((label_text, placeholder))
            continue

        # Crop student with 25% margin
        h_img, w_img = full_frame_rgb.shape[:2]
        x1, y1 = float(detection.get("bbox_x1", 0)), float(detection.get("bbox_y1", 0))
        x2, y2 = float(detection.get("bbox_x2", 0)), float(detection.get("bbox_y2", 0))
        bw = max(10.0, x2 - x1)
        bh = max(10.0, y2 - y1)
        mx = bw * 0.25
        my = bh * 0.20

        cx1 = int(max(0, x1 - mx))
        cy1 = int(max(0, y1 - my))
        cx2 = int(min(w_img, x2 + mx))
        cy2 = int(min(h_img, y2 + my))

        if cx2 <= cx1 + 10 or cy2 <= cy1 + 10:
            placeholder = _create_placeholder(f"{label_text}\nBị che ngoài khung")
            results.append((label_text, placeholder))
            continue

        crop = full_frame_rgb[cy1:cy2, cx1:cx2].copy()
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Draw skeleton on cropped image with adjusted coordinates
        kpts = np.zeros((17, 3), dtype=np.float32)
        has_kpts = False
        for i in range(17):
            kx = float(detection.get(f"kpt_{i:02d}_x", 0.0)) - cx1
            ky = float(detection.get(f"kpt_{i:02d}_y", 0.0)) - cy1
            kc = float(detection.get(f"kpt_{i:02d}_conf", 0.0))
            kpts[i] = [kx, ky, kc]
            if kc >= conf_threshold and (kx > 0 or ky > 0):
                has_kpts = True

        if blur_face:
            rel_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
            apply_face_blur(crop_bgr, rel_box, keypoints=kpts, conf_threshold=conf_threshold)

        if show_skeleton and has_kpts:
            max_len = max(bw, bh) * 1.6 + 30.0
            for pt1_idx, pt2_idx in COCO_LIMBS:
                kx1, ky1, c1 = kpts[pt1_idx]
                kx2, ky2, c2 = kpts[pt2_idx]
                if c1 >= conf_threshold and c2 >= conf_threshold and (kx1 > 0 or ky1 > 0) and (kx2 > 0 or ky2 > 0):
                    if np.hypot(kx2 - kx1, ky2 - ky1) <= max_len:
                        p1 = (int(round(kx1)), int(round(ky1)))
                        p2 = (int(round(kx2)), int(round(ky2)))
                        cv2.line(crop_bgr, p1, p2, (0, 215, 255), thickness=2, lineType=cv2.LINE_AA)

            for i in range(17):
                kx, ky, kc = kpts[i]
                if kc >= conf_threshold and (kx > 0 or ky > 0):
                    pt = (int(round(kx)), int(round(ky)))
                    cv2.circle(crop_bgr, pt, radius=4, color=(20, 20, 20), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(crop_bgr, pt, radius=3, color=(0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        # Resize to standard height (e.g. 220px) keeping aspect ratio
        target_h = 220
        aspect = crop_bgr.shape[1] / max(1, crop_bgr.shape[0])
        target_w = max(100, int(round(target_h * aspect)))
        resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        results.append((label_text, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

    return results


def _create_placeholder(text: str, width: int = 180, height: int = 220) -> np.ndarray:
    """Create a dark placeholder image with text."""
    img = np.full((height, width, 3), 35, dtype=np.uint8)
    lines = text.split("\n")
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = height // 2 - len(lines) * 12
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, 0.45, 1)
        x = max(5, (width - tw) // 2)
        cv2.putText(img, line, (x, y), font, 0.45, (160, 160, 160), 1, lineType=cv2.LINE_AA)
        y += 24
    return img
