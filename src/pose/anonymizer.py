"""Anonymization and visualization module for CVF25PRJ10.

Implements Task T33 (Anonymization) and Task T36 (Anonymized video preview).
Privacy contract:
- Head/face ROI is blurred or pixelated.
- Identifiers are anonymous track IDs, never real student names or IDs.
- Skeleton lines visualize observable body posture.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
    to_display_label,
)
from src.pose.extractor import SKELETON_PAIRS, TrackedPose

# State color palette (BGR for OpenCV)
STATE_COLORS_BGR = {
    HIGH: (46, 204, 113),         # Green
    MODERATE: (241, 196, 15),     # Amber/Yellow
    LOW: (52, 73, 94),            # Gray/Dark
    UNCERTAIN_LABEL: (149, 165, 166), # Light Gray
}
DEFAULT_COLOR_BGR = (255, 180, 0)


def blur_head_roi(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    keypoints: np.ndarray,
    blur_kernel_size: int = 31,
) -> np.ndarray:
    """Blur the head/face region using keypoints (nose, eyes, ears) or upper 25% of bbox."""
    h, w = image.shape[:2]
    x1, y1, x2, y3 = bbox

    # Estimate head box from face keypoints if available (points 0..4)
    face_kpts = keypoints[:5]
    valid_face = face_kpts[face_kpts[:, 2] > 0.2]

    if len(valid_face) >= 2:
        hx1 = max(0, int(np.min(valid_face[:, 0]) - 15))
        hy1 = max(0, int(np.min(valid_face[:, 1]) - 15))
        hx2 = min(w, int(np.max(valid_face[:, 0]) + 15))
        hy2 = min(h, int(np.max(valid_face[:, 1]) + 15))
    else:
        # Fallback: top 25% of bounding box
        box_w = x2 - x1
        box_h = y3 - y1
        hx1 = max(0, int(x1 + 0.15 * box_w))
        hy1 = max(0, int(y1))
        hx2 = min(w, int(x2 - 0.15 * box_w))
        hy2 = min(h, int(y1 + 0.28 * box_h))

    if hx2 > hx1 + 4 and hy2 > hy1 + 4:
        head_roi = image[hy1:hy2, hx1:hx2]
        ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        blurred = cv2.GaussianBlur(head_roi, (ksize, ksize), 30)
        image[hy1:hy2, hx1:hx2] = blurred

    return image


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = DEFAULT_COLOR_BGR,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Draw 17-keypoint skeleton lines on the frame."""
    # Draw connections
    for pt1_idx, pt2_idx in SKELETON_PAIRS:
        x1, y1, c1 = keypoints[pt1_idx]
        x2, y2, c2 = keypoints[pt2_idx]
        if c1 >= conf_threshold and c2 >= conf_threshold:
            cv2.line(
                image,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                2,
                cv2.LINE_AA,
            )

    # Draw joint points
    for idx in range(len(keypoints)):
        x, y, c = keypoints[idx]
        if c >= conf_threshold:
            cv2.circle(image, (int(round(x)), int(round(y))), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)

    return image


def render_anonymized_frame(
    frame: np.ndarray,
    poses: List[TrackedPose],
    participation_labels: Optional[Dict[int, str]] = None,
    blur_heads: bool = True,
    draw_skeletons: bool = True,
    draw_boxes: bool = True,
) -> np.ndarray:
    """Render an anonymized frame with head blur, skeleton, and anonymous ID tags."""
    output = frame.copy()
    labels = participation_labels or {}

    for pose in poses:
        color = STATE_COLORS_BGR.get(labels.get(pose.track_id, ""), DEFAULT_COLOR_BGR)

        # 1. Blur head region
        if blur_heads:
            blur_head_roi(output, pose.bbox, pose.keypoints)

        # 2. Draw skeleton
        if draw_skeletons:
            draw_skeleton(output, pose.keypoints, color=color)

        # 3. Draw bounding box and label badge
        if draw_boxes:
            x1, y1, x2, y2 = [int(v) for v in pose.bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # Badge text
            canonical = labels.get(pose.track_id)
            status_text = to_display_label(canonical) if canonical else "DETECTED"
            badge = f"#{pose.track_id} [{status_text}]"

            (tw, th), baseline = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color, -1)
            cv2.putText(
                output,
                badge,
                (x1 + 3, max(th, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return output
