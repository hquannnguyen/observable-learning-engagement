"""YOLOv8-Pose extraction and ByteTrack integration for CVF25PRJ10.

Implements Task T06 (YOLOv8-Pose extractor) and Task T07 (ByteTrack integration).
Outputs conform to the pose schema defined in docs/dataset_schema.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# COCO 17 Keypoint names
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

# COCO 17 Keypoint skeleton connections
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12),                         # Torso
    (11, 12),                                 # Hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]


def assign_zone_id(bbox: Tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    """Assign 3x3 classroom zone based on normalized center of bounding box.

    Camera layout follows docs/data_collection_protocol.md:
    FRONT (closest to camera, lower Y): F-L, F-C, F-R
    MIDDLE (intermediate rows):        M-L, M-C, M-R
    BACK (furthest rows, upper Y):     B-L, B-C, B-R
    """
    if img_w <= 0 or img_h <= 0:
        return "UNKNOWN"

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h

    # Y-axis (Row depth in typical elevated front classroom camera)
    # Higher Y in image coordinates = closer to camera (FRONT)
    if cy >= 0.66:
        row = "F"
    elif cy >= 0.33:
        row = "M"
    else:
        row = "B"

    # X-axis (Column)
    if cx < 0.33:
        col = "L"
    elif cx <= 0.66:
        col = "C"
    else:
        col = "R"

    return f"{row}-{col}"


@dataclass
class TrackedPose:
    """Representation of a single tracked student's pose in a frame."""
    frame_idx: int
    timestamp_ms: int
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    bbox_conf: float
    keypoints: np.ndarray  # Shape: (17, 3) -> [x, y, conf]
    visible_ratio: float
    mean_keypoint_conf: float
    zone_id: str

    def to_dict(self, session_id: str = "session_001", camera_id: str = "cam_A") -> Dict[str, Any]:
        """Convert to dict matching docs/dataset_schema.md Section 5."""
        row: Dict[str, Any] = {
            "session_id": session_id,
            "camera_id": camera_id,
            "frame_idx": self.frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "track_id": self.track_id,
            "bbox_x1": round(self.bbox[0], 2),
            "bbox_y1": round(self.bbox[1], 2),
            "bbox_x2": round(self.bbox[2], 2),
            "bbox_y2": round(self.bbox[3], 2),
            "bbox_conf": round(self.bbox_conf, 4),
        }
        for idx in range(17):
            x, y, conf = self.keypoints[idx]
            row[f"kpt_{idx:02d}_x"] = round(float(x), 2)
            row[f"kpt_{idx:02d}_y"] = round(float(y), 2)
            row[f"kpt_{idx:02d}_conf"] = round(float(conf), 4)

        row["visible_ratio"] = round(self.visible_ratio, 4)
        row["mean_keypoint_conf"] = round(self.mean_keypoint_conf, 4)
        row["zone_id"] = self.zone_id
        return row


class PoseExtractor:
    """YOLOv8-Pose extractor with integrated ByteTrack."""

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.30,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = YOLO(model_name)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: int,
        tracker: str = "bytetrack.yaml",
    ) -> List[TrackedPose]:
        """Process a single frame with pose tracking."""
        img_h, img_w = frame.shape[:2]
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        tracked_poses: List[TrackedPose] = []
        if not results or len(results) == 0:
            return tracked_poses

        res = results[0]
        if res.boxes is None or res.keypoints is None:
            return tracked_poses

        boxes = res.boxes
        keypoints = res.keypoints

        # Handle case when boxes or keypoints are empty
        if len(boxes) == 0 or len(keypoints) == 0:
            return tracked_poses

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None

        kpts_data = keypoints.data.cpu().numpy()  # Shape: (N, 17, 3) -> x, y, conf

        for i in range(len(boxes)):
            track_id = int(ids[i]) if ids is not None else (i + 1)
            bbox = (float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3]))
            bbox_conf = float(confs[i])
            kpts = kpts_data[i]  # (17, 3)

            kpt_confs = kpts[:, 2]
            valid_mask = kpt_confs >= self.confidence_threshold
            visible_ratio = float(np.mean(valid_mask))
            mean_kpt_conf = float(np.mean(kpt_confs))

            zone_id = assign_zone_id(bbox, img_w, img_h)

            tracked_poses.append(
                TrackedPose(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    track_id=track_id,
                    bbox=bbox,
                    bbox_conf=bbox_conf,
                    keypoints=kpts,
                    visible_ratio=visible_ratio,
                    mean_keypoint_conf=mean_kpt_conf,
                    zone_id=zone_id,
                )
            )

        return tracked_poses

    def process_video(
        self,
        video_path: str | Path,
        sample_fps: int = 10,
        max_duration_seconds: Optional[float] = None,
        start_seconds: float = 0.0,
    ) -> Generator[Tuple[int, int, np.ndarray, List[TrackedPose]], None, None]:
        """Yield (frame_idx, timestamp_ms, frame, tracked_poses) for sampled frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(round(native_fps / sample_fps)))

        start_frame = int(start_seconds * native_fps)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame_idx = start_frame
        sampled_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time_s = current_frame_idx / native_fps
                if max_duration_seconds is not None and (current_time_s - start_seconds) >= max_duration_seconds:
                    break

                if (current_frame_idx - start_frame) % frame_interval == 0:
                    timestamp_ms = int(current_time_s * 1000)
                    poses = self.process_frame(frame, sampled_idx, timestamp_ms)
                    yield (sampled_idx, timestamp_ms, frame, poses)
                    sampled_idx += 1

                current_frame_idx += 1
        finally:
            cap.release()
