"""Tests for pose extraction module."""

import numpy as np
import pytest

from src.pose.extractor import (
    COCO_KEYPOINTS,
    POSE_BASE_COLUMNS,
    POSE_COLUMNS,
    POSE_EXTRA_COLUMNS,
    POSE_KEYPOINT_COLUMNS,
    extract_frame_poses,
    load_pose_config,
)


def test_coco_keypoints_count():
    assert len(COCO_KEYPOINTS) == 17
    assert COCO_KEYPOINTS[0] == "nose"
    assert COCO_KEYPOINTS[16] == "right_ankle"


def test_schema_column_names_and_counts():
    assert len(POSE_BASE_COLUMNS) == 10
    assert len(POSE_KEYPOINT_COLUMNS) == 51  # 17 * 3
    assert len(POSE_EXTRA_COLUMNS) == 3
    assert len(POSE_COLUMNS) == 64

    # Verify keypoint naming format
    for i in range(17):
        assert f"kpt_{i:02d}_x" in POSE_COLUMNS
        assert f"kpt_{i:02d}_y" in POSE_COLUMNS
        assert f"kpt_{i:02d}_conf" in POSE_COLUMNS

    assert "visible_ratio" in POSE_COLUMNS
    assert "mean_keypoint_conf" in POSE_COLUMNS
    assert "zone_id" in POSE_COLUMNS


def test_load_pose_config_defaults():
    cfg = load_pose_config("configs/baseline.yaml")
    assert cfg["model"] == "yolov8n-pose.pt"
    assert cfg["confidence_threshold"] == 0.30
    assert cfg["sample_fps"] == 10.0
    assert "bytetrack" in cfg["tracker"].lower()


class DummyBox:
    def __init__(self, xyxy, conf, track_id=None):
        import torch
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf], dtype=torch.float32)
        self.id = torch.tensor([track_id], dtype=torch.float32) if track_id is not None else None


class DummyBoxes:
    def __init__(self, boxes_list):
        import torch
        self.boxes = boxes_list
        if boxes_list:
            self.xyxy = torch.stack([b.xyxy[0] for b in boxes_list])
            self.conf = torch.stack([b.conf[0] for b in boxes_list])
            if all(b.id is not None for b in boxes_list):
                self.id = torch.stack([b.id[0] for b in boxes_list])
            else:
                self.id = None
        else:
            self.xyxy = torch.empty((0, 4))
            self.conf = torch.empty((0,))
            self.id = None

    def __len__(self):
        return len(self.boxes)


class DummyKeypoints:
    def __init__(self, data_tensor):
        self.data = data_tensor


class DummyResult:
    def __init__(self, boxes, keypoints):
        self.boxes = boxes
        self.keypoints = keypoints


class DummyModel:
    def __init__(self, results):
        self._results = results

    def predict(self, frame, **kwargs):
        return [self._results]

    def track(self, frame, **kwargs):
        return [self._results]


def test_missing_keypoints_encoded_as_nan():
    import torch

    # Create 1 person with keypoint 0 missing (conf=0.1, <0.3 threshold)
    # and keypoint 1 at (0, 0)
    # and keypoint 2 valid at (150, 200, 0.85)
    kp_data = torch.zeros((1, 17, 3), dtype=torch.float32)
    kp_data[0, 0] = torch.tensor([100.0, 100.0, 0.10])  # conf below threshold
    kp_data[0, 1] = torch.tensor([0.0, 0.0, 0.90])      # (0, 0) coords -> missing
    kp_data[0, 2] = torch.tensor([150.0, 200.0, 0.85])  # valid

    dummy_boxes = DummyBoxes([DummyBox([10.0, 20.0, 100.0, 200.0], 0.92, track_id=1)])
    dummy_kpts = DummyKeypoints(kp_data)
    dummy_res = DummyResult(dummy_boxes, dummy_kpts)
    model = DummyModel(dummy_res)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rows = extract_frame_poses(
        frame=dummy_frame,
        model=model,
        session_id="S001",
        camera_id="cam_A",
        frame_idx=0,
        timestamp_ms=0,
        conf_threshold=0.30,
        track=True,
    )

    assert len(rows) == 1
    r = rows[0]

    # kp 0 should be NaN due to low confidence
    assert np.isnan(r["kpt_00_x"])
    assert np.isnan(r["kpt_00_y"])
    assert r["kpt_00_conf"] == 0.10

    # kp 1 should be NaN because coordinates are (0, 0)
    assert np.isnan(r["kpt_01_x"])
    assert np.isnan(r["kpt_01_y"])

    # kp 2 should be valid
    assert r["kpt_02_x"] == 150.0
    assert r["kpt_02_y"] == 200.0
    assert r["kpt_02_conf"] == 0.85

    # Check track_id and bounding box
    assert r["track_id"] == 1
    assert r["bbox_x1"] == 10.0
    assert r["bbox_y1"] == 20.0
    assert r["bbox_conf"] == 0.92
    assert 0.0 <= r["visible_ratio"] <= 1.0
