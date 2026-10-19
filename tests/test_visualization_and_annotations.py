"""Unit tests for pose visualization and annotation generation."""

import numpy as np
import pandas as pd
import pytest

from src.contracts.labels import CANONICAL_TRAINING_LABELS
from src.pose.visualization import (
    apply_face_blur,
    draw_hud_header,
    draw_skeleton_on_frame,
    get_limb_color,
)
from scripts.dataset_qa import validate_annotation_dataframe
from scripts.generate_annotations import analyze_track_interval


def test_draw_skeleton_on_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    # Nose
    keypoints[0] = [320, 100, 0.9]
    # Eyes
    keypoints[1] = [310, 95, 0.9]
    keypoints[2] = [330, 95, 0.9]
    # Shoulders
    keypoints[5] = [280, 160, 0.9]
    keypoints[6] = [360, 160, 0.9]

    bbox = (260.0, 80.0, 380.0, 400.0)

    # Test rendering without crash
    draw_skeleton_on_frame(
        frame=frame,
        keypoints=keypoints,
        bbox=bbox,
        track_id=1,
        conf_threshold=0.30,
        blur_face=True,
    )

    # Frame should have non-zero pixels
    assert np.any(frame > 0)


def test_hud_header():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud_header(
        frame=frame,
        session_title="class_1",
        frame_idx=10,
        timestamp_ms=1000,
        active_tracks=2,
    )
    # HUD banner modified the top 36 rows
    assert np.any(frame[:36] > 0)


def test_analyze_track_interval():
    # Synthetic track interval
    rows = []
    for f in range(10):
        rows.append({
            "timestamp_ms": f * 100,
            "visible_ratio": 0.85,
            "mean_keypoint_conf": 0.80,
            "kpt_00_y": 100.0,
            "kpt_00_x": 320.0,
            "kpt_01_x": 315.0,
            "kpt_02_x": 325.0,
            "kpt_05_y": 160.0,
            "kpt_06_y": 160.0,
            "kpt_09_y": 180.0,
            "kpt_10_y": 180.0,
        })
    df = pd.DataFrame(rows)
    behavior, label, state, conf, occl = analyze_track_interval(df)
    assert state == "VALID"
    assert label in CANONICAL_TRAINING_LABELS
    assert conf > 0.5


def test_saved_annotations_qa():
    # Load master annotations file
    ann_file = "data/annotations/annotations_all_v0.1.csv"
    df = pd.read_csv(ann_file)
    is_valid, errors = validate_annotation_dataframe(df)
    assert is_valid, f"Annotations QA failed with errors: {errors}"
