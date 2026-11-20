"""Unit tests for newly implemented pose, baseline heuristic, and reporting modules."""

import numpy as np
import pytest

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
    CANONICAL_TRAINING_LABELS,
)
from src.models.baseline_heuristic import (
    HeuristicBaselineClassifier,
    WindowAssessment,
)
from src.pose.anonymizer import blur_head_roi, draw_skeleton, render_anonymized_frame
from src.pose.extractor import TrackedPose, assign_zone_id
from src.reporting.aggregator import (
    build_results_dataframe,
    build_timeline_dataframe,
    build_zone_summary_dataframe,
)
from src.uncertainty.gate import UncertaintyThresholds


def test_assign_zone_id_geometry():
    img_w, img_h = 1000, 1000
    # Bottom-left (FRONT-LEFT)
    assert assign_zone_id((100, 700, 200, 900), img_w, img_h) == "F-L"
    # Center (MIDDLE-CENTER)
    assert assign_zone_id((450, 450, 550, 550), img_w, img_h) == "M-C"
    # Top-right (BACK-RIGHT)
    assert assign_zone_id((700, 100, 900, 200), img_w, img_h) == "B-R"


def test_tracked_pose_to_dict():
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.8
    pose = TrackedPose(
        frame_idx=10,
        timestamp_ms=1000,
        track_id=1,
        bbox=(50, 60, 150, 200),
        bbox_conf=0.95,
        keypoints=kpts,
        visible_ratio=1.0,
        mean_keypoint_conf=0.8,
        zone_id="F-C",
    )
    d = pose.to_dict(session_id="S01", camera_id="A")
    assert d["session_id"] == "S01"
    assert d["track_id"] == 1
    assert d["zone_id"] == "F-C"
    assert "kpt_00_x" in d
    assert "kpt_16_conf" in d


def test_anonymizer_renders_without_crash():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [100, 40, 0.9]  # nose
    kpts[1] = [95, 35, 0.9]   # L eye
    kpts[2] = [105, 35, 0.9]  # R eye
    kpts[5] = [80, 80, 0.9]   # L shoulder
    kpts[6] = [120, 80, 0.9]  # R shoulder

    blurred = blur_head_roi(img.copy(), (80, 30, 120, 180), kpts)
    assert blurred.shape == img.shape

    with_skel = draw_skeleton(img.copy(), kpts)
    assert with_skel.shape == img.shape

    pose = TrackedPose(
        frame_idx=1,
        timestamp_ms=100,
        track_id=1,
        bbox=(80, 30, 120, 180),
        bbox_conf=0.9,
        keypoints=kpts,
        visible_ratio=0.8,
        mean_keypoint_conf=0.8,
        zone_id="M-C",
    )
    rendered = render_anonymized_frame(img, [pose], participation_labels={1: HIGH})
    assert rendered.shape == img.shape


def test_heuristic_classifier_uncertain_on_low_visibility():
    classifier = HeuristicBaselineClassifier(
        thresholds=UncertaintyThresholds(min_visible_ratio=0.60)
    )
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Visible ratio 0.2 < 0.60
    pose = TrackedPose(
        frame_idx=1,
        timestamp_ms=0,
        track_id=1,
        bbox=(0, 0, 10, 10),
        bbox_conf=0.5,
        keypoints=kpts,
        visible_ratio=0.2,
        mean_keypoint_conf=0.2,
        zone_id="M-C",
    )
    assessment = classifier.assess_window([pose])
    assert assessment is not None
    assert assessment.canonical_label == UNCERTAIN_LABEL


def test_reporting_aggregators():
    assessments = [
        WindowAssessment(
            track_id=1,
            t_start_ms=0,
            t_end_ms=4000,
            canonical_label=HIGH,
            probabilities=(0.8, 0.1, 0.1),
            visible_ratio=0.9,
            mean_keypoint_conf=0.85,
            atomic_behaviors=["BOARD_OR_INSTRUCTOR_FACING"],
            zone_id="F-C",
        ),
        WindowAssessment(
            track_id=2,
            t_start_ms=0,
            t_end_ms=4000,
            canonical_label=UNCERTAIN_LABEL,
            probabilities=(0.33, 0.34, 0.33),
            visible_ratio=0.4,
            mean_keypoint_conf=0.25,
            atomic_behaviors=["OCCLUDED"],
            zone_id="B-L",
        ),
    ]

    results_df = build_results_dataframe(assessments)
    assert len(results_df) == 2
    assert "canonical_label" in results_df.columns
    assert "display_label" in results_df.columns

    timeline_df = build_timeline_dataframe(results_df, bucket_sec=5)
    assert not timeline_df.empty
    assert "HIGH_pct" in timeline_df.columns

    zone_df = build_zone_summary_dataframe(results_df)
    assert len(zone_df) == 9  # 3x3 zones
    assert "sample_count_N" in zone_df.columns
    # Zone F-C has 1 sample
    fc_row = zone_df[zone_df["zone_id"] == "F-C"].iloc[0]
    assert fc_row["sample_count_N"] == 1
    assert fc_row["HIGH_pct"] == 100.0
