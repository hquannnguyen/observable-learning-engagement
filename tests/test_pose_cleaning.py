"""Unit tests for tracking statistics, quality gate, and pose cleaning."""

import numpy as np
import pandas as pd
import pytest

from src.pose.tracker import compute_track_statistics, filter_short_tracks
from src.preprocessing.cleaner import (
    clean_and_normalize_poses,
    interpolate_missing_keypoints,
    normalize_keypoints,
    smooth_keypoints,
)
from src.preprocessing.quality_gate import (
    evaluate_pose_quality,
    filter_quality_poses,
    load_quality_gate_config,
)


def make_dummy_poses_df(num_frames=25, sample_fps=10.0, track_id=1):
    """Generate synthetic pose DataFrame for testing."""
    rows = []
    for f in range(num_frames):
        ts = int(f * (1000.0 / sample_fps))
        row = {
            "session_id": "test_sess",
            "camera_id": "cam_A",
            "frame_idx": f,
            "timestamp_ms": ts,
            "track_id": track_id,
            "bbox_x1": 100.0,
            "bbox_y1": 50.0,
            "bbox_x2": 200.0,
            "bbox_y2": 250.0,
            "bbox_conf": 0.90,
            "visible_ratio": 0.85,
            "mean_keypoint_conf": 0.80,
            "zone_id": "zone_1",
        }
        # Add 17 keypoints
        for i in range(17):
            # Mid-shoulder at (150, 80)
            if i == 5:
                row[f"kpt_{i:02d}_x"] = 130.0
                row[f"kpt_{i:02d}_y"] = 80.0
            elif i == 6:
                row[f"kpt_{i:02d}_x"] = 170.0
                row[f"kpt_{i:02d}_y"] = 80.0
            # Mid-hip at (150, 180)
            elif i == 11:
                row[f"kpt_{i:02d}_x"] = 135.0
                row[f"kpt_{i:02d}_y"] = 180.0
            elif i == 12:
                row[f"kpt_{i:02d}_x"] = 165.0
                row[f"kpt_{i:02d}_y"] = 180.0
            else:
                row[f"kpt_{i:02d}_x"] = 150.0 + i * 2
                row[f"kpt_{i:02d}_y"] = 100.0 + i * 5
            row[f"kpt_{i:02d}_conf"] = 0.85

        rows.append(row)
    return pd.DataFrame(rows)


def test_compute_track_statistics():
    df1 = make_dummy_poses_df(num_frames=30, sample_fps=10.0, track_id=1)  # 2.9s -> valid
    df2 = make_dummy_poses_df(num_frames=5, sample_fps=10.0, track_id=2)   # 0.4s -> short
    combined = pd.concat([df1, df2], ignore_index=True)

    stats = compute_track_statistics(combined, min_track_seconds=2.0)
    assert len(stats) == 2
    assert set(stats["track_id"]) == {1, 2}

    t1 = stats[stats["track_id"] == 1].iloc[0]
    t2 = stats[stats["track_id"] == 2].iloc[0]
    assert t1["is_valid_duration"] is True or t1["is_valid_duration"] == 1
    assert t2["is_valid_duration"] is False or t2["is_valid_duration"] == 0

    filtered = filter_short_tracks(combined, min_track_seconds=2.0)
    assert len(filtered) == 30
    assert set(filtered["track_id"]) == {1}


def test_interpolate_short_missing_gaps():
    df = make_dummy_poses_df(num_frames=10, track_id=1)
    # Inject 2-frame gap at frame 3 and 4 for keypoint 0 (nose)
    df.loc[3, "kpt_00_x"] = np.nan
    df.loc[3, "kpt_00_y"] = np.nan
    df.loc[4, "kpt_00_x"] = np.nan
    df.loc[4, "kpt_00_y"] = np.nan

    interpolated = interpolate_missing_keypoints(df, max_gap_frames=3)

    # Frame 3 and 4 should be interpolated (not NaN)
    assert not np.isnan(interpolated.loc[3, "kpt_00_x"])
    assert not np.isnan(interpolated.loc[4, "kpt_00_x"])

    # Now inject 5-frame gap (exceeds max_gap_frames=3)
    for f in range(2, 7):
        df.loc[f, "kpt_01_x"] = np.nan
        df.loc[f, "kpt_01_y"] = np.nan

    interpolated_long = interpolate_missing_keypoints(df, max_gap_frames=3)
    # Middle frame should still be NaN because gap was too large
    assert np.isnan(interpolated_long.loc[4, "kpt_01_x"])


def test_smooth_keypoints_preserves_shape():
    df = make_dummy_poses_df(num_frames=15, track_id=1)
    # Add artificial jitter to frame 5
    df.loc[5, "kpt_00_x"] += 20.0

    smoothed = smooth_keypoints(df, window_size=3, method="moving_average")
    assert len(smoothed) == len(df)
    # Value at frame 5 should be damped by smoothing
    assert smoothed.loc[5, "kpt_00_x"] < df.loc[5, "kpt_00_x"]


def test_normalize_keypoints_torso_and_bbox():
    df = make_dummy_poses_df(num_frames=5, track_id=1)

    norm_torso = normalize_keypoints(df, method="torso")
    assert "norm_kpt_00_x" in norm_torso.columns
    assert "norm_kpt_05_x" in norm_torso.columns

    # Mid-shoulder is centered around (0, 0)
    mid_shoulder_norm_x = (
        norm_torso["norm_kpt_05_x"].iloc[0] + norm_torso["norm_kpt_06_x"].iloc[0]
    ) / 2.0
    mid_shoulder_norm_y = (
        norm_torso["norm_kpt_05_y"].iloc[0] + norm_torso["norm_kpt_06_y"].iloc[0]
    ) / 2.0
    assert abs(mid_shoulder_norm_x) < 0.05
    assert abs(mid_shoulder_norm_y) < 0.05

    norm_bbox = normalize_keypoints(df, method="bbox")
    # Coordinates inside bbox [100, 200] x [50, 250] should fall in [0, 1]
    val_x = norm_bbox["norm_kpt_00_x"].iloc[0]
    val_y = norm_bbox["norm_kpt_00_y"].iloc[0]
    assert 0.0 <= val_x <= 1.0
    assert 0.0 <= val_y <= 1.0


def test_quality_gate_evaluation():
    df = make_dummy_poses_df(num_frames=25, track_id=1)
    # Set row 0 to low visible ratio
    df.loc[0, "visible_ratio"] = 0.40  # < 0.60
    # Set row 1 to low mean confidence
    df.loc[1, "mean_keypoint_conf"] = 0.20  # < 0.30

    evaluated = evaluate_pose_quality(
        df,
        min_visible_ratio=0.60,
        min_mean_keypoint_confidence=0.30,
        min_track_seconds=2.0,
    )

    assert evaluated.loc[0, "quality_state"] == "OCCLUDED"
    assert evaluated.loc[1, "quality_state"] == "INVALID"
    assert evaluated.loc[2, "quality_state"] == "VALID"

    valid_only = filter_quality_poses(evaluated, allowed_states=("VALID",))
    assert len(valid_only) == 23


def test_full_cleaning_pipeline():
    df = make_dummy_poses_df(num_frames=25, track_id=1)
    # Add a short 1-frame gap
    df.loc[10, "kpt_00_x"] = np.nan
    df.loc[10, "kpt_00_y"] = np.nan

    cleaned = clean_and_normalize_poses(
        df,
        min_track_seconds=2.0,
        max_gap_frames=3,
        smooth_window=3,
        normalization_method="torso",
    )

    assert len(cleaned) == 25
    # Gap at frame 10 should be interpolated
    assert not np.isnan(cleaned.loc[10, "kpt_00_x"])
    assert "norm_kpt_00_x" in cleaned.columns
