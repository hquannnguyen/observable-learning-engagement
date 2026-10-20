"""Pose and motion feature extraction conforming to feature_schema_version: posefeat-v1.

Implements Task T15 (Pose/motion feature engineering).
Produces [T, 64] feature tensors consumed by LSTM and TCN models.
Validated against src/preprocessing/feature_contract.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.preprocessing.feature_contract import validate_temporal_features

FEATURE_SCHEMA_VERSION = "posefeat-v1"
FEATURE_DIM = 64


def extract_frame_features(kpts: np.ndarray, prev_kpts: np.ndarray | None = None) -> np.ndarray:
    """Extract 64-dimensional feature vector for a single frame.

    kpts: Shape (17, 3) -> [x, y, conf]
    Components:
    - 17 normalized keypoints (x, y, conf): 51 features
    - 13 geometry and motion features: 13 features
    Total: 64 features.
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 1. Torso reference scale and center
    # 5: L_shoulder, 6: R_shoulder, 11: L_hip, 12: R_hip
    shoulder_center_x = (kpts[5, 0] + kpts[6, 0]) / 2.0
    shoulder_center_y = (kpts[5, 1] + kpts[6, 1]) / 2.0
    hip_center_x = (kpts[11, 0] + kpts[12, 0]) / 2.0
    hip_center_y = (kpts[11, 1] + kpts[12, 1]) / 2.0

    torso_h = np.sqrt((shoulder_center_x - hip_center_x) ** 2 + (shoulder_center_y - hip_center_y) ** 2)
    shoulder_w = np.sqrt((kpts[5, 0] - kpts[6, 0]) ** 2 + (kpts[5, 1] - kpts[6, 1]) ** 2)
    ref_scale = max(float(torso_h), float(shoulder_w), 40.0)

    # 2. Normalized keypoints (centered at shoulder center, scaled by ref_scale)
    norm_kpts = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        if kpts[i, 2] > 0.05:  # Valid keypoint
            norm_kpts[i, 0] = (kpts[i, 0] - shoulder_center_x) / ref_scale
            norm_kpts[i, 1] = (kpts[i, 1] - shoulder_center_y) / ref_scale
            norm_kpts[i, 2] = kpts[i, 2]
        else:
            norm_kpts[i, 0] = 0.0
            norm_kpts[i, 1] = 0.0
            norm_kpts[i, 2] = 0.0

    feats[:51] = norm_kpts.flatten()

    # 3. Geometry and posture features (13 dims)
    # Geo 1: Shoulder slope (tilt)
    shoulder_dy = kpts[6, 1] - kpts[5, 1]
    shoulder_dx = kpts[6, 0] - kpts[5, 0] + 1e-6
    shoulder_slope = np.arctan2(shoulder_dy, shoulder_dx)
    feats[51] = float(shoulder_slope)

    # Geo 2: Normalized shoulder width
    feats[52] = float(shoulder_w / ref_scale)

    # Geo 3: Torso angle (relative to vertical)
    torso_dy = hip_center_y - shoulder_center_y
    torso_dx = hip_center_x - shoulder_center_x + 1e-6
    feats[53] = float(np.arctan2(torso_dx, torso_dy))

    # Geo 4: Head orientation proxy (nose x relative to ear midpoint)
    ear_mid_x = (kpts[3, 0] + kpts[4, 0]) / 2.0
    head_yaw_proxy = (kpts[0, 0] - ear_mid_x) / (ref_scale + 1e-6)
    feats[54] = float(head_yaw_proxy)

    # Geo 5: Head down vertical delta (nose y below ear level)
    ear_mid_y = (kpts[3, 1] + kpts[4, 1]) / 2.0
    head_pitch_proxy = (kpts[0, 1] - ear_mid_y) / (ref_scale + 1e-6)
    feats[55] = float(head_pitch_proxy)

    # Geo 6: Left wrist vertical delta relative to shoulder
    feats[56] = float((kpts[9, 1] - kpts[5, 1]) / ref_scale)

    # Geo 7: Right wrist vertical delta relative to shoulder
    feats[57] = float((kpts[10, 1] - kpts[6, 1]) / ref_scale)

    # Geo 8: Left elbow to shoulder distance
    feats[58] = float(np.linalg.norm(kpts[7, :2] - kpts[5, :2]) / ref_scale)

    # Geo 9: Right elbow to shoulder distance
    feats[59] = float(np.linalg.norm(kpts[8, :2] - kpts[6, :2]) / ref_scale)

    # Motion features (from previous frame if available)
    if prev_kpts is not None:
        nose_vel = np.linalg.norm(kpts[0, :2] - prev_kpts[0, :2]) / ref_scale
        l_wrist_vel = np.linalg.norm(kpts[9, :2] - prev_kpts[9, :2]) / ref_scale
        r_wrist_vel = np.linalg.norm(kpts[10, :2] - prev_kpts[10, :2]) / ref_scale
    else:
        nose_vel = 0.0
        l_wrist_vel = 0.0
        r_wrist_vel = 0.0

    feats[60] = float(nose_vel)
    feats[61] = float(l_wrist_vel)
    feats[62] = float(r_wrist_vel)

    # Overall visible ratio
    feats[63] = float(np.mean(kpts[:, 2] > 0.2))

    return feats


def build_window_features(
    keypoints_sequence: np.ndarray,
    target_timesteps: int = 40,
) -> np.ndarray:
    """Build [T=40, F=64] model-ready feature tensor for one window.

    keypoints_sequence: Shape (T_obs, 17, 3)
    target_timesteps: 40 (10 FPS x 4s)
    """
    t_obs = len(keypoints_sequence)
    if t_obs == 0:
        raise ValueError("Cannot extract features from empty sequence.")

    # Frame-by-frame feature extraction
    frame_feats = []
    prev = None
    for t in range(t_obs):
        f = extract_frame_features(keypoints_sequence[t], prev_kpts=prev)
        frame_feats.append(f)
        prev = keypoints_sequence[t]

    feats_arr = np.array(frame_feats, dtype=np.float32)  # Shape (T_obs, 64)

    # Resample or pad/truncate to exact target_timesteps (e.g. 40)
    if len(feats_arr) == target_timesteps:
        final_tensor = feats_arr
    elif len(feats_arr) < target_timesteps:
        # Interpolate / pad
        indices = np.linspace(0, len(feats_arr) - 1, target_timesteps)
        final_tensor = np.zeros((target_timesteps, FEATURE_DIM), dtype=np.float32)
        for dim in range(FEATURE_DIM):
            final_tensor[:, dim] = np.interp(indices, np.arange(len(feats_arr)), feats_arr[:, dim])
    else:
        # Downsample uniformly
        indices = np.linspace(0, len(feats_arr) - 1, target_timesteps).astype(int)
        final_tensor = feats_arr[indices]

    # Validate against feature contract
    return validate_temporal_features(final_tensor, expected_timesteps=target_timesteps)
