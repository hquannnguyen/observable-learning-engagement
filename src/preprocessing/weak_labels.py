"""Weak supervision labeling functions for observable participation (Task T16).

Owner: Vũ Văn Phong (Data & Annotation Lead)
Contract:
- Generates weak labels marked as `label_source=WEAK_RULE`.
- Always attaches `label_rule_version` (e.g. `weak-v0.1`).
- Strictly adheres to canonical 3 classes and state contracts in src/contracts/labels.py.
- Does NOT infer internal emotion or intent; operates strictly on observable cues.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    CANONICAL_TRAINING_LABELS,
)

RULE_VERSION = "weak-v0.1"


def apply_labeling_rules(
    window_poses_df,
    class_phase: str = "lecture",
    min_visible_ratio: float = 0.50,
) -> Tuple[str, Optional[str], List[str], float]:
    """Apply labeling functions on a window of poses.

    Returns:
        (annotation_state, participation_label, atomic_behaviors, confidence)
    """
    if len(window_poses_df) < 3:
        return "INVALID", None, ["UNKNOWN"], 0.0

    mean_vis = float(window_poses_df["visible_ratio"].mean())
    mean_conf = float(window_poses_df["mean_keypoint_conf"].mean())

    # 1. State: Occluded check
    if mean_vis < min_visible_ratio or mean_conf < 0.25:
        return "OCCLUDED", None, ["OCCLUDED"], round(1.0 - mean_vis, 2)

    # Keypoint extraction
    # 0: nose, 1: L_eye, 2: R_eye, 3: L_ear, 4: R_ear
    # 5: L_shoulder, 6: R_shoulder, 7: L_elbow, 8: R_elbow, 9: L_wrist, 10: R_wrist
    nose_y = window_poses_df["kpt_00_y"].values
    nose_c = window_poses_df["kpt_00_conf"].values
    l_ear_y = window_poses_df["kpt_03_y"].values
    r_ear_y = window_poses_df["kpt_04_y"].values
    ears_y = (l_ear_y + r_ear_y) / 2.0

    l_shoulder_y = window_poses_df["kpt_05_y"].values
    r_shoulder_y = window_poses_df["kpt_06_y"].values
    l_shoulder_c = window_poses_df["kpt_05_conf"].values
    r_shoulder_c = window_poses_df["kpt_06_conf"].values

    l_wrist_y = window_poses_df["kpt_09_y"].values
    r_wrist_y = window_poses_df["kpt_10_y"].values
    l_wrist_c = window_poses_df["kpt_09_conf"].values
    r_wrist_c = window_poses_df["kpt_10_conf"].values

    l_wrist_x = window_poses_df["kpt_09_x"].values
    r_wrist_x = window_poses_df["kpt_10_x"].values

    # Hand raised: wrist significantly above shoulder
    l_raised = (l_wrist_c > 0.3) & (l_shoulder_c > 0.3) & (l_wrist_y < l_shoulder_y - 20)
    r_raised = (r_wrist_c > 0.3) & (r_shoulder_c > 0.3) & (r_wrist_y < r_shoulder_y - 20)
    hand_raised_ratio = float(np.mean(l_raised | r_raised))

    # Head down: nose noticeably lower than ears
    head_down_mask = (nose_y > ears_y + 12) & (nose_c > 0.3)
    head_down_ratio = float(np.mean(head_down_mask))

    # Wrist motion (typing / writing)
    wrist_motion = float(np.std(l_wrist_x) + np.std(r_wrist_x))

    behaviors: List[str] = []
    participation_label: str
    confidence: float

    # LF1: Hand raised (Strong high participation)
    if hand_raised_ratio >= 0.15:
        behaviors.append("HAND_RAISED")
        participation_label = HIGH
        confidence = 0.90

    # LF2: Reading/Writing / Laptop interaction
    elif (head_down_ratio >= 0.35) and (wrist_motion > 2.0):
        behaviors.append("READING_WRITING")
        if class_phase in {"individual_work", "discussion"}:
            participation_label = HIGH
            confidence = 0.85
        else:
            participation_label = MODERATE
            confidence = 0.75

    # LF3: Board or Instructor Facing in Lecture
    elif class_phase == "lecture" and head_down_ratio < 0.40 and mean_vis >= 0.65:
        behaviors.append("BOARD_OR_INSTRUCTOR_FACING")
        participation_label = HIGH
        confidence = 0.80

    # LF4: Prolonged Head Down with no desk/typing movement
    elif head_down_ratio > 0.75 and wrist_motion < 1.0:
        behaviors.append("HEAD_DOWN")
        participation_label = LOW
        confidence = 0.75

    # LF5: Moderate steady baseline
    else:
        behaviors.append("BOARD_OR_INSTRUCTOR_FACING" if head_down_ratio < 0.5 else "HEAD_DOWN")
        participation_label = MODERATE
        confidence = 0.70

    return "VALID", participation_label, behaviors, confidence
