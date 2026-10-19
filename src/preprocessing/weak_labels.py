"""Weak supervision and heuristic labeling module.

Implements Task T16:
Labeling functions based on observable signals and class phases to generate
weak labels tagged strictly as WEAK_RULE with label_rule_version.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    validate_annotation_label,
)

HIGH_LABEL = HIGH
MODERATE_LABEL = MODERATE
LOW_LABEL = LOW

logger = logging.getLogger(__name__)

WEAK_RULE_VERSION = "weak-v0.1"


def apply_weak_label_rules(
    window_record: Dict[str, Any],
    window_tensor: np.ndarray,
) -> Tuple[Optional[str], str]:
    """Assign weak participation label from observable behavior tensor and class phase.

    Args:
        window_record: Window metadata dictionary (must contain 'class_phase', 'quality_state').
        window_tensor: [40, 77] temporal feature matrix for the window.

    Returns:
        (participation_label, rule_name)
    """
    if window_record.get("quality_state") != "VALID":
        return None, "invalid_quality"

    class_phase = window_record.get("class_phase", "lecture")

    # Feature column indices based on src/preprocessing/features.py:
    # 0..33: norm_kpt
    # 34..67: velocities
    # 68: shoulder_slope
    # 69: head_proxy_x
    # 70: head_proxy_y
    # 71: left_hand_raised
    # 72: right_hand_raised
    # 73: head_down
    # 74: motion_energy
    # 75: visible_ratio
    # 76: mean_keypoint_conf

    left_raised = window_tensor[:, 71]
    right_raised = window_tensor[:, 72]
    head_down = window_tensor[:, 73]
    motion = window_tensor[:, 74]
    head_proxy_x = window_tensor[:, 69]

    # Rule 1: Hand Raised in lecture/discussion -> HIGH
    hand_raised_timesteps = np.sum((left_raised > 0.05) | (right_raised > 0.05))
    if hand_raised_timesteps >= 5:  # at least 0.5s in 4s window
        return HIGH_LABEL, "rule_hand_raised"

    # Rule 2: Active writing/reading in individual_work -> HIGH
    if class_phase == "individual_work":
        # Head slightly down with active hand motion
        is_writing = np.mean(head_down > 0.02) > 0.50 and np.mean(motion) > 0.01
        if is_writing:
            return HIGH_LABEL, "rule_writing_individual"
        elif np.mean(head_down > 0.30) > 0.80 and np.mean(motion) < 0.005:
            # Gục đầu ngủ bất động
            return LOW_LABEL, "rule_head_down_sleeping"
        else:
            return MODERATE_LABEL, "rule_moderate_individual"

    # Rule 3: Head down sleeping in lecture -> LOW
    if class_phase == "lecture":
        if np.mean(head_down > 0.25) > 0.70:
            return LOW_LABEL, "rule_head_down_lecture"

        # Body/head turned sharply away (>0.4 relative offset) -> LOW
        if np.mean(np.abs(head_proxy_x) > 0.45) > 0.60:
            return LOW_LABEL, "rule_turned_away"

        # Attentive facing board with small natural motion -> HIGH
        if np.mean(np.abs(head_proxy_x) < 0.25) > 0.70 and np.mean(head_down < 0.15) > 0.70:
            if np.mean(motion) > 0.015:
                return HIGH_LABEL, "rule_active_listening"
            else:
                return MODERATE_LABEL, "rule_passive_listening"

    # Rule 4: Discussion phase
    if class_phase == "discussion":
        if np.mean(motion) > 0.02:
            return HIGH_LABEL, "rule_active_discussion"
        elif np.mean(motion) < 0.005 and np.mean(head_down > 0.20) > 0.50:
            return LOW_LABEL, "rule_disengaged_discussion"
        else:
            return MODERATE_LABEL, "rule_moderate_discussion"

    return MODERATE_LABEL, "rule_default_moderate"


def assign_weak_labels_to_windows(
    window_records: List[Dict[str, Any]],
    window_tensors: np.ndarray,
) -> List[Dict[str, Any]]:
    """Label windows using weak supervision rules, preserving provenance contract."""
    labeled_records = []
    for idx, rec in enumerate(window_records):
        rec_copy = rec.copy()
        if rec_copy.get("quality_state") == "VALID":
            w_tensor = window_tensors[idx]
            label, rule_name = apply_weak_label_rules(rec_copy, w_tensor)
            rec_copy["participation_label"] = label
            rec_copy["label_source"] = "WEAK_RULE"
            rec_copy["label_rule_version"] = WEAK_RULE_VERSION

            # Validate against contract
            validate_annotation_label(
                participation_label=label,
                annotation_state=rec_copy["quality_state"],
                label_source=rec_copy["label_source"],
                label_rule_version=rec_copy["label_rule_version"],
            )
        else:
            rec_copy["participation_label"] = None
            rec_copy["label_source"] = None
            rec_copy["label_rule_version"] = None

        labeled_records.append(rec_copy)

    return labeled_records
