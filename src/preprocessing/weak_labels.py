"""Weak supervision and heuristic labeling module.

Implements Task T16:
Labeling functions based on observable signals and class phases to generate
weak labels tagged strictly as WEAK_RULE with label_rule_version.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

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
SUPPORTED_WEAK_RULE_VERSIONS = ("weak-v0.1", "weak-v0.2")


def _apply_weak_v01_rules(
    class_phase: str,
    window_tensor: np.ndarray,
) -> Tuple[Optional[str], str]:
    """Legacy weak-v0.1 heuristic rules over normalized pose/motion channels."""
    left_raised = window_tensor[:, 71]
    right_raised = window_tensor[:, 72]
    head_down = window_tensor[:, 73]
    motion = window_tensor[:, 74]
    head_proxy_x = window_tensor[:, 69]

    hand_raised_timesteps = np.sum((left_raised > 0.05) | (right_raised > 0.05))
    if hand_raised_timesteps >= 5:
        return HIGH_LABEL, "rule_hand_raised"

    if class_phase == "individual_work":
        is_writing = np.mean(head_down > 0.02) > 0.50 and np.mean(motion) > 0.01
        if is_writing:
            return HIGH_LABEL, "rule_writing_individual"
        if np.mean(head_down > 0.30) > 0.80 and np.mean(motion) < 0.005:
            return LOW_LABEL, "rule_head_down_sleeping"
        return MODERATE_LABEL, "rule_moderate_individual"

    if class_phase == "lecture":
        if np.mean(head_down > 0.25) > 0.70:
            return LOW_LABEL, "rule_head_down_lecture"
        if np.mean(np.abs(head_proxy_x) > 0.45) > 0.60:
            return LOW_LABEL, "rule_turned_away"
        if np.mean(np.abs(head_proxy_x) < 0.25) > 0.70 and np.mean(head_down < 0.15) > 0.70:
            if np.mean(motion) > 0.015:
                return HIGH_LABEL, "rule_active_listening"
            return MODERATE_LABEL, "rule_passive_listening"

    if class_phase == "discussion":
        if np.mean(motion) > 0.02:
            return HIGH_LABEL, "rule_active_discussion"
        if np.mean(motion) < 0.005 and np.mean(head_down > 0.20) > 0.50:
            return LOW_LABEL, "rule_disengaged_discussion"
        return MODERATE_LABEL, "rule_moderate_discussion"

    return MODERATE_LABEL, "rule_default_moderate"


def _apply_weak_v02_rules(
    class_phase: str,
    window_tensor: np.ndarray,
) -> Tuple[Optional[str], str]:
    """Calibrated weak-v0.2 rules using signed head_proxy_y, wrist velocity, and YOLO11m motion scale."""
    head_proxy_x = window_tensor[:, 69]
    head_proxy_y = window_tensor[:, 70]
    left_raised = window_tensor[:, 71]
    right_raised = window_tensor[:, 72]
    motion = window_tensor[:, 74]
    wrist_vel_rms = float(np.sqrt(np.mean(window_tensor[:, 52:56] ** 2)))
    motion_mean = float(np.mean(motion))

    hand_raised_steps = int(np.sum((left_raised > 0.10) | (right_raised > 0.10)))
    if hand_raised_steps >= 8:
        return HIGH_LABEL, "v2_hand_raised"

    if class_phase == "lecture":
        if np.mean(head_proxy_y > -0.02) > 0.65 and wrist_vel_rms < 0.15:
            return LOW_LABEL, "v2_head_down_lecture"
        if np.mean(np.abs(head_proxy_x) > 0.30) > 0.60:
            return LOW_LABEL, "v2_turned_away_lecture"
        if (
            np.mean(np.abs(head_proxy_x) < 0.18) > 0.70
            and np.mean(head_proxy_y < -0.22) > 0.70
            and 0.03 <= motion_mean <= 0.18
        ):
            return HIGH_LABEL, "v2_attentive_lecture"
        return MODERATE_LABEL, "v2_moderate_lecture"

    if class_phase == "individual_work":
        if np.mean(head_proxy_y > 0.03) > 0.65 and wrist_vel_rms < 0.12:
            return LOW_LABEL, "v2_head_down_individual"
        if np.mean(np.abs(head_proxy_x) > 0.32) > 0.60 and wrist_vel_rms < 0.12:
            return LOW_LABEL, "v2_turned_away_individual"
        if (
            np.mean((head_proxy_y >= -0.22) & (head_proxy_y <= 0.08)) > 0.55
            and wrist_vel_rms > 0.12
            and np.mean(np.abs(head_proxy_x) < 0.28) > 0.60
        ):
            return HIGH_LABEL, "v2_writing_individual"
        return MODERATE_LABEL, "v2_moderate_individual"

    if class_phase == "discussion":
        if np.mean(head_proxy_y > 0.02) > 0.60 and motion_mean < 0.05:
            return LOW_LABEL, "v2_disengaged_discussion"
        if (
            np.mean(np.abs(head_proxy_x) > 0.15) > 0.50
            and wrist_vel_rms > 0.15
            and 0.04 <= motion_mean <= 0.25
        ):
            return HIGH_LABEL, "v2_active_discussion"
        return MODERATE_LABEL, "v2_moderate_discussion"

    return MODERATE_LABEL, "v2_default_moderate"


def apply_weak_label_rules(
    window_record: Dict[str, Any],
    window_tensor: np.ndarray,
    rule_version: str = WEAK_RULE_VERSION,
) -> Tuple[Optional[str], str]:
    """Assign weak participation label from observable behavior tensor and class phase.

    Args:
        window_record: Window metadata dictionary (must contain 'class_phase', 'quality_state').
        window_tensor: [40, D] temporal feature matrix for the window (D >= 77).
        rule_version: Weak rule set version ('weak-v0.1' or 'weak-v0.2').

    Returns:
        (participation_label, rule_name)
    """
    if rule_version not in SUPPORTED_WEAK_RULE_VERSIONS:
        raise ValueError(
            f"Unsupported weak rule version {rule_version!r}; expected one of {SUPPORTED_WEAK_RULE_VERSIONS}"
        )

    if window_record.get("quality_state") != "VALID":
        return None, "invalid_quality"

    class_phase = window_record.get("class_phase", "UNKNOWN")
    if class_phase not in {"lecture", "individual_work", "discussion"}:
        return None, "unknown_phase"

    if rule_version == "weak-v0.2":
        return _apply_weak_v02_rules(class_phase, window_tensor)
    return _apply_weak_v01_rules(class_phase, window_tensor)


def assign_weak_labels_to_windows(
    window_records: List[Dict[str, Any]],
    window_tensors: np.ndarray,
    rule_version: str = WEAK_RULE_VERSION,
) -> List[Dict[str, Any]]:
    """Label windows using weak supervision rules, preserving provenance contract."""
    if rule_version not in SUPPORTED_WEAK_RULE_VERSIONS:
        raise ValueError(
            f"Unsupported weak rule version {rule_version!r}; expected one of {SUPPORTED_WEAK_RULE_VERSIONS}"
        )
    labeled_records = []
    for idx, rec in enumerate(window_records):
        rec_copy = rec.copy()
        if rec_copy.get("quality_state") == "VALID":
            w_tensor = window_tensors[idx]
            label, rule_name = apply_weak_label_rules(rec_copy, w_tensor, rule_version=rule_version)
            if label is None:
                rec_copy.update(participation_label=None, label_source=None, label_rule_version=None,
                                quality_state="INVALID", quality_reason="UNKNOWN_PHASE")
                labeled_records.append(rec_copy)
                continue
            rec_copy["participation_label"] = label
            rec_copy["label_source"] = "WEAK_RULE"
            rec_copy["label_rule_version"] = rule_version

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
