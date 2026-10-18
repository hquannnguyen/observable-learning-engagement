"""Rule-based heuristic baseline (M0) for CVF25PRJ10.

Implements Task T20 (Baseline M0).
Operates strictly on observable spatial and temporal motion features.
Does NOT infer internal emotion, boredom, motivation, or intent.
Integrates directly with the Uncertainty Gate (src/uncertainty/gate.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
    CANONICAL_TRAINING_LABELS,
)
from src.pose.extractor import TrackedPose
from src.uncertainty.gate import UncertaintyThresholds, apply_uncertainty_gate


@dataclass
class WindowAssessment:
    """Output assessment of a temporal window for a single track."""
    track_id: int
    t_start_ms: int
    t_end_ms: int
    canonical_label: str       # HIGH_OBSERVED_PARTICIPATION, MODERATE..., LOW..., UNCERTAIN
    probabilities: Tuple[float, float, float]  # (P_high, P_moderate, P_low)
    visible_ratio: float
    mean_keypoint_conf: float
    atomic_behaviors: List[str]
    zone_id: str


class HeuristicBaselineClassifier:
    """Baseline M0: Heuristic rule-based observable participation estimator."""

    def __init__(
        self,
        thresholds: UncertaintyThresholds = UncertaintyThresholds(),
    ) -> None:
        self.thresholds = thresholds

    def assess_window(
        self,
        poses: List[TrackedPose],
        class_phase: str = "lecture",
    ) -> Optional[WindowAssessment]:
        """Assess a sequence of poses representing one track within a temporal window."""
        if not poses:
            return None

        track_id = poses[0].track_id
        t_start_ms = poses[0].timestamp_ms
        t_end_ms = poses[-1].timestamp_ms
        zone_id = poses[-1].zone_id

        # Calculate mean visibility and confidence
        vis_ratios = [p.visible_ratio for p in poses]
        conf_scores = [p.mean_keypoint_conf for p in poses]
        mean_vis = float(np.mean(vis_ratios))
        mean_conf = float(np.mean(conf_scores))

        # Check early occlusion/low-quality
        if mean_vis < self.thresholds.min_visible_ratio or mean_conf < self.thresholds.min_mean_keypoint_confidence:
            return WindowAssessment(
                track_id=track_id,
                t_start_ms=t_start_ms,
                t_end_ms=t_end_ms,
                canonical_label=UNCERTAIN_LABEL,
                probabilities=(0.33, 0.34, 0.33),
                visible_ratio=mean_vis,
                mean_keypoint_conf=mean_conf,
                atomic_behaviors=["OCCLUDED"],
                zone_id=zone_id,
            )

        # Extract observable features across window
        kpts_seq = np.array([p.keypoints for p in poses])  # Shape: (T, 17, 3)

        # 1. Hand raised detection (wrist above shoulder)
        # 5: L_shoulder, 6: R_shoulder, 9: L_wrist, 10: R_wrist
        # In image coords, smaller y = higher elevation
        l_wrist_raised = (kpts_seq[:, 9, 2] > 0.3) & (kpts_seq[:, 5, 2] > 0.3) & (kpts_seq[:, 9, 1] < kpts_seq[:, 5, 1] - 20)
        r_wrist_raised = (kpts_seq[:, 10, 2] > 0.3) & (kpts_seq[:, 6, 2] > 0.3) & (kpts_seq[:, 10, 1] < kpts_seq[:, 6, 1] - 20)
        hand_raised_frames = np.sum(l_wrist_raised | r_wrist_raised)
        hand_raised_ratio = hand_raised_frames / len(poses)

        # 2. Head posture / orientation proxy
        # 0: nose, 1: L_eye, 2: R_eye, 3: L_ear, 4: R_ear
        # Nose significantly lower than ears indicates head down (looking at desk/laptop/phone)
        nose_y = kpts_seq[:, 0, 1]
        ears_y = np.mean([kpts_seq[:, 3, 1], kpts_seq[:, 4, 1]], axis=0)
        head_down_mask = (nose_y > ears_y + 15) & (kpts_seq[:, 0, 2] > 0.3)
        head_down_ratio = float(np.mean(head_down_mask))

        # 3. Motion energy (std of nose/shoulder movement)
        nose_motion = np.std(kpts_seq[:, 0, :2], axis=0).mean()
        shoulders_motion = np.std(kpts_seq[:, [5, 6], :2], axis=0).mean()
        overall_motion = float((nose_motion + shoulders_motion) / 2.0)

        # 4. Wrist/hand movement near desk level (typing/writing cue)
        wrist_motion = float(np.std(kpts_seq[:, [9, 10], :2], axis=0).mean())
        active_typing_writing = (head_down_ratio > 0.3) and (wrist_motion > 1.5)

        # Determine atomic observable behaviors
        behaviors: List[str] = []
        if hand_raised_ratio > 0.15:
            behaviors.append("HAND_RAISED")
        if active_typing_writing:
            behaviors.append("READING_WRITING")
        elif head_down_ratio > 0.6:
            behaviors.append("HEAD_DOWN")
        else:
            behaviors.append("BOARD_OR_INSTRUCTOR_FACING")

        # Heuristic probability estimation based on phase and observable behaviors
        if hand_raised_ratio > 0.15:
            # Strong signal of active observable participation
            p_high, p_mod, p_low = 0.85, 0.10, 0.05
        elif class_phase == "individual_work" and active_typing_writing:
            # Normal individual work on laptop or writing on paper
            p_high, p_mod, p_low = 0.75, 0.18, 0.07
        elif class_phase == "lecture" and not head_down_mask.all() and head_down_ratio < 0.4:
            # Looking towards board/instructor
            p_high, p_mod, p_low = 0.70, 0.22, 0.08
        elif head_down_ratio > 0.75 and wrist_motion < 0.8:
            # Prolonged static head down without typing/writing cues
            p_high, p_mod, p_low = 0.10, 0.25, 0.65
        else:
            # Moderate/intermediate signals
            p_high, p_mod, p_low = 0.25, 0.60, 0.15

        probabilities = (p_high, p_mod, p_low)

        # Apply formal Uncertainty Gate
        canonical_label = apply_uncertainty_gate(
            probabilities,
            visible_ratio=mean_vis,
            mean_keypoint_confidence=mean_conf,
            thresholds=self.thresholds,
            class_labels=CANONICAL_TRAINING_LABELS,
        )

        return WindowAssessment(
            track_id=track_id,
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms,
            canonical_label=canonical_label,
            probabilities=probabilities,
            visible_ratio=mean_vis,
            mean_keypoint_conf=mean_conf,
            atomic_behaviors=behaviors,
            zone_id=zone_id,
        )
