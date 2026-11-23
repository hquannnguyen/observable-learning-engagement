#!/usr/bin/env python3
"""Baseline M0: Heuristic Rule-Based Classifier Evaluation.

Implements Task T20 (Baseline M0) as specified in IMPLEMENTATION_PLAN.md:
1. Evaluates rule-based heuristic classifier on temporal windows.
2. Uses observable spatial/temporal signals (hand raise, head down proxy, motion energy)
   and class phase to estimate participation without learning parameters.
3. Applies Uncertainty Gate for low visibility / confidence.
4. Outputs classification metrics and confusion matrix for Train, Val, and Test splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.labels import (
    CANONICAL_TRAINING_LABELS,
    HIGH,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
)
from src.uncertainty.gate import UncertaintyThresholds, apply_uncertainty_gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_m0")

LABEL_TO_INT = {lb: i for i, lb in enumerate(CANONICAL_TRAINING_LABELS)}

def assess_tensor_window(
    tensor_40x94: np.ndarray,
    class_phase: str,
    thresholds: UncertaintyThresholds,
) -> Dict[str, Any]:
    """Assess a single (40, 94) window tensor using M0 heuristic rules."""
    # Channel indices:
    # 71: left_hand_raised, 72: right_hand_raised
    # 73: head_down_proxy
    # 74: motion_energy
    # 75: visible_ratio
    # 76: mean_keypoint_conf
    mean_vis = float(np.mean(tensor_40x94[:, 75]))
    mean_conf = float(np.mean(tensor_40x94[:, 76]))

    # Early rejection by quality gate
    if mean_vis < thresholds.min_visible_ratio or mean_conf < thresholds.min_mean_keypoint_confidence:
        return {
            "pred_label": UNCERTAIN_LABEL,
            "gate_reason": "LOW_VISIBILITY" if mean_vis < thresholds.min_visible_ratio else "LOW_KEYPOINT_CONF",
            "probabilities": (0.33, 0.34, 0.33),
            "p_max": 0.34,
            "accepted": False,
        }

    # Observable cues
    hand_raised = (tensor_40x94[:, 71] > 0.5) | (tensor_40x94[:, 72] > 0.5)
    hand_raised_ratio = float(np.mean(hand_raised))

    head_down = tensor_40x94[:, 73] > 0.5
    head_down_ratio = float(np.mean(head_down))

    motion = float(np.mean(tensor_40x94[:, 74]))

    # Heuristic probability estimation
    if hand_raised_ratio > 0.125:  # at least 5 frames of 40
        p_high, p_mod, p_low = 0.85, 0.10, 0.05
    elif class_phase == "individual_work" and head_down_ratio > 0.3 and motion > 0.02:
        p_high, p_mod, p_low = 0.75, 0.18, 0.07
    elif class_phase == "lecture" and head_down_ratio < 0.4:
        p_high, p_mod, p_low = 0.70, 0.22, 0.08
    elif head_down_ratio > 0.75 and motion < 0.01:
        p_high, p_mod, p_low = 0.10, 0.25, 0.65
    else:
        p_high, p_mod, p_low = 0.25, 0.60, 0.15

    probs = (p_high, p_mod, p_low)
    p_max = float(max(probs))

    # Apply uncertainty gate
    gated_label = apply_uncertainty_gate(
        probs,
        visible_ratio=mean_vis,
        mean_keypoint_confidence=mean_conf,
        thresholds=thresholds,
        class_labels=CANONICAL_TRAINING_LABELS,
    )

    accepted = gated_label != UNCERTAIN_LABEL
    gate_reason = "ACCEPTED" if accepted else "LOW_MODEL_CONF"

    return {
        "pred_label": gated_label,
        "gate_reason": gate_reason,
        "probabilities": probs,
        "p_max": p_max,
        "accepted": accepted,
    }

def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    phases: List[str],
    thresholds: UncertaintyThresholds,
    split_name: str,
) -> Dict[str, Any]:
    preds = []
    accepted_mask = []
    pred_ints = []

    for i in range(len(X)):
        res = assess_tensor_window(X[i], phases[i], thresholds)
        preds.append(res)
        accepted_mask.append(res["accepted"])
        # If UNCERTAIN, assign argmax of probabilities for metric computation
        arg_max_class = int(np.argmax(res["probabilities"]))
        pred_ints.append(arg_max_class)

    pred_ints = np.array(pred_ints)
    accepted_mask = np.array(accepted_mask)

    # Metrics on ALL supervised samples (argmax)
    macro_f1_all = float(f1_score(y, pred_ints, average="macro", zero_division=0))
    bal_acc_all = float(balanced_accuracy_score(y, pred_ints))
    cm_all = confusion_matrix(y, pred_ints, labels=[0, 1, 2]).tolist()
    report_all = classification_report(
        y, pred_ints, labels=[0, 1, 2], target_names=CANONICAL_TRAINING_LABELS, output_dict=True, zero_division=0
    )

    # Metrics on ACCEPTED samples only
    if accepted_mask.sum() > 0:
        y_acc = y[accepted_mask]
        p_acc = pred_ints[accepted_mask]
        macro_f1_acc = float(f1_score(y_acc, p_acc, average="macro", zero_division=0))
        bal_acc_acc = float(balanced_accuracy_score(y_acc, p_acc))
        coverage = float(np.mean(accepted_mask))
    else:
        macro_f1_acc, bal_acc_acc, coverage = 0.0, 0.0, 0.0

    logger.info(
        "[%s M0-Heuristic] Macro-F1 (All): %.4f | Macro-F1 (Accepted): %.4f | Coverage: %.2f%%",
        split_name, macro_f1_all, macro_f1_acc, coverage * 100.0
    )

    return {
        "macro_f1_all": macro_f1_all,
        "balanced_accuracy_all": bal_acc_all,
        "macro_f1_accepted": macro_f1_acc,
        "balanced_accuracy_accepted": bal_acc_acc,
        "coverage": coverage,
        "accepted_count": int(accepted_mask.sum()),
        "total_count": len(y),
        "confusion_matrix_all": cm_all,
        "classification_report_all": report_all,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline M0 Heuristic.")
    parser.add_argument("--features", default="data/features_v02_yolo11_weak/window_features.npz")
    parser.add_argument("--windows", default="data/features_v02_yolo11_weak/participation_windows.csv")
    parser.add_argument("--out-dir", default="reports/experiments/m0_heuristic_baseline")
    parser.add_argument("--min-probability", type=float, default=0.60)
    args = parser.parse_args()

    feat_path = Path(args.features)
    windows_path = Path(args.windows)

    if not feat_path.is_file() or not windows_path.is_file():
        raise FileNotFoundError("Features or windows CSV not found.")

    data = np.load(feat_path, allow_pickle=False)
    windows_df = pd.read_csv(windows_path)

    thresholds = UncertaintyThresholds(min_probability=args.min_probability)

    results: Dict[str, Any] = {
        "model": "M0-HeuristicBaselineClassifier",
        "timestamp": datetime.now().isoformat(),
        "thresholds": {
            "min_probability": thresholds.min_probability,
            "min_visible_ratio": thresholds.min_visible_ratio,
            "min_mean_keypoint_confidence": thresholds.min_mean_keypoint_confidence,
        },
        "splits": {},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        X_split = data[f"X_{split}"]
        y_split = data[f"y_{split}"]
        split_meta = windows_df[windows_df["split"] == split]

        # Only evaluate on labeled samples (y >= 0)
        labeled_mask = y_split >= 0
        X_eval = X_split[labeled_mask]
        y_eval = y_split[labeled_mask]
        phases_eval = split_meta.loc[labeled_mask, "class_phase"].fillna("UNKNOWN").tolist()

        logger.info("Evaluating M0 on %s split (N=%d)...", split, len(y_eval))
        results["splits"][split] = evaluate_split(X_eval, y_eval, phases_eval, thresholds, split)

    out_file = out_dir / "m0_baseline_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved M0 baseline results -> %s", out_file)

if __name__ == "__main__":
    main()
