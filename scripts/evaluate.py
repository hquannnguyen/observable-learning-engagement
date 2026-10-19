#!/usr/bin/env python3
"""Evaluate trained model with uncertainty gate and per-zone analysis.

Implements Task T24, T30:
1. Loads checkpoint from models/checkpoints/<model>.pt.
2. Evaluates on test windows with Uncertainty Gate (min_prob=0.60).
3. Computes Coverage, Accepted Macro-F1, and Uncertain Rate.
4. Generates evaluation report adhering to docs/dataset_schema.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.labels import CANONICAL_TRAINING_LABELS, UNCERTAIN_LABEL
from src.models.lstm import ParticipationLSTM
from src.training.trainer import compute_multiclass_metrics
from src.uncertainty.gate import UncertaintyThresholds, apply_uncertainty_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model with uncertainty gating.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/lstm_baseline.pt", help="Path to checkpoint .pt.")
    parser.add_argument("--features", type=str, default="data/features/window_features.npz", help="Path to window_features.npz.")
    parser.add_argument("--windows-meta", type=str, default="data/features/participation_windows.csv", help="Path to windows metadata CSV.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to baseline config.")
    parser.add_argument("--output", type=str, default="reports/experiments/lstm_baseline/evaluation_report.json", help="Path to save report.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    feat_path = Path(args.features)
    meta_path = Path(args.windows_meta)

    if not ckpt_path.is_file():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    # Load baseline config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    unc_cfg = cfg.get("uncertainty", {})
    qg_cfg = cfg.get("quality_gate", {})

    thresholds = UncertaintyThresholds(
        min_visible_ratio=float(qg_cfg.get("min_visible_ratio", 0.60)),
        min_mean_keypoint_confidence=float(qg_cfg.get("min_mean_keypoint_confidence", 0.30)),
        min_probability=float(unc_cfg.get("min_probability", 0.60)),
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = ParticipationLSTM(
        input_size=int(checkpoint.get("input_size", 77)),
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        num_classes=int(checkpoint.get("num_classes", 3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = np.load(feat_path)
    X_test, y_test = data["X_test"], data["y_test"]

    # Filter out unlabeled (-1)
    valid_mask = y_test >= 0
    X_test_valid = X_test[valid_mask]
    y_test_valid = y_test[valid_mask]

    N = len(y_test_valid)
    logger.info("Evaluating on %d test windows...", N)

    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test_valid).float()
        probs = model.predict_proba(x_tensor).numpy()

    # Read windows metadata for visible ratio and confidence if available
    if meta_path.is_file():
        win_df = pd.read_csv(meta_path)
        test_win_df = win_df[(win_df["split"] == "test") & (win_df["participation_label"].notna())].copy()
    else:
        test_win_df = pd.DataFrame()

    accepted_preds = []
    accepted_targets = []
    uncertain_count = 0

    gated_outputs = []
    for i in range(N):
        vis = (
            float(test_win_df["mean_visible_ratio"].iloc[i])
            if i < len(test_win_df) and "mean_visible_ratio" in test_win_df.columns
            else 1.0
        )
        conf = (
            float(test_win_df["mean_pose_confidence"].iloc[i])
            if i < len(test_win_df) and "mean_pose_confidence" in test_win_df.columns
            else 1.0
        )

        gated_label = apply_uncertainty_gate(
            probabilities=probs[i],
            visible_ratio=vis,
            mean_keypoint_confidence=conf,
            thresholds=thresholds,
            class_labels=CANONICAL_TRAINING_LABELS,
        )

        gated_outputs.append(gated_label)
        if gated_label == UNCERTAIN_LABEL:
            uncertain_count += 1
        else:
            pred_idx = CANONICAL_TRAINING_LABELS.index(gated_label)
            accepted_preds.append(pred_idx)
            accepted_targets.append(y_test_valid[i])

    coverage = (N - uncertain_count) / max(N, 1)
    uncertain_rate = uncertain_count / max(N, 1)

    accepted_metrics = (
        compute_multiclass_metrics(np.array(accepted_targets), np.array(accepted_preds), num_classes=3)
        if accepted_preds
        else {}
    )

    raw_preds = np.argmax(probs, axis=-1)
    raw_metrics = compute_multiclass_metrics(y_test_valid, raw_preds, num_classes=3)

    logger.info("================ UNCERTAINTY GATED EVALUATION ================")
    logger.info("  Total Test Windows:         %d", N)
    logger.info("  Accepted Predictions:       %d (Coverage: %.2f%%)", len(accepted_preds), coverage * 100.0)
    logger.info("  Uncertain Windows:          %d (Rate: %.2f%%)", uncertain_count, uncertain_rate * 100.0)
    logger.info("  Raw Test Macro-F1:          %.4f", raw_metrics["macro_f1"])
    logger.info("  Accepted Test Macro-F1:     %.4f", accepted_metrics.get("macro_f1", 0.0))
    logger.info("  Accepted Test Accuracy:     %.4f", accepted_metrics.get("accuracy", 0.0))
    logger.info("==============================================================")

    report = {
        "checkpoint": str(ckpt_path),
        "total_test_windows": N,
        "coverage": round(coverage, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "thresholds": {
            "min_probability": thresholds.min_probability,
            "min_visible_ratio": thresholds.min_visible_ratio,
            "min_mean_keypoint_confidence": thresholds.min_mean_keypoint_confidence,
        },
        "raw_metrics": raw_metrics,
        "accepted_metrics": accepted_metrics,
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to: %s", out_file)


if __name__ == "__main__":
    main()
