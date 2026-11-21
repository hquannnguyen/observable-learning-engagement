#!/usr/bin/env python3
"""Evaluate a trained model with the uncertainty gate and per-window audit trail.

Implements remediation A4:
1. Loads a checkpoint from models/checkpoints/<model>.pt.
2. Joins test tensors to window metadata by ``window_id`` (never by row order).
3. Applies the quality + model-confidence gate and records the gate reason.
4. Writes predictions.csv plus coverage / accepted / per-session metrics.
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

from src.contracts.labels import (
    CANONICAL_TRAINING_LABELS,
    INDEX_TO_LABEL,
    UNCERTAIN_LABEL,
)
from src.models.factory import model_from_checkpoint
from src.training.trainer import (
    compute_accuracy_coverage_curve,
    compute_ece,
    predict_logits,
    compute_multiclass_metrics,
)
from src.uncertainty.gate import UncertaintyThresholds, explain_uncertainty_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")

REQUIRED_META_COLUMNS = (
    "window_id",
    "split",
    "session_id",
    "track_id",
    "participation_label",
    "mean_visible_ratio",
    "mean_pose_confidence",
)


def load_test_metadata(meta_path: Path) -> pd.DataFrame:
    """Load test-split window metadata indexed by unique window_id."""
    if not meta_path.is_file():
        raise FileNotFoundError(f"Window metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path)
    missing_cols = [c for c in REQUIRED_META_COLUMNS if c not in meta.columns]
    if missing_cols:
        raise ValueError(f"Window metadata missing columns: {missing_cols}")

    test_meta = meta[meta["split"] == "test"].copy()
    if test_meta.empty:
        raise ValueError("No rows with split=='test' in window metadata")
    duplicated = test_meta["window_id"].duplicated()
    if duplicated.any():
        dupes = test_meta.loc[duplicated, "window_id"].head(5).tolist()
        raise ValueError(f"Duplicate window_id in test metadata: {dupes}")
    return test_meta.set_index("window_id")


def metrics_or_none(y_true: List[int], y_pred: List[int]) -> Dict[str, Any] | None:
    if not y_true:
        return None
    return compute_multiclass_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        num_classes=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model with uncertainty gating.")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Experiment folder created by scripts/train.py.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt (defaults to experiment metadata).")
    parser.add_argument("--features", type=str, default="data/features/window_features.npz", help="Path to window_features.npz.")
    parser.add_argument("--windows-meta", type=str, default="data/features/participation_windows.csv", help="Path to windows metadata CSV.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to baseline config.")
    parser.add_argument("--output", type=str, default=None, help="Path to save report (defaults to <experiment>/evaluation_report.json).")
    parser.add_argument("--predictions", type=str, default=None, help="Path to save per-window predictions CSV.")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir) if args.experiment_dir else None
    exp_metadata: Dict[str, Any] = {}
    if exp_dir is not None:
        meta_file = exp_dir / "metadata.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"Experiment metadata not found: {meta_file}")
        exp_metadata = json.loads(meta_file.read_text(encoding="utf-8"))

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif exp_metadata.get("checkpoint_path"):
        ckpt_path = Path(exp_metadata["checkpoint_path"])
    else:
        ckpt_path = Path("models/checkpoints/lstm_baseline.pt")

    feat_path = Path(args.features)
    meta_path = Path(args.windows_meta)
    if args.output:
        out_file = Path(args.output)
    elif exp_dir is not None:
        out_file = exp_dir / "evaluation_report.json"
    else:
        out_file = Path("reports/experiments/lstm_baseline/evaluation_report.json")

    if not ckpt_path.is_file():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)
    if not feat_path.is_file():
        logger.error("Features not found: %s. Run scripts/build_dataset.py first.", feat_path)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    unc_cfg = cfg.get("uncertainty", {})
    qg_cfg = cfg.get("quality_gate", {})

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    calibration = checkpoint.get("calibration") or {}
    temperature = float(calibration.get("temperature", 1.0))
    thresholds = UncertaintyThresholds(
        min_visible_ratio=float(qg_cfg.get("min_visible_ratio", 0.60)),
        min_mean_keypoint_confidence=float(qg_cfg.get("min_mean_keypoint_confidence", 0.30)),
        min_probability=float(calibration.get("min_probability", unc_cfg.get("min_probability", 0.60))),
        upper_body_min_visible_ratio=float(qg_cfg.get("upper_body_min_visible_ratio", 0.72))
        if qg_cfg.get("allow_upper_body_visibility") else None,
    )

    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = np.load(feat_path, allow_pickle=False)
    for key in ("X_test", "y_test", "window_ids_test"):
        if key not in data.files:
            raise KeyError(
                f"'{key}' missing from {feat_path}. Rebuild the dataset with "
                "scripts/build_dataset.py so window ids are stored."
            )

    X_test = data["X_test"]
    y_test = data["y_test"].astype(np.int64)
    window_ids = data["window_ids_test"].astype(str)

    if not (len(X_test) == len(y_test) == len(window_ids)):
        raise ValueError(
            "Tensor/metadata length mismatch: "
            f"X={len(X_test)}, y={len(y_test)}, ids={len(window_ids)}"
        )

    meta_by_id = load_test_metadata(meta_path)

    valid_mask = y_test >= 0
    X_valid = X_test[valid_mask]
    y_valid = y_test[valid_mask]
    ids_valid = window_ids[valid_mask]
    N = len(y_valid)
    excluded = int((~valid_mask).sum())
    logger.info(
        "Test windows: %d total, %d evaluated, %d excluded (no valid label)",
        len(y_test),
        N,
        excluded,
    )

    missing_ids = [wid for wid in ids_valid if wid not in meta_by_id.index]
    if missing_ids:
        raise ValueError(
            "Test tensors reference window ids absent from metadata: "
            f"{missing_ids[:5]} (total {len(missing_ids)}). Rebuild metadata and tensors together."
        )

    if N == 0:
        raise ValueError("Test split has no labeled windows.")
    logits = predict_logits(model, X_valid)
    probs = (logits / temperature).softmax(dim=-1).numpy()

    records: List[Dict[str, Any]] = []
    accepted_preds: List[int] = []
    accepted_targets: List[int] = []
    gate_reason_counts: Dict[str, int] = {}

    for i in range(N):
        wid = str(ids_valid[i])
        row = meta_by_id.loc[wid]
        true_idx = int(y_valid[i])
        true_label = INDEX_TO_LABEL[true_idx]

        # Contract check: metadata and tensor must describe the same label.
        meta_label = row["participation_label"]
        if pd.isna(meta_label) or str(meta_label) != true_label:
            raise ValueError(
                f"Label mismatch for window {wid}: tensor={true_label}, metadata={meta_label!r}. "
                "Tensors and metadata are out of sync; rebuild the dataset."
            )

        vis = float(row["mean_visible_ratio"])
        conf = float(row["mean_pose_confidence"])
        gated_label, reason = explain_uncertainty_gate(
            probabilities=probs[i],
            visible_ratio=vis,
            mean_keypoint_confidence=conf,
            upper_body_visible_ratio=row.get("mean_upper_body_visible_ratio"),
            thresholds=thresholds,
            class_labels=CANONICAL_TRAINING_LABELS,
        )
        gate_reason_counts[reason] = gate_reason_counts.get(reason, 0) + 1

        pred_idx = int(np.argmax(probs[i]))
        records.append(
            {
                "window_id": wid,
                "session_id": row["session_id"],
                "track_id": int(row["track_id"]),
                "t_start_ms": int(row["t_start_ms"]) if "t_start_ms" in row else None,
                "t_end_ms": int(row["t_end_ms"]) if "t_end_ms" in row else None,
                "true_label": true_label,
                "pred_label": INDEX_TO_LABEL[pred_idx],
                "gated_label": gated_label,
                "p_max": round(float(probs[i][pred_idx]), 6),
                "gate_reason": reason,
                "label_source": row["label_source"] if "label_source" in row else None,
                "class_phase": (
                    None
                    if "class_phase" not in row or pd.isna(row["class_phase"])
                    else str(row["class_phase"])
                ),
                "zone_id": (
                    None
                    if "zone_id" not in row or pd.isna(row["zone_id"])
                    else str(row["zone_id"])
                ),
                "mean_visible_ratio": vis,
                "mean_pose_confidence": conf,
            }
        )

        if gated_label != UNCERTAIN_LABEL:
            accepted_preds.append(pred_idx)
            accepted_targets.append(true_idx)

    predictions_df = pd.DataFrame(records)
    pred_path = (
        Path(args.predictions)
        if args.predictions
        else out_file.parent / "predictions.csv"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(pred_path, index=False)
    logger.info("Saved per-window predictions to: %s", pred_path)

    coverage = len(accepted_preds) / max(N, 1)
    uncertain_rate = 1.0 - coverage
    accepted_metrics = metrics_or_none(accepted_targets, accepted_preds) or {}

    raw_preds = np.argmax(probs, axis=-1)
    raw_metrics = compute_multiclass_metrics(y_valid, raw_preds, num_classes=3)

    # Per-session breakdown (N is always reported with the metric).
    by_session: Dict[str, Dict[str, Any]] = {}
    for session_id, group in predictions_df.groupby("session_id"):
        accepted = group[group["gated_label"] != UNCERTAIN_LABEL]
        yt = [CANONICAL_TRAINING_LABELS.index(v) for v in accepted["true_label"]]
        yp = [CANONICAL_TRAINING_LABELS.index(v) for v in accepted["pred_label"]]
        by_session[str(session_id)] = {
            "windows": int(len(group)),
            "accepted": int(len(accepted)),
            "uncertain": int(len(group) - len(accepted)),
            "coverage": round(len(accepted) / max(len(group), 1), 4),
            "macro_f1": (metrics_or_none(yt, yp) or {}).get("macro_f1"),
        }

    label_source_counts = (
        predictions_df["label_source"].fillna("NONE").astype(str).value_counts().to_dict()
    )

    logger.info("================ UNCERTAINTY GATED EVALUATION ================")
    logger.info("  Test windows (total):       %d", len(y_test))
    logger.info("  Evaluated windows:          %d", N)
    logger.info("  Accepted Predictions:       %d (Coverage: %.2f%%)", len(accepted_preds), coverage * 100.0)
    logger.info("  Uncertain Windows:          %d (Rate: %.2f%%)", N - len(accepted_preds), uncertain_rate * 100.0)
    logger.info("  Gate reasons:               %s", gate_reason_counts)
    logger.info("  Raw Test Macro-F1:          %.4f", raw_metrics["macro_f1"])
    logger.info("  Accepted Test Macro-F1:     %.4f", accepted_metrics.get("macro_f1", 0.0))
    logger.info("  Accepted Test Accuracy:     %.4f", accepted_metrics.get("accuracy", 0.0))
    logger.info("==============================================================")

    sources = {
        str(s) for s in predictions_df["label_source"].dropna().unique().tolist()
    }
    if sources == {"WEAK_RULE"}:
        metric_scope = "weak_rule_labels"
    elif sources and sources <= {"HUMAN", "ADJUDICATED"}:
        metric_scope = "human_labels"
    else:
        metric_scope = "mixed_labels"

    ece = compute_ece(probs, y_valid)
    eligible = predictions_df.gate_reason.isin(["ACCEPTED", "LOW_MODEL_CONF"]).to_numpy()
    acc_cov_df = compute_accuracy_coverage_curve(probs, y_valid, eligible=eligible)
    logger.info("  Expected Calibration Error (ECE): %.4f", ece)

    report = {
        "checkpoint": str(ckpt_path),
        "experiment_id": exp_metadata.get("experiment_id"),
        "total_test_windows": int(len(y_test)),
        "evaluated_windows": int(N),
        "excluded_unlabeled_windows": excluded,
        "coverage": round(coverage, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "calibration": {
            "temperature": temperature,
            "validation_fit": calibration or None,
            "expected_calibration_error": ece,
            "accuracy_vs_coverage": acc_cov_df.astype(object).where(acc_cov_df.notna(), None).to_dict(orient="records"),
        },
        "thresholds": {
            "min_probability": thresholds.min_probability,
            "min_visible_ratio": thresholds.min_visible_ratio,
            "min_mean_keypoint_confidence": thresholds.min_mean_keypoint_confidence,
            "upper_body_min_visible_ratio": thresholds.upper_body_min_visible_ratio,
        },
        "label_source_counts": {str(k): int(v) for k, v in label_source_counts.items()},
        "gate_reason_counts": gate_reason_counts,
        "raw_metrics": raw_metrics,
        "accepted_metrics": accepted_metrics,
        "by_session": by_session,
        "predictions_file": str(pred_path),
        "metric_scope": metric_scope,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to: %s", out_file)

    if exp_dir is not None:
        exp_metadata.update(
            {
                "evaluation_report": str(out_file),
                "predictions_file": str(pred_path),
                "metric_scope": metric_scope,
                "evaluated_windows": int(N),
                "coverage": round(coverage, 4),
                "accepted_macro_f1": accepted_metrics.get("macro_f1"),
                "raw_macro_f1": raw_metrics.get("macro_f1"),
            }
        )
        (exp_dir / "metadata.json").write_text(
            json.dumps(exp_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
