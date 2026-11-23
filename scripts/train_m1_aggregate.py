#!/usr/bin/env python3
"""Baseline M1: Aggregated Linear & MLP Classifier for Observable Learning Participation.

Implements Task T21 (Baseline M1) as specified in IMPLEMENTATION_PLAN.md:
1. Takes temporal window tensors (N, T, F) and extracts non-temporal summary features:
   mean, std, min, max across the time axis (resulting in 4 * F dimensions).
2. Fits StandardScaler on the training split only.
3. Trains Logistic Regression and MLP baselines with balanced class weights.
4. Evaluates on Validation and Test splits, saving metrics and predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_m1")

CLASS_NAMES = [
    "HIGH_OBSERVED_PARTICIPATION",
    "MODERATE_OBSERVED_PARTICIPATION",
    "LOW_OBSERVED_PARTICIPATION",
]

def pool_features(X: np.ndarray) -> np.ndarray:
    """Aggregate (N, T, F) -> (N, 4 * F) using mean, std, min, max."""
    mean_feat = np.mean(X, axis=1)
    std_feat = np.std(X, axis=1)
    min_feat = np.min(X, axis=1)
    max_feat = np.max(X, axis=1)
    return np.hstack([mean_feat, std_feat, min_feat, max_feat])

def evaluate_classifier(clf: Any, X: np.ndarray, y: np.ndarray, split_name: str) -> Dict[str, Any]:
    y_pred = clf.predict(X)
    probs = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

    # Macro F1 (accounting for possible absent classes in test split)
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2]).tolist()

    report = classification_report(
        y, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    logger.info(
        "[%s] Macro-F1: %.4f | Balanced Acc: %.4f",
        split_name, macro_f1, bal_acc
    )
    for idx, cname in enumerate(CLASS_NAMES):
        sub = report.get(cname, {})
        logger.info(
            "  -> %s: F1=%.4f (P=%.4f, R=%.4f, Support=%d)",
            cname.replace("_OBSERVED_PARTICIPATION", ""),
            sub.get("f1-score", 0.0),
            sub.get("precision", 0.0),
            sub.get("recall", 0.0),
            sub.get("support", 0),
        )

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

def main():
    parser = argparse.ArgumentParser(description="Train Baseline M1 (Aggregated features).")
    parser.add_argument("--features", default="data/features_v02_yolo11_weak/window_features.npz")
    parser.add_argument("--out-dir", default="reports/experiments/m1_aggregate_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", choices=("logistic", "mlp", "both"), default="both")
    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.is_file():
        raise FileNotFoundError(f"Tensor features file not found: {feat_path}")

    data = np.load(feat_path, allow_pickle=False)
    X_train_raw, y_train_raw = data["X_train"], data["y_train"]
    X_val_raw, y_val_raw = data["X_val"], data["y_val"]
    X_test_raw, y_test_raw = data["X_test"], data["y_test"]

    # Filter out y == -1 (invalid/unlabeled)
    train_mask = y_train_raw >= 0
    val_mask = y_val_raw >= 0
    test_mask = y_test_raw >= 0

    X_train_seq, y_train = X_train_raw[train_mask], y_train_raw[train_mask]
    X_val_seq, y_val = X_val_raw[val_mask], y_val_raw[val_mask]
    X_test_seq, y_test = X_test_raw[test_mask], y_test_raw[test_mask]

    logger.info("Loaded dataset: Train=%d, Val=%d, Test=%d windows", len(y_train), len(y_val), len(y_test))

    # Aggregate features across time: (N, 40, 94) -> (N, 376)
    logger.info("Aggregating temporal features (mean, std, min, max)...")
    X_train_agg = pool_features(X_train_seq)
    X_val_agg = pool_features(X_val_seq)
    X_test_agg = pool_features(X_test_seq)
    logger.info("Pooled feature dimension: %d", X_train_agg.shape[1])

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_agg)
    X_val = scaler.transform(X_val_agg)
    X_test = scaler.transform(X_test_agg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "dataset": str(feat_path),
        "seed": args.seed,
        "feature_dim": int(X_train.shape[1]),
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    # 1. Logistic Regression
    if args.model_type in ("logistic", "both"):
        logger.info("--- Training Logistic Regression Baseline (M1-LogReg) ---")
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed)
        lr.fit(X_train, y_train)

        results["models"]["logistic_regression"] = {
            "val": evaluate_classifier(lr, X_val, y_val, "Val-Logistic"),
            "test": evaluate_classifier(lr, X_test, y_test, "Test-Logistic"),
        }

    # 2. MLP
    if args.model_type in ("mlp", "both"):
        logger.info("--- Training MLP Baseline (M1-MLP) ---")
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=args.seed, early_stopping=True)
        mlp.fit(X_train, y_train)

        results["models"]["mlp"] = {
            "val": evaluate_classifier(mlp, X_val, y_val, "Val-MLP"),
            "test": evaluate_classifier(mlp, X_test, y_test, "Test-MLP"),
        }

    out_file = out_dir / "m1_baseline_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved M1 baseline results -> %s", out_file)

if __name__ == "__main__":
    main()
