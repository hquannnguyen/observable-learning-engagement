"""Model training and evaluation loop with Macro-F1 metric and early stopping.

Implements Task T23 according to docs/experiment_protocol.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.contracts.labels import CANONICAL_TRAINING_LABELS

logger = logging.getLogger(__name__)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> Dict[str, Any]:
    """Compute Macro-F1, per-class Precision/Recall/F1, and Confusion Matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores = []
    recalls = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        label_name = CANONICAL_TRAINING_LABELS[c] if c < len(CANONICAL_TRAINING_LABELS) else f"class_{c}"
        per_class[label_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(cm[c, :])),
        }
        f1_scores.append(f1)
        recalls.append(rec)

    macro_f1 = float(np.mean(f1_scores))
    balanced_acc = float(np.mean(recalls))
    overall_acc = float(np.trace(cm) / np.maximum(np.sum(cm), 1))

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "accuracy": round(overall_acc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate model on a DataLoader. Returns (avg_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = y.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    avg_loss = total_loss / max(len(all_targets), 1)
    metrics = compute_multiclass_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        num_classes=num_classes,
    )
    return avg_loss, metrics
