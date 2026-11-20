"""Model training and evaluation loop with Macro-F1 metric and early stopping.

Implements Task T23 according to docs/experiment_protocol.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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


class TemperatureScaler(nn.Module):
    """Post-processing temperature scaling for probability calibration (Guo et al. 2017)."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    def fit(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Optimize temperature parameter T by minimizing NLL on validation set."""
        self.train()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.eval()
        return float(torch.clamp(self.temperature, min=0.01).item())


def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if len(probs) == 0:
        return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == targets

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return round(float(ece), 4)


def compute_accuracy_coverage_curve(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute Accuracy vs Coverage curve across varying confidence thresholds."""
    if thresholds is None:
        thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    total = len(targets)

    curve_rows = []
    for tau in thresholds:
        accepted_mask = confidences >= tau
        n_accepted = int(np.sum(accepted_mask))
        coverage = round(n_accepted / max(total, 1), 4)

        if n_accepted > 0:
            acc = round(float(np.mean(predictions[accepted_mask] == targets[accepted_mask])), 4)
        else:
            acc = 1.0

        curve_rows.append({
            "threshold": tau,
            "coverage": coverage,
            "accuracy": acc,
            "n_accepted": n_accepted,
            "n_total": total,
        })

    return pd.DataFrame(curve_rows)
