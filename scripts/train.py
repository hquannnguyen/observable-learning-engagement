#!/usr/bin/env python3
"""CLI script to train LSTM classifier for observable learning participation.

Usage:
    python scripts/train.py --features data/features/window_features.npz \
                           --config configs/baseline.yaml \
                           --name lstm_baseline \
                           --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.lstm import ParticipationLSTM
from src.training.dataset import WindowFeatureDataset
from src.training.trainer import compute_multiclass_metrics, evaluate_model, train_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM Participation Classifier.")
    parser.add_argument("--features", type=str, default="data/features/window_features.npz", help="Path to window_features.npz.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML.")
    parser.add_argument("--name", type=str, default="lstm_baseline", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs to train.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda').")
    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.is_file():
        logger.error("Features file not found: %s. Run scripts/build_dataset.py first.", feat_path)
        sys.exit(1)

    # Load baseline config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    proj_cfg = cfg.get("project", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    seed = int(proj_cfg.get("seed", 42))
    set_seed(seed)

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    lr = float(args.lr if args.lr is not None else train_cfg.get("learning_rate", 0.001))
    hidden_size = int(model_cfg.get("hidden_size", 128))
    num_layers = int(model_cfg.get("num_layers", 2))
    num_classes = int(model_cfg.get("num_classes", 3))
    dropout = float(model_cfg.get("dropout", 0.30))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s | Seed: %d | Batch size: %d | LR: %f", device, seed, batch_size, lr)

    data = np.load(feat_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info("Loaded datasets: Train=%d, Val=%d, Test=%d windows", len(X_train), len(X_val), len(X_test))

    train_ds = WindowFeatureDataset(X_train, y_train)
    val_ds = WindowFeatureDataset(X_val, y_val)
    test_ds = WindowFeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[-1] if len(X_train) > 0 else 77
    model = ParticipationLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    # Class-weighted CrossEntropyLoss for imbalance
    class_weights = train_ds.compute_class_weights(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    patience_counter = 0
    best_epoch = 0

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{args.name}.pt"

    exp_dir = Path("reports/experiments") / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    history = []
    logger.info("Starting LSTM training for %d epochs...", args.epochs)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
        val_f1 = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": val_f1,
            "val_acc": val_metrics["accuracy"],
        })

        logger.info(
            "Epoch %02d/%02d | Tr Loss: %.4f, Tr Acc: %.3f | Val Loss: %.4f, Val F1: %.4f",
            epoch, args.epochs, tr_loss, tr_acc, val_loss, val_f1
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_classes": num_classes,
            }, best_ckpt_path)
            logger.info("  --> Saved new best model checkpoint (Val Macro-F1: %.4f)", best_val_f1)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d (patience=%d).", epoch, args.patience)
                break

    # Load best model for test set evaluation
    logger.info("Loading best checkpoint from epoch %d for Test evaluation...", best_epoch)
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes)
    logger.info("================ FINAL TEST EVALUATION ================")
    logger.info("  Test Macro-F1:         %.4f", test_metrics["macro_f1"])
    logger.info("  Test Balanced Acc:     %.4f", test_metrics["balanced_accuracy"])
    logger.info("  Test Overall Acc:      %.4f", test_metrics["accuracy"])
    logger.info("  Confusion Matrix:      %s", test_metrics["confusion_matrix"])
    logger.info("  Per-class metrics:     %s", json.dumps(test_metrics["per_class"], indent=2))
    logger.info("=======================================================")

    results = {
        "experiment_name": args.name,
        "model_type": "LSTM",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(best_ckpt_path),
    }

    metrics_out = exp_dir / "metrics.json"
    metrics_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved experiment artifacts to: %s", exp_dir)


if __name__ == "__main__":
    main()
