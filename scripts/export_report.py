#!/usr/bin/env python3
"""Export anonymized report artifacts from an evaluation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.exporter import export_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results/timeline/zone/summary.")
    parser.add_argument("--experiment-dir", default=None, help="Experiment folder with metadata.json + predictions.csv.")
    parser.add_argument("--predictions", default=None, help="Direct path to predictions.csv.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <experiment>/export).")
    parser.add_argument("--bucket-seconds", type=int, default=60)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir) if args.experiment_dir else None
    metadata = {}
    if exp_dir is not None:
        meta_file = exp_dir / "metadata.json"
        if meta_file.is_file():
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))

    if args.predictions:
        pred_path = Path(args.predictions)
    elif exp_dir is not None:
        pred_path = exp_dir / "predictions.csv"
    else:
        parser.error("Provide --experiment-dir or --predictions")

    if not pred_path.is_file():
        raise FileNotFoundError(f"predictions not found: {pred_path}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (exp_dir / "export" if exp_dir else pred_path.parent / "export")
    )

    predictions = pd.read_csv(pred_path)
    paths = export_bundle(
        predictions,
        out_dir,
        experiment_id=metadata.get("experiment_id"),
        metric_scope=metadata.get("metric_scope", "weak_rule_labels"),
        bucket_seconds=args.bucket_seconds,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
