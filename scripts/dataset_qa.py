#!/usr/bin/env python3
"""Dataset QA & Annotation Agreement tool.

Validates annotation CSVs against docs/dataset_schema.md contract and computes
inter-annotator agreement (Cohen's Kappa) for double-annotated subsets (Task T12, T19).

Usage examples:
    # 1. Validate a single annotation file:
    python scripts/dataset_qa.py --validate data/annotations/session_001_phong.csv

    # 2. Check double annotation agreement between two annotators:
    python scripts/dataset_qa.py --annotator-a data/annotations/session_001_phong.csv \
                                --annotator-b data/annotations/session_001_tai.csv \
                                --adjudication-out data/annotations/session_001_disagreements.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_cohen_kappa(y1: List[str], y2: List[str], labels: List[str]) -> Tuple[float, List[List[int]]]:
    """Calculate Cohen's Kappa and confusion matrix without external ML libraries."""
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]

    for a, b in zip(y1, y2):
        if a in label_to_idx and b in label_to_idx:
            cm[label_to_idx[a]][label_to_idx[b]] += 1

    total = sum(sum(row) for row in cm)
    if total == 0:
        return 0.0, cm

    # Observed agreement po
    po = sum(cm[i][i] for i in range(k)) / total

    # Expected agreement pe
    row_sums = [sum(cm[i][j] for j in range(k)) for i in range(k)]
    col_sums = [sum(cm[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_sums[i] / total) * (col_sums[i] / total) for i in range(k))

    if abs(1.0 - pe) < 1e-9:
        kappa = 1.0 if abs(po - 1.0) < 1e-9 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return round(float(kappa), 4), cm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    validate_annotation_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dataset_qa")

REQUIRED_COLUMNS = [
    "session_id",
    "camera_id",
    "track_id",
    "t_start_ms",
    "t_end_ms",
    "class_phase",
    "atomic_behavior",
    "participation_label",
    "annotation_state",
    "label_source",
    "label_rule_version",
    "annotation_guide_version",
    "annotation_confidence",
    "occlusion_level",
    "zone_id",
    "annotator_id",
    "notes",
]


def validate_annotation_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate DataFrame against canonical annotation schema and rules.

    Returns:
        (is_valid, list of error messages)
    """
    errors: List[str] = []

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return False, errors

    for idx, row in df.iterrows():
        row_id = f"Row {idx} (track_id={row.get('track_id')}, t={row.get('t_start_ms')}-{row.get('t_end_ms')})"

        # Check time interval
        try:
            t_start = int(row["t_start_ms"])
            t_end = int(row["t_end_ms"])
            if t_start < 0:
                errors.append(f"{row_id}: t_start_ms must be >= 0, got {t_start}")
            if t_end <= t_start:
                errors.append(f"{row_id}: t_end_ms ({t_end}) must be > t_start_ms ({t_start})")
        except (ValueError, TypeError):
            errors.append(f"{row_id}: Invalid integer in t_start_ms / t_end_ms")

        # Check annotation state
        state = str(row["annotation_state"]).strip()
        if state not in ANNOTATION_STATES:
            errors.append(f"{row_id}: Unknown annotation_state '{state}'. Must be one of {ANNOTATION_STATES}")

        # Check confidence range
        try:
            conf = float(row["annotation_confidence"])
            if not (0.0 <= conf <= 1.0):
                errors.append(f"{row_id}: annotation_confidence must be in [0, 1], got {conf}")
        except (ValueError, TypeError):
            errors.append(f"{row_id}: Invalid float in annotation_confidence")

        # Check canonical participation label contract
        try:
            part_label = str(row["participation_label"]).strip() if pd.notna(row["participation_label"]) else None
            src = str(row["label_source"]).strip() if pd.notna(row["label_source"]) else None
            rule_v = str(row["label_rule_version"]).strip() if pd.notna(row["label_rule_version"]) else None

            validate_annotation_label(
                participation_label=part_label or None,
                annotation_state=state,
                label_source=src or None,
                label_rule_version=rule_v or None,
            )
        except Exception as e:
            errors.append(f"{row_id}: Contract violation - {e}")

    return len(errors) == 0, errors


def compute_double_annotation_agreement(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    annotator_a_name: str = "annotator_1",
    annotator_b_name: str = "annotator_2",
) -> Dict[str, Any]:
    """Compute Cohen's Kappa agreement and identify disagreement windows between two annotators."""
    # Match by key (session_id, camera_id, track_id, t_start_ms, t_end_ms)
    merge_keys = ["session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms"]

    merged = pd.merge(
        df_a,
        df_b,
        on=merge_keys,
        suffixes=(f"_{annotator_a_name}", f"_{annotator_b_name}"),
        how="inner",
    )

    if merged.empty:
        logger.warning("No overlapping windows matched by exact start/end time. Attempting track-time overlap matching.")
        return {"matched_count": 0, "kappa": None, "agreement_rate": None, "disagreements": pd.DataFrame()}

    col_a = f"participation_label_{annotator_a_name}"
    col_b = f"participation_label_{annotator_b_name}"

    # Filter to pairs where both gave valid labels
    valid_pairs = merged[
        merged[col_a].isin(CANONICAL_TRAINING_LABELS)
        & merged[col_b].isin(CANONICAL_TRAINING_LABELS)
    ].copy()

    total_valid = len(valid_pairs)
    if total_valid == 0:
        return {"matched_count": len(merged), "valid_pairs": 0, "kappa": None, "agreement_rate": None, "disagreements": merged}

    agree_count = int((valid_pairs[col_a] == valid_pairs[col_b]).sum())
    agreement_rate = round(float(agree_count / total_valid), 4)

    labels = list(CANONICAL_TRAINING_LABELS)
    y_a = list(valid_pairs[col_a].astype(str))
    y_b = list(valid_pairs[col_b].astype(str))

    kappa, cm = compute_cohen_kappa(y_a, y_b, labels=labels)

    disagreements = valid_pairs[valid_pairs[col_a] != valid_pairs[col_b]].copy()

    return {
        "matched_count": len(merged),
        "valid_pairs_count": total_valid,
        "agree_count": agree_count,
        "agreement_rate": agreement_rate,
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "labels": labels,
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Annotation QA & Agreement Checker.")
    parser.add_argument("--validate", type=str, help="Validate an annotation CSV against canonical schema.")
    parser.add_argument("--annotator-a", type=str, help="First annotator CSV for double annotation check.")
    parser.add_argument("--annotator-b", type=str, help="Second annotator CSV for double annotation check.")
    parser.add_argument("--adjudication-out", type=str, help="Path to save disagreements CSV for adjudication review.")
    args = parser.parse_args()

    if args.validate:
        p = Path(args.validate)
        if not p.is_file():
            logger.error("File not found: %s", p)
            sys.exit(1)
        df = pd.read_csv(p)
        is_valid, errors = validate_annotation_dataframe(df)
        if is_valid:
            logger.info("PASS: '%s' strictly satisfies dataset schema! (Total rows: %d)", p.name, len(df))
        else:
            logger.error("FAIL: Found %d schema errors in '%s':", len(errors), p.name)
            for e in errors[:15]:
                logger.error("  - %s", e)
            if len(errors) > 15:
                logger.error("  ... and %d more errors.", len(errors) - 15)
            sys.exit(1)

    elif args.annotator_a and args.annotator_b:
        df_a = pd.read_csv(args.annotator_a)
        df_b = pd.read_csv(args.annotator_b)

        res = compute_double_annotation_agreement(df_a, df_b)
        logger.info("Matched windows: %d | Valid labeled pairs: %d", res["matched_count"], res.get("valid_pairs_count", 0))
        logger.info("Agreement Rate: %.2f%% | Cohen's Kappa: %s",
                    (res["agreement_rate"] or 0) * 100.0, res["cohen_kappa"])

        disagreements = res["disagreements"]
        if not disagreements.empty and args.adjudication_out:
            out_p = Path(args.adjudication_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            disagreements.to_csv(out_p, index=False)
            logger.info("Saved %d disagreements for adjudication review to: %s", len(disagreements), out_p)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
