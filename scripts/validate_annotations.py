"""CLI tool to validate annotation CSV files against schema and label contracts (Task T11 / T12).

Owner: Vũ Văn Phong (Data & Annotation QA)
Usage:
    python scripts/validate_annotations.py --input data/annotations/session_001_annotated.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Counter, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    validate_annotation_label,
)

REQUIRED_ANNOTATION_COLUMNS = [
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


def validate_annotation_file(file_path: Path) -> Tuple[bool, List[str], Counter]:
    errors: List[str] = []
    stats: Counter = Counter()

    if not file_path.exists():
        return False, [f"File not found: {file_path}"], stats

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        # Check required columns
        missing_cols = set(REQUIRED_ANNOTATION_COLUMNS) - set(fields)
        if missing_cols:
            errors.append(f"Missing required columns in CSV header: {sorted(missing_cols)}")
            return False, errors, stats

        for row_idx, row in enumerate(reader, start=2):  # 1-indexed, header is row 1
            stats["total_rows"] += 1
            state = row.get("annotation_state", "").strip()
            label = row.get("participation_label", "").strip()
            source = row.get("label_source", "").strip()
            rule_ver = row.get("label_rule_version", "").strip()

            stats[f"state_{state}"] += 1
            if label:
                stats[f"label_{label}"] += 1

            # 1. Check time range
            try:
                start_ms = int(row.get("t_start_ms", -1))
                end_ms = int(row.get("t_end_ms", -1))
                if start_ms < 0:
                    errors.append(f"Row {row_idx}: 't_start_ms' must be >= 0 (got {start_ms}).")
                if end_ms <= start_ms:
                    errors.append(f"Row {row_idx}: 't_end_ms' must be > 't_start_ms' (got {end_ms} <= {start_ms}).")
            except ValueError:
                errors.append(f"Row {row_idx}: 't_start_ms' and 't_end_ms' must be integers.")

            # 2. Check confidence
            try:
                conf_val = float(row.get("annotation_confidence", 1.0))
                if not 0.0 <= conf_val <= 1.0:
                    errors.append(f"Row {row_idx}: 'annotation_confidence' must be in [0, 1] (got {conf_val}).")
            except ValueError:
                errors.append(f"Row {row_idx}: 'annotation_confidence' must be a float number.")

            # 3. Check labels contract
            try:
                validate_annotation_label(
                    participation_label=label if label else None,
                    annotation_state=state,
                    label_source=source if source else None,
                    label_rule_version=rule_ver if rule_ver else None,
                )
            except ValueError as exc:
                errors.append(f"Row {row_idx}: {exc}")

    is_valid = len(errors) == 0
    return is_valid, errors, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate annotation CSV conforming to CVF25PRJ10 contracts.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to annotation CSV")
    args = parser.parse_args()

    file_path = Path(args.input)
    print(f"[INFO] Validating annotation file: {file_path} ...")
    valid, errors, stats = validate_annotation_file(file_path)

    print("\n[DATASET STATISTICS]")
    print(f"  * Total Windows: {stats['total_rows']}")
    for state in ANNOTATION_STATES:
        cnt = stats.get(f"state_{state}", 0)
        pct = cnt / stats['total_rows'] * 100 if stats['total_rows'] else 0
        print(f"    - State {state}: {cnt} ({pct:.1f}%)")

    print("\n[CANONICAL CLASS LABELS]")
    for lbl in CANONICAL_TRAINING_LABELS:
        cnt = stats.get(f"label_{lbl}", 0)
        pct = cnt / stats['total_rows'] * 100 if stats['total_rows'] else 0
        print(f"    - {lbl}: {cnt} ({pct:.1f}%)")

    if valid:
        print(f"\n[OK] Annotation CSV '{file_path.name}' is 100% valid and ready for training!")
        sys.exit(0)
    else:
        print(f"\n[FAIL] Found {len(errors)} error(s):")
        for err in errors[:20]:  # Show first 20 errors
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
