"""CLI script to generate weak supervision labels automatically (Task T16).

Owner: Vũ Văn Phong (Data & Annotation Lead)
Usage:
    python scripts/auto_label_weak.py --poses data/poses/cam_A_pilot_poses.parquet --output data/annotations/session_001_weak_labels.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.preprocessing.weak_labels import RULE_VERSION, apply_labeling_rules
from scripts.validate_annotations import validate_annotation_file


def get_phase_at_timestamp(timestamp_ms: int, phases: List[Dict[str, Any]]) -> str:
    for p in phases:
        if p.get("start_ms", 0) <= timestamp_ms < p.get("end_ms", 0):
            return str(p.get("name", "lecture"))
    return "lecture"


def auto_label_session(
    poses_path: Path,
    metadata_path: Path,
    output_path: Path,
    window_sec: float = 4.0,
    stride_sec: float = 2.0,
) -> None:
    if not poses_path.exists():
        raise FileNotFoundError(f"Pose file not found: {poses_path}")

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    session_id = metadata.get("session_id", "session_001")
    camera_id = metadata.get("camera_id", "cam_A")
    phases = metadata.get("class_phase", [])

    print(f"[INFO] Reading poses: {poses_path}")
    df_poses = pd.read_parquet(poses_path) if poses_path.suffix == ".parquet" else pd.read_csv(poses_path)

    window_ms = int(window_sec * 1000)
    stride_ms = int(stride_sec * 1000)

    rows = []
    print("[INFO] Applying Weak Supervision Labeling Functions...")

    for track_id, group in df_poses.groupby("track_id"):
        group = group.sort_values("timestamp_ms")
        min_ts = int(group["timestamp_ms"].min())
        max_ts = int(group["timestamp_ms"].max())

        t_start = min_ts
        while t_start + window_ms <= max_ts:
            t_end = t_start + window_ms
            w_df = group[(group["timestamp_ms"] >= t_start) & (group["timestamp_ms"] <= t_end)]

            if len(w_df) >= 3:
                last_zone = str(w_df["zone_id"].iloc[-1])
                phase = get_phase_at_timestamp(t_start, phases)
                mean_vis = float(w_df["visible_ratio"].mean())

                # Apply weak labeling rules
                state, label, behaviors, conf = apply_labeling_rules(w_df, class_phase=phase)

                is_valid = (state == "VALID")
                occlusion_lvl = "none" if mean_vis >= 0.75 else ("moderate" if mean_vis >= 0.50 else "heavy")

                rows.append({
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "t_start_ms": t_start,
                    "t_end_ms": t_end,
                    "class_phase": phase,
                    "atomic_behavior": ";".join(behaviors),
                    "participation_label": label if is_valid else "",
                    "annotation_state": state,
                    "label_source": "WEAK_RULE" if is_valid else "",
                    "label_rule_version": RULE_VERSION if is_valid else "",
                    "annotation_guide_version": "v0.2",
                    "annotation_confidence": str(round(conf, 2)),
                    "occlusion_level": occlusion_lvl,
                    "zone_id": last_zone,
                    "annotator_id": f"RuleBot_{RULE_VERSION}",
                    "notes": f"auto_vis_ratio={mean_vis:.2f}",
                })

            t_start += stride_ms

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(out_df)} weak supervision annotations.")
    print(f"[INFO] Saved to: {output_path}")

    # Validate output
    print(f"\n[INFO] Validating generated file against contracts...")
    is_valid, errors, stats = validate_annotation_file(output_path)
    if is_valid:
        print("[OK] Generated weak labels are 100% compliant with CVF25PRJ10 contracts!")
        print("\n[SUMMARY BREAKDOWN]")
        for k, v in stats.items():
            if k.startswith("label_") or k.startswith("state_"):
                pct = v / stats["total_rows"] * 100
                print(f"  * {k}: {v} ({pct:.1f}%)")
    else:
        print(f"[FAIL] Found {len(errors)} validation error(s):")
        for err in errors[:5]:
            print(f"  - {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate weak supervision labels.")
    parser.add_argument("--poses", type=str, default="data/poses/cam_A_pilot_poses.parquet", help="Path to pose parquet")
    parser.add_argument("--metadata", type=str, default="data/raw/session_001/metadata.json", help="Path to metadata JSON")
    parser.add_argument("--output", "-o", type=str, default="data/annotations/session_001_weak_labels.csv", help="Output CSV path")

    args = parser.parse_args()
    auto_label_session(
        poses_path=Path(args.poses),
        metadata_path=Path(args.metadata),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
