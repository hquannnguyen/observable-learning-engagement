"""Generate sliding-window annotation template CSV from extracted poses (Task T11).

Owner: Vũ Văn Phong (Data & Annotation QA)
Usage:
    python scripts/prepare_annotation_template.py --poses data/poses/cam_A_pilot_poses.parquet --metadata data/raw/session_001/metadata.json --output data/annotations/session_001_annotation_template.csv --annotator-id Phong
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def get_phase_at_timestamp(timestamp_ms: int, phases: List[Dict[str, Any]]) -> str:
    """Find active class phase for a given timestamp."""
    for p in phases:
        if p.get("start_ms", 0) <= timestamp_ms < p.get("end_ms", 0):
            return str(p.get("name", "lecture"))
    return "lecture"


def generate_template(
    poses_path: Path,
    metadata_path: Path,
    output_path: Path,
    annotator_id: str = "Phong",
    window_sec: float = 4.0,
    stride_sec: float = 2.0,
) -> None:
    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_path}")

    # Load metadata
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    session_id = metadata.get("session_id", "session_001")
    camera_id = metadata.get("camera_id", "A")
    phases = metadata.get("class_phase", [])

    print(f"[INFO] Reading poses: {poses_path}")
    df_poses = pd.read_parquet(poses_path) if poses_path.suffix == ".parquet" else pd.read_csv(poses_path)

    window_ms = int(window_sec * 1000)
    stride_ms = int(stride_sec * 1000)

    rows = []
    # Group by track_id
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

                initial_state = "VALID" if mean_vis >= 0.50 else "OCCLUDED"
                initial_occlusion = "none" if mean_vis >= 0.75 else ("moderate" if mean_vis >= 0.50 else "heavy")

                rows.append({
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "t_start_ms": t_start,
                    "t_end_ms": t_end,
                    "class_phase": phase,
                    "atomic_behavior": "",
                    "participation_label": "",
                    "annotation_state": initial_state,
                    "label_source": "HUMAN" if initial_state == "VALID" else "",
                    "label_rule_version": "",
                    "annotation_guide_version": "v0.2",
                    "annotation_confidence": "1.0",
                    "occlusion_level": initial_occlusion,
                    "zone_id": last_zone,
                    "annotator_id": annotator_id,
                    "notes": f"auto_vis_ratio={mean_vis:.2f}",
                })

            t_start += stride_ms

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(out_df)} annotation windows.")
    print(f"[INFO] Template saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sliding window annotation template.")
    parser.add_argument("--poses", type=str, required=True, help="Path to pose parquet")
    parser.add_argument("--metadata", type=str, default="data/raw/session_001/metadata.json", help="Path to metadata JSON")
    parser.add_argument("--output", "-o", type=str, default="data/annotations/session_001_annotation_template.csv", help="Output CSV path")
    parser.add_argument("--annotator-id", type=str, default="Phong", help="Annotator name or ID")

    args = parser.parse_args()
    generate_template(
        poses_path=Path(args.poses),
        metadata_path=Path(args.metadata),
        output_path=Path(args.output),
        annotator_id=args.annotator_id,
    )


if __name__ == "__main__":
    main()
