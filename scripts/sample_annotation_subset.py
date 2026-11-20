#!/usr/bin/env python3
"""Create a blind human-annotation template from frozen participation windows.

This script does NOT create labels. It only samples windows and writes an empty
template that real annotators fill in. Weak labels stay in the sampling manifest
as aggregate counts and are intentionally omitted from the row-level template to
avoid anchoring the annotator.

Usage:
    python scripts/sample_annotation_subset.py \
        --windows data/features/participation_windows.csv \
        --out-dir data/annotations/human

Outputs:
    data/annotations/human/annotation_template.csv
    data/annotations/human/sampling_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEMPLATE_COLUMNS = [
    "window_id",
    "session_id",
    "camera_id",
    "track_id",
    "t_start_ms",
    "t_end_ms",
    "class_phase",
    "declared_class_phase",
    "phase_source",
    "phase_verified",
    "split",
    "zone_id",
    "video_path",
    "pose_overlay_path",
    "participation_label",
    "annotation_state",
    "atomic_behavior",
    "annotation_guide_version",
    "occlusion_level",
    "label_source",
    "label_rule_version",
    "annotator_id",
    "annotation_confidence",
    "notes",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_path_for(session_id: str) -> str:
    candidates = [
        Path("data/raw") / session_id / "cam_A.mp4",
        Path("data/raw") / f"{session_id}.mp4",
        Path("data/raw") / f"{session_id}.MOV",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def pose_overlay_for(session_id: str) -> str:
    candidates = [
        Path("data/poses") / f"{session_id}_cam_A_pose_overlay.mp4",
        Path("data/poses") / f"{session_id}_cam_A_overlay.mp4",
        Path("data/poses") / f"{session_id}_pose_overlay.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample up to n rows, stratified by (session, weak label), deterministically."""
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).copy()

    rng = np.random.default_rng(seed)
    picks: List[pd.DataFrame] = []
    groups = [
        g for _, g in df.groupby(["session_id", "participation_label"], sort=True, dropna=False)
    ]
    for group in groups:
        quota = max(1, int(round(n * len(group) / len(df))))
        quota = min(quota, len(group))
        picked = group.sample(n=quota, random_state=int(rng.integers(0, 2**31 - 1)))
        picks.append(picked)

    sampled = pd.concat(picks, ignore_index=True).drop_duplicates("window_id")
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remaining = df[~df["window_id"].isin(sampled["window_id"])]
        need = min(n - len(sampled), len(remaining))
        if need:
            sampled = pd.concat(
                [sampled, remaining.sample(n=need, random_state=seed)],
                ignore_index=True,
            )
    return sampled.sort_values(["session_id", "t_start_ms"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample blind human annotation windows.")
    parser.add_argument("--windows", default="data/features/participation_windows.csv")
    parser.add_argument("--out-dir", default="data/annotations/human")
    parser.add_argument("--n-test", type=int, default=200, help="Test windows to sample (0 = all).")
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.n_test, args.n_train, args.n_val) < 0:
        parser.error("Sample sizes must be non-negative; --n-test 0 means all test windows")

    windows_path = Path(args.windows)
    if not windows_path.is_file():
        raise FileNotFoundError(f"Window metadata not found: {windows_path}")

    df = pd.read_csv(windows_path)
    required = {"window_id", "session_id", "split", "participation_label", "quality_state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Window metadata missing columns: {sorted(missing)}")

    if df["window_id"].duplicated().any():
        raise ValueError("Duplicate window_id in source windows")
    # Include unlabelled/occluded cases; humans must be able to assess coverage.
    usable = df.copy()

    test_n = int((usable["split"] == "test").sum()) if args.n_test == 0 else args.n_test
    parts = [
        stratified_sample(usable[usable["split"] == "test"], test_n, args.seed),
        stratified_sample(usable[usable["split"] == "train"], args.n_train, args.seed),
        stratified_sample(usable[usable["split"] == "val"], args.n_val, args.seed),
    ]
    sampled = pd.concat(parts, ignore_index=True).drop_duplicates("window_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = pd.DataFrame(
        {
            "window_id": sampled["window_id"],
            "session_id": sampled["session_id"],
            "camera_id": sampled.get("camera_id", "cam_A"),
            "track_id": sampled["track_id"],
            "t_start_ms": sampled["t_start_ms"],
            "t_end_ms": sampled["t_end_ms"],
            "class_phase": sampled.get("class_phase", "UNKNOWN"),
            "declared_class_phase": sampled.get("declared_class_phase", "UNKNOWN"),
            "phase_source": sampled.get("phase_source", "UNKNOWN"),
            "phase_verified": sampled.get("phase_verified", False),
            "split": sampled["split"],
            "zone_id": sampled.get("zone_id", ""),
            "video_path": sampled["session_id"].map(video_path_for),
            "pose_overlay_path": sampled["session_id"].map(pose_overlay_for),
            "participation_label": "",
            "annotation_state": "",
            "atomic_behavior": "",
            "annotation_guide_version": "v0.2",
            "occlusion_level": "",
            "label_source": "",
            "label_rule_version": "",
            "annotator_id": "",
            "annotation_confidence": "",
            "notes": "",
        }
    )
    template = template[TEMPLATE_COLUMNS]

    template_path = out_dir / "annotation_template.csv"
    template.to_csv(template_path, index=False)

    manifest: Dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "source_windows": str(windows_path),
        "source_windows_sha256": sha256_file(windows_path),
        "seed": args.seed,
        "sampled_windows": int(len(template)),
        "by_split": sampled["split"].value_counts().to_dict(),
        "by_session": template["session_id"].value_counts().to_dict(),
        "weak_label_counts_in_sample": sampled["participation_label"]
        .value_counts()
        .to_dict(),
        "blind": True,
        "note": (
            "Row-level weak labels are intentionally omitted. Annotators must label "
            "from video only, then run scripts/dataset_qa.py --strict-human."
        ),
    }
    manifest_path = out_dir / "sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Template: {template_path} ({len(template)} windows)")
    print(f"Manifest: {manifest_path}")
    print("This script created no labels; human annotation is required.")


if __name__ == "__main__":
    main()
