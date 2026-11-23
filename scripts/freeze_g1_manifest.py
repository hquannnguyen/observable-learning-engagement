#!/usr/bin/env python3
"""Freeze G1 source manifests and SHA256 hashes for dataset v0.2-yolo11-feat94-weak."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    reports_dir = ROOT / "reports" / "dataset_v02"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Split files
    splits_dir = ROOT / "data" / "splits"
    split_hashes = {
        f.name: {
            "path": str(f.relative_to(ROOT)),
            "sha256": sha256_file(f),
            "size_bytes": f.stat().st_size,
        }
        for f in sorted(splits_dir.glob("*.txt"))
    }

    # 2. Pose files
    poses_dir = ROOT / "data" / "poses"
    pose_records = []
    for f in sorted(poses_dir.glob("*")):
        if f.is_file() and f.suffix in (".csv", ".parquet") and not f.name.startswith("."):
            pose_records.append({
                "filename": f.name,
                "relative_path": str(f.relative_to(ROOT)),
                "format": f.suffix.lstrip("."),
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            })
    pose_df = pd.DataFrame(pose_records)
    pose_manifest_csv = reports_dir / "pose_manifest.csv"
    pose_df.to_csv(pose_manifest_csv, index=False)
    print(f"Saved pose manifest ({len(pose_records)} files) -> {pose_manifest_csv}")

    # 3. Features v0.2 files
    features_dir = ROOT / "data" / "features_v02_yolo11_weak"
    feat_hashes = {}
    if features_dir.is_dir():
        for f in sorted(features_dir.glob("*")):
            if f.is_file():
                feat_hashes[f.name] = {
                    "path": str(f.relative_to(ROOT)),
                    "sha256": sha256_file(f),
                    "size_bytes": f.stat().st_size,
                }

    # 4. Configs
    config_path = ROOT / "configs" / "experiments" / "yolo11_feat94_weak_v02.yaml"
    config_info = {
        "path": str(config_path.relative_to(ROOT)),
        "sha256": sha256_file(config_path) if config_path.is_file() else None,
    }

    # Assemble Source Manifest
    manifest = {
        "gate": "G1",
        "dataset_version": "v0.2-yolo11-feat94-weak",
        "feature_schema_version": "posefeat-v2",
        "split_version": "split-v0.1",
        "created_at": datetime.now().isoformat(),
        "canonical_recordings": [
            "class_1", "class_2", "class_3", "class_4", "class_5",
            "session_001", "session_002", "session_003", "session_004", "session_005", "session_006"
        ],
        "split_hashes": split_hashes,
        "config": config_info,
        "features": feat_hashes,
    }

    manifest_json = reports_dir / "source_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved source manifest -> {manifest_json}")

if __name__ == "__main__":
    main()
