#!/usr/bin/env python3
"""Create blank independent human-annotation templates for full train/val/test splits.

Never writes HUMAN labels or overwrites existing annotator sheets unless --force is passed.
Pairs directly with `scripts/prepare_ai_bundles.py --features-npz` and Studio Mode 2.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.annotation_io import write_annotation_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prepare_full_split_annotations")

ANNOTATION_TEMPLATE_COLUMNS = [
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
    "phase_evidence",
    "annotation_evidence",
]


def _resolve_relative_video(session_id: str, camera_id: str, raw_dir: Path, root: Path) -> str:
    for cand in (
        raw_dir / session_id / f"{camera_id}.mp4",
        raw_dir / session_id / "cam_A.mp4",
        raw_dir / f"{session_id}.mp4",
    ):
        if cand.is_file():
            try:
                return str(cand.relative_to(root))
            except ValueError:
                return str(cand)
    return f"data/raw/{session_id}/{camera_id}.mp4"


def build_blank_split_dataframe(
    windows_df: pd.DataFrame,
    split_name: str,
    *,
    raw_dir: Path,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Build a blind human-review template DataFrame for all rows in one split."""
    sub = windows_df[windows_df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if sub.empty:
        return pd.DataFrame(columns=ANNOTATION_TEMPLATE_COLUMNS)

    sub["_t0"] = pd.to_numeric(sub["t_start_ms"], errors="coerce").fillna(0.0)
    sub["_trk"] = sub["track_id"].astype(str)
    sub = sub.sort_values(["session_id", "_t0", "_trk", "window_id"]).drop(columns=["_t0", "_trk"])

    video_cache: Dict[tuple, str] = {}
    records: List[Dict[str, str]] = []
    for _, row in sub.iterrows():
        sess = str(row["session_id"])
        cam = str(row.get("camera_id", "cam_A") or "cam_A")
        if (sess, cam) not in video_cache:
            video_cache[(sess, cam)] = _resolve_relative_video(sess, cam, raw_dir, root)
        declared = str(row.get("declared_class_phase") or row.get("class_phase") or "UNKNOWN").strip() or "UNKNOWN"
        records.append({
            "window_id": str(row["window_id"]),
            "session_id": sess,
            "camera_id": cam,
            "track_id": str(row["track_id"]),
            "t_start_ms": str(row["t_start_ms"]),
            "t_end_ms": str(row["t_end_ms"]),
            "class_phase": declared,
            "declared_class_phase": declared,
            "phase_source": "METADATA_UNVERIFIED",
            "phase_verified": "False",
            "split": split_name.lower(),
            "zone_id": "",
            "video_path": video_cache[(sess, cam)],
            "pose_overlay_path": "",
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
            "phase_evidence": "",
            "annotation_evidence": "",
        })
    return pd.DataFrame(records, columns=ANNOTATION_TEMPLATE_COLUMNS)


def prepare_full_split_sheets(
    windows_csv: Path,
    *,
    out_root: Path,
    raw_dir: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    force: bool = False,
    root: Path = ROOT,
) -> Dict[str, int]:
    """Write annotation_template.csv, annotator_a.csv, annotator_b.csv for each requested split."""
    windows_df = pd.read_csv(windows_csv, dtype=object, keep_default_na=False)
    counts: Dict[str, int] = {}
    for split_name in splits:
        split_key = split_name.strip().lower()
        df_split = build_blank_split_dataframe(windows_df, split_key, raw_dir=raw_dir, root=root)
        split_dir = out_root / f"{split_key}_full_ai_assist"
        split_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("annotation_template.csv", "annotator_a.csv", "annotator_b.csv"):
            target = split_dir / fname
            if target.exists() and not force:
                logger.info("Preserving existing file: %s", target)
                continue
            write_annotation_csv(target, df_split)
            logger.info("Wrote %d rows -> %s", len(df_split), target)
        counts[split_key] = len(df_split)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create full train/val/test annotation templates.")
    parser.add_argument("--windows-csv", default="data/features/participation_windows.csv")
    parser.add_argument("--out-root", default="data/annotations")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing annotator CSVs")
    args = parser.parse_args()

    counts = prepare_full_split_sheets(
        Path(args.windows_csv),
        out_root=Path(args.out_root),
        raw_dir=Path(args.raw_dir),
        splits=args.splits,
        force=args.force,
    )
    logger.info("Done preparing full split templates: %s", counts)


if __name__ == "__main__":
    main()
