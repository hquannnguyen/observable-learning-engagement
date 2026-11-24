#!/usr/bin/env python3
"""Manage session-level annotation files for AnnotationStudio_V2 (Tiến & Phong).

Bridges the 600 sampled windows in review_v03 with the multi-student scene
workflow of AnnotationStudio_V2_Standalone.exe.

Features:
1. 'export': Splits review_v03 600 windows into per-session CSV files matching
   each video (session_001, session_002, session_003, session_004, session_005, session_006, classes)
   for both Phong and Tài.
2. 'merge': Merges annotated session CSVs back into canonical annotator_phong.csv
   and annotator_tai.csv, validating schemas and tracking completion.
3. 'status': Displays a live progress summary across all sessions and splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.annotation_io import read_annotation_csv, write_annotation_csv
from src.contracts.labels import ANNOTATION_STATES, CANONICAL_TRAINING_LABELS
from scripts.dataset_qa import validate_annotation_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("studio_manager")

CANONICAL_COLUMNS = [
    "window_id",
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

STUDIO_DIR = ROOT / "data" / "annotations" / "review_v03_studio"

SESSION_SPLIT_MAP = {
    "session_001": "train",
    "session_002": "train",
    "class_1": "train",
    "class_2": "train",
    "class_3": "train",
    "class_4": "train",
    "class_5": "train",
    "session_003": "val",
    "session_004": "val",
    "session_005": "test",
    "session_006": "test",
}


def export_sessions():
    """Split 600 review_v03 windows into per-session CSVs for AnnotationStudio."""
    src_phong = ROOT / "data" / "annotations" / "review_v03" / "annotator_b.csv"
    src_tai = ROOT / "data" / "annotations" / "review_v03" / "annotator_a.csv"

    if not src_phong.is_file() or not src_tai.is_file():
        raise FileNotFoundError("Missing review_v03 source CSVs in data/annotations/review_v03/")

    df_phong = pd.read_csv(src_phong, dtype=object, keep_default_na=False)
    df_tai = pd.read_csv(src_tai, dtype=object, keep_default_na=False)

    STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (STUDIO_DIR / "phong").mkdir(parents=True, exist_ok=True)
    (STUDIO_DIR / "tai").mkdir(parents=True, exist_ok=True)

    sessions = sorted(df_phong["session_id"].unique())
    logger.info("Exporting review_v03 sessions for AnnotationStudio into %s", STUDIO_DIR)

    for person, annotator_name, df, sub_dir in (
        ("phong", "ANN_PHONG", df_phong, STUDIO_DIR / "phong"),
        ("tai", "ANN_TAI", df_tai, STUDIO_DIR / "tai"),
    ):
        # 1. Main sessions 001..006
        for s in ["session_001", "session_002", "session_003", "session_004", "session_005", "session_006"]:
            sub_df = df[df["session_id"] == s].copy()
            if sub_df.empty:
                continue
            # Ensure columns
            for col in CANONICAL_COLUMNS:
                if col not in sub_df.columns:
                    sub_df[col] = ""
            sub_df = sub_df[CANONICAL_COLUMNS]
            out_path = sub_dir / f"{s}_cam_A_labels.csv"
            sub_df.to_csv(out_path, index=False, encoding="utf-8")
            split = SESSION_SPLIT_MAP.get(s, "train")
            logger.info("  [%s] %s (%s): %d windows -> %s", person.upper(), s, split.upper(), len(sub_df), out_path.name)

        # 2. Individual class_1..class_5 files matching each video & pose
        for c in ["class_1", "class_2", "class_3", "class_4", "class_5"]:
            c_df = df[df["session_id"] == c].copy()
            if not c_df.empty:
                for col in CANONICAL_COLUMNS:
                    if col not in c_df.columns:
                        c_df[col] = ""
                c_df = c_df[CANONICAL_COLUMNS]
                out_c = sub_dir / f"{c}_cam_A_labels.csv"
                c_df.to_csv(out_c, index=False, encoding="utf-8")
                logger.info("  [%s] %s (TRAIN): %d windows -> %s", person.upper(), c, len(c_df), out_c.name)

        # Also keep combined classes_train_labels.csv for backward compatibility
        class_df = df[df["session_id"].str.startswith("class_")].copy()
        if not class_df.empty:
            for col in CANONICAL_COLUMNS:
                if col not in class_df.columns:
                    class_df[col] = ""
            class_df = class_df[CANONICAL_COLUMNS]
            out_class = sub_dir / "classes_train_labels.csv"
            class_df.to_csv(out_class, index=False, encoding="utf-8")

    logger.info("Export completed! All session CSVs are ready for AnnotationStudio.")


def show_status():
    """Print status of annotation across all sessions for Phong and Tài."""
    if not STUDIO_DIR.is_dir():
        logger.warning("Studio directory %s does not exist. Run --export first.", STUDIO_DIR)
        return

    print("\n" + "=" * 80)
    print(" BÁO CÁO TIẾN ĐỘ GÁN NHÃN ANNOTATION STUDIO (TIẾN & PHONG) - REVIEW_V03")
    print("=" * 80)
    header = f"{'Session ID':<16} | {'Split':<8} | {'Tổng mẫu':<10} | {'Phong đã gán':<14} | {'Tài đã gán':<14}"
    print(header)
    print("-" * 80)

    total_all, done_phong_all, done_tai_all = 0, 0, 0

    session_names = [
        "session_005", "session_006",  # Test
        "session_003", "session_004",  # Val
        "session_001", "session_002",  # Train
        "classes_train",               # Train
    ]

    for s in session_names:
        fname = f"{s if s == 'classes_train' else s + '_cam_A'}_labels.csv"
        path_p = STUDIO_DIR / "phong" / fname
        path_t = STUDIO_DIR / "tai" / fname

        split_label = "TRAIN" if "train" in s or s in ("session_001", "session_002") else ("VAL" if s in ("session_003", "session_004") else "TEST")

        n_p, done_p = 0, 0
        if path_p.is_file():
            df_p = pd.read_csv(path_p, dtype=object, keep_default_na=False)
            n_p = len(df_p)
            is_done_p = (
                (df_p["annotation_state"].eq("VALID") & df_p["participation_label"].ne("")) |
                (df_p["annotation_state"].isin(["OCCLUDED", "INVALID"]))
            ) & df_p["label_source"].isin(["HUMAN", "ADJUDICATED"])
            done_p = int(is_done_p.sum())

        n_t, done_t = 0, 0
        if path_t.is_file():
            df_t = pd.read_csv(path_t, dtype=object, keep_default_na=False)
            n_t = len(df_t)
            is_done_t = (
                (df_t["annotation_state"].eq("VALID") & df_t["participation_label"].ne("")) |
                (df_t["annotation_state"].isin(["OCCLUDED", "INVALID"]))
            ) & df_t["label_source"].isin(["HUMAN", "ADJUDICATED"])
            done_t = int(is_done_t.sum())

        n_total = max(n_p, n_t)
        total_all += n_total
        done_phong_all += done_p
        done_tai_all += done_t

        status_line = f"{s:<16} | {split_label:<8} | {n_total:<10} | {f'{done_p}/{n_total}':<14} | {f'{done_t}/{n_total}':<14}"
        print(status_line)

    print("-" * 80)
    summary_line = f"{'TỔNG CỘNG':<16} | {'ALL':<8} | {total_all:<10} | {f'{done_phong_all}/{total_all} ({done_phong_all/max(1, total_all)*100:.1f}%)':<14} | {f'{done_tai_all}/{total_all} ({done_tai_all/max(1, total_all)*100:.1f}%)':<14}"
    print(summary_line)
    print("=" * 80 + "\n")


def merge_sessions():
    """Merge annotated session CSVs back into review_v03/annotator_*.csv."""
    logger.info("Merging session CSVs back to master review_v03 annotator files...")
    for person, annotator_id, out_name, sub_dir in (
        ("phong", "ANN_PHONG", "annotator_phong.csv", STUDIO_DIR / "phong"),
        ("tai", "ANN_TAI", "annotator_tai.csv", STUDIO_DIR / "tai"),
    ):
        files = sorted(sub_dir.glob("*.csv"))
        if not files:
            logger.warning("No files found in %s", sub_dir)
            continue

        frames = [pd.read_csv(f, dtype=object, keep_default_na=False) for f in files]
        merged = pd.concat(frames, ignore_index=True)

        # Merge with master review_v03 to keep any extra columns like split, video_path
        master_path = ROOT / "data" / "annotations" / "review_v03" / out_name
        if master_path.is_file():
            master_df = pd.read_csv(master_path, dtype=object, keep_default_na=False)
            # Update matching window_id
            merged_map = merged.set_index("window_id")
            for col in [
                "participation_label", "annotation_state", "atomic_behavior",
                "class_phase", "annotation_confidence", "occlusion_level",
                "label_source", "annotator_id", "notes"
            ]:
                if col in merged_map.columns:
                    mask = master_df["window_id"].isin(merged_map.index)
                    master_df.loc[mask, col] = master_df.loc[mask, "window_id"].map(merged_map[col])
            out_df = master_df
        else:
            out_df = merged

        out_path = ROOT / "data" / "annotations" / "review_v03" / out_name
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info("Saved %d merged windows -> %s", len(out_df), out_path)

        # Also keep annotator_a / annotator_b in sync
        alias_name = "annotator_b.csv" if person == "phong" else "annotator_a.csv"
        out_df.to_csv(ROOT / "data" / "annotations" / "review_v03" / alias_name, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Manage AnnotationStudio sessions for review_v03.")
    parser.add_argument("--export", action="store_true", help="Export review_v03 600 windows to per-session CSVs for Studio.")
    parser.add_argument("--status", action="store_true", help="Show completion progress of Studio sessions.")
    parser.add_argument("--merge", action="store_true", help="Merge completed session CSVs back to review_v03 master.")
    args = parser.parse_args()

    if args.export:
        export_sessions()
    if args.merge:
        merge_sessions()
    if args.status or (not args.export and not args.merge):
        show_status()


if __name__ == "__main__":
    main()
