#!/usr/bin/env python3
"""Prepare AI-assisted pre-annotation bundles and local/Web-LLM suggestions.

Workflow:
1. Load annotation sheet (e.g. review_v03_independent_20261124/annotator_a.csv)
   and select target windows (--split, --session, --only-unfinished, --limit).
2. Load matching session video + pose CSV and validate session/camera/timestamp/track.
3. For each selected window:
   - Render a composite contact-sheet JPG (annotated classroom overview + 5 timestamped
     cropped frames with skeleton + pose quality metrics + verified phase + v0.2 guide).
   - Run Cách 1 (Local AI proposer: temporal checkpoint + Uncertainty Gate + weak-v0.2)
     to write structured suggestions into `ai_suggestions.json`.
   - Organize Cách 2 batches (JPGs + copy-paste `prompt.txt`) for ChatGPT Plus / Gemini Pro web.
4. Also supports `--import-json` to merge JSON pasted from ChatGPT Plus / Gemini Pro web
   into `ai_suggestions.json` without ever touching HUMAN labels directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.annotation_io import read_annotation_csv
from src.contracts.labels import ANNOTATION_STATES
from src.models.factory import model_from_checkpoint
from src.pose.ai_bundle import (
    build_web_llm_prompt,
    build_window_contact_sheet,
    load_suggestions_map,
    parse_and_validate_suggestions,
    propose_local_suggestion,
    propose_local_suggestions_from_features,
    save_suggestions_map,
)
from src.uncertainty.gate import UncertaintyThresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prepare_ai_bundles")

DEFAULT_CHECKPOINT = "models/checkpoints/20261125-205918-lstm-yolo11-feat94-weak-r2-lecture.pt"
DEFAULT_CONFIG = "configs/experiments/yolo11_feat94_weak_v02_lecture.yaml"


def _resolve_video_path(row: pd.Series, raw_dir: Path) -> Path:
    v_str = str(row.get("video_path", "")).strip()
    if v_str:
        p = Path(v_str) if Path(v_str).is_absolute() else ROOT / v_str
        if p.is_file():
            return p
    session_id = str(row["session_id"])
    camera_id = str(row.get("camera_id", "cam_A"))
    for cand in (
        raw_dir / session_id / f"{camera_id}.mp4",
        raw_dir / session_id / "cam_A.mp4",
        raw_dir / f"{session_id}.mp4",
    ):
        if cand.is_file():
            return cand
    return raw_dir / session_id / f"{camera_id}.mp4"


def _resolve_pose_path(row: pd.Series, poses_dir: Path) -> Path:
    session_id = str(row["session_id"])
    camera_id = str(row.get("camera_id", "cam_A"))
    for cand in (
        poses_dir / f"{session_id}_{camera_id}_poses.csv",
        poses_dir / f"{session_id}_poses.csv",
    ):
        if cand.is_file():
            return cand
    return poses_dir / f"{session_id}_{camera_id}_poses.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid Local + ChatGPT/Gemini pre-annotation bundles.")
    parser.add_argument("--annotations", default="data/annotations/review_v03_independent_20261124/annotator_a.csv")
    parser.add_argument("--out-dir", default=None, help="Output bundle dir (default: <annotations.parent>/ai_bundles)")
    parser.add_argument("--poses-dir", default="data/poses")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--phase", default="lecture", choices=("lecture", "individual_work", "discussion"))
    parser.add_argument("--split", default=None, choices=("train", "val", "test"))
    parser.add_argument("--session", default=None)
    parser.add_argument("--window-ids", nargs="*", default=None)
    parser.add_argument("--only-unfinished", action="store_true", help="Only process rows without completed human labels")
    parser.add_argument("--limit", type=int, default=None, help="Maximum windows to process")
    parser.add_argument("--skip-images", action="store_true", help="Only run Local AI suggestions without rendering JPGs")
    parser.add_argument("--blur-face", action="store_true", help="Blur faces in exported contact sheets")
    parser.add_argument("--batch-size", type=int, default=6, help="Images per ChatGPT/Gemini web batch folder")
    parser.add_argument("--export-batches", choices=("uncertain", "all", "none"), default="uncertain")
    parser.add_argument("--overwrite-vision", action="store_true", help="Overwrite existing VISION_LLM_IMPORT suggestions")
    parser.add_argument("--features-npz", default=None, help="Optional precomputed window_features.npz for fast batched Local AI suggestions")
    parser.add_argument("--import-json", default=None, help="Import JSON returned by ChatGPT Plus / Gemini Pro into ai_suggestions.json")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    out_dir = Path(args.out_dir) if args.out_dir else ann_path.parent / "ai_bundles"
    suggestions_path = out_dir / "ai_suggestions.json"

    if args.import_json:
        raw_text = Path(args.import_json).read_text(encoding="utf-8")
        imported = parse_and_validate_suggestions(raw_text, suggestion_source="VISION_LLM_IMPORT")
        merged = save_suggestions_map(suggestions_path, imported)
        logger.info("Imported %d Vision LLM suggestions -> %s (total stored: %d)", len(imported), suggestions_path, len(merged))
        return

    header_cols = pd.read_csv(ann_path, nrows=0).columns
    if "annotation_confidence" in header_cols:
        df = read_annotation_csv(ann_path)
    else:
        df = pd.read_csv(ann_path, dtype=object, keep_default_na=False)
        if "window_id" not in df or df["window_id"].eq("").any() or df["window_id"].duplicated().any():
            raise ValueError(f"Window table {ann_path} requires unique non-empty window_id.")
    mask = pd.Series(True, index=df.index)
    if args.split and "split" in df.columns:
        mask &= df["split"].astype(str).str.lower().eq(args.split.lower())
    if args.session:
        mask &= df["session_id"].astype(str).eq(args.session)
    if args.window_ids:
        mask &= df["window_id"].astype(str).isin(set(args.window_ids))
    if args.only_unfinished and "annotator_id" in df.columns:
        state_col = df["annotation_state"] if "annotation_state" in df.columns else pd.Series("", index=df.index)
        label_col = df["participation_label"] if "participation_label" in df.columns else pd.Series("", index=df.index)
        done = (
            (state_col.eq("VALID") & label_col.ne(""))
            | state_col.isin(["OCCLUDED", "INVALID", "TRANSITION"])
        ) & df["annotator_id"].ne("")
        mask &= ~done

    target_df = df[mask].copy()
    if args.limit is not None:
        target_df = target_df.head(max(0, args.limit))
    if target_df.empty:
        logger.info("No matching windows to process.")
        return

    # Load model + calibration thresholds if checkpoint exists
    model = None
    thresholds = UncertaintyThresholds()
    temperature = 1.0
    input_size = 94
    model_id = "weak-v0.2-rule-only"
    sample_fps = 10.0

    cfg_path = Path(args.config)
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        sample_fps = float(cfg.get("video", {}).get("sample_fps", 10.0))
        qg = cfg.get("quality_gate", {})
        unc = cfg.get("uncertainty", {})
        thresholds = UncertaintyThresholds(
            min_visible_ratio=float(qg.get("min_visible_ratio", 0.60)),
            min_mean_keypoint_confidence=float(qg.get("min_mean_keypoint_confidence", 0.30)),
            min_probability=float(unc.get("min_probability", 0.60)),
        )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = model_from_checkpoint(ckpt)
        model.eval()
        input_size = int(ckpt.get("input_size", 94))
        calib = ckpt.get("calibration") or {}
        temperature = float(calib.get("temperature", 1.0))
        if "min_probability" in calib:
            thresholds = UncertaintyThresholds(
                min_visible_ratio=thresholds.min_visible_ratio,
                min_mean_keypoint_confidence=thresholds.min_mean_keypoint_confidence,
                min_probability=float(calib["min_probability"]),
                upper_body_min_visible_ratio=thresholds.upper_body_min_visible_ratio,
            )
        model_id = ckpt_path.stem
        logger.info("Loaded model %s (input_size=%d, T=%.3f, min_prob=%.2f)", model_id, input_size, temperature, thresholds.min_probability)

    existing_map = load_suggestions_map(suggestions_path)
    poses_cache: Dict[Path, pd.DataFrame] = {}
    bundles_meta: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []

    images_dir = out_dir / "images"
    raw_dir = Path(args.raw_dir)
    poses_dir = Path(args.poses_dir)

    if args.features_npz:
        npz = np.load(args.features_npz, allow_pickle=True)
        npz_wids = [str(w) for w in npz["window_ids"]]
        wid_to_idx = {w: i for i, w in enumerate(npz_wids)}
        row_indices: List[int] = []
        for wid in target_df["window_id"].astype(str):
            if wid not in wid_to_idx:
                raise ValueError(f"Window ID {wid} not found in {args.features_npz}")
            row_indices.append(wid_to_idx[wid])
        feat_tensor = npz["all_features"][row_indices]
        batch_sugs = propose_local_suggestions_from_features(
            target_df,
            feat_tensor,
            raw_dir=raw_dir,
            poses_dir=poses_dir,
            model=model,
            thresholds=thresholds,
            temperature=temperature,
            input_size=input_size,
            class_phase=args.phase,
            sample_fps=sample_fps,
            model_id=model_id,
        )
        for s in batch_sugs:
            prev_sug = existing_map.get(s["window_id"])
            if prev_sug and prev_sug.get("suggestion_source") == "VISION_LLM_IMPORT" and not args.overwrite_vision:
                suggestions.append(prev_sug)
            else:
                suggestions.append(s)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_suggestions_map(suggestions_path, suggestions)
        n_valid = sum(1 for s in suggestions if s["suggested_state"] == "VALID")
        n_unc = sum(1 for s in suggestions if s["needs_vision_review"])
        logger.info(
            "Done (batched features)! Suggestions=%d (VALID=%d, needs_review/gate=%d) -> %s & %s",
            len(suggestions), n_valid, n_unc, suggestions_path, suggestions_path.with_suffix(".csv"),
        )
        return

    for idx, (_, row) in enumerate(target_df.iterrows(), start=1):
        p_path = _resolve_pose_path(row, poses_dir)
        v_path = _resolve_video_path(row, raw_dir)
        if p_path not in poses_cache:
            poses_cache[p_path] = pd.read_csv(p_path) if p_path.is_file() else pd.DataFrame()
        pose_df = poses_cache[p_path]

        img_path_str = ""
        meta_rec: Dict[str, Any]
        if not args.skip_images:
            img_path = images_dir / f"{row['window_id']}.jpg"
            _, meta_rec = build_window_contact_sheet(
                v_path,
                pose_df,
                row.to_dict(),
                output_path=img_path,
                class_phase=args.phase,
                phase_verified=True,
                show_skeleton=True,
                blur_face=args.blur_face,
                sample_fps=sample_fps,
            )
            img_path_str = str(img_path)
        else:
            meta_rec = {
                "window_id": str(row["window_id"]),
                "session_id": str(row["session_id"]),
                "camera_id": str(row.get("camera_id", "cam_A")),
                "track_id": str(row["track_id"]),
                "class_phase": args.phase,
                "pose_quality": {},
                "image_path": "",
            }

        prev_sug = existing_map.get(str(row["window_id"]))
        if prev_sug and prev_sug.get("suggestion_source") == "VISION_LLM_IMPORT" and not args.overwrite_vision:
            suggestion = {**prev_sug, "bundle_image_path": img_path_str or prev_sug.get("bundle_image_path", "")}
        else:
            suggestion = propose_local_suggestion(
                row.to_dict(),
                pose_df,
                video_path=v_path,
                model=model,
                thresholds=thresholds,
                temperature=temperature,
                input_size=input_size,
                class_phase=args.phase,
                sample_fps=sample_fps,
                model_id=model_id,
                bundle_image_path=img_path_str,
            )
        if not meta_rec.get("pose_quality"):
            meta_rec["pose_quality"] = {"mean_visible_ratio": 0.0, "mean_pose_confidence": 0.0, "frame_coverage": 0.0}
        bundles_meta.append(meta_rec)
        suggestions.append(suggestion)
        if idx % 25 == 0 or idx == len(target_df):
            logger.info("Processed %d/%d windows...", idx, len(target_df))

    out_dir.mkdir(parents=True, exist_ok=True)
    save_suggestions_map(suggestions_path, suggestions)
    (out_dir / "bundle_manifest.json").write_text(json.dumps(bundles_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Export ChatGPT Plus / Gemini Pro batches
    if args.export_batches != "none" and not args.skip_images:
        batches_root = out_dir / "web_llm_batches"
        if batches_root.exists():
            shutil.rmtree(batches_root)
        selected_pairs = [
            (m, s) for m, s in zip(bundles_meta, suggestions)
            if args.export_batches == "all" or s.get("needs_vision_review")
        ]
        bs = max(1, int(args.batch_size))
        for b_idx in range(0, len(selected_pairs), bs):
            chunk = selected_pairs[b_idx:b_idx + bs]
            b_dir = batches_root / f"batch_{b_idx // bs + 1:03d}"
            b_dir.mkdir(parents=True, exist_ok=True)
            chunk_metas = []
            for m, _ in chunk:
                src_img = Path(m["image_path"])
                if src_img.is_file():
                    shutil.copy2(src_img, b_dir / src_img.name)
                chunk_metas.append(m)
            (b_dir / "prompt.txt").write_text(build_web_llm_prompt(chunk_metas), encoding="utf-8")
        logger.info("Exported %d window(s) into %s for ChatGPT Plus / Gemini Pro web.", len(selected_pairs), batches_root)

    n_valid = sum(1 for s in suggestions if s["suggested_state"] == "VALID")
    n_unc = sum(1 for s in suggestions if s["needs_vision_review"])
    logger.info(
        "Done! Suggestions=%d (VALID=%d, needs_review/gate=%d) -> %s & %s",
        len(suggestions), n_valid, n_unc, suggestions_path, suggestions_path.with_suffix(".csv"),
    )


if __name__ == "__main__":
    main()
