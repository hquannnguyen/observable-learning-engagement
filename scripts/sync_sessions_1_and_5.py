"""Synchronize and standardize manual annotations for session_001 and session_005."""

import json
import logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CANON_COLS = [
    "window_id",
    "session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms",
    "class_phase", "atomic_behavior", "participation_label", "annotation_state",
    "label_source", "label_rule_version", "annotation_guide_version",
    "annotation_confidence", "occlusion_level", "zone_id", "annotator_id", "notes"
]

def update_metadata_verified(session_id: str, verified_phase: str, duration_ms: int = 60000):
    meta_path = ROOT / "data" / "raw" / session_id / "metadata.json"
    if not meta_path.is_file():
        logging.warning(f"Metadata file not found: {meta_path}")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["camera_id"] = "cam_A"
    
    # Update class_phase structure
    meta["class_phase"] = [
        {
            "name": verified_phase,
            "start_ms": 0,
            "end_ms": duration_ms,
            "verified": True,
            "phase_source": "METADATA_VERIFIED",
            "verified_by": "HUMAN_ANNOTATOR",
            "verified_date": "2026-11-24",
            "notes": f"Human-verified {verified_phase} phase for first 60 seconds (0-60s) on 2026-11-24."
        },
        {
            "name": verified_phase if session_id == "session_005" else "discussion",
            "start_ms": duration_ms,
            "end_ms": 900000 if session_id == "session_005" else 4750000,
            "verified": False,
            "phase_source": "METADATA_UNVERIFIED",
            "notes": "Unverified remaining duration of session recording."
        }
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info(f"Updated metadata for {session_id} -> {meta_path}")


def process_session(session_id: str, input_csv: Path, target_phase: str, pose_parquet: Path):
    logging.info(f"================ Processing {session_id} ================")
    df = pd.read_csv(input_csv, dtype=object, keep_default_na=False)
    logging.info(f"Loaded {len(df)} rows from {input_csv}")

    # 1. Standardize basic columns
    df["session_id"] = session_id
    df["camera_id"] = "cam_A"
    df["class_phase"] = target_phase
    if "label_source" not in df.columns or df["label_source"].eq("").all():
        df["label_source"] = "HUMAN"
    if "annotator_id" not in df.columns or df["annotator_id"].eq("").all():
        df["annotator_id"] = "ANN_PHONG"
    if "annotation_guide_version" not in df.columns or df["annotation_guide_version"].eq("").all():
        df["annotation_guide_version"] = "v0.2"

    # 2. Clean atomic_behavior
    for idx, r in df.iterrows():
        behs = [b.strip() for b in r["atomic_behavior"].split(";") if b.strip()]
        cleaned = [b for b in behs if b != "UNKNOWN"]
        if not cleaned:
            cleaned = ["OCCLUDED"] if r["annotation_state"] == "OCCLUDED" else ["BOARD_OR_INSTRUCTOR_FACING"]
        df.loc[idx, "atomic_behavior"] = ";".join(cleaned)

    # 3. Detect and correct sparse pose windows (< 20 pose frames in 4s window)
    df_pose = pd.read_parquet(pose_parquet)
    sparse_count = 0
    for idx, r in df.iterrows():
        if r["annotation_state"] == "VALID":
            trk = int(float(r["track_id"]))
            t_s = float(r["t_start_ms"])
            t_e = float(r["t_end_ms"])
            sub_pose = df_pose[(df_pose["track_id"] == trk) & (df_pose["timestamp_ms"] >= t_s) & (df_pose["timestamp_ms"] <= t_e)]
            if len(sub_pose) < 20:
                sparse_count += 1
                logging.info(f"  Adjusting sparse window row {idx} (Track #{trk}, {t_s:.0f}ms): {len(sub_pose)} pose frames -> OCCLUDED")
                df.loc[idx, "annotation_state"] = "OCCLUDED"
                df.loc[idx, "participation_label"] = ""
                df.loc[idx, "label_source"] = ""
                df.loc[idx, "label_rule_version"] = ""
                df.loc[idx, "atomic_behavior"] = "OCCLUDED"
                df.loc[idx, "occlusion_level"] = "heavy"
                cur_notes = str(df.loc[idx, "notes"])
                df.loc[idx, "notes"] = (cur_notes + " | corrected_sparse_pose_under_20").strip(" |")
        elif r["annotation_state"] == "OCCLUDED":
            # Ensure OCCLUDED contract: participation_label and label_source must be empty
            df.loc[idx, "participation_label"] = ""
            df.loc[idx, "label_source"] = ""
            df.loc[idx, "label_rule_version"] = ""

    logging.info(f"Total sparse windows converted to OCCLUDED: {sparse_count}")

    # 4. Match with tensor features_v02 and attach window_id
    feat_path = ROOT / "data" / "features_v02_yolo11_weak" / "participation_windows.csv"
    df_feat = pd.read_csv(feat_path, dtype={"track_id": int, "t_start_ms": int})
    feat_s = df_feat[df_feat["session_id"] == session_id].copy()

    window_ids = []
    alignment_status = []
    match_diff_ms = []

    for idx, r in df.iterrows():
        trk = int(float(r["track_id"]))
        ts = int(float(r["t_start_ms"]))
        st = r["annotation_state"]

        m_exact = feat_s[(feat_s["track_id"] == trk) & (feat_s["t_start_ms"] == ts)]
        if not m_exact.empty:
            window_ids.append(m_exact["window_id"].iloc[0])
            alignment_status.append("EXACT_TENSOR_MATCH")
            match_diff_ms.append(0)
        else:
            m_near = feat_s[(feat_s["track_id"] == trk) & ((feat_s["t_start_ms"] - ts).abs() <= 250)]
            if not m_near.empty:
                diff = int(abs(m_near["t_start_ms"].iloc[0] - ts))
                window_ids.append(m_near["window_id"].iloc[0])
                alignment_status.append("NEAR_TENSOR_MATCH")
                match_diff_ms.append(diff)
            else:
                window_ids.append("")
                match_diff_ms.append(-1)
                if st == "OCCLUDED":
                    alignment_status.append("UNMATCHED_OCCLUDED_TRACK_GAP")
                else:
                    alignment_status.append("UNMATCHED_DISCONTINUOUS_TRACK")

    df["window_id"] = window_ids
    df["alignment_status"] = alignment_status
    df["match_diff_ms"] = match_diff_ms

    matched_total = int(df["window_id"].ne("").sum())
    logging.info(f"Tensor alignment: {matched_total} / {len(df)} windows matched with features_v02.")

    # 5. Save aligned file
    out_dir = ROOT / "data" / "annotations" / "aligned_v0.3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_aligned = out_dir / f"{session_id}_cam_A_aligned.csv"
    df.to_csv(out_aligned, index=False, encoding="utf-8")
    logging.info(f"Saved aligned file -> {out_aligned}")

    # 6. Update canonical session_xxx_cam_A_annotations.csv
    canon_path = ROOT / "data" / "annotations" / f"{session_id}_cam_A_annotations.csv"
    df[CANON_COLS].to_csv(canon_path, index=False, encoding="utf-8")
    logging.info(f"Saved canonical annotation file -> {canon_path}")

    # 7. Update review_v03 studio file for Phong
    studio_p = ROOT / "data" / "annotations" / "review_v03_studio" / "phong" / f"{session_id}_cam_A_labels.csv"
    if studio_p.is_file():
        df_target = pd.read_csv(studio_p, dtype=object, keep_default_na=False)
        matched_st = 0
        for idx, r in df_target.iterrows():
            trk = str(int(float(r["track_id"])))
            ts = float(r["t_start_ms"])
            matches = df[
                (df["track_id"].astype(float).astype(int).astype(str) == trk) &
                ((df["t_start_ms"].astype(float) - ts).abs() <= 1000)
            ]
            if not matches.empty:
                matched_st += 1
                m = matches.iloc[0]
                df_target.loc[idx, "participation_label"] = m["participation_label"]
                df_target.loc[idx, "annotation_state"] = m["annotation_state"]
                df_target.loc[idx, "atomic_behavior"] = m["atomic_behavior"]
                df_target.loc[idx, "class_phase"] = m["class_phase"]
                df_target.loc[idx, "annotation_confidence"] = m["annotation_confidence"]
                df_target.loc[idx, "occlusion_level"] = m["occlusion_level"]
                df_target.loc[idx, "label_source"] = "HUMAN"
                df_target.loc[idx, "annotator_id"] = "ANN_PHONG"
                df_target.loc[idx, "notes"] = m["notes"]

        df_target.to_csv(studio_p, index=False, encoding="utf-8")
        logging.info(f"Updated {matched_st} / {len(df_target)} sampled windows in Phong's studio file for {session_id}.")


def main():
    # 1. Update metadata
    update_metadata_verified("session_001", "lecture", duration_ms=60000)
    update_metadata_verified("session_005", "individual_work", duration_ms=60000)

    # 2. Process Session 1
    process_session(
        session_id="session_001",
        input_csv=Path(r"D:\ThiGiacMayTinh\cam_A_labels_final01.csv"),
        target_phase="lecture",
        pose_parquet=ROOT / "data" / "poses" / "session_001_cam_A_poses.parquet"
    )

    # 3. Process Session 5
    process_session(
        session_id="session_005",
        input_csv=Path(r"D:\ThiGiacMayTinh\cam_A_labels_final05.csv"),
        target_phase="individual_work",
        pose_parquet=ROOT / "data" / "poses" / "session_005_cam_A_poses.parquet"
    )

    logging.info("Synchronization complete for Session 1 and Session 5!")

if __name__ == "__main__":
    main()
