"""Process and standardize session_003 manual annotations."""

import json
import logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def process_session_003():
    in_path = Path(r"D:\ThiGiacMayTinh\cam_A_labels_final003.csv")
    df = pd.read_csv(in_path, dtype=object, keep_default_na=False)
    logging.info(f"Loaded {len(df)} rows from {in_path}")

    # 1. Standardize camera_id & session_id
    df["camera_id"] = "cam_A"
    df["session_id"] = "session_003"

    # 2. Set class_phase = lecture across all 313 rows (confirmed by user)
    df["class_phase"] = "lecture"

    # 3. Clean up UNKNOWN atomic behaviors
    for idx, r in df.iterrows():
        behs = [b.strip() for b in r["atomic_behavior"].split(";") if b.strip()]
        cleaned = [b for b in behs if b != "UNKNOWN"]
        if not cleaned:
            cleaned = ["OCCLUDED"] if r["annotation_state"] == "OCCLUDED" else ["BOARD_OR_INSTRUCTOR_FACING"]
        df.loc[idx, "atomic_behavior"] = ";".join(cleaned)

    # 4. Correct the 4 sparse pose windows (< 20 frames) to OCCLUDED
    sparse_rows = [293, 294, 297, 312]
    for r_idx in sparse_rows:
        trk = df.loc[r_idx, "track_id"]
        ts = df.loc[r_idx, "t_start_ms"]
        logging.info(f"Adjusting sparse window Row {r_idx} (Track #{trk}, {ts}ms) -> OCCLUDED")
        df.loc[r_idx, "annotation_state"] = "OCCLUDED"
        df.loc[r_idx, "participation_label"] = ""
        df.loc[r_idx, "label_source"] = ""
        df.loc[r_idx, "label_rule_version"] = ""
        df.loc[r_idx, "atomic_behavior"] = "OCCLUDED"
        df.loc[r_idx, "occlusion_level"] = "heavy"
        df.loc[r_idx, "notes"] = df.loc[r_idx, "notes"] + " | corrected_sparse_pose_under_20"

    # 5. Match with tensor features_v02 and attach window_id
    feat_path = ROOT / "data" / "features_v02_yolo11_weak" / "participation_windows.csv"
    df_feat = pd.read_csv(feat_path, dtype={"track_id": int, "t_start_ms": int})
    feat_s3 = df_feat[df_feat["session_id"] == "session_003"].copy()

    window_ids = []
    alignment_status = []
    match_diff_ms = []

    for idx, r in df.iterrows():
        trk = int(r["track_id"])
        ts = int(r["t_start_ms"])
        st = r["annotation_state"]

        m_exact = feat_s3[(feat_s3["track_id"] == trk) & (feat_s3["t_start_ms"] == ts)]
        if not m_exact.empty:
            window_ids.append(m_exact["window_id"].iloc[0])
            alignment_status.append("EXACT_TENSOR_MATCH")
            match_diff_ms.append(0)
        else:
            m_near = feat_s3[(feat_s3["track_id"] == trk) & ((feat_s3["t_start_ms"] - ts).abs() <= 200)]
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

    # 6. Save aligned file
    out_dir = ROOT / "data" / "annotations" / "aligned_v0.3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_aligned = out_dir / "session_003_cam_A_aligned.csv"
    df.to_csv(out_aligned, index=False, encoding="utf-8")
    logging.info(f"Saved aligned session_003 -> {out_aligned}")

    # 7. Update canonical session_003_cam_A_annotations.csv
    canon_path = ROOT / "data" / "annotations" / "session_003_cam_A_annotations.csv"
    # Keep canonical columns for training annotations including official window_id
    canon_cols = [
        "window_id",
        "session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms",
        "class_phase", "atomic_behavior", "participation_label", "annotation_state",
        "label_source", "label_rule_version", "annotation_guide_version",
        "annotation_confidence", "occlusion_level", "zone_id", "annotator_id", "notes"
    ]
    df[canon_cols].to_csv(canon_path, index=False, encoding="utf-8")
    logging.info(f"Updated canonical {canon_path} with 313 human-annotated rows (including window_id).")

    # 8. Update review_v03 studio file for Phong
    studio_p = ROOT / "data" / "annotations" / "review_v03_studio" / "phong" / "session_003_cam_A_labels.csv"
    if studio_p.is_file():
        df_target = pd.read_csv(studio_p, dtype=object, keep_default_na=False)
        matched_st = 0
        for idx, r in df_target.iterrows():
            trk = str(r["track_id"])
            ts = float(r["t_start_ms"])
            matches = df[
                (df["track_id"].astype(str) == trk) &
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
        logging.info(f"Updated {matched_st} / {len(df_target)} windows in Phong's studio file for session_003.")

if __name__ == "__main__":
    process_session_003()
