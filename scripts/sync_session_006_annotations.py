"""Sync human annotations for session_006 into review_v03 and session_006_cam_A_annotations.csv."""

import logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def sync_session_006():
    user_csv = ROOT / "data" / "annotations" / "session_006_user_input.csv"
    if not user_csv.is_file():
        raise FileNotFoundError(f"Missing {user_csv}")

    df_user = pd.read_csv(user_csv, dtype=object, keep_default_na=False)
    logging.info(f"Loaded {len(df_user)} user annotated windows from {user_csv.name}")

    # 1. Update session_006_cam_A_annotations.csv
    canon_file = ROOT / "data" / "annotations" / "session_006_cam_A_annotations.csv"
    df_user.to_csv(canon_file, index=False, encoding="utf-8")
    logging.info(f"Updated canonical {canon_file.name} with {len(df_user)} human-annotated rows.")

    # 2. Update review_v03 studio file for Phong
    phong_studio = ROOT / "data" / "annotations" / "review_v03_studio" / "phong" / "session_006_cam_A_labels.csv"
    if phong_studio.is_file():
        df_target = pd.read_csv(phong_studio, dtype=object, keep_default_na=False)
        matched_count = 0
        for idx, r in df_target.iterrows():
            trk = str(r["track_id"])
            ts = float(r["t_start_ms"])
            matches = df_user[
                (df_user["track_id"].astype(str) == trk) &
                ((df_user["t_start_ms"].astype(float) - ts).abs() <= 1000)
            ]
            if not matches.empty:
                matched_count += 1
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

        df_target.to_csv(phong_studio, index=False, encoding="utf-8")
        logging.info(f"Updated {matched_count} / {len(df_target)} windows in Phong's studio file.")

if __name__ == "__main__":
    sync_session_006()
