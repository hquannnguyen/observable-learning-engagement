"""Recover positioning for a frozen inference review, checking every window key."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.pose.extractor import extract_poses_from_video
from src.preprocessing.cleaner import clean_and_normalize_poses
from src.preprocessing.windows import build_track_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", required=True)
    args = parser.parse_args()
    folder = Path(args.inference_dir)
    meta = json.loads((folder / "inference_meta.json").read_text())
    output = ROOT / "data/poses" / f"{meta['session_id']}_cam_A_poses.csv"
    if output.exists():
        raise FileExistsError(f"Preserving existing tracks: {output}")
    poses, extraction = extract_poses_from_video(meta["video"], session_id=meta["session_id"],
        config_path=meta["config"], track=True, device="cpu")
    if extraction["total_detections"] != meta["detections"]:
        raise ValueError("Detection counts differ from frozen run; refusing to associate IDs")
    cleaned = clean_and_normalize_poses(poses, min_track_seconds=2.0)
    records = []
    for _, group in cleaned.groupby(["session_id", "camera_id", "track_id"]):
        windows, _ = build_track_windows(group, split="inference")
        records.extend(windows)
    keys = ["window_id", "session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms"]
    frozen = pd.read_csv(folder / "predictions.csv")
    actual = pd.DataFrame(records)
    if set(map(tuple, actual[keys].to_numpy())) != set(map(tuple, frozen[keys].to_numpy())):
        raise ValueError("Track/window keys differ from frozen run; refusing to associate IDs")
    poses.to_csv(output, index=False)
    audit = {"purpose":"ANNOTATION_POSITIONING_ONLY", "frozen_predictions_unchanged":True,
             "matching_frozen_windows":len(actual), "detections":len(poses),
             "source_predictions_sha256":hashlib.sha256((folder / "predictions.csv").read_bytes()).hexdigest(),
             "poses_sha256":hashlib.sha256(output.read_bytes()).hexdigest(), "config":meta["config"]}
    output.with_suffix(".annotation_provenance.json").write_text(json.dumps(audit, indent=2))
    print(output, flush=True)


if __name__ == "__main__":
    main()
