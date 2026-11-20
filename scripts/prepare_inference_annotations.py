"""Create blind review forms for supplied videos, without retraining or relabelling."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    frames, sources = [], []
    for name in args.predictions:
        source = Path(name)
        df = pd.read_csv(source)
        frames.append(pd.DataFrame({
            "window_id": df["window_id"], "session_id": df["session_id"],
            "camera_id": df.get("camera_id", "cam_A"), "track_id": df["track_id"],
            "t_start_ms": df["t_start_ms"], "t_end_ms": df["t_end_ms"],
            "class_phase": "UNKNOWN", "zone_id": df.get("zone_id", "UNKNOWN"),
            "quality_state": df.get("quality_state", "UNREVIEWED"),
            "participation_label": None, "label_source": None, "split": "test"}))
        sources.append({"predictions": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    combined = pd.concat(frames, ignore_index=True)
    if combined["window_id"].duplicated().any():
        raise ValueError("Duplicate windows across inference runs")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    window_file = out / "review_windows.csv"
    combined.to_csv(window_file, index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/sample_annotation_subset.py"), "--windows", str(window_file), "--out-dir", str(out), "--n-test", "0", "--n-train", "0", "--n-val", "0"], cwd=ROOT, check=True)
    template = pd.read_csv(out / "annotation_template.csv")
    previews = {}
    for name in args.predictions:
        path = Path(name)
        preview = path.parent / "head_blurred_preview.mp4"
        if preview.is_file():
            for session_id in pd.read_csv(path, usecols=["session_id"])["session_id"].unique():
                previews[session_id] = str(preview)
    template["pose_overlay_path"] = template["session_id"].map(previews).fillna("")
    template.to_csv(out / "annotation_template.csv", index=False)
    # Independent blank forms, not completed/copied annotations.
    for annotator in ("annotator_a", "annotator_b"):
        target = out / f"{annotator}.csv"
        if target.exists():
            raise FileExistsError(f"Do not overwrite a reviewer form: {target}")
        template.to_csv(target, index=False)
    manifest = {"scope": "external_video_review", "windows": len(combined), "sources": sources,
                "status": "AWAITING_INDEPENDENT_HUMAN_ANNOTATION", "has_human_labels": False,
                "note": "The review split does not alter frozen training splits. No prediction labels are shown in the forms."}
    (out / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Blind forms for {len(combined)} windows -> {out}")

if __name__ == "__main__":
    main()
