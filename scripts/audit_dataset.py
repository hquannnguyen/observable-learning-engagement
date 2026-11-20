"""Write dataset readiness evidence; no inferred sessions, phases or human labels."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def audit(windows_path: Path, raw_dir: Path):
    df = pd.read_csv(windows_path)
    metadata = []
    for session in sorted(df["session_id"].unique()):
        path = raw_dir / str(session) / "metadata.json"
        item = {"session_id": str(session), "metadata_exists": path.is_file()}
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            item.update(phase_verified=meta.get("phase_verified", False),
                        source_recording_id=meta.get("source_recording_id"),
                        participant_count_declared=meta.get("participants"),
                        consent_declared=meta.get("consent_confirmed"),
                        source_independence_verified=meta.get("source_independence_verified", False))
        metadata.append(item)
    duplicates = int(df["window_id"].duplicated().sum())
    sources = df["label_source"].fillna("NONE").value_counts().to_dict()
    human_count = int(df["label_source"].isin(["HUMAN", "ADJUDICATED"]).sum())
    return {"windows": len(df), "duplicate_window_ids": duplicates,
            "label_source_counts": sources, "human_windows": human_count,
            "phase_counts": df["class_phase"].fillna("UNKNOWN").value_counts().to_dict(),
            "split_counts": df["split"].value_counts().to_dict(),
            "metadata": metadata,
            "official_evaluation_ready": False if not human_count or duplicates else "REVIEW_REQUIRED",
            "pending": ["Independent human annotation and adjudication",
                        "Verify actual classroom phase intervals; legacy generated phases are not evidence",
                        "Verify source recordings/participants across splits; session folder names alone do not establish independence"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="data/features/participation_windows.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="reports/dataset_audit.json")
    args = parser.parse_args()
    report = audit(Path(args.windows), Path(args.raw_dir))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{report['windows']} windows; human={report['human_windows']}; duplicates={report['duplicate_window_ids']} -> {path}")

if __name__ == "__main__":
    main()
