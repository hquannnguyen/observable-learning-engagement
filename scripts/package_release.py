"""Create a local review bundle using an explicit allowlist; excludes raw video."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--additional-review-dir", action="append", default=[], help="Include another independent review set, without raw videos")
    parser.add_argument("--pose-weights", default=None)
    parser.add_argument("--output", default="releases/cvf25prj10-review-20261109.zip")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace an existing generated review bundle")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace bundle: {output}")
    files = set()
    suffixes = {".py", ".md", ".html", ".yaml", ".yml", ".json", ".csv", ".txt"}
    for folder in ("src", "scripts", "app", "tests", "configs", "docs", "data/splits"):
        files.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    for name in ("README.md", "requirements.txt", "AGENTS.md"):
        files.add(ROOT / name)
    for folder in map(Path, [args.features_dir, args.review_dir, *args.additional_review_dir]):
        folder = folder.resolve()
        if not folder.is_relative_to(ROOT / "data") or folder.is_relative_to(ROOT / "data/raw"):
            raise ValueError("Only derived datasets/review forms inside project data are allowed")
        files.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in suffixes | {".npz", ".parquet"})
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_relative_to(ROOT / "models/checkpoints"):
        raise ValueError("Checkpoint must be in models/checkpoints")
    files.add(checkpoint)
    if args.pose_weights:
        pose_weights = Path(args.pose_weights).resolve()
        if not pose_weights.is_relative_to(ROOT) or pose_weights.suffix != ".pt":
            raise ValueError("Pose weights must be a .pt file inside the project")
        files.add(pose_weights)
    for pattern in ("dataset_audit_20261109*.json", "annotation_tool_verification_20261109.json", "REMEDIATION_20261109.md", "progress_log.csv"):
        files.update((ROOT / "reports").glob(pattern))
    for folder in (ROOT / "reports/position_audit_20261109", ROOT / "reports/inference/external_106705_yolo11", ROOT / "reports/inference/external_215475_yolo11"):
        files.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".md"})
    for exp in (ROOT / "reports/experiments").glob("*"):
        if exp.is_dir():
            files.update(p for p in exp.iterdir() if p.is_file() and p.suffix in {".json", ".yaml", ".csv", ".md"})
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "LOCAL_REVIEW_BUNDLE", "acceptance_complete": False,
                "raw_video_included": False,
                "pending": ["Annotation rubric pilot and team agreement", "Human annotations and actual phase review", "Independent accuracy and positional evaluation"],
                "files": []}
    temporary = output.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for file in sorted(files):
            if not file.is_relative_to(ROOT):
                raise ValueError(f"Outside repository: {file}")
            relative = str(file.relative_to(ROOT)).replace("\\", "/")
            archive.write(file, relative)
            manifest["files"].append({"path": relative, "bytes": file.stat().st_size,
                                      "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        archive.writestr("REVIEW_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(output)
    print(f"{len(files)} files, {output.stat().st_size} bytes -> {output}")

if __name__ == "__main__":
    main()
