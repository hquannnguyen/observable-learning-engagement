#!/usr/bin/env python3
"""Package the adjudicated 600-window benchmark + derived v1 bundle (excluding raw video)."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src_feat = ROOT / "data" / "features" / "adjudicated_v03_600"
    src_ann_dir = ROOT / "data" / "annotations" / "review_v03_independent_20261124"
    v1_dir = ROOT / "data" / "features" / "dataset_v1"
    v1_dir.mkdir(parents=True, exist_ok=True)

    src_npz = np.load(src_feat / "window_features.npz", allow_pickle=False)
    npz_dict = {k: src_npz[k] for k in src_npz.files}
    npz_dict["dataset_version"] = np.array("v1")
    np.savez_compressed(v1_dir / "window_features.npz", **npz_dict)

    pw_df = pd.read_csv(src_feat / "participation_windows.csv", dtype=object, keep_default_na=False)
    pw_df["dataset_version"] = "v1"
    pw_df.to_csv(v1_dir / "participation_windows.csv", index=False)

    shutil.copy2(src_ann_dir / "adjudicated_v03.csv", v1_dir / "annotations_v1.csv")
    shutil.copy2(src_ann_dir / "adjudication_log_20261126.json", v1_dir / "adjudication_log.json")
    shutil.copy2(src_ann_dir / "qa_report_tai_vs_phong.json", v1_dir / "qa_report_pre_adjudication.json")
    shutil.copy2(src_feat / "split_report.json", v1_dir / "split_report.json")

    ann_v1 = pd.read_csv(v1_dir / "annotations_v1.csv", dtype=object, keep_default_na=False)
    split_rep = json.loads((v1_dir / "split_report.json").read_text(encoding="utf-8"))

    readme_text = (
        "# Dataset v1 — CVF25PRJ10 (Observable Learning Participation)\n\n"
        "- **Version:** `v1` (2026-11-26)\n"
        "- **Feature schema:** `posefeat-v2` (`(N, 40, 94)` at `10 FPS`, 4s window, 2s stride)\n"
        "- **Split version:** `split-v0.1` (session-disjoint train/val/test)\n"
        "- **Provenance:** Reconciled from independent annotations by Vũ Đức Tài (`annotator_a.csv`) "
        "and Vũ Văn Phong (`annotator_b.csv`); all 154 `VALID` disagreements adjudicated in favor of Tài (`ADJUDICATED`).\n\n"
        "## Counts (600-window benchmark)\n"
        "- **Total packaged windows:** `600` (`540 VALID` + `60 OCCLUDED`)\n"
        "  - `HUMAN` (agreed): `386`\n"
        "  - `ADJUDICATED` (154 resolved disagreements): `154`\n"
        "  - `OCCLUDED` (46 agreed + 14 state-mismatch conservative): `60`\n"
        "- **VALID windows by split (`540` total):**\n"
        "  - `train` (`175` — `session_001`, `session_002`, `class_1..5`): `72 HIGH` / `63 MODERATE` / `40 LOW`\n"
        "  - `val` (`92` — `session_003`, `session_004`): `23 HIGH` / `53 MODERATE` / `16 LOW`\n"
        "  - `test` (`273` — `session_005`, `session_006`): `58 HIGH` / `170 MODERATE` / `45 LOW`\n\n"
        "## Benchmark on v1 (`TCN` — `20261126-110817-tcn-adjudicated-v03-600`)\n"
        "- Val Macro-F1: `0.6999`\n"
        "- Raw Test Macro-F1: `0.4989` (`Accuracy: 56.04%`, `Balanced Acc: 51.16%`)\n"
        "- Accepted Test Macro-F1 (Uncertainty Gate, `77.66%` coverage): `0.5514` (`Accuracy: 61.32%`)\n\n"
        "## Notes\n"
        "- Pre-adjudication inter-annotator Cohen's Kappa: `0.4989` overall (`Train 0.6373`, `Val 0.6282`, `Test 0.2361`).\n"
        "- Raw video files (`data/raw/`) are strictly excluded for privacy.\n"
    )
    (v1_dir / "README.md").write_text(readme_text, encoding="utf-8")

    manifest = {
        "dataset_version": "v1",
        "created_at": datetime.now().isoformat(),
        "split_version": "split-v0.1",
        "feature_schema_version": "posefeat-v2",
        "raw_video_included": False,
        "total_windows_annotated_package": int(len(ann_v1)),
        "total_valid_windows": int((ann_v1["annotation_state"] == "VALID").sum()),
        "total_occluded_windows": int((ann_v1["annotation_state"] == "OCCLUDED").sum()),
        "pre_adjudication_kappa": 0.4989,
        "splits": split_rep,
        "files": {},
    }
    for p in sorted(v1_dir.iterdir()):
        if p.is_file() and p.name != "DATASET_V1_MANIFEST.json":
            manifest["files"][p.name] = {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
    (v1_dir / "DATASET_V1_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rel_dir = ROOT / "releases"
    rel_dir.mkdir(parents=True, exist_ok=True)
    zip_path = rel_dir / "cvf25prj10-dataset-v1-20261126.zip"
    tmp_zip = zip_path.with_suffix(".tmp.zip")

    include_paths = [p for p in v1_dir.iterdir() if p.is_file()]
    for rel_p in [
        "data/splits/train_sessions.txt",
        "data/splits/val_sessions.txt",
        "data/splits/test_sessions.txt",
        "data/annotations/review_v03_independent_20261124/annotator_a.csv",
        "data/annotations/review_v03_independent_20261124/annotator_b.csv",
        "data/annotations/review_v03_independent_20261124/adjudicated_v03.csv",
        "data/annotations/review_v03_independent_20261124/adjudication_log_20261126.json",
        "data/annotations/review_v03_independent_20261124/qa_report_tai_vs_phong.json",
        "data/annotations/review_v03_independent_20261124/disagreements_tai_vs_phong.csv",
        "configs/experiments/adjudicated_v03_600.yaml",
        "configs/experiments/adjudicated_v03_600_compact.yaml",
        "models/checkpoints/20261126-110817-tcn-adjudicated-v03-600.pt",
    ]:
        fp = ROOT / rel_p
        if fp.is_file():
            include_paths.append(fp)

    tcn_exp = ROOT / "reports" / "experiments" / "20261126-110817-tcn-adjudicated-v03-600"
    if tcn_exp.is_dir():
        include_paths.extend(p for p in tcn_exp.iterdir() if p.is_file())

    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(set(include_paths)):
            arc = str(f.relative_to(ROOT)).replace("\\", "/")
            zf.write(f, arc)
    tmp_zip.replace(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        n_files = len(zf.namelist())
    print(f"Packaged v1 dir: {v1_dir}")
    print(f"Packaged ZIP: {zip_path} ({zip_path.stat().st_size:,} bytes, {n_files} files, testzip={bad})")

    # Build standalone Dataset v1 Submission package (includes 11 anonymized pose parquets + metadata + docs; excludes raw mp4)
    sub_dir = rel_dir / "CVF25PRJ10_Dataset_v1"
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    for folder in ("annotations", "poses", "features", "metadata", "splits", "docs"):
        (sub_dir / folder).mkdir(parents=True, exist_ok=True)

    # Annotations
    for fn in (
        "adjudicated_v03.csv",
        "annotator_a.csv",
        "annotator_b.csv",
        "adjudication_log_20261126.json",
        "qa_report_tai_vs_phong.json",
        "disagreements_tai_vs_phong.csv",
    ):
        p = src_ann_dir / fn
        if p.is_file():
            shutil.copy2(p, sub_dir / "annotations" / fn)

    # Features (v1 600-window benchmark)
    for fn in ("window_features.npz", "participation_windows.csv", "split_report.json", "annotations_v1.csv"):
        p = v1_dir / fn
        if p.is_file():
            shutil.copy2(p, sub_dir / "features" / fn)

    # Anonymized Pose/Tracking Parquets (all 11 recordings)
    pose_parquets = sorted((ROOT / "data" / "poses").glob("*_poses.parquet"))
    for p in pose_parquets:
        shutil.copy2(p, sub_dir / "poses" / p.name)

    # Splits
    for fn in ("train_sessions.txt", "val_sessions.txt", "test_sessions.txt"):
        p = ROOT / "data" / "splits" / fn
        if p.is_file():
            shutil.copy2(p, sub_dir / "splits" / fn)

    # Session metadata (exclude raw mp4)
    for sess in ("session_001", "session_002", "session_003", "session_004", "session_005", "session_006"):
        meta_p = ROOT / "data" / "raw" / sess / "metadata.json"
        if meta_p.is_file():
            dst = sub_dir / "metadata" / sess / "metadata.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_p, dst)

    # Docs
    for fn in ("dataset_schema.md", "annotation_guide.md", "data_collection_protocol.md", "privacy_and_consent.md", "scope.md"):
        p = ROOT / "docs" / fn
        if p.is_file():
            shutil.copy2(p, sub_dir / "docs" / fn)

    # Submission README & Manifest
    sub_readme = (
        "# CVF25PRJ10 — Dataset v1 Submission Package (CVFP26BCPPRJC012)\n\n"
        "- **Tên đề tài:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được\n"
        "- **Phiên bản Dataset:** `v1` (2026-11-26)\n"
        "- **Quyền riêng tư:** Đã loại bỏ hoàn toàn video thô (`.mp4`), chỉ giữ dữ liệu khung xương & track ẩn danh (`poses/*.parquet`), đặc trưng chuỗi thời gian (`features/window_features.npz`), metadata phiên học và nhãn hành vi.\n\n"
        "## Cấu trúc gói nộp\n"
        "- `annotations/`: `adjudicated_v03.csv` (600 cửa sổ chuẩn sau phân xử: 386 `HUMAN` + 154 `ADJUDICATED` + 60 `OCCLUDED`), kèm 2 phiếu gán mù gốc (`annotator_a.csv` - Tài, `annotator_b.csv` - Phong), log phân xử & báo cáo QA Cohen's Kappa (`0.4989` toàn cục; Train `0.6373`, Val `0.6282`).\n"
        "- `poses/`: 11 file Parquet dữ liệu pose YOLOv8/YOLO11 + ByteTrack ẩn danh (`session_001..006` và `class_1..5`).\n"
        "- `features/`: `window_features.npz` (`175 train / 92 val / 273 test`, shape `(N, 40, 94)`, `posefeat-v2`) và `participation_windows.csv`.\n"
        "- `splits/`: Tách phiên học cố định `split-v0.1` (`train_sessions.txt`, `val_sessions.txt`, `test_sessions.txt`), không rò rỉ khung hình.\n"
        "- `metadata/`: `metadata.json` của 6 phiên học (`session_001..006`).\n"
        "- `docs/`: Hợp đồng schema (`dataset_schema.md`), hướng dẫn gán nhãn (`annotation_guide.md`), quy trình thu thập & bảo mật.\n\n"
        "## Cách đọc nhanh bằng Python\n"
        "```python\n"
        "import numpy as np, pandas as pd\n"
        "ann = pd.read_csv('annotations/adjudicated_v03.csv')\n"
        "pose_s1 = pd.read_parquet('poses/session_001_cam_A_poses.parquet')\n"
        "feat = np.load('features/window_features.npz')\n"
        "X_train, y_train = feat['X_train'], feat['y_train']\n"
        "```\n"
    )
    (sub_dir / "README.md").write_text(sub_readme, encoding="utf-8")

    sub_manifest = {
        "package_name": "CVF25PRJ10_Dataset_v1",
        "dataset_version": "v1",
        "created_at": datetime.now().isoformat(),
        "raw_video_included": False,
        "pose_parquet_files": len(pose_parquets),
        "annotated_windows": int(len(ann_v1)),
        "valid_windows": int((ann_v1["annotation_state"] == "VALID").sum()),
        "occluded_windows": int((ann_v1["annotation_state"] == "OCCLUDED").sum()),
        "splits": split_rep,
        "files": {},
    }
    sub_files = [p for p in sorted(sub_dir.rglob("*")) if p.is_file() and p.name != "DATASET_MANIFEST.json"]
    for p in sub_files:
        rel_key = str(p.relative_to(sub_dir)).replace("\\", "/")
        sub_manifest["files"][rel_key] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    (sub_dir / "DATASET_MANIFEST.json").write_text(
        json.dumps(sub_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sub_zip = rel_dir / "CVF25PRJ10_Dataset_v1_Submission.zip"
    tmp_sub_zip = sub_zip.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_sub_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(sub_dir.rglob("*")):
            if p.is_file():
                arc = "CVF25PRJ10_Dataset_v1/" + str(p.relative_to(sub_dir)).replace("\\", "/")
                zf.write(p, arc)
    tmp_sub_zip.replace(sub_zip)
    with zipfile.ZipFile(sub_zip, "r") as zf:
        bad_sub = zf.testzip()
        n_sub = len(zf.namelist())
    print(f"Submission dir: {sub_dir}")
    print(f"Submission ZIP: {sub_zip} ({sub_zip.stat().st_size:,} bytes, {n_sub} files, testzip={bad_sub})")


if __name__ == "__main__":
    main()
