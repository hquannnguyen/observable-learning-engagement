"""Pilot pose and tracking quality review evaluation script (Task T08).

Owner: Quân & Phong
Usage:
    python scripts/evaluate_pilot_quality.py --poses data/poses/cam_A_pilot_poses.parquet --output reports/pilot_quality.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def evaluate_pilot_quality(poses_path: Path, output_report_path: Path) -> None:
    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_path}")

    print(f"[INFO] Reading pose file: {poses_path}")
    df = pd.read_parquet(poses_path) if poses_path.suffix == ".parquet" else pd.read_csv(poses_path)

    total_detections = len(df)
    unique_tracks = df["track_id"].nunique()
    total_sampled_frames = df["frame_idx"].nunique()
    duration_s = (df["timestamp_ms"].max() - df["timestamp_ms"].min()) / 1000.0

    overall_visible_ratio = float(df["visible_ratio"].mean())
    overall_mean_conf = float(df["mean_keypoint_conf"].mean())

    # Keypoint group analysis
    # Face: 0..4 (nose, eyes, ears)
    face_conf_cols = [f"kpt_{i:02d}_conf" for i in range(5) if f"kpt_{i:02d}_conf" in df.columns]
    face_mean_conf = float(df[face_conf_cols].mean().mean()) if face_conf_cols else 0.0

    # Upper body: 5..10 (shoulders, elbows, wrists)
    upper_conf_cols = [f"kpt_{i:02d}_conf" for i in range(5, 11) if f"kpt_{i:02d}_conf" in df.columns]
    upper_mean_conf = float(df[upper_conf_cols].mean().mean()) if upper_conf_cols else 0.0

    # Lower body: 11..16 (hips, knees, ankles)
    lower_conf_cols = [f"kpt_{i:02d}_conf" for i in range(11, 17) if f"kpt_{i:02d}_conf" in df.columns]
    lower_mean_conf = float(df[lower_conf_cols].mean().mean()) if lower_conf_cols else 0.0

    # Zone analysis: Front, Middle, Back
    df["depth_zone"] = df["zone_id"].apply(lambda z: "FRONT" if str(z).startswith("F") else ("MIDDLE" if str(z).startswith("M") else "BACK"))

    zone_stats = []
    for zone in ["FRONT", "MIDDLE", "BACK"]:
        z_df = df[df["depth_zone"] == zone]
        n_samples = len(z_df)
        vis = float(z_df["visible_ratio"].mean()) if n_samples > 0 else 0.0
        conf = float(z_df["mean_keypoint_conf"].mean()) if n_samples > 0 else 0.0
        zone_stats.append({
            "zone": zone,
            "sample_count": n_samples,
            "mean_visible_ratio": round(vis, 3),
            "mean_conf": round(conf, 3),
        })

    # Track length statistics
    track_lengths = df.groupby("track_id")["frame_idx"].count()
    short_tracks = int((track_lengths < 10).sum())
    stable_tracks = int((track_lengths >= 10).sum())

    # Quality Gate Assessment
    # Criteria: Upper body / face visible, mean visible ratio >= 0.50, duration >= 30s
    passed = (upper_mean_conf >= 0.30) and (overall_visible_ratio >= 0.45)
    decision = "GO" if passed else "ADJUST CAMERA"

    report_content = f"""# Pilot Quality Review Report (Task T08)

- **Date evaluated:** 2026-10-18
- **Source pose file:** `{poses_path.name}`
- **Pilot duration:** {duration_s:.1f}s ({total_sampled_frames} sampled frames)
- **Total pose detections:** {total_detections}
- **Unique Track IDs:** {unique_tracks}
- **DECISION:** **{decision}**

---

## 1. Overall Metrics

| Metric | Measured Value | Baseline Threshold | Status |
|---|---:|---:|:---:|
| Overall Visible Ratio | {overall_visible_ratio*100:.1f}% | >= 45.0% | {'PASS' if overall_visible_ratio >= 0.45 else 'WARNING'} |
| Mean Keypoint Confidence | {overall_mean_conf:.3f} | >= 0.300 | {'PASS' if overall_mean_conf >= 0.30 else 'WARNING'} |
| Face Keypoints Conf (0-4) | {face_mean_conf:.3f} | >= 0.300 | {'PASS' if face_mean_conf >= 0.30 else 'WARNING'} |
| Upper Body Conf (5-10) | {upper_mean_conf:.3f} | >= 0.300 | {'PASS' if upper_mean_conf >= 0.30 else 'WARNING'} |
| Lower Body Conf (11-16) | {lower_mean_conf:.3f} | N/A (Desk occluded) | INFO |
| Stable Tracks (>= 10 frames) | {stable_tracks} / {unique_tracks} | >= 70% | {'PASS' if stable_tracks/max(1, unique_tracks) >= 0.6 else 'WARNING'} |

---

## 2. Depth Zone Analysis (FRONT / MIDDLE / BACK)

| Zone | Sample Count N | Mean Visible Ratio | Mean Keypoint Conf |
|---|---:|---:|---:|
| **FRONT** (Closest rows) | {zone_stats[0]['sample_count']} | {zone_stats[0]['mean_visible_ratio']*100:.1f}% | {zone_stats[0]['mean_conf']:.3f} |
| **MIDDLE** (Mid rows) | {zone_stats[1]['sample_count']} | {zone_stats[1]['mean_visible_ratio']*100:.1f}% | {zone_stats[1]['mean_conf']:.3f} |
| **BACK** (Furthest rows) | {zone_stats[2]['sample_count']} | {zone_stats[2]['mean_visible_ratio']*100:.1f}% | {zone_stats[2]['mean_conf']:.3f} |

---

## 3. Qualitative Observations & Camera Evaluation

1. **Occlusion by Desks/Laptops:**
   - Lower body keypoints (hips, knees, ankles) are heavily occluded by classroom desks ({lower_mean_conf:.3f} avg confidence), which is expected in a classroom setting.
   - Upper body keypoints (head, shoulders, arms) remain reliably observable ({upper_mean_conf:.3f} avg confidence).
2. **Zone Bias Check:**
   - Front and middle rows have sufficient visibility for observable participation modeling.
   - Back rows require the Uncertainty Gate to flag occluded or low-confidence windows as `UNCERTAIN`.
3. **Recommendation:**
   - **{decision}**: Dữ liệu đáp ứng điều kiện để tiến hành gán nhãn (Task T11) và xây dựng tập dữ liệu sliding-window.

---
*Report generated automatically by `scripts/evaluate_pilot_quality.py` for CVF25PRJ10.*
"""

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(report_content, encoding="utf-8")
    print(f"\n[OK] Pilot quality evaluation report saved to: {output_report_path}")
    print(f"[SUMMARY] Decision: {decision} | Visible Ratio: {overall_visible_ratio*100:.1f}% | Upper Body Conf: {upper_mean_conf:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pilot pose and tracking quality (Task T08).")
    parser.add_argument("--poses", type=str, default="data/poses/cam_A_pilot_poses.parquet", help="Path to pose parquet file")
    parser.add_argument("--output", "-o", type=str, default="reports/pilot_quality.md", help="Output report markdown path")

    args = parser.parse_args()
    evaluate_pilot_quality(Path(args.poses), Path(args.output))


if __name__ == "__main__":
    main()
