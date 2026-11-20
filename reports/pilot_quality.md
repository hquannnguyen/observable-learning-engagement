# Pilot Quality Review Report (Task T08)

- **Date evaluated:** 2026-10-18
- **Source pose file:** `cam_A_pilot_poses.parquet`
- **Pilot duration:** 59.8s (300 sampled frames)
- **Total pose detections:** 4375
- **Unique Track IDs:** 38
- **DECISION:** **GO**

---

## 1. Overall Metrics

| Metric | Measured Value | Baseline Threshold | Status |
|---|---:|---:|:---:|
| Overall Visible Ratio | 70.4% | >= 45.0% | PASS |
| Mean Keypoint Confidence | 0.569 | >= 0.300 | PASS |
| Face Keypoints Conf (0-4) | 0.706 | >= 0.300 | PASS |
| Upper Body Conf (5-10) | 0.742 | >= 0.300 | PASS |
| Lower Body Conf (11-16) | 0.281 | N/A (Desk occluded) | INFO |
| Stable Tracks (>= 10 frames) | 25 / 38 (65.8%) | >= 60.0% | PASS |

---

## 2. Depth Zone Analysis (FRONT / MIDDLE / BACK)

| Zone | Sample Count N | Mean Visible Ratio | Mean Keypoint Conf |
|---|---:|---:|---:|
| **FRONT** (Closest rows) | 86 | 73.6% | 0.571 |
| **MIDDLE** (Mid rows) | 1941 | 67.8% | 0.565 |
| **BACK** (Furthest rows) | 2348 | 72.5% | 0.572 |

---

## 3. Qualitative Observations & Camera Evaluation

1. **Occlusion by Desks/Laptops:**
   - Lower body keypoints (hips, knees, ankles) are heavily occluded by classroom desks (0.281 avg confidence), which is expected in a classroom setting.
   - Upper body keypoints (head, shoulders, arms) remain reliably observable (0.742 avg confidence).
2. **Zone Bias Check:**
   - Front and middle rows have sufficient visibility for observable participation modeling.
   - Back rows require the Uncertainty Gate to flag occluded or low-confidence windows as `UNCERTAIN`.
3. **Recommendation:**
   - **GO**: Dữ liệu đáp ứng điều kiện để tiến hành gán nhãn (Task T11) và xây dựng tập dữ liệu sliding-window.

---
*Report generated automatically by `scripts/evaluate_pilot_quality.py` for CVF25PRJ10.*
