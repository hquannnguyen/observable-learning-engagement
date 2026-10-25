# BÁO CÁO KỸ THUẬT CHI TIẾT THEO LUỒNG CÔNG VIỆC

**Dự án:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được  
**Mã đề tài:** CVFP26BCPPRJC012 — **Nhóm:** CVF25PRJ10  
**Ngày:** 25/10/2026 — **Phiên bản:** v3.0 (đính chính provenance, split, evaluation)  
**Nguồn trạng thái:** `docs/PROJECT_STATUS.md` — **Hạn chế:** `docs/LIMITATIONS.md`

---

## 1. TỔNG QUAN LUỒNG KỸ THUẬT

```text
11 video lớp học 1080p
   → sampling 10 FPS
   → YOLOv8-Pose (17 keypoints, bbox)
   → ByteTrack (track_id ẩn danh)
   → Pose Quality Gate
   → Cleaning + normalization (nội suy gap ≤3 frame, torso scale)
   → 77 features/frame
   → Sliding window 4s / stride 2s → tensor [N, 40, 77]
   → LSTM 2 lớp (hidden 128, dropout 0.30)
   → 3 xác suất canonical
   → Uncertainty gate (visibility / keypoint conf / model conf)
   → HIGH / MODERATE / LOW / UNCERTAIN
```

**Trạng thái thực tế:** pipeline chạy end-to-end (kể cả upload video mới), có LSTM/TCN,
zone, export và dashboard. Phần còn thiếu: human ground truth, ablation/calibration,
benchmark full-flow trên video dài/GPU.

---

## 2. DỮ LIỆU, ANNOTATION & SPLIT (Vũ Văn Phong)

### 2.1. Dữ liệu

- 11 video: `class_1`–`class_5`, `session_001`–`session_006`.
- `session_001`: 15 phút, metadata 8 người, một phase `lecture`.
- Một góc máy chính (`cam_A`), 1080p, 30 FPS gốc.
- **Pose coverage:** đã full — `session_001–005` 900 s (9.000 frame),
  `session_006` 250,4 s (2.505 frame). Tổng 559.499 detection.
- Dataset full: **32.959 window** (train 13.481 / val 12.011 / test 7.467).

### 2.2. Annotation — đính chính quan trọng

- `data/annotations/annotations_all_v0.1.csv`: **8.532 dòng**, trong đó
  **7.613 dòng `WEAK_RULE`** và **919 dòng không có nhãn** (occluded/invalid).
- **Không có nhãn HUMAN thật.** Bản cũ có cột `HUMAN` là do
  `scripts/generate_annotations.py` tự gắn; đã bị loại bỏ.
- Hai file `pilot_double_annotator_*.csv` cũ là bản copy synthetic; đã chuyển vào
  `data/annotations/deprecated/`. **Kappa 1.0 cũ không có giá trị.**
- Quy trình nhãn người: `data/annotations/human/README.md`,
  `scripts/sample_annotation_subset.py`, `scripts/dataset_qa.py --strict-human`.

### 2.3. Split `split-v0.1` (đóng băng)

| Split | Session | Tổng window | Window có nhãn weak |
|---|---|---:|---:|
| Train | `class_1..class_5`, `session_001`, `session_002` | 13.481 | 11.629 |
| Val | `session_003`, `session_004` | 12.011 | 10.466 |
| Test | `session_005`, `session_006` | 7.467 | 6.518 |

Hash của 3 file split nằm trong `data/features/split_report.json`.
Builder đọc file split, không tự chia lại; nếu thiếu/sai sẽ báo lỗi.

### 2.4. Class distribution (weak)

- HIGH: 16.747
- MODERATE: 9.943
- LOW: 1.923
- Không nhãn: 4.346

Mất cân bằng lớp rõ rệt; training dùng class-weighted CrossEntropy.

---

## 3. POSE, TRACKING, FEATURE (Nguyễn Hữu Quân)

### 3.1. Pose extraction

- Model: `yolov8n-pose.pt`, `imgsz=640`, conf 0.30.
- Sampling 10 FPS; 17 COCO keypoints.
- Keypoint thiếu lưu `NaN` (không lưu `(0,0)`), kèm `visible_ratio`,
  `mean_keypoint_conf`.
- Thử nghiệm `yolo11m-pose` @1280 trong `configs/experiments/pose_yolo11m_hd.yaml`.

### 3.2. Tracking

- ByteTrack; lưu `track_id` ẩn danh, track statistics, lọc track ngắn.
- Chưa có so sánh ByteTrack vs BoT-SORT (T34).

### 3.3. Cleaning & normalization

- Nội suy tuyến tính gap ≤3 frame; gap dài giữ NaN.
- Chuẩn hóa gốc vai + scale torso/bbox.
- **Caveat:** `src/preprocessing/features.py` dùng `np.nan_to_num(..., 0.0)` nên
  mask khuyết điểm không còn tường minh ở vector 77 chiều. Feature v2 (94 chiều,
  thêm 17 mask) là việc B1.

### 3.4. 77 features

| Nhóm | Số chiều |
|---|---:|
| Normalized keypoints (17×2) | 34 |
| Velocity (Δ/dt) | 34 |
| Shoulder slope, head proxy x/y, hand raised L/R, head down | 6 |
| Motion energy (RMS velocity) | 1 |
| visible_ratio, mean_keypoint_conf | 2 |
| **Tổng** | **77** |

`head_orientation_proxy` chỉ là hình học mũi–vai, **không phải gaze**.

### 3.5. Sliding window

- 4.0s = 40 frame @10 FPS; stride 2.0s = 20 frame.
- Tensor `[7259, 40, 77]`.
- `quality_state` hiện dùng ngưỡng hardcode 0.60/0.30 (B2 sẽ đưa vào config).

---

## 4. MÔ HÌNH, TRAINING, EVALUATION (Vũ Đức Tài)

### 4.1. LSTM

- Input `[B, 40, 77]`.
- 2 tầng LSTM, hidden 128, dropout 0.30.
- Head: mean-pool + last-step → Linear(128→64) → ReLU → Dropout → Linear(64→3).
- Softmax 3 lớp canonical.

### 4.2. Training thực tế

| Thông số | Giá trị |
|---|---|
| Optimizer | **Adam** (không phải AdamW) |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Loss | CrossEntropy + class weights |
| Early stopping | patience 10, monitor `val_macro_f1` |
| Best epoch | 24 |
| Best val Macro-F1 | 0.9159 |
| Training time | 538 s (CPU) |
| Device | CPU |

Báo cáo cũ ghi AdamW/batch 64/patience 5 là **không khớp code**; bản này đã sửa.

### 4.3. Kết quả test (weak-label)

**Experiment:** `20261025-162855-lstm-full-v0-1`  
**Split:** `split-v0.1` — test 7.467 window, 6.518 có nhãn.

| Chỉ số | Raw | Accepted |
|---|---:|---:|
| Macro-F1 | 0.9098 | 0.9151 |
| Balanced Accuracy | 0.9107 | 0.9162 |
| Accuracy | 0.9228 | 0.9281 |
| Coverage | 100% | 98.54% |
| Uncertain rate | 0% | 1.46% |

Confusion matrix raw:

```text
                HIGH   MODERATE   LOW
HIGH             3302      130       6
MODERATE          172     1907     103
LOW                 0       92     806
```

Per-session accepted Macro-F1: `session_005` 0.9143 (N=5.241),
`session_006` 0.9183 (N=1.277).

**Diễn giải:** đây là mức độ khớp với **luật yếu sinh nhãn**, không phải độ chính xác
trên hành vi người thật. Không được trích dẫn như kết luận về người học.

### 4.4. Uncertainty gate

- `visible_ratio < 0.60` → `LOW_VISIBILITY`
- `mean_keypoint_conf < 0.30` → `LOW_KEYPOINT_CONF`
- `max_prob < 0.60` → `LOW_MODEL_CONF`
- Ngược lại → canonical label, reason `ACCEPTED`

Evaluation lưu `predictions.csv` với từng window, `true_label`, `pred_label`,
`gated_label`, `p_max`, `gate_reason`, `label_source`.

### 4.5. Reproducibility

Mỗi experiment có:
- `metadata.json`: experiment ID, created_at, seed, git commit, dataset/split version,
  config SHA256, checkpoint SHA256, class counts, metric scope.
- `config.yaml` snapshot.
- `metrics.json`, `training_history.csv`, `evaluation_report.json`, `predictions.csv`.

### 4.6. TCN và so sánh

`src/models/tcn.py` dùng causal dilated convolution, cùng input `[B, 40, 77]`.

| Chỉ số | LSTM | TCN |
|---|---:|---:|
| Raw Macro-F1 | **0.9098** | 0.9027 |
| Accepted Macro-F1 | 0.9151 | **0.9198** |
| Accepted Accuracy | 0.9281 | **0.9332** |
| Coverage | **98.54%** | 95.95% |
| Params | 246.531 | 63.363 |
| CPU ms/122 window | 31.74 | 17.05 |

TCN nhẹ/nhanh hơn; accepted nhỉnh hơn rất nhẹ nhưng coverage thấp hơn. Chưa tune.
Chi tiết: `reports/tcn_vs_lstm.md`.

End-to-end CPU (class_1, 150 frame sampled): 23,96 s, **6,26 FPS**,
**95,8 s mỗi phút video**, 66 window (63 accepted). Xem `reports/benchmark.md`.

---

## 5. DASHBOARD, ZONE & EXPORT (Nguyễn Dũng Tiến)

**Trạng thái: ĐÃ TRIỂN KHAI (viewer + export + upload inference).**

- `app/streamlit_app.py`: 6 tab — Tổng quan, Timeline, Vùng lớp, Preview ẩn danh,
  Export, Video mới.
- Tab “Video mới”: upload → `scripts/predict_video.py` → hiển thị predictions/summary.
- Timeline theo **session + phút** (không gộp session).
- Zone 3×3 từ `scripts/assign_zones.py` + `configs/zones/grid_3x3.yaml`;
  bảng zone luôn có `N`, cảnh báo khi N < 20.
- `src/reporting/aggregation.py` + `exporter.py`: `results.csv`, `timeline.csv`,
  `zone_summary.csv`, `summary.json`, `summary_report.md`.
- Preview overlay skeleton + anonymous track ID; upload inference chạy CPU, chưa có progress chi tiết.
- AppTest: 0 exception; HTTP smoke test 200.

**Chưa có:** upload video → xử lý → báo cáo trong một flow (T35/T39).

---

## 6. ARTIFACT & LỆNH CHẠY

| Artifact | Đường dẫn |
|---|---|
| Annotation weak | `data/annotations/annotations_all_v0.1.csv` |
| Split | `data/splits/{train,val,test}_sessions.txt`, `split_metadata.json` |
| Window tensor | `data/features/window_features.npz` |
| Split report | `data/features/split_report.json` |
| Checkpoint | `models/checkpoints/20261025-162855-lstm-full-v0-1.pt` |
| Metrics | `reports/experiments/20261025-162855-lstm-full-v0-1/metrics.json` |
| Evaluation | `reports/experiments/20261025-162855-lstm-full-v0-1/evaluation_report.json` |
| Predictions | `reports/experiments/20261025-162855-lstm-full-v0-1/predictions.csv` |
| Export bundle | `reports/experiments/20261025-162855-lstm-full-v0-1/export/` |
| TCN experiment | `reports/experiments/20261025-163801-tcn-full-v0-1/` |
| So sánh | `reports/tcn_vs_lstm.md` |
| Benchmark | `reports/benchmark.md` |
| Dashboard | `app/streamlit_app.py` |

```bash
python scripts/build_dataset.py
python scripts/train.py --name lstm_weaklabel_v0.1 --epochs 30
python scripts/evaluate.py --experiment-dir reports/experiments/<experiment_id>
python scripts/assign_zones.py
python scripts/build_dataset.py --poses-dir data/poses_zones --output-dir data/features_zones
python scripts/export_report.py --experiment-dir reports/experiments/<experiment_id>
python scripts/benchmark.py --video data/raw/class_1.mp4 --max-frames 60
python -m streamlit run app/streamlit_app.py
python -m pytest -q --basetemp=D:\ThiGiacMayTinh\.pytest_tmp
```

---

## 7. HẠN CHẾ & VIỆC TIẾP THEO

1. **Chưa có human ground truth** → mọi metric là weak-label agreement.
2. Test N = 6.518 window nhưng chỉ từ 2 session (`session_005`, `session_006`);
   vẫn chưa đủ để kết luận tổng quát theo người/zone.
3. Feature v2 (mask) đã có nhưng model chính vẫn dùng v1; chưa có ablation.
4. Dashboard đã upload/chạy pipeline; benchmark chưa có ID switch GT và full-flow video dài.
5. Chưa có calibration (ECE), M0/M1, threshold tuning trên validation.

Ưu tiên: human annotation (A6/A7) → full-flow benchmark (T39) →
M0/M1 + ablation + calibration (T20/T21/T26–T31).
