# BÁO CÁO TIẾN ĐỘ DỰ ÁN NGHIÊN CỨU & PHÁT TRIỂN

> **Đề tài:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được  
> **Tên tiếng Anh:** Observable Learning Participation Analysis  
> **Mã đề tài:** CVFP26BCPPRJC012  
> **Nhóm thực hiện:** CVF25PRJ10  
> **Ngày báo cáo:** 25/10/2026 (bản đính chính)  
> **Nguồn trạng thái:** `docs/PROJECT_STATUS.md`  
> **Nguồn hạn chế:** `docs/LIMITATIONS.md`

---

## 1. THÔNG TIN CHUNG & PHÂN CÔNG

### 1.1. Mục tiêu
Xây dựng hệ thống thị giác máy tính + phân tích chuỗi thời gian để **ước lượng mức độ
tham gia học tập quan sát được** từ video lớp học.

Hệ thống chỉ mô tả tín hiệu quan sát được (tư thế, chuyển động, độ ổn định, hướng thân/đầu
gần đúng). Hệ thống **không** suy diễn cảm xúc, động cơ, ý định, mức độ chú ý nội tại hay
danh tính; không gắn nhãn tiêu cực như “lười”, “chán”, “mất tập trung”.

### 1.2. Thành viên

| STT | Họ và tên | MSSV | Trách nhiệm chính |
|---:|---|---|---|
| 1 | Vũ Văn Phong | B23DCPT250 | Dữ liệu, annotation, QA dataset, split |
| 2 | Nguyễn Hữu Quân | B23DCPT262 | Pose, tracking, preprocessing, feature/window |
| 3 | Vũ Đức Tài | B23DCPT277 | LSTM, training/evaluation, uncertainty |
| 4 | Nguyễn Dũng Tiến | B23DCPT306 | Streamlit, reporting, anonymization |

---

## 2. TÓM TẮT ĐIỀU HÀNH

Đợt cập nhật 2026-10-25 tập trung vào **tính trung thực và tái lập** của kết quả:

1. **Đính chính provenance:** toàn bộ 8.532 dòng annotation hiện là `WEAK_RULE`.
   Các nhãn “HUMAN” cũ do script tự sinh đã bị loại bỏ và archive.
2. **Đính chính split & dữ liệu:** `split-v0.1` đóng băng — train `class_1..class_5`,
   `session_001`, `session_002`; val `session_003`, `session_004`; test `session_005`,
   `session_006`. Sau khi pose full, dataset có **32.959 window** (train 13.481 /
   val 12.011 / test 7.467); test có **6.518** window nhãn weak hợp lệ.
3. **Đính chính evaluation:** join theo `window_id`, lưu `predictions.csv` và lý do gate;
   không còn phụ thuộc thứ tự dòng.
4. **Đính chính metric:** mọi con số hiện tại là **weak-label agreement**, không phải
   accuracy trên hành vi người thật. Chưa có human test set.
5. **Đính chính trạng thái sản phẩm:** dashboard/export/zone/TCN/benchmark **đã có**; đánh giá độc lập còn chờ nhãn người.

---

## 3. TRẠNG THÁI THEO PHÂN HỆ

| Phân hệ | Trạng thái | Ghi chú |
|---|---|---|
| Scope & label contract | DONE | `docs/scope.md`, `src/contracts/labels.py` |
| Pose extraction | DONE | YOLOv8-Pose, 17 keypoints, 10 FPS |
| Tracking | DONE | ByteTrack, anonymous `track_id` |
| Quality gate + cleaning | DONE | Ngưỡng lấy từ config; mask tường minh ở `posefeat-v2` |
| Feature | DONE | `posefeat-v1` (77) + `posefeat-v2` (94, 17 mask) |
| Sliding window | DONE | 4s/40 frame, stride 2s/20 frame |
| Weak labels | PARTIAL | Có rule + contract; **không phải ground truth** |
| Frozen split | DONE | `split-v0.1`, hash trong `split_report.json` |
| LSTM | DONE | 2 lớp, hidden 128, dropout 0.3 |
| TCN | DONE | Causal dilated conv, 4 level, 63.363 params |
| Training/eval | DONE | Metadata + config snapshot + predictions + gate reason |
| Uncertainty gate | DONE | 3 lý do: visibility/conf/model-conf |
| Human annotation | BLOCKED | Chưa có người gán; template đã sẵn sàng |
| Dashboard/export | DONE | 6 tab (gồm upload → pipeline) + 5 artifact export |
| Zone 3×3 | PARTIAL | Gán hình học; chưa có F1/bias theo zone |
| Benchmark | DONE | Pose 4,1–4,3 FPS; LSTM/TCN latency; E2E 6,26 FPS, 95,8 s/video-min |

---

## 4. KẾT QUẢ ĐỊNH LƯỢNG (WEAK-LABEL AGREEMENT)

**Experiment chính thức hiện tại (dataset full-pose):**

```text
experiment_id : 20261025-162855-lstm-full-v0-1
model         : LSTM (2 layers, hidden 128, dropout 0.30)
seed          : 42
split         : split-v0.1
metric_scope  : weak_rule_labels
test windows  : 7.467 total / 6.518 evaluated / 949 unlabeled
```

| Chỉ số | Raw | Sau uncertainty gate (accepted) |
|---|---:|---:|
| Macro-F1 | **0.9098** | **0.9151** |
| Balanced Accuracy | 0.9107 | 0.9162 |
| Accuracy | 0.9228 | 0.9281 |
| Coverage | 100% | **98.54%** (6.423/6.518) |
| Uncertain rate | 0% | **1.46%** (95/6.518) |

**Confusion matrix (raw, 6.518 window):**

```text
                HIGH   MODERATE   LOW
HIGH             3302      130       6
MODERATE          172     1907     103
LOW                 0       92     806
```

**Gate reason:** `ACCEPTED=6423`, `LOW_MODEL_CONF=95`.

**So sánh LSTM vs TCN (cùng split, cùng 6.518 window):**

| Chỉ số | LSTM | TCN |
|---|---:|---:|
| Experiment ID | `20261025-162855…` | `20261025-163801…` |
| Raw Macro-F1 | **0.9098** | 0.9027 |
| Accepted Macro-F1 | 0.9151 | **0.9198** |
| Accepted Accuracy | 0.9281 | **0.9332** |
| Coverage | **98.54%** | 95.95% |
| Tham số | 246.531 | 63.363 |
| CPU ms/122 window | 31.74 | **17.05** |

Chi tiết: `reports/tcn_vs_lstm.md`. TCN accepted nhỉnh hơn rất nhẹ nhưng coverage thấp hơn;
raw Macro-F1 LSTM cao hơn. Không kết luận “model nào tốt hơn” nếu bỏ qua coverage.

**Diễn giải đúng:**
- Đây là mức độ mô hình học lại **luật yếu** đã sinh nhãn, không phải bằng chứng mô hình
  nhận diện đúng hành vi người thật.
- **Pose đã full:** `session_001–005` 900 s, `session_006` 250 s; dataset 32.959 window.
  Metric giảm so với bản 15 s cũ (0.94 → 0.91) vì dữ liệu full khó hơn, không phải model kém đi.
- Không được trích dẫn “độ chính xác 91%” như kết luận về người học.
- Chỉ sau khi có human test set (A7) mới được báo cáo accuracy/F1 trên người thật.

---

## 5. CÁC VẤN ĐỀ ĐÃ SỬA TRONG ĐỢT NÀY

1. **Bỏ nhãn HUMAN giả:** `scripts/generate_annotations.py` chỉ sinh `WEAK_RULE`;
   không còn tạo “double annotator” giả. File cũ chuyển vào
   `data/annotations/deprecated/` kèm giải thích.
2. **Đóng băng split:** `scripts/build_dataset.py` đọc `data/splits/*.txt`, kiểm tra
   overlap/missing session, ghi hash; không tự chia lại.
3. **Evaluation theo `window_id`:** lưu `predictions.csv`, kiểm tra label tensor khớp
   metadata, báo lỗi thay vì lệch im lặng.
4. **Gate reason:** thêm `LOW_VISIBILITY`, `LOW_KEYPOINT_CONF`, `LOW_MODEL_CONF`.
5. **Experiment metadata:** mỗi run có experiment ID, config snapshot, SHA256, seed,
   git commit, dataset/split version, checkpoint hash.
6. **Đồng bộ tài liệu:** README/AGENTS/IMPLEMENTATION_PLAN trỏ về `PROJECT_STATUS.md`;
   thêm `docs/LIMITATIONS.md`.
7. **Feature v2:** thêm 17 mask khuyết điểm (`posefeat-v2`, 94 dims), giữ v1 cho checkpoint cũ.
8. **Zone 3×3:** `scripts/assign_zones.py` + `configs/zones/grid_3x3.yaml`; zone là gán hình học,
   không phải sơ đồ chỗ ngồi.
9. **Reporting/export:** `src/reporting/aggregation.py`, `exporter.py`; 5 artifact
   `results/timeline/zone_summary/summary.json/summary_report.md`.
10. **Dashboard:** `app/streamlit_app.py` 6 tab (gồm upload → `predict_video.py`); AppTest 0 exception.
11. **TCN:** `src/models/tcn.py` + `configs/experiments/tcn_baseline.yaml` + benchmark.
12. **Full pose + rebuild:** chạy lại 5 session (38.505 frame, 559.499 detection), rebuild
    dataset 32.959 window, retrain LSTM/TCN trên dữ liệu full.

---

## 6. ĐỐI SOÁT VỚI MỤC TIÊU ĐỀ TÀI

| Mục tiêu | Trạng thái thực tế | Đánh giá |
|---|---|---|
| Baseline: video → pose → tracking → feature → classifier → demo | Pipeline + LSTM + dashboard + upload inference | 95% |
| Extended 1: dataset + customized temporal model + eval | Có dataset/window/LSTM/TCN; **nhãn người + M0/M1 chưa có** | 60% |
| Extended 2: full flow + TCN comparison + failure analysis | Có full flow + LSTM vs TCN + benchmark; thiếu benchmark video dài/GPU | 70% |
| Extended 3: zone aggregation + uncertainty + anonymization | Uncertainty + zone 3×3 + dashboard/export; chưa có F1/bias zone, blur mặc định | 70% |

Không dùng các con số trên như “đạt 100%” cho bất kỳ mục nào.

---

## 7. KẾ HOẠCH TIẾP THEO

1. **A6/A7 — nhãn người thật:** template đã có tại `data/annotations/human/`;
   hai annotator gán blind, chạy `dataset_qa.py --strict-human`, đóng băng `test-human-v1`.
2. **T39 — benchmark end-to-end** trên video dài/GPU (upload flow đã có).
3. **T20/T21 + ablation + calibration:** M0/M1, representation/window/motion, ECE.
4. **A9:** cập nhật lại báo cáo này bằng số human khi có.

---

## 8. KẾT LUẬN

Nhóm đã hoàn thành được pipeline kỹ thuật cốt lõi và biết chính xác giới hạn của nó.
Điểm mạnh hiện tại là **tính tái lập và contract dữ liệu**; điểm yếu lớn nhất là
**chưa có ground truth người thật**. Báo cáo này cố ý không tuyên bố độ chính xác hành vi
khi chưa có bằng chứng tương ứng.


> Cập nhật 09/11/2026: xem docs/HUMAN_REVIEW_RUNBOOK.md và docs/PROJECT_STATUS.md. Nhãn phase cũ sinh tự động chưa được xác minh; dataset chờ review mới có 27.116 window. Không dùng số liệu baseline lịch sử như metric trên nhãn người.
