# LIMITATIONS — CVF25PRJ10

> Mọi báo cáo/demo phải đọc mục này trước khi trích dẫn số liệu.  
> Cập nhật: 2026-10-25.

## 1. Giới hạn phạm vi (không thương lượng)

Hệ thống chỉ ước lượng **mức độ tham gia học tập quan sát được** từ tư thế/chuyển động.
Không được diễn giải đầu ra thành:

- cảm xúc (vui/buồn/chán),
- động cơ, ý định, mức độ chú ý nội tại,
- mức độ hiểu bài, thông minh, nhân cách,
- danh tính người học.

`LOW` **không** có nghĩa là “lười”, “mất tập trung” hay “không có động lực”.
`UNCERTAIN` là trạng thái thiếu căn cứ của hệ thống, không phải lớp hành vi thứ tư.

## 2. Giới hạn nhãn — quan trọng nhất

- Toàn bộ window hiện tại được gán bằng **weak rule** (`scripts/generate_annotations.py`,
  `src/preprocessing/weak_labels.py`), không phải nhãn người.
- Chưa có human test set. Vì vậy các con số trong `reports/experiments/*` là
  **weak-label agreement**, không phải accuracy trên hành vi người thật.
- Các file `pilot_double_annotator_*.csv` cũ là synthetic copy; Kappa 1.0 cũ **không hợp lệ**.
  Chúng đã được chuyển vào `data/annotations/deprecated/`.
- Không được dùng weak label làm ground truth test chính thức.
- Quy trình nhãn người: `data/annotations/human/README.md`, `docs/annotation_guide.md`,
  `scripts/sample_annotation_subset.py`, `scripts/dataset_qa.py --strict-human`.

## 3. Giới hạn dữ liệu

- 11 video (`class_1..5`, `session_001..006`), chủ yếu một góc máy, một buổi/loop.
- `session_001` có 8 người trong metadata, thấp hơn target ≥15 người.
- Chưa chứng minh đủ ≥90 phút usable, đủ 3 phase (`lecture`, `individual_work`, `discussion`).
- **Pose đã phủ full video** (cập nhật 2026-10-25): 5 session dài được xử lý lại
  toàn bộ; `class_1..5` vốn là video ngắn 12–18 s.

| Video | Thời lượng | Pose | Frame sampled |
|---|---:|---:|---:|
| `session_001` (train) | 900 s | 899.9 s | 9.000 |
| `session_002` (train) | 900 s | 899.9 s | 9.000 |
| `session_003` (val) | 900 s | 899.9 s | 9.000 |
| `session_004` (val) | 900 s | 899.9 s | 9.000 |
| `session_005` (test) | 900 s | 899.9 s | 9.000 |
| `session_006` (test) | 250.5 s | 250.4 s | 2.505 |

- Dataset full sau rebuild: **32.959 window** — train 13.481 / val 12.011 / test 7.467.
  Test có 7.467 window (949 không nhãn), LOW 898, MODERATE 2.182, HIGH 3.438.
- Val/test gần bằng train về số window; không còn tình trạng 94% train đến từ
  `session_001`. Tuy nhiên class imbalance vẫn rõ (LOW ít hơn HIGH).
- **Metric hiện hành trên dataset full:** LSTM `20261025-162855…` raw 0.9098 / accepted
  0.9151 / coverage 98.54%; TCN `20261025-163801…` raw 0.9027 / accepted 0.9198 /
  coverage 95.95%. Các experiment 15 s cũ vẫn còn trong repo nhưng **không dùng báo cáo**.
- `zone_id` đã có trong `data/features` (gán hình học 3×3), không phải sơ đồ chỗ ngồi
  đã xác minh; mọi bảng zone phải kèm `N`.

## 4. Giới hạn mô hình/đặc trưng

- Pose 2D từ YOLO; không có gaze estimator thật. `head_orientation_proxy` chỉ là
  hình học mũi–vai, không phải hướng nhìn chính xác.
- `posefeat-v1` zero-fill keypoint thiếu (`np.nan_to_num`), làm mất mask ở tầng feature.
  `visible_ratio`/`mean_keypoint_conf` chỉ phản ánh một phần chất lượng.
- Đã có LSTM và TCN; chưa có ablation representation/window/motion, M0/M1, calibration.
- Ngưỡng quality (`visible_ratio=0.60`, `conf=0.30`) và uncertainty (`max_prob=0.60`)
  là baseline ban đầu, chưa được tune trên validation một cách hệ thống.
- Chưa có calibration/reliability (ECE, temperature scaling).

## 5. Giới hạn đánh giá

- Metric chính: Macro-F1, balanced accuracy, per-class P/R/F1, confusion matrix.
- Với uncertainty: coverage, accepted performance, uncertain rate.
- Chưa có:
  - human-labeled test,
  - inter-annotator Kappa thật,
  - per-zone F1/bias analysis,
  - tracker ID switch/fragmentation có ground truth.
- Benchmark hiện tại là **single-run CPU** (`reports/benchmark.md`): pose 4,1–4,3 FPS
  trên 60 frame; LSTM full 0,260 ms/window, TCN full 0,140 ms/window (best-of-3, 100 reps);
  end-to-end 6,26 FPS / 95,8 s mỗi phút video trên `class_1` (150 frame sampled).
  Chưa có GPU/VRAM và chưa đo video dài.
- Không được chọn threshold/window trên test set.

## 6. Giới hạn triển khai/demo

- Dashboard (`app/streamlit_app.py`) có 6 tab, gồm upload → `predict_video.py` → kết quả;
  pipeline chạy CPU, video dài sẽ chậm và chưa có progress chi tiết.
- End-to-end mới smoke trên video ngắn; chưa benchmark full-flow trên video dài/GPU.
- Zone 3×3 là **gán hình học**, không phải sơ đồ chỗ ngồi đã xác minh; mọi bảng phải kèm `N`.
- Chưa bật head-blur/pixelation mặc định cho mọi demo; overlay skeleton + anonymous track ID
  là biện pháp ẩn danh hiện có (`apply_face_blur` có sẵn nhưng không dùng mặc định).
- Raw video người thật nằm local, không được commit public.

## 7. Cách trích dẫn số liệu đúng

Khi báo cáo kết quả LSTM hiện tại, phải ghi kèm:

```text
Experiment ID: 20261025-162855-lstm-full-v0-1 (LSTM)
             : 20261025-163801-tcn-full-v0-1 (TCN)
metric_scope: weak_rule_labels
test windows: 7.467 total / 6.518 evaluated
split: split-v0.1 (test = session_005 + session_006)
```

Không viết “độ chính xác 91%” như thể đó là độ chính xác trên người thật.
Cách viết đúng: “weak-label agreement đạt Macro-F1 0.915 (accepted) trên
6.423/6.518 cửa sổ test; chưa có human ground truth để kết luận độ chính xác hành vi.”
