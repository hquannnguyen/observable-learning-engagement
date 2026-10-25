# PROJECT STATUS — CVF25PRJ10

> **Nguồn sự thật duy nhất về trạng thái task.**  
> Cập nhật lần cuối: 2026-10-25  
> Xác minh gần nhất: `python -m pytest -q --basetemp=D:\ThiGiacMayTinh\.pytest_tmp` → **103 passed**;  
> Streamlit `AppTest` → **0 exception**.  
> Không dùng báo cáo cũ làm bằng chứng; báo cáo đã được đính chính tại `reports/`.

## Quy ước trạng thái

| Trạng thái | Nghĩa |
|---|---|
| **DONE** | Code + artifact + test/verify; đạt DoD trong `docs/IMPLEMENTATION_PLAN.md` |
| **PARTIAL** | Có code/artifact nhưng chưa đạt DoD, còn caveat hoặc thiếu verify |
| **DRAFT** | Chỉ có tài liệu, chưa được duyệt/áp dụng |
| **BLOCKED** | Không thể tiếp tục nếu thiếu dữ liệu/đầu vào bên ngoài |
| **NOT STARTED** | Chưa có implementation thật |

## Bảng trạng thái 41 task

| ID | Task | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|---|
| T01 | Scope & terminology | **DONE** | `docs/scope.md`, `README.md`, `src/contracts/labels.py`; test `test_config.py` |
| T02 | Repo skeleton & experiment log | **DONE** | `configs/baseline.yaml`, `scripts/check_setup.py`, `scripts/new_experiment.py`; 101 test pass |
| T03 | Data collection/privacy protocol | **DRAFT** | `docs/data_collection_protocol.md`, `docs/privacy_and_consent.md`; chưa có reviewer duyệt |
| T04 | Annotation guide v0.1 | **PARTIAL** | `docs/annotation_guide.md`, `docs/cvat_labels_spec.json`; đã thêm quy trình human blind, chưa double-review thật |
| T05 | Pilot plan & camera layout | **PARTIAL** | 11 video + metadata 6 session; `session_001` 8 người (<15 target); chưa có pilot plan/layout |
| T06 | YOLOv8-Pose extractor | **DONE** | `src/pose/extractor.py`, `scripts/extract_pose.py`; 11 file pose; `test_pose_extractor.py` |
| T07 | ByteTrack integration | **DONE** | `src/pose/tracker.py`, `track_id` trong pose; `test_pose_cleaning.py` |
| T08 | Pilot pose/tracking quality review | **PARTIAL** | `reports/class_1_track_quality.csv` chỉ 10 frame/track; ID switch/fragmentation chưa có GT |
| T09 | Metadata & data-layout validation | **PARTIAL** | `data/raw/session_00*/metadata.json`; `class_1..5` chưa có metadata |
| T10 | Thu dataset chính | **PARTIAL** | 11 video thực tế; chưa chứng minh ≥90 phút usable, ≥15 người, đủ 3 phase |
| T11 | Annotation template/tooling | **PARTIAL** | `scripts/generate_annotations.py` (weak only) + `scripts/sample_annotation_subset.py` (human template) |
| T12 | Double annotation pilot | **BLOCKED** | Chưa có nhãn người thật; file synthetic cũ đã archive tại `data/annotations/deprecated/` |
| T13 | Pose Quality Gate | **DONE** | `src/preprocessing/quality_gate.py`; config-driven trong `windows.py`; test |
| T14 | Keypoint cleaning & normalization | **DONE** | `src/preprocessing/cleaner.py`; mask tường minh ở feature v2 |
| T15 | Pose/motion feature engineering | **DONE** | `posefeat-v1` (77) + `posefeat-v2` (94, có 17 mask); test `test_feature_v2_mask_marks_missing_joint` |
| T16 | Weak supervision | **PARTIAL** | `src/preprocessing/weak_labels.py` + contract; không phải ground truth |
| T17 | Sliding-window builder | **DONE** | `src/preprocessing/windows.py` (40 step/4s, stride 20); test |
| T18 | Group train/val/test split | **DONE** | `split-v0.1` đóng băng; builder đọc file split, validate leak; `split_report.json` có hash |
| T19 | Dataset acceptance & versioning | **PARTIAL** | 8.532 annotation, 32.959 window; chưa có acceptance report/Kappa thật |
| T20 | Rule-based baseline M0 | **NOT STARTED** | — |
| T21 | ML baseline M1 | **NOT STARTED** | — |
| T22 | Customized LSTM M2 | **DONE** | `src/models/lstm.py`; checkpoint + metrics |
| T23 | Training/evaluation pipeline | **DONE** | `scripts/train.py`, `scripts/evaluate.py`; experiment metadata, config snapshot, hash, predictions |
| T24 | Initial comparison M0/M1/M2 | **PARTIAL** | Có LSTM vs TCN (`reports/tcn_vs_lstm.md`), chưa có M0/M1 |
| T25 | Customized TCN M3 | **DONE** | `src/models/tcn.py`, `configs/experiments/tcn_baseline.yaml`, experiment `20261025-101644-tcn-weaklabel-v0-1` |
| T26 | Representation ablation | **NOT STARTED** | — |
| T27 | Motion ablation | **NOT STARTED** | — |
| T28 | Temporal-window ablation | **NOT STARTED** | — |
| T29 | Class imbalance strategy | **PARTIAL** | Class-weighted CE có; chưa có experiment no-weight vs weighted |
| T30 | Uncertainty gate | **DONE** | `src/uncertainty/gate.py`; reason `LOW_VISIBILITY/LOW_KEYPOINT_CONF/LOW_MODEL_CONF`; test |
| T31 | Calibration/reliability | **NOT STARTED** | — |
| T32 | Zone/bias aggregation | **PARTIAL** | `src/reporting/aggregation.py`, zone 3×3; chưa có F1/bias theo zone |
| T33 | Anonymization | **PARTIAL** | Skeleton overlay + anonymous track ID + `apply_face_blur`; chưa bật blur mặc định mọi demo |
| T34 | ByteTrack vs BoT-SORT | **PARTIAL** | `reports/benchmark.md`: FPS/track stats; ID switch/fragmentation N/A (không có GT) |
| T35 | Upload/config/processing UI | **DONE** | Tab “Video mới”: upload → `scripts/predict_video.py` → hiển thị kết quả; config checkpoint/max frames |
| T36 | Anonymized video preview | **DONE** | Tab “Preview ẩn danh” trong `app/streamlit_app.py`; `st.video` overlay |
| T37 | Timeline & zone dashboard | **DONE** | Timeline theo session/phút + heatmap 3×3 (có cảnh báo N) |
| T38 | Export result/report | **DONE** | `src/reporting/exporter.py`, `scripts/export_report.py`; 5 artifact |
| T39 | Full-flow integration & performance | **PARTIAL** | `predict_video.py` end-to-end; benchmark 4.95 FPS / 121 s/video-min (class_1, 150 frame); chưa video dài/GPU |
| T40 | Demo & final QA | **NOT STARTED** | — |
| T41 | Final package | **NOT STARTED** | — |

## Tổng hợp

| Nhóm | Số task |
|---|---:|
| DONE | 17 |
| PARTIAL | 14 |
| DRAFT | 1 |
| BLOCKED | 1 |
| NOT STARTED | 8 |

## Kết quả đã xác minh (weak-label)

| Hạng mục | Giá trị |
|---|---|
| Dataset | 11 video, 8.532 annotation weak, **32.959 window**, tensor `[32959, 40, 77]` (224,8 MB npz) |
| Split | train **13.481** / val **12.011** / test **7.467**; test 6.518 window có nhãn weak |
| **Pose coverage** | Full: `session_001–005` 900 s, `session_006` 250 s; `class_1..5` full video ngắn |
| LSTM full (`20261025-162855…`) | raw Macro-F1 **0.9098**, accepted **0.9151**, coverage **98.54%**; test 6.518 |
| TCN full (`20261025-163801…`) | raw **0.9027**, accepted **0.9198**, coverage **95.95%**; 63.363 params |
| Legacy 15 s (không dùng báo cáo) | LSTM 0.9408/0.9528, TCN 0.8726/0.8863; v2 0.9096/0.9226 |
| Benchmark CPU | pose 4,1–4,3 FPS (60 frame); LSTM full 0,260 ms/window, TCN full 0,140 ms/window; E2E 6,26 FPS / 95,8 s per video minute (class_1, 150 frame) |
| Dashboard | Streamlit AppTest 0 exception; 6 tab (gồm upload → pipeline) |

## Tuyên bố cần đính chính (đã sửa trong repo)

1. **Không có nhãn HUMAN thật**: `generate_annotations.py` chỉ sinh `WEAK_RULE`.
2. **Kappa 1.0 cũ không hợp lệ**: file double-annotator là bản copy; đã archive.
3. **Metric 0.90–0.92 là weak-label agreement**, không phải accuracy trên người thật.
4. **Split đang dùng**: test = `session_005` + `session_006` (7.467 window, 6.518 có nhãn).
5. **Dashboard/TCN/export/upload đã có**; human ground truth và benchmark video dài/GPU thì chưa.
