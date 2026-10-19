# CVF25PRJ10 — Implementation Plan

> **Project:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được  
> **Task code:** CVFP26BCPPRJC012  
> **Group:** CVF25PRJ10  
> **Plan version:** 1.0  
> **Status baseline:** T01, T02 completed; implementation of pose/tracking/model starts from the next tasks.

---

## 1. Mục đích của tài liệu

Tài liệu này là kế hoạch triển khai kỹ thuật chính của dự án. Mọi thành viên và AI Agent phải dùng tài liệu này để biết:

- Việc nào cần làm trước/sau.
- Ai là owner chính và ai hỗ trợ.
- Artifact nào phải được tạo ra.
- Điều kiện nào được coi là hoàn thành (Definition of Done — DoD).
- Task nào thuộc Baseline, Extended Level 1, 2 hoặc 3.
- Không tự ý mở rộng scope nếu chưa có task/chấp thuận của nhóm.

Khi task thay đổi, cập nhật file này cùng commit thay đổi.

---

## 2. Thành viên và owner chính

| Thành viên | MSSV | Owner chính | Phối hợp |
|---|---|---|---|
| Vũ Văn Phong | B23DCPT250 | Data protocol, collection, annotation QA | Weak supervision, dataset quality, zone metadata |
| Nguyễn Hữu Quân | B23DCPT262 | YOLOv8-Pose, tracking, preprocessing, integration | Baseline, feature pipeline, performance/failure analysis |
| Vũ Đức Tài | B23DCPT277 | LSTM/TCN, training, evaluation | Ablation, uncertainty, calibration |
| Nguyễn Dũng Tiến | B23DCPT306 | Streamlit, anonymization, reporting | Bias dashboard, export, demo/package |

**Quy tắc:** owner chịu trách nhiệm đưa task tới DoD; người hỗ trợ review/chạy thử nhưng không làm thay trách nhiệm owner.

---

## 3. Các mốc yêu cầu của đề tài

### Baseline
Video → pose → tracking → feature/basic classifier → demo Streamlit cơ bản.

### Extended Level 1
Dataset đầy đủ + customized temporal model + đánh giá so sánh.

### Extended Level 2
Full flow + LSTM/TCN comparison/ablation + demo thực tế + failure/performance analysis.

### Extended Level 3
Aggregation theo thời gian/vùng lớp + uncertainty + anonymization + đánh giá sai lệch theo vị trí quan sát.

---

## 4. Quy trình kỹ thuật mục tiêu

```text
Video / Camera
    ↓
Video Validation + Frame Sampling
    ↓
YOLOv8-Pose
    ↓
ByteTrack (BoT-SORT optional comparison)
    ↓
Pose Quality Gate
    ↓
Keypoint Cleaning + Normalization
    ↓
Pose / Motion Feature Extraction
    ↓
Sliding Window
    ↓
Customized LSTM / TCN
    ↓
Canonical 3-class probabilities (see `src/contracts/labels.py`)
    ↓
Uncertainty Gate
    ↓
HIGH / MODERATE / LOW / UNCERTAIN
    ↓
Time + Classroom Zone Aggregation
    ↓
Anonymized Streamlit Dashboard
    ↓
CSV / JSON / Summary Report
```

---

## 5. Kế hoạch 10 tuần — 41 task

### Tuần 1 — Scope, repository, data protocol

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T01** | Quân | Cả nhóm | Foundation | Chốt phạm vi & thuật ngữ | Chốt input/output, train label vs system output, giới hạn suy luận, terminology | `README.md`, `docs/scope.md` | README/scope thống nhất canonical 3-class labels + display mapping + UNCERTAIN gate; không emotion/intention claim | — |
| **T02** | Quân | Cả nhóm | Foundation | Repo skeleton & experiment log | Tạo cấu trúc source/config/data/report; baseline config; experiment ID/seed/git commit/dataset version | repo skeleton, `baseline.yaml`, `experiment_protocol.md`, scripts | `python scripts/check_setup.py` pass; tạo experiment smoke-test được | T01 |
| **T03** | Phong | Tiến | Foundation | Protocol thu thập & quyền riêng tư | Viết protocol quay, đồng thuận, lưu trữ raw video, policy public demo | `docs/data_collection_protocol.md` | Có checklist trước quay; nêu ai truy cập dữ liệu, cách ẩn danh, không public raw video | T01 |
| **T04** | Phong | Tài | Foundation | Annotation guide v0.1 | Định nghĩa atomic behaviors, 3 participation labels, uncertain/occluded/transition, ví dụ dương/âm | `docs/annotation_guide.md` | Hai thành viên đọc và có thể gán cùng quy tắc trên pilot | T01 |

### Tuần 2 — Pilot, pose và tracking baseline

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T05** | Phong | Quân | Baseline | Kịch bản pilot & bố trí camera | Pilot 10–15 phút, 6–10 người, lecture/individual/discussion; bố trí zone 3×3 | pilot plan + metadata | Có camera layout, phase timeline, participant consent và metadata template | T03,T04 |
| **T06** | Quân | Phong | Baseline | YOLOv8-Pose extractor | Đọc video, sampling FPS, chạy pose, lưu bbox + 17 keypoints + confidence | `src/pose/extractor.py`, `scripts/extract_pose.py` | Chạy được video pilot, output đúng schema, frame/timestamp khớp | T02,T05 |
| **T07** | Quân | Tiến | Baseline | ByteTrack integration | Gắn track ID theo người, lưu track length, preview ID | `src/pose/tracker.py` | Cùng người giữ ID tương đối ổn định trên đoạn pilot đã chọn | T06 |
| **T08** | Quân | Phong | Baseline | Pilot pose/tracking quality review | Đo visible ratio, mean confidence, miss detection, ID switch/fragment | `reports/pilot_quality.md/csv` | Có quyết định camera đạt/chưa đạt trước thu dataset chính | T06,T07 |
| **T09** | Phong | Quân | Foundation | Metadata & data-layout validation | Chuẩn hóa session/camera/phase/zone metadata và folder convention | `data/README.md`, metadata schema | Một session pilot được lưu đúng cấu trúc và validate không thiếu field bắt buộc | T05 |

### Tuần 3 — Dataset chính và preprocessing

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T10** | Phong | Cả nhóm | Extended 1 | Thu dataset chính | Thu nhiều session/ngày/vị trí; đổi vị trí người để giảm confounding person-zone | raw sessions + metadata | Đạt target tối thiểu đã chốt trong Dataset Acceptance Criteria | T08,T09 |
| **T11** | Phong | Tiến | Extended 1 | Annotation template/tooling | Chuẩn CSV/schema hoặc notebook/tool hỗ trợ gán window + behavior + phase + zone | annotation template/tool | Annotator nhập được dữ liệu thống nhất, validate enum/time range | T04,T09 |
| **T12** | Phong | Tài | Extended 1 | Double annotation pilot | Hai annotator gán độc lập ≥15–20% subset; xử lý disagreement | agreement report | Báo Cohen’s kappa + cập nhật guideline; không sửa test label theo model result | T11 |
| **T13** | Quân | Tài | Baseline | Pose Quality Gate | visible ratio, mean keypoint confidence, minimum track duration | `src/preprocessing/...` | Window/track quality thấp bị đánh dấu invalid/uncertain trước classifier | T06,T07 |
| **T14** | Quân | Tài | Baseline | Keypoint cleaning & normalization | Missing mask, short interpolation, smoothing nhẹ, bbox/body-scale normalization | cleaner module + tests | Không dùng `(0,0)` như keypoint thật; normalized feature ổn định theo scale | T13 |

### Tuần 4 — Features, weak supervision, windows, splits

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T15** | Quân | Tài | Extended 1 | Pose/motion feature engineering | normalized keypoints, torso/shoulder/arm geometry, motion/velocity, stability, visibility | `src/preprocessing/features.py` | Feature schema versioned; NaN/missing được xử lý có mask | T14 |
| **T16** | Phong | Tài | Extended 1 | Weak supervision | Viết labeling functions dựa trên observable behavior + phase để gợi ý weak labels | `src/.../weak_labels.py`, docs | Weak label luôn phân biệt với human ground truth; dùng để hỗ trợ annotation/train experiment riêng | T04,T15 |
| **T17** | Quân | Tài | Extended 1 | Sliding-window builder | Build 2/4/6s windows, default 4s stride 2s, align label + quality | `src/preprocessing/windows.py` | Window timestamps không overlap sai track; output reproducible | T15 |
| **T18** | Tài | Phong | Extended 1 | Group train/val/test split | Split theo participant/session; cố định random seed/split version | split files | Không session/participant leakage; test split đóng băng | T10,T12,T17 |
| **T19** | Phong | Tài | Extended 1 | Dataset acceptance & versioning | Thống kê duration, participants, sessions, class/zone counts, missing metadata, annotation agreement | `reports/data_quality.csv`, dataset version | Dataset đạt tiêu chí hoặc có documented exception trước train chính thức | T10,T12,T18 |

### Tuần 5 — Baselines và LSTM

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T20** | Quân | Tài | Baseline | Rule-based baseline M0 | Heuristic observable rules từ features/context; không dùng emotion inference | baseline code + metrics | Có Macro-F1/per-class metrics trên fixed split | T17,T18 |
| **T21** | Tài | Quân | Extended 1 | ML baseline M1 | Logistic Regression/MLP trên aggregate feature, không temporal order | model + experiment | Có config, metrics, confusion matrix | T17,T18 |
| **T22** | Tài | Quân | Extended 1 | Customized LSTM M2 | LSTM 3-class, configurable hidden/layers/dropout | `src/models/lstm.py` | Train/val chạy end-to-end, checkpoint và config snapshot | T17,T18 |
| **T23** | Tài | Quân | Extended 1 | Training/evaluation pipeline | Dataset loader, weighted loss optional, early stopping, seed, checkpoint, metrics | `src/training/*` | Tạo experiment artifact đầy đủ, reproduce được bằng command/config | T18,T22 |
| **T24** | Tài | Phong | Extended 1 | Initial comparison | M0/M1/M2, confusion matrix, per-class recall/F1, Macro-F1 | results table/report | Có bảng so sánh trên cùng fixed test split | T20,T21,T23 |

### Tuần 6 — TCN và ablation

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T25** | Tài | Quân | Extended 1 | Customized TCN M3 | Causal/temporal convolution blocks, same input/features/split as LSTM | `src/models/tcn.py` | Train/evaluate được bằng cùng training pipeline | T23 |
| **T26** | Tài | Quân | Extended 2 | Representation ablation | normalized keypoints vs engineered features vs combined | experiment set | Chỉ thay representation, giữ split/eval protocol cố định | T24,T25 |
| **T27** | Quân | Tài | Extended 2 | Motion ablation | So sánh không velocity/motion và có dynamic cues | experiment/config | Có kết luận dựa metric, không dựa cảm quan | T15,T23 |
| **T28** | Tài | Quân | Extended 2 | Temporal-window ablation | 2s / 4s / 6s và có thể FPS 5/10 nếu đủ thời gian | experiment set | Chọn default window từ validation, không tune bằng test | T17,T23 |
| **T29** | Tài | Phong | Extended 2 | Class imbalance strategy | class weights/sampler; theo dõi class counts, Macro-F1 | experiment note | So sánh ít nhất no-weight vs selected strategy nếu imbalance đáng kể | T19,T23 |

### Tuần 7 — Uncertainty, bias, anonymization, tracker comparison

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T30** | Tài | Quân | Extended 3 | Uncertainty gate | Input quality gate + max probability threshold; 3-class train → 4-state system output | `src/uncertainty/gate.py` | Có coverage vs accepted-prediction performance curve | T23,T25 |
| **T31** | Tài | Tiến | Extended 3 | Calibration/reliability analysis | Tune threshold trên validation; optional temperature scaling/ECE | calibration report | Threshold cuối không chọn từ test set; report coverage/reliability trade-off | T30 |
| **T32** | Tiến | Tài, Phong | Extended 3 | Zone/bias aggregation | Front/Middle/Back × Left/Center/Right; report N, visible ratio, F1, uncertain rate | `src/reporting/bias_analysis.py` | Mọi zone table có sample count; không kết luận bias từ nhóm quá nhỏ mà không cảnh báo | T19,T30 |
| **T33** | Tiến | Quân | Extended 3 | Anonymization | Anonymous track ID; blur/pixelation head ROI hoặc skeleton-only mode | anonymization module | Demo mode không lộ danh tính; không có name/MSSV mapping | T07 |
| **T34** | Quân | Tiến | Extended 2 | ByteTrack vs BoT-SORT | Đo ID switches/min, fragmentation, FPS trên subset | tracker comparison | Có lý do chọn tracker final dựa metric/complexity | T07,T08 |

### Tuần 8 — Streamlit full flow

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T35** | Tiến | Quân | Extended 2 | Upload/config/processing UI | Upload video, choose model/config, processing state/error handling | `app/streamlit_app.py` modules | Video hợp lệ chạy pipeline; lỗi input có thông báo rõ | T23,T25,T30 |
| **T36** | Tiến | Quân | Extended 3 | Anonymized video preview | Skeleton/bbox + anonymous ID + system state/confidence | UI preview | Privacy mode bật mặc định cho demo có người thật | T33,T35 |
| **T37** | Tiến | Tài | Extended 3 | Timeline & classroom zone dashboard | Time-series distribution, uncertain rate, zone heatmap | dashboard components | Aggregation khớp output backend; hiển thị quality warning | T32,T35 |
| **T38** | Tiến | Phong,Tài | Extended 3 | Export result/report | Export `results.csv`, `timeline.csv`, `zone_summary.csv`, JSON/summary report | reporting/export module | File export có session/config/model version + limitations | T32,T37 |

### Tuần 9 — Integration, performance, failure analysis

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T39** | Quân | Cả nhóm | Extended 2 | Full-flow integration & performance | Video → pose → tracking → features → model → uncertainty → report; đo latency/FPS/RAM/VRAM; failure cases | integration/performance report | Chạy được ít nhất một video chưa dùng để debug pipeline; log failure modes và performance | T30,T34,T35,T38 |

### Tuần 10 — Demo, QA và bàn giao

| ID | Owner | Hỗ trợ | Level | Công việc | Nội dung thực hiện | Artifact | Definition of Done | Dependency |
|---|---|---|---|---|---|---|---|---|
| **T40** | Tiến | Cả nhóm | Extended 2/3 | Demo thực tế & final QA | Chạy kịch bản demo, test privacy/report/error path, capture evidence | demo checklist/video/screenshots | Demo lặp lại được theo README, không cần sửa tay source/config giữa chừng | T39 |
| **T41** | Tiến | Cả nhóm | Delivery | Final package | Technical report/NCKH, source, dataset metadata/annotations/splits, model/configs, logs, README reproduction guide | release package | Người khác clone/cài/chạy demo hoặc reproduce experiment chính theo hướng dẫn | T40 |

---

## 6. Dataset Acceptance Criteria

Các giá trị dưới đây là **target triển khai**, được phép điều chỉnh nếu có lý do và phải ghi vào data-quality report.

| Tiêu chí | Target |
|---|---:|
| Người tham gia | ≥ 15 (khuyến nghị 15–25) |
| Session | ≥ 6 |
| Tổng video usable | ≥ 90 phút |
| Pha hoạt động | lecture + individual work + discussion |
| Double annotation | ≥ 15–20% subset |
| Zone coverage | Front/Middle/Back và Left/Center/Right có mẫu |
| Metadata bắt buộc thiếu | < 5% |
| Train/test leakage | 0 |
| Annotation agreement | Phải báo Cohen’s kappa; nếu thấp phải sửa guideline trước train chính thức |

**Không ép số window từng class bằng cách tạo nhãn giả.** Báo class imbalance thực tế và xử lý bằng loss/sampling nếu cần.

---

## 7. Model/Evaluation Protocol

### Training classes

```text
HIGH_OBSERVED_PARTICIPATION
MODERATE_OBSERVED_PARTICIPATION
LOW_OBSERVED_PARTICIPATION
```

### System/API output

```text
Canonical: HIGH_OBSERVED_PARTICIPATION / MODERATE_OBSERVED_PARTICIPATION / LOW_OBSERVED_PARTICIPATION / UNCERTAIN
Display:   HIGH / MODERATE / LOW / UNCERTAIN
```

### Metric chính
- Macro-F1.
- Precision/Recall/F1 theo class.
- Balanced Accuracy.
- Confusion Matrix.
- Coverage khi bật uncertainty.
- Accepted-prediction Macro-F1/Accuracy.

### Tracking quality
- ID switches/minute trên annotated subset.
- Track fragmentation/person.
- Missed detection ratio.
- Processing FPS.

### Bias/quality stratification
Mỗi nhóm phải báo ít nhất:
- `N windows`.
- `visible_ratio`.
- `Macro-F1` hoặc per-class recall khi có ground truth.
- `UNCERTAIN rate`.

Không so sánh hai zone chỉ bằng F1 nếu kích thước mẫu chênh lệch lớn mà không ghi `N`.

---

## 8. System Performance Test

Đối với full flow phải lưu:

| Metric | Đơn vị |
|---|---|
| Pose inference FPS | frame/s |
| End-to-end FPS | frame/s |
| Processing time | seconds / minute video |
| Average inference latency | ms/window hoặc ms/frame |
| Peak RAM | MB/GB |
| Peak VRAM (nếu GPU) | MB/GB |
| Model checkpoint size | MB |

Thiết bị test, OS, CPU/GPU và config phải được ghi cùng kết quả.

---

## 9. Chuẩn artifact theo task

Mọi task kỹ thuật cần tối thiểu:

1. **Source/config/doc artifact** đúng vị trí.
2. **Cách chạy** bằng command hoặc README.
3. **Verification**: test, metric, screenshot/log hoặc output sample.
4. **Experiment ID** nếu có train/evaluation.
5. **Không sửa ngoài phạm vi task** trừ dependency bắt buộc và phải ghi lý do.

Task không được đánh dấu `Done` nếu chỉ “đã code” nhưng chưa chạy/kiểm tra.

---

## 10. Quy trình thực hiện một task

```text
1. Đọc AGENTS.md + scope + task này
2. Kiểm tra dependency đã Done
3. Tạo branch/task scope
4. Implement thay đổi nhỏ nhất cần thiết
5. Test / smoke test
6. Sinh artifact/metrics
7. Review với người hỗ trợ
8. Cập nhật plan/progress log
9. Commit rõ task ID
```

Commit message gợi ý:

```text
feat(T06): add YOLOv8 pose extraction pipeline
fix(T14): preserve missing keypoint mask during normalization
exp(T28): compare 2s 4s 6s temporal windows
docs(T04): refine annotation guide examples
```

---

## 11. Trạng thái hiện tại

### Hoàn thành
- **T01 — Scope & Terminology** ✅
- **T02 — Repo & Experiment Logging** ✅
- **T06 — YOLOv8-Pose extractor** ✅
- **T07 — ByteTrack integration & tracking QA** ✅
- **T13 — Pose Quality Gate** ✅
- **T14 — Keypoint cleaning & normalization** ✅
- **T15 — Pose/motion feature engineering** ✅
- **T16 — Weak supervision labeling functions** ✅
- **T17 — Sliding-window builder** ✅
- **T18 — Group train/val/test split** ✅
- **T22 — Customized LSTM M2** ✅
- **T23 — Training & evaluation pipeline** ✅
- **T30 — Uncertainty gate integration** ✅

### Task tiếp theo được ưu tiên
1. **T03 — Data collection/privacy protocol** — draft v0.1 đã có trong repo, cần owner/reviewer duyệt checklist trước khi Done.
2. **T04 — Annotation guide v0.1** — draft v0.1 + sample schema đã có, cần double-review trên pilot trước khi Done.
3. Sau khi T03/T04 ổn định mới thực hiện T05 pilot.

### Repository readiness đã bổ sung
- Dataset schema + sample metadata/annotation/window.
- Privacy/consent guide.
- Google Colab notebook templates (orchestration only).
- Model checkpoint storage rule.
- Contract tests + GitHub CI.

Các bổ sung này chuẩn bị hạ tầng, **không thay đổi trạng thái các task thuật toán chưa triển khai**.

Không bắt đầu train LSTM/TCN trước khi pilot camera, annotation guide, data schema và split strategy đủ tin cậy.

---

## 12. Change-control

Nếu muốn thay đổi một trong các mục sau, phải cập nhật `docs/scope.md`, config/doc liên quan và log lý do:

- Label semantics.
- Input/output pipeline.
- Dataset split.
- Pose/tracker/model chính.
- Uncertainty definition.
- Privacy/anonymization policy.
- Metric chính.

Không thay đổi test split hoặc label test chỉ vì kết quả model thấp.
