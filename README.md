# CVF25PRJ10 — Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được

**Mã đề tài:** CVFP26BCPPRJC012  
**Nhóm:** CVF25PRJ10

## Thành viên

| STT | Mã sinh viên | Họ và tên |
|---:|---|---|
| 1 | B23DCPT250 | Vũ Văn Phong |
| 2 | B23DCPT262 | Nguyễn Hữu Quân |
| 3 | B23DCPT277 | Vũ Đức Tài |
| 4 | B23DCPT306 | Nguyễn Dũng Tiến |

## 1. Mục tiêu

Xây dựng hệ thống phân tích chuỗi video lớp học để **ước lượng mức độ tham gia học tập quan sát được** từ tư thế cơ thể và diễn biến theo thời gian.

Hệ thống **không** được diễn giải đầu ra như phép đo trực tiếp về cảm xúc, động cơ, ý định hay trạng thái chú ý nội tại của người học.

## 2. Input / Output chính thức

### Input
- Video lớp học hoặc video demo được phép sử dụng.
- Khuyến nghị: 1080p, 25–30 FPS; hệ thống có thể sampling xuống 5–10 FPS để xử lý.
- Không yêu cầu âm thanh trong baseline.

### Output ở mức hệ thống
- `HIGH_OBSERVED_PARTICIPATION`
- `MODERATE_OBSERVED_PARTICIPATION`
- `LOW_OBSERVED_PARTICIPATION`
- `UNCERTAIN`

> Lưu ý: dataset/model lưu 3 canonical label đầy đủ ở trên. `HIGH/MODERATE/LOW` chỉ là display label. `UNCERTAIN` được tạo bởi quality/confidence gate, không phải training class. Mapping chính thức nằm tại `src/contracts/labels.py`.

## 3. Pipeline kỹ thuật

```text
Video / Camera
    ↓
Frame Sampling
    ↓
YOLOv8-Pose
    ↓
ByteTrack / BoT-SORT
    ↓
Pose Quality Gate
    ↓
Keypoint Cleaning + Normalization
    ↓
Pose / Motion Features
    ↓
Sliding Window
    ↓
Customized LSTM / TCN
    ↓
Probability
    ↓
Uncertainty Gate
    ↓
HIGH / MODERATE / LOW / UNCERTAIN
    ↓
Aggregation theo thời gian + vùng lớp
    ↓
Streamlit Dashboard + Export Report
```

## 4. Thuật ngữ bắt buộc

Dùng:
- **mức độ tham gia học tập quan sát được**
- **observable participation**
- **pose features / motion features**
- **head-orientation proxy** nếu dùng đặc trưng hướng đầu gần đúng
- **anonymous track ID** cho ID theo dõi trong video

Không dùng như kết luận của hệ thống:
- “sinh viên lười”
- “buồn chán”
- “mất tập trung”
- “không có động lực”
- “đang nhìn chính xác vào …” nếu chỉ có pose 2D

Chi tiết xem: [`docs/scope.md`](docs/scope.md).

## 5. Customized model

Trong đề tài này:
- **YOLOv8-Pose**: bộ trích xuất pose pretrained.
- **ByteTrack/BoT-SORT**: tracker.
- **LSTM/TCN**: **customized participation classifier** do nhóm thiết kế và huấn luyện trên dataset của đề tài.

## 6. Cấu trúc repository

```text
CVF25PRJ10/
├── AGENTS.md
├── configs/
│   ├── baseline.yaml
│   └── experiments/
├── data/
│   ├── raw/
│   ├── sample/
│   ├── annotations/
│   ├── poses/
│   ├── features/
│   └── splits/
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   ├── data_collection_protocol.md
│   ├── annotation_guide.md
│   ├── dataset_schema.md
│   └── privacy_and_consent.md
├── notebooks/
├── src/
│   ├── pose/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── uncertainty/
│   └── reporting/
├── models/checkpoints/
├── app/
├── scripts/
├── reports/experiments/
├── tests/
├── .github/workflows/ci.yml
├── .gitignore
└── README.md
```

## 7. Experiment reproducibility

Mỗi lần train/đánh giá chính thức phải có:
- `experiment_id` duy nhất.
- Config snapshot.
- Random seed.
- Git commit hiện tại.
- Dataset version và `split_version`.
- Metrics và artifact riêng.

Tạo experiment mới:

```bash
python scripts/new_experiment.py --name lstm-baseline --seed 42 --dataset-version v0.1 --split-version split-v0.1
```

Kiểm tra repo skeleton:

```bash
python scripts/check_setup.py
```

## 8. Quy tắc dữ liệu

- Không commit raw video có người thật vào repository public.
- Không lưu mapping `track_id ↔ tên/MSSV` trong dataset phân tích nếu không cần thiết.
- Train/validation/test phải tách theo participant hoặc session; không random frame/window cùng video sang nhiều split.
- Demo công khai ưu tiên skeleton/anonymous ID hoặc làm mờ vùng đầu.

## 9. Repository readiness additions

Repo đã có thêm các contract/template để bắt đầu T03+ mà không thay đổi kiến trúc baseline:

- Data collection: [`docs/data_collection_protocol.md`](docs/data_collection_protocol.md)
- Annotation: [`docs/annotation_guide.md`](docs/annotation_guide.md)
- Dataset schema: [`docs/dataset_schema.md`](docs/dataset_schema.md)
- Privacy/consent: [`docs/privacy_and_consent.md`](docs/privacy_and_consent.md)
- Sample metadata/annotation/windows: [`data/sample/`](data/sample/)
- Colab orchestration templates: [`notebooks/`](notebooks/)
- Model/checkpoint storage rules: [`models/README.md`](models/README.md)
- Contract tests: [`tests/`](tests/)
- GitHub CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

> Các file T03/T04 hiện là **draft v0.1 cần team review**. Có file không đồng nghĩa task đã Done.

## 10. Trạng thái triển khai (cập nhật 2026-10-25)

> Nguồn sự thật duy nhất: [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).  
> Hạn chế khoa học: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

**Đã có code + artifact:**
- Pose extraction (YOLOv8-Pose), ByteTrack, quality gate, cleaning/normalization.
- 77-dim features (`posefeat-v1`) + 94-dim mask (`posefeat-v2`), sliding window 4s/2s.
- Weak-label rule engine + contract validation; human annotation template + strict QA.
- LSTM + TCN, training/evaluation pipeline, uncertainty gate, per-window predictions.
- Frozen session split `split-v0.1`, experiment metadata + config snapshot + hash.
- Reporting/export (`results.csv`, `timeline.csv`, `zone_summary.csv`, `summary.json`).
- Geometric 3×3 zone assignment (`scripts/assign_zones.py`).
- Streamlit dashboard: overview, timeline, zone heatmap, anonymized preview, export, upload → full inference.
- Benchmark script + `reports/benchmark.md`, `reports/tcn_vs_lstm.md`.

**Chưa có / còn thiếu:**
- **Nhãn người thật cho test set** — mọi metric hiện là *weak-label agreement*, không phải accuracy trên hành vi người thật.
- End-to-end benchmark trên video dài/GPU (mới smoke trên video ngắn CPU).
- Ablation representation/window/motion, calibration/reliability.
- ID switch/fragmentation tracker có ground truth.

## 11. Tài liệu triển khai & AI Agent context

- **Kế hoạch triển khai chi tiết:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — 10 tuần, 41 task, owner, dependency, artifact và Definition of Done.
- **Kế hoạch chỉnh sửa & cập nhật:** [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md).
- **Hướng dẫn tiếp theo (nhãn người, benchmark dài/GPU, M0/M1, ablation, calibration):** [`docs/NEXT_STEPS_GUIDE.md`](docs/NEXT_STEPS_GUIDE.md).
- **AI Agent / onboarding guide:** [`AGENTS.md`](AGENTS.md) — context dự án, kiến trúc, quy tắc dữ liệu/model, cách AI Agent và thành viên mới thực hiện task mà không sửa code base lan man.

**Trạng thái hiện tại:** xem bảng 41 task trong [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md). Ưu tiên tiếp theo: nhãn người thật cho test set, calibration, M0/M1 và ablation.
