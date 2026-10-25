# KẾ HOẠCH CHỈNH SỬA & CẬP NHẬT — CVF25PRJ10

> **Trạng thái:** 🟡 IN PROGRESS — Track A/B và C1–C5 đã triển khai; A7 (human test set) BLOCKED vì chưa có nhãn người.  
> **Ngày lập:** 2026-10-25 — **Cập nhật:** 2026-10-25  
> **Căn cứ:** kết quả đọc repo ngày 2026-10-25; `docs/scope.md`, `AGENTS.md`, `docs/IMPLEMENTATION_PLAN.md`  
> **Nguyên tắc:** minimal-change, không tạo dữ liệu/nhãn giả, không tuyên bố kết quả chưa đo, giữ mọi thứ tái lập được.

### Tiến độ thực hiện

| Nhóm | Trạng thái | Bằng chứng |
|---|---|---|
| A1 status doc | ✅ DONE | `docs/PROJECT_STATUS.md` |
| A2 bỏ HUMAN giả | ✅ DONE | `generate_annotations.py`; test; `deprecated/` |
| A3 split đóng băng | ✅ DONE | `split_report.json`, `window_ids`, test |
| A4 evaluation theo window_id | ✅ DONE | `predictions.csv`, gate reason, test |
| A5 experiment metadata | ✅ DONE | `metadata.json`, config snapshot, hash |
| A6 human tooling | ✅ DONE | `sample_annotation_subset.py`, `--strict-human`, template 322 window |
| A7 human evaluation | ⛔ BLOCKED | Chưa có nhãn người thật |
| A8/A9 tài liệu & báo cáo | ✅ DONE | README/AGENTS/PLAN + 2 báo cáo |
| B1 feature mask v2 | ✅ DONE | `posefeat-v2` 94 dims; test |
| B2 config thresholds | ✅ DONE | `windows.py`; `quality_reason` |
| B3 test/CI | ✅ DONE | 103 test pass; CI cập nhật |
| B4 pin môi trường | ✅ DONE | `requirements.txt`, `docs/ENVIRONMENT.md` |
| C1 zone | ✅ DONE | `assign_zones.py`, 11 file zone, test |
| C2 aggregation/export | ✅ DONE | `src/reporting/*`, 5 artifact, test |
| C3 Streamlit | ✅ DONE | 6 tab, AppTest 0 exception; upload → `predict_video.py` |
| C4 TCN | ✅ DONE | `src/models/tcn.py`, experiment, `tcn_vs_lstm.md` |
| C5 benchmark | ✅ DONE | `reports/benchmark.md`: pose, model, end-to-end; ID switch N/A |

---

## 1. Mục đích

Đưa repo về trạng thái **trung thực — nhất quán — tái lập — kiểm chứng được**, trước khi bổ sung các tính năng còn thiếu (dashboard, TCN, benchmark).

Kế hoạch này không thay đổi scope nghiên cứu. Nó sửa 4 nhóm vấn đề:

1. **Nhãn & metric:** metric hiện tại đo trên nhãn do rule sinh ra, không phải nhãn người.
2. **Split & evaluation:** split trong báo cáo khác split thực tế; evaluation phụ thuộc thứ tự dòng.
3. **Tài liệu & trạng thái:** README/AGENTS/IMPLEMENTATION_PLAN/báo cáo mâu thuẫn với code.
4. **Tính năng còn thiếu:** dashboard, export, zone, TCN, benchmark chưa có nhưng báo cáo mô tả như đã có.

---

## 2. Hiện trạng cần sửa (tóm tắt có bằng chứng)

| # | Vấn đề | Bằng chứng | Mức độ |
|---|---|---|---|
| 1 | Toàn bộ window được gán bằng weak rule; file annotation người không được dùng | `scripts/build_dataset.py` L163–L167 | P0 |
| 2 | Nhãn "HUMAN" được script tự sinh và gắn cứ 5 segment một lần | `scripts/generate_annotations.py` L158–L169 | P0 |
| 3 | Hai file "double annotator" là bản copy, sửa 1 mẫu → Kappa 1.0 vô nghĩa | `scripts/generate_annotations.py` L239–L258 | P0 |
| 4 | Test split báo cáo (`class_5`, `session_006`) khác file split (`session_005`, `session_006`) | `data/splits/test_sessions.txt`, `reports/BAO_CAO_*` | P0 |
| 5 | Số test window báo cáo 141, evaluation thực tế 122 | `reports/experiments/lstm_baseline/evaluation_report.json` | P1 |
| 6 | `build_dataset.py` tự chia lại split, không đọc `data/splits/*.txt` | `scripts/build_dataset.py` L46–L67, L126 | P0 |
| 7 | Evaluation join metadata theo `iloc[i]` | `scripts/evaluate.py` L97–L118 | P1 |
| 8 | Báo cáo nói AdamW/batch 64/patience 5; code dùng Adam/batch 32/patience 10 | `scripts/train.py` L113–L119 | P1 |
| 9 | Dashboard/export/zone/TCN được mô tả nhưng chưa triển khai | `app/streamlit_app.py` 11 dòng; `src/reporting/__init__.py` rỗng | P1 |
| 10 | Feature zero-fill keypoint thiếu, làm mất mask | `src/preprocessing/features.py` L67–L69 | P2 |
| 11 | `reports/experiments/*/` bị gitignore; metrics thiếu experiment metadata | `.gitignore` L32–L33; `metrics.json` | P1 |
| 12 | README/AGENTS nói chỉ xong T01–T02; plan nói xong T06–T30 | `README.md` L197; `AGENTS.md` L134–L158 | P1 |

---

## 3. Phạm vi & phân track

| Track | Tên | Mục tiêu | Ước lượng |
|---|---|---|---|
| **A** | Trung thực hóa dữ liệu, split, metric, báo cáo | Không còn tuyên bố sai; có human test set và metric minh bạch | 9–11 người-ngày + công gán nhãn |
| **B** | Gia cố pipeline & tái lập | Contract dữ liệu, experiment metadata, test, CI | 5–6 người-ngày |
| **C** | Hoàn thiện sản phẩm | Zone, aggregation, export, Streamlit, TCN, benchmark | 10–12 người-ngày |

**Thứ tự bắt buộc:** A1 → A2 → A3 → A4 → A5 → A6; B làm song song sau A3; C chỉ bắt đầu sau khi A4/A6 xong (tránh làm dashboard trên số liệu sai).

---

## 4. TRACK A — Trung thực hóa dữ liệu & báo cáo

### A1. Chốt hiện trạng & nguồn sự thật duy nhất

| Nội dung | Chi tiết |
|---|---|
| Việc | Tạo `docs/PROJECT_STATUS.md` liệt kê từng task T01–T41: `DONE / PARTIAL / NOT STARTED`, bằng chứng (file, test, artifact), ngày kiểm chứng |
| File | `docs/PROJECT_STATUS.md` (mới); link từ `README.md`, `AGENTS.md`, `docs/IMPLEMENTATION_PLAN.md` |
| Cách làm | Đối chiếu code thực tế, không dựa vào báo cáo; ghi rõ task nào chỉ có placeholder |
| Verify | Mọi file trạng thái trỏ về cùng một bảng; không còn câu "T01/T02 hoàn thành" đơn lẻ |
| DoD | Một người mới đọc 5 phút biết chính xác cái gì có, cái gì không |
| Est | 0.5 ngày |

### A2. Sửa nguồn sinh annotation (bỏ nhãn HUMAN giả)

| Nội dung | Chi tiết |
|---|---|
| Việc | `generate_annotations.py` chỉ sinh `WEAK_RULE`; bỏ logic `is_human_pilot`; bỏ sinh file double-annotator giả |
| File | `scripts/generate_annotations.py` |
| Cách làm | 1) Mọi row: `label_source="WEAK_RULE"`, `label_rule_version="weak-v0.1"`, `annotator_id="SYSTEM"`. 2) Xóa khối L158–L169 và L239–L258. 3) Di chuyển 2 file `pilot_double_annotator_*.csv` hiện có vào `data/annotations/deprecated/` kèm README giải thích (không xóa hẳn). 4) Thêm test `test_generate_annotations_has_no_human_source` |
| Verify | `pytest tests/test_visualization_and_annotations.py -q`; grep `HUMAN` trong output sinh mới phải rỗng |
| DoD | Không còn đường code nào tự gắn HUMAN; artifact cũ được đánh dấu deprecated |
| Est | 0.5 ngày |

### A3. Đóng băng split & builder đọc split cố định

| Nội dung | Chi tiết |
|---|---|
| Việc | `build_dataset.py` đọc `data/splits/{train,val,test}_sessions.txt` nếu có; chỉ tự chia khi thiếu; validate chống leakage |
| File | `scripts/build_dataset.py`; `data/splits/split_metadata.json` |
| Cách làm | 1) Hàm `load_or_create_splits()`: nếu đủ 3 file → dùng nguyên; nếu thiếu → chia deterministic và ghi file. 2) Assert: không session nào nằm 2 split; union = toàn bộ session; mỗi split > 0 window. 3) Ghi `split_report.json`: session→số window, tổng theo split, SHA256 của 3 file split. 4) Lưu `window_ids` vào `window_features.npz`. 5) Thêm `--splits-dir` bắt buộc đọc. |
| Verify | Chạy builder 2 lần cho ra cùng split hash; test `test_split_no_session_overlap`, `test_split_is_frozen` |
| DoD | Split hiện tại (session_005/006 test) được đóng băng; mọi báo cáo dùng đúng số thực tế |
| Est | 1 ngày |

> **Quyết định cần chốt:** giữ split hiện tại (`split-v0.1`: test = session_005 + session_006; 122 window) hay tạo `split-v0.2` theo ý báo cáo cũ (test = class_5 + session_006). Nếu đổi split, **bắt buộc retrain + re-evaluate + version mới**, không sửa số cũ.

### A4. Sửa evaluation: join theo `window_id`, lưu prediction

| Nội dung | Chi tiết |
|---|---|
| Việc | Không dùng `iloc[i]`; join metadata theo `window_id`; xuất `predictions.csv` và lý do gate |
| File | `scripts/evaluate.py`, `src/uncertainty/gate.py` |
| Cách làm | 1) Đọc `window_ids` từ npz (A3). 2) Map id→dòng metadata; nếu thiếu id → fail rõ ràng, không fallback im lặng. 3) Thêm `explain_uncertainty_gate()` trả `(label, reason)` với reason ∈ `LOW_VISIBILITY / LOW_KEYPOINT_CONF / LOW_MODEL_CONF / ACCEPTED`; giữ API cũ. 4) Ghi `predictions.csv`: window_id, session, track, true, pred, p_max, label_source, gate_reason. 5) Metrics theo session và theo label_source. |
| Verify | Test `test_evaluate_join_by_window_id`; predictions có đúng N dòng = số test window; tổng confusion matrix = N accepted + uncertain |
| DoD | Không thể lệch nhãn/dòng dữ liệu; mọi con số truy vết được xuống từng window |
| Est | 1 ngày |

### A5. Experiment metadata & tái lập

| Nội dung | Chi tiết |
|---|---|
| Việc | Mỗi lần train/evaluate ghi metadata chuẩn; cho phép version hóa metrics |
| File | `scripts/train.py`, `scripts/evaluate.py`, `.gitignore`, `reports/experiments/<id>/metadata.json` |
| Cách làm | 1) Tạo `metadata.json`: `experiment_id`, `created_at`, `seed`, `git_commit` (hoặc `NO_GIT_COMMIT`), `dataset_version`, `split_version`, `feature_schema_version`, `config_sha256`, `checkpoint_sha256`, `class_counts`. 2) `config.yaml` snapshot. 3) `.gitignore`: ignore checkpoint nhưng **allow** `metrics.json`, `evaluation_report.json`, `metadata.json`, `split_report.json`. 4) Đổi tên experiment hiện tại thành `lstm_baseline_weaklabel_v0.1` khi chạy lại (không ghi đè). |
| Verify | `python scripts/train.py --name ... --epochs 1` tạo đủ file; metadata khớp config; hash kiểm tra lại được |
| DoD | Một experiment tái lập được từ metadata + config + split hash |
| Est | 1 ngày |

### A6. Gói gán nhãn người thật (không tự sinh)

| Nội dung | Chi tiết |
|---|---|
| Việc | Chuẩn bị tooling + protocol để con người gán nhãn; **không** tạo nhãn giả |
| File | `scripts/sample_annotation_subset.py` (mới), `data/annotations/human/`, `docs/annotation_guide.md`, `scripts/dataset_qa.py` |
| Cách làm | 1) Sample phân tầng theo session/phase/quality/predicted-label; ưu tiên test: 150–200 window, train/val: 150 window. 2) Xuất CSV template rỗng cột nhãn + `window_id` + video/timestamp; kèm hướng dẫn xem video. 3) 2 annotator độc lập, blind; adjudication ghi `ADJUDICATED`. 4) `dataset_qa.py` thêm `--strict-human`: chỉ tính Kappa trên cặp `HUMAN`; cảnh báo nếu 2 file giống nhau >95% (dấu hiệu copy). 5) Freeze `test-human-v1` + SHA256. |
| Verify | `python scripts/dataset_qa.py --annotator-a ... --annotator-b ... --strict-human`; Kappa + confusion matrix + danh sách disagreement |
| DoD | Có bộ nhãn người thật, có Kappa thật, có manifest đóng băng; không có nhãn nào do AI sinh |
| Est | 2 ngày tooling + 2–3 người-ngày gán nhãn |

> **Điều kiện:** nếu nhóm chưa thể gán nhãn người, A7 bị **BLOCKED** và toàn bộ báo cáo phải ghi "weak-label agreement", không được gọi là accuracy.

### A7. Đánh giá trên human test set

| Nội dung | Chi tiết |
|---|---|
| Việc | `build_dataset.py --labels human` (chỉ giữ window có nhãn người) và `--labels hybrid`; evaluate lại |
| File | `scripts/build_dataset.py`, `scripts/evaluate.py` |
| Cách làm | 1) Join human annotation theo `(session_id, track_id)` + overlap thời gian ≥ 50%. 2) `human`: chỉ window có nhãn HUMAN/ADJUDICATED. 3) `hybrid`: human ưu tiên, còn lại weak, giữ `label_source`. 4) Train lại từ đầu với human train; test chỉ human. 5) Báo cáo song song: weak-label metrics (cũ) vs human metrics (mới), kèm N và Kappa. |
| Verify | Bảng kết quả có 2 cột nguồn nhãn; test set không overlap train; N test human ≥ 100 |
| DoD | Có con số trung thực để báo cáo; nếu thấp hơn 94.77% thì ghi đúng và phân tích |
| Est | 1 ngày (sau khi có nhãn) |

### A8. Đồng bộ tài liệu trạng thái

| Nội dung | Chi tiết |
|---|---|
| Việc | Sửa `README.md` §10, `AGENTS.md` §5/§24, `docs/IMPLEMENTATION_PLAN.md` §11 theo `PROJECT_STATUS.md` |
| File | 3 file trên + `docs/LIMITATIONS.md` (mới) |
| Cách làm | 1) Bỏ tuyên bố "chỉ xong T01/T02". 2) Đánh dấu rõ task nào DONE/PARTIAL. 3) `LIMITATIONS.md`: weak labels, 2D pose, không gaze thật, zone chưa có, test N nhỏ, chưa có TCN/benchmark. |
| Verify | Không còn mâu thuẫn giữa 4 file; mọi task DONE đều có artifact/test |
| DoD | Tài liệu phản ánh đúng code tại thời điểm commit |
| Est | 1 ngày |

### A9. Viết lại 2 báo cáo theo số liệu thật

| Nội dung | Chi tiết |
|---|---|
| Việc | Sửa `reports/BAO_CAO_TIEN_DO_DU_AN.md` và `reports/BAO_CAO_CHI_TIET_LUONG_KY_THUAT.md` |
| File | 2 file trên |
| Cách làm | 1) Bỏ/đổi nhãn "HUMAN 1.726", "Kappa 1.0", "67/67" nếu chưa kiểm chứng lại. 2) Sửa split: train/val/test đúng file; test 122 window. 3) Sửa optimizer/batch/patience theo code. 4) Dashboard ghi "chưa triển khai" cho đến khi C3 xong. 5) Thêm mục "Hạn chế & độ tin cậy": metric weak-label, N nhỏ, không suy diễn tâm lý. 6) Mọi bảng số phải ghi nguồn file JSON. |
| Verify | Mọi số trong báo cáo khớp `metrics.json`/`evaluation_report.json`/`split_report.json` |
| DoD | Báo cáo không còn tuyên bố vượt quá bằng chứng |
| Est | 1 ngày |

---

## 5. TRACK B — Gia cố pipeline & tái lập

### B1. Mask khuyết điểm tường minh (feature v2)

| Nội dung | Chi tiết |
|---|---|
| Việc | Không zero-fill im lặng; thêm mask 17 chiều hoặc `missing_ratio` theo keypoint |
| File | `src/preprocessing/features.py`, `src/preprocessing/cleaner.py`, `src/preprocessing/feature_contract.py` |
| Cách làm | Tạo `posefeat-v2`: 77 + 17 mask = 94 chiều; giữ `posefeat-v1` cho checkpoint cũ; config experiment riêng; retrain như experiment mới |
| Verify | Test khẳng định keypoint thiếu có mask=0 và tọa độ không được coi là thật; shape 94 |
| DoD | Contract "không mã hóa (0,0) là điểm thật" đúng ở cả tầng feature |
| Est | 1.5 ngày |

### B2. Threshold lấy từ config + lý do quality

| Nội dung | Chi tiết |
|---|---|
| Việc | `windows.py` không hardcode 0.60/0.30; `quality_gate` trả reason |
| File | `src/preprocessing/windows.py`, `src/preprocessing/quality_gate.py`, `configs/baseline.yaml` |
| Cách làm | Truyền `QualityThresholds` từ config; thêm `quality_reason` vào window record; test config override |
| Verify | Đổi config → quality_state đổi theo; test |
| DoD | Không còn ngưỡng ma thuật trong code |
| Est | 0.5 ngày |

### B3. Test bổ sung & CI

| Nội dung | Chi tiết |
|---|---|
| Việc | Test cho split, join, gate reason, annotation provenance, exporter |
| File | `tests/test_split_contract.py`, `tests/test_evaluation_join.py`, `tests/test_annotation_provenance.py`, `tests/test_reporting_export.py`, `.github/workflows/ci.yml` |
| Cách làm | Fixture nhỏ 3 session giả; không cần video; chạy trong <30s; CI chạy pytest + import check |
| Verify | `python -m pytest -q` xanh; CI xanh |
| DoD | Các lỗi P0/P1 ở trên đều có test chống tái phát |
| Est | 1.5 ngày |

### B4. Pin môi trường

| Nội dung | Chi tiết |
|---|---|
| Việc | `requirements.txt` đang không pin; ghi rõ Python 3.14 + torch/ultralytics version đã dùng |
| File | `requirements.txt`, `README.md` |
| Cách làm | Pin version tối thiểu đã test; thêm `docs/ENVIRONMENT.md`; ghi CPU/GPU |
| Verify | Tạo venv mới cài theo file, chạy `scripts/check_setup.py` |
| DoD | Người khác cài được không cần hỏi |
| Est | 0.5 ngày |

---

## 6. TRACK C — Hoàn thiện sản phẩm

### C1. Gán zone thật (bắt buộc trước heatmap)

| Nội dung | Chi tiết |
|---|---|
| Việc | Hiện `zone_id` gần như rỗng/"M-C"; heatmap 3×3 chưa có cơ sở |
| File | `scripts/assign_zones.py` (mới), `configs/zones/<session>.yaml` |
| Cách làm | Cho phép ROI polygon 3×3 hoặc grid theo tọa độ người; gán zone theo tâm bbox; lưu vào pose/window; báo coverage theo zone |
| Verify | Mỗi window test có zone; bảng N theo 9 zone; zone không suy ra từ nhãn |
| DoD | Heatmap có dữ liệu thật, có N từng ô |
| Est | 1.5 ngày |

### C2. Aggregation + Export

| Nội dung | Chi tiết |
|---|---|
| Việc | `src/reporting` rỗng; cần module tổng hợp và xuất báo cáo |
| File | `src/reporting/aggregation.py`, `src/reporting/exporter.py`, `src/reporting/__init__.py` |
| Cách làm | `timeline.csv` theo phút; `zone_summary.csv` theo 9 zone kèm N/coverage/uncertain; `results.csv` theo window; `summary.json` kèm version + limitations |
| Verify | Test schema; tổng timeline = tổng window; zone N cộng lại = tổng |
| DoD | Export đủ 4 artifact, số liệu khớp backend |
| Est | 2 ngày |

### C3. Streamlit full flow (T35–T38)

| Nội dung | Chi tiết |
|---|---|
| Việc | Thay placeholder bằng app thật |
| File | `app/streamlit_app.py` + `app/pages/*` hoặc module con |
| Cách làm | 1) Chọn session có sẵn hoặc upload. 2) Chạy/đọc pipeline có cache. 3) Preview overlay ẩn danh (skeleton + track_id). 4) Timeline phân phối 4 trạng thái. 5) Heatmap 3×3 (chỉ khi C1 xong). 6) Panel uncertainty/quality. 7) Nút export. Không copy logic xử lý vào UI. |
| Verify | `streamlit run app/streamlit_app.py` chạy; chọn session demo hiển thị đúng số; export tải được |
| DoD | Demo lặp lại được, không sửa tay; privacy mặc định bật |
| Est | 3 ngày |

### C4. TCN + so sánh (T25)

| Nội dung | Chi tiết |
|---|---|
| Việc | Thêm `src/models/tcn.py`, cùng input [B,40,F], cùng split |
| File | `src/models/tcn.py`, `scripts/train_tcn.py` hoặc `--model tcn` |
| Cách làm | Causal dilated conv, residual; config `configs/experiments/tcn_baseline.yaml`; bảng so sánh LSTM/TCN (Macro-F1, latency, size) |
| Verify | Train/eval chạy; cùng test set; kết quả ghi experiment riêng |
| DoD | Có bảng so sánh công bằng, không tune bằng test |
| Est | 2 ngày |

### C5. Tracking comparison + benchmark (T34, T39)

| Nội dung | Chi tiết |
|---|---|
| Việc | ByteTrack vs BoT-SORT; đo FPS/RAM/VRAM/latency |
| File | `scripts/benchmark.py`, `reports/benchmark.md` |
| Cách làm | Subset video; ID switches/min, fragmentation; đo trên máy cụ thể; ghi cấu hình |
| Verify | Số đo lặp lại ±10%; ghi hardware |
| DoD | Có lý do chọn tracker dựa metric |
| Est | 2 ngày |

---

## 7. Thứ tự & phụ thuộc

```text
A1 ─┬─> A2 ─┐
    ├─> A3 ─┼─> A4 ─> A5 ─> A9
    └─> A8  │
A6 ─────────┴─> A7 (cần nhãn người)
A3/A4 ─> B1/B2/B3/B4 (song song)
A4 + A6 ─> C1 ─> C2 ─> C3
A5 ─> C4/C5
```

**Mốc bắt buộc:**
- Hết A5: repo không còn tuyên bố metric sai; split đóng băng.
- Hết A7: có metric human (hoặc tuyên bố rõ weak-label nếu chưa gán).
- Hết C3: dashboard thật, khớp backend.

---

## 8. Lệnh kiểm chứng

```bash
# Contract & unit
python -m pytest -q

# Annotation QA (chỉ HUMAN)
python scripts/dataset_qa.py --annotator-a data/annotations/human/ann_a.csv \
                             --annotator-b data/annotations/human/ann_b.csv --strict-human

# Build dataset từ split đóng băng
python scripts/build_dataset.py --poses-dir data/poses --splits-dir data/splits

# Train/eval (experiment mới, không ghi đè)
python scripts/train.py --features data/features/window_features.npz --name lstm_human_v1
python scripts/evaluate.py --checkpoint models/checkpoints/lstm_human_v1.pt \
                           --output reports/experiments/lstm_human_v1/evaluation_report.json

# UI
streamlit run app/streamlit_app.py
```

---

## 9. Rủi ro & cách xử lý

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Không có nhãn người thật | Không thể báo cáo accuracy thật | A7 BLOCKED; mọi báo cáo ghi "weak-label agreement"; không suy diễn |
| Đổi split/feature làm số cũ thay đổi | Mất so sánh lịch sử | Version hóa experiment; giữ artifact cũ; không sửa số cũ |
| Xóa/rename file synthetic | Mất dấu vết | Chuyển vào `deprecated/` + README, không xóa |
| Shell lỗi ACL sandbox | Không chạy được test | Sửa quyền workspace hoặc chạy trên máy nhóm; ghi rõ trong PR |
| Dashboard làm trước khi sửa số | Hiển thị sai | Chặn C sau A4/A6 |
| Zone không có dữ liệu | Heatmap vô nghĩa | C1 bắt buộc; nếu chưa có, ẩn heatmap và ghi N/A |

---

## 10. Những việc KHÔNG làm

- Không tự sinh nhãn HUMAN, không sửa tay file Kappa.
- Không đổi test split/label để tăng metric.
- Không xóa artifact cũ để "che" số liệu.
- Không thêm emotion/gaze/face recognition.
- Không refactor ngoài phạm vi task.
- Không tuyên bố T35–T38 xong khi Streamlit còn placeholder.

---

## 11. Definition of Done toàn dự án (bản trung thực)

1. `PROJECT_STATUS.md` đúng với code; 4 tài liệu trạng thái không mâu thuẫn.
2. Không còn nhãn HUMAN do script sinh; provenance kiểm tra bằng test.
3. Split đóng băng, có hash; build_dataset không tự chia lại.
4. Evaluation join theo `window_id`; có `predictions.csv`; metric truy vết được.
5. Có human test set + Kappa thật, **hoặc** tuyên bố rõ chỉ có weak-label agreement.
6. Mọi experiment có metadata + config snapshot + seed + commit + version.
7. Dashboard/export chỉ hiển thị cái đã có; zone có N thật.
8. `pytest` xanh; CI xanh; README chạy lại được từ đầu.

---

## 12. Quyết định cần nhóm chốt

1. **Phạm vi đợt này:** chỉ Track A (P0), A+B, hay A+B+C?
2. **Nhãn người:** đã có ai gán tay chưa, hay cần lập kế hoạch gán mới?
3. **Split:** giữ `split-v0.1` (session_005/006) hay đổi sang `split-v0.2` (class_5/session_006) và retrain?
4. **Deadline:** mốc nào cần báo cáo/bảo vệ để ưu tiên đúng?
