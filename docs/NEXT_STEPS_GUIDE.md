# HƯỚNG DẪN TIẾP THEO — NHÃN NGƯỜI, BENCHMARK, M0/M1, ABLATION, CALIBRATION

> **Bối cảnh hiện tại (2026-10-25):**
> - Dataset full-pose: **32.959 window** — train 13.481 / val 12.011 / test 7.467 (6.518 có nhãn weak).
> - LSTM: raw Macro-F1 **0.9098**, accepted **0.9151**, coverage **98.54%**.
> - TCN: raw **0.9027**, accepted **0.9198**, coverage **95.95%**.
> - **Chưa có nhãn người thật**; mọi metric hiện là weak-label agreement.
> - Split `split-v0.1` đã đóng băng; không đổi test set.
>
> **Nguyên tắc:** không tạo nhãn giả, không tune trên test, mọi experiment phải có ID +
> config snapshot + seed + git commit + dataset/split version.

---

## 1. NHÃN NGƯỜI THẬT (A6/A7, T12, T19)

### 1.1. Chuẩn bị

Template hiện tại trong `data/annotations/human/` được sinh **trước** khi rebuild full-pose,
nên phải sinh lại từ dataset 32.959 window:

```bash
python scripts/sample_annotation_subset.py \
  --windows data/features/participation_windows.csv \
  --out-dir data/annotations/human \
  --n-test 300 --n-train 200 --n-val 100
```

- 300 test window (~4,6% của 6.518) đủ để ước lượng Macro-F1 với sai số ~±5%.
- 200 train + 100 val để kiểm tra quy trình và làm hybrid training.
- Script đã **ẩn weak label** khỏi template để tránh anchoring.
- Ghi lại `sampling_manifest.json` (seed, hash nguồn, phân bố).

### 1.2. Quy trình gán

1. Copy template thành 2 file độc lập:
   ```bash
   cp data/annotations/human/annotation_template.csv data/annotations/human/ann_a.csv
   cp data/annotations/human/annotation_template.csv data/annotations/human/ann_b.csv
   ```
2. Hai người gán **blind**, không trao đổi, không xem weak label.
3. Mỗi dòng điền:
   - `participation_label`: `HIGH_OBSERVED_PARTICIPATION` / `MODERATE…` / `LOW…`
   - `annotation_state`: `VALID` / `OCCLUDED` / `TRANSITION` / `INVALID`
   - `label_source`: `HUMAN`
   - `label_rule_version`: để trống
   - `annotator_id`, `annotation_confidence` (0–1), `notes`
4. State khác `VALID` → để trống label/source/rule_version.
5. Xem đủ 4 giây mỗi window; dùng `video_path` + `t_start_ms`/`t_end_ms`.
6. Không sửa `annotation_template.csv` gốc.

### 1.3. QA và adjudication

```bash
python scripts/dataset_qa.py \
  --annotator-a data/annotations/human/ann_a.csv \
  --annotator-b data/annotations/human/ann_b.csv \
  --strict-human --min-kappa 0.60 \
  --report-json data/annotations/human/qa_report.json \
  --adjudication-out data/annotations/human/disagreements.csv
```

- `--strict-human` từ chối `WEAK_RULE` và cảnh báo nếu 2 file giống nhau.
- Kappa < 0.60 → sửa guideline, gán lại một phần; không đưa vào test.
- Review `disagreements.csv`, chốt `adjudicated.csv` với `label_source=ADJUDICATED`.
- Đóng băng test:
  ```text
  data/annotations/human/test-human-v1.manifest.json
  ```
  ghi SHA256 của `adjudicated.csv`, số window, Kappa, ngày, người review.

### 1.4. Build dataset và đánh giá

**Train hybrid** (human ưu tiên, weak cho phần còn lại — dùng cho training):

```bash
python scripts/build_dataset.py \
  --labels hybrid \
  --human-annotations-dir data/annotations/human \
  --output-dir data/features_hybrid
```

**Human-only** (chỉ giữ window có nhãn HUMAN/ADJUDICATED — dùng cho test):

```bash
python scripts/build_dataset.py \
  --labels human \
  --human-annotations-dir data/annotations/human \
  --output-dir data/features_human
```

Train trên hybrid, evaluate trên human:

```bash
python scripts/train.py \
  --features data/features_hybrid/window_features.npz \
  --name lstm_human_v1 --epochs 30

python scripts/evaluate.py \
  --experiment-dir reports/experiments/<lstm_human_id> \
  --features data/features_human/window_features.npz \
  --windows-meta data/features_human/participation_windows.csv \
  --output reports/experiments/<lstm_human_id>/human_evaluation.json
```

Báo cáo phải có **2 cột**: weak-label (cũ) và human-label (mới), kèm N và Kappa.
Không được trộn weak label vào test human.

### 1.5. DoD

- [ ] Template mới sinh từ dataset 32.959 window.
- [ ] 2 annotator gán blind; Kappa thật ≥ 0.60 (hoặc có giải trình).
- [ ] `test-human-v1` đóng băng + SHA256.
- [ ] Có `human_evaluation.json`, metric human, confusion matrix.
- [ ] Báo cáo ghi rõ metric nào là human, metric nào là weak.

**Ước lượng:** 600 window × 2 người ≈ 2–3 giờ/người + 0,5 ngày adjudication.

---

## 2. BENCHMARK VIDEO DÀI / GPU (T34, T39)

### 2.1. Chuẩn bị GPU

Kiểm tra:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu `False`, cài PyTorch CUDA đúng driver, rồi ghi version vào báo cáo.
Không so sánh FPS CPU và GPU trong cùng một bảng nếu khác hardware.

### 2.2. End-to-end trên video dài

Dùng `session_005` (900 s, 9.000 frame sampled):

```bash
python scripts/predict_video.py \
  --video data/raw/session_005/cam_A.mp4 \
  --checkpoint models/checkpoints/20261025-162855-lstm-full-v0-1.pt \
  --out-dir reports/inference/benchmark_session_005_gpu \
  --device cuda
```

Kết quả đo tự động trong `inference_meta.json`:

```text
sampled_frames, windows, accepted, elapsed_s,
end_to_end_fps, seconds_per_video_minute
```

Chạy thêm 1 lần CPU trên cùng video để có bảng so sánh nếu cần.

### 2.3. Đo VRAM / RAM

`predict_video.py` chưa tự đo VRAM. Thêm vào cuối script (khi `--device cuda`):

```python
if torch.cuda.is_available():
    inference_meta["peak_vram_mb"] = round(
        torch.cuda.max_memory_allocated() / (1024 * 1024), 1
    )
```

Peak RAM: sample `WorkingSet64` của tiến trình bằng PowerShell trong lúc chạy, hoặc
dùng `psutil` (cài thêm) và ghi `peak_rss_mb`.

### 2.4. Tracking

```bash
python scripts/benchmark.py \
  --video data/raw/session_005/cam_A.mp4 \
  --max-frames 9000 --pose-frames 300 \
  --trackers bytetrack.yaml,botsort.yaml \
  --out reports/benchmark_session_005.md
```

- Báo cáo FPS, số track, median duration, peak RAM.
- **ID switch/fragmentation chỉ đo được nếu có ground-truth track**; nếu không, ghi `N/A`,
  không suy đoán.

### 2.5. Bảng cần điền

| Metric | CPU | GPU |
|---|---:|---:|
| Pose FPS | | |
| End-to-end FPS | | |
| giây / phút video | | |
| Peak RAM | | |
| Peak VRAM | | |
| Checkpoint size | | |
| Hardware / CUDA / PyTorch | | |

### 2.6. DoD

- [ ] Chạy full `session_005` (hoặc video chưa dùng để debug).
- [ ] Có `inference_meta.json` + bảng CPU/GPU.
- [ ] Ghi hardware, driver, torch version.
- [ ] Không dùng số cũ 60 frame để tuyên bố full-flow.

**Ước lượng:** 0,5–1 ngày (chưa tính thời gian chạy GPU).

---

## 3. M0/M1 + ABLATION (T20, T21, T26–T29)

### 3.1. M0 — rule-based baseline

Tạo `src/models/rule_baseline.py`:

- Input: 77-dim window (hoặc metadata).
- Rule gợi ý:
  - `hand_raised` nhiều frame → HIGH;
  - `head_down` + motion thấp → LOW;
  - còn lại → MODERATE.
- Ngưỡng chọn trên **validation**, không test.
- Script `scripts/train_m0.py` xuất metrics cùng format `compute_multiclass_metrics`.

### 3.2. M1 — sklearn baseline

Tạo `scripts/train_sklearn_baseline.py`:

1. Load `X_train/X_val/X_test`.
2. Aggregate theo thời gian: mean, std, min, max mỗi feature → vector ~308 chiều.
3. `StandardScaler` **fit trên train only**.
4. `LogisticRegression(class_weight="balanced", max_iter=2000)` và `MLPClassifier`.
5. Chọn model/threshold trên validation; test chỉ chạy một lần.
6. Lưu `metrics.json`, `confusion_matrix.png`, `experiment metadata`.

Cài đặt:

```bash
python -m pip install scikit-learn
```

### 3.3. Ablation

| Task | Biến đổi | Cách làm |
|---|---|---|
| T26 representation | keypoints-only / engineered-only / combined | Slice feature dims, giữ nguyên window/split; train cùng LSTM |
| T27 motion | bỏ velocity 34–67 + `motion_energy` | Slice dims; so với full |
| T28 window | 2 s / 4 s / 6 s; FPS 5 / 10 | `build_dataset --window-seconds ... --stride-seconds ... --output-dir data/features_wX` |
| T29 imbalance | no-weight vs class-weight vs sampler | Thêm CLI `--no-class-weights` vào `train.py`; cùng split |

Quy tắc:
- Mỗi biến thể là một experiment riêng, cùng `split-v0.1`.
- Chọn cấu hình cuối bằng **validation**, không test.
- Báo cáo bảng: Macro-F1, per-class F1, balanced accuracy, confusion matrix, N.

### 3.4. Bảng so sánh cần có

```text
Model | Representation | Window | Imbalance | Val Macro-F1 | Test Macro-F1 | Test Balanced Acc
M0    | engineered     | 4 s    | -         | ...          | ...           | ...
M1    | aggregate      | 4 s    | weighted  | ...          | ...           | ...
M2    | combined       | 4 s    | weighted  | ...          | ...           | ...
M3    | combined       | 4 s    | weighted  | ...          | ...           | ...
```

### 3.5. DoD

- [ ] M0/M1 chạy trên cùng test set.
- [ ] Có bảng so sánh M0/M1/M2/M3.
- [ ] Ablation chỉ thay 1 biến mỗi lần.
- [ ] Kết luận dựa validation + test frozen.

**Ước lượng:** M0/M1 2–3 ngày; ablation 3–5 ngày.

---

## 4. CALIBRATION / RELIABILITY (T31)

### 4.1. Temperature scaling

Tạo `src/uncertainty/calibration.py`:

```python
def fit_temperature(logits, y, max_iter=200):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp(min=1e-3), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-3))
```

- Fit **chỉ trên validation**.
- Áp dụng cho test; báo cáo ECE trước/sau.

### 4.2. ECE và reliability

```python
def expected_calibration_error(probs, y, n_bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = (pred[mask] == y[mask]).mean()
        ece += mask.mean() * abs(acc - conf[mask].mean())
    return float(ece)
```

Vẽ reliability diagram + histogram confidence.

### 4.3. Chọn threshold uncertainty trên validation

- Quét `min_probability` ∈ [0.40, 0.95].
- Với mỗi threshold: coverage, accepted Macro-F1, uncertain rate.
- Chọn threshold theo tiêu chí nhóm (ví dụ coverage ≥ 90% và accepted F1 cao nhất).
- **Không chọn bằng test.**
- Báo cáo thêm coverage/uncertain theo session và zone (kèm N).

### 4.4. Script đề xuất

```bash
python scripts/calibrate.py \
  --experiment-dir reports/experiments/20261025-162855-lstm-full-v0-1 \
  --features data/features/window_features.npz \
  --windows-meta data/features/participation_windows.csv \
  --output reports/experiments/<id>/calibration_report.json
```

### 4.5. DoD

- [ ] Có `calibration_report.json`: T, ECE before/after, coverage-risk table.
- [ ] Threshold cuối chọn trên validation.
- [ ] Báo cáo ghi rõ calibration không thay đổi nhãn, chỉ thay độ tin cậy.

**Ước lượng:** 1–2 ngày.

---

## 5. THỨ TỰ ƯU TIÊN

1. **Nhãn người** (blocking mọi kết luận khoa học) — 3–4 ngày.
2. **Calibration** trên model hiện có — 1–2 ngày (có thể làm song song).
3. **M0/M1** — 2–3 ngày.
4. **Ablation** — 3–5 ngày.
5. **Benchmark dài/GPU** — 0,5–1 ngày.

## 6. CHECKLIST CHUNG

- [ ] Không có nhãn `HUMAN` do script sinh.
- [ ] Test set `test-human-v1` đóng băng + hash.
- [ ] Mọi experiment có metadata + config snapshot + seed + commit.
- [ ] Không chọn model/threshold/window bằng test.
- [ ] Báo cáo ghi rõ `metric_scope` (human vs weak).
- [ ] Cập nhật `PROJECT_STATUS.md`, `LIMITATIONS.md`, 2 báo cáo sau mỗi mốc.
