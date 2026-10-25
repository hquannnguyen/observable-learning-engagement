# LSTM vs TCN — so sánh trên dataset full-pose

> **Cảnh báo:** cả hai mô hình được đánh giá trên **weak-rule labels**, không phải
> human ground truth. Bảng này so sánh kiến trúc trên cùng dữ liệu, không chứng minh
> hiệu quả trên hành vi người thật.

## 1. Thiết lập

| Hạng mục | Giá trị |
|---|---|
| Dataset | `data/features/window_features.npz`, tensor `[32959, 40, 77]` |
| Split | `split-v0.1` — test 7.467 window, 6.518 có nhãn |
| Train / Val | 13.481 / 12.011 window |
| Seed | 42 |
| Optimizer | Adam, lr 1e-3, batch 32 |
| Loss | class-weighted CrossEntropy |
| Early stopping | patience 10, monitor `val_macro_f1` |
| Hardware | CPU (không đo GPU) |

## 2. Kết quả

| Chỉ số | LSTM | TCN |
|---|---:|---:|
| Experiment ID | `20261025-162855-lstm-full-v0-1` | `20261025-163801-tcn-full-v0-1` |
| Best val Macro-F1 | **0.9159** | 0.8801 |
| Raw Macro-F1 | **0.9098** | 0.9027 |
| Accepted Macro-F1 | 0.9151 | **0.9198** |
| Accepted Accuracy | 0.9281 | **0.9332** |
| Coverage | **98.54%** | 95.95% |
| Uncertain rate | **1.46%** | 4.05% |
| Params | 246.531 | 63.363 |
| Latency CPU (122 window) | 31.74 ms | **17.05 ms** |

## 3. Nhận xét

- Raw Macro-F1: LSTM nhỉnh hơn (0.9098 vs 0.9027). Sau gate, TCN nhỉnh hơn rất nhẹ
  (0.9198 vs 0.9151) nhưng **coverage thấp hơn** (95.95% vs 98.54%) — không phải
  so sánh “tốt hơn” tuyệt đối nếu bỏ qua coverage.
- TCN nhẹ hơn ~3,9× và nhanh hơn ~1,8× trên CPU.
- Cả hai đều thấp hơn số công bố cũ (0.94–0.95) vì dataset full khó hơn nhiều:
  32.959 window thay vì 7.259, test 6.518 thay vì 122.
- TCN yếu nhất ở `MODERATE` (recall raw 0.8451); LSTM cân bằng hơn giữa các lớp.
- Chưa tune hyperparameter; không được chọn model/window/threshold bằng test set.

## 4. Cách tái lập

```bash
python scripts/train.py --name lstm_full_v0.1 --epochs 30
python scripts/evaluate.py --experiment-dir reports/experiments/<lstm_id>
python scripts/train.py --config configs/experiments/tcn_baseline.yaml \
  --name tcn_full_v0.1 --epochs 30 --model tcn
python scripts/evaluate.py --experiment-dir reports/experiments/<tcn_id>
```
