# Hướng dẫn Kỹ thuật: Bàn giao Dữ liệu cho Bạn VŨ ĐỨC TÀI

> **Người gửi:** Vũ Văn Phong (Data Lead)  
> **Người nhận:** Vũ Đức Tài (Temporal Models & Training Lead - B23DCPT277)  
> **Nhiệm vụ trọng tâm:** Task T18 (Splits), T22 (Customized LSTM), T23 (Training Pipeline), T25 (TCN)

---

## 1. Dữ liệu Tài nhận được (Sau khi `git pull origin main`)

1. **Nhãn huấn luyện:** `data/annotations/session_001_weak_labels.csv` (427 cửa sổ, 100% đúng chuẩn 3 class).
2. **Khung xương đầu vào:** `data/poses/cam_A_pilot_poses.parquet` (Tọa độ 17 keypoints theo thời gian).
3. **Cấu hình siêu tham số chuẩn:** `configs/baseline.yaml`.
4. **Notebook chạy Google Colab:** `notebooks/03_train_lstm_colab.ipynb`.

---

## 2. Các bước Tài cần làm để hoàn thành nhiệm vụ

### Bước 1: Chia tập Train / Validation / Test (Task T18)
- **Quy tắc bất di bất dịch của đề tài:** Không được chia ngẫu nhiên (random split) các window của cùng một người hoặc cùng buổi học sang cả Train và Test.
- Tài chia tập dựa theo `track_id` hoặc chia theo mốc thời gian (ví dụ: 70% thời gian đầu làm Train/Val, 30% thời gian cuối đóng băng làm Test).
- Lưu thông tin split vào cột `split` (`train`, `val`, `test`) cố định random seed = 42.

### Bước 2: Thiết kế mô hình Customized LSTM M2 (Task T22)
- Mở file `src/models/lstm.py` và thiết kế class `CustomizedLSTMClassifier(nn.Module)`:
  - Input: Tensor chuỗi thời gian `[Batch, Timesteps=20..40, Features]`.
  - Backbone: `nn.LSTM(input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.30)`.
  - Output Head: `nn.Linear(hidden_size, 3)` tương ứng 3 canonical classes:
    - 0: `HIGH_OBSERVED_PARTICIPATION`
    - 1: `MODERATE_OBSERVED_PARTICIPATION`
    - 2: `LOW_OBSERVED_PARTICIPATION`

### Bước 3: Huấn luyện mô hình (Task T23)
- Sử dụng hàm mất mát có trọng số `Weighted CrossEntropyLoss` nếu có sự chênh lệch số lượng mẫu giữa các class.
- Huấn luyện trên Colab GPU bằng notebook `notebooks/03_train_lstm_colab.ipynb` hoặc máy local:
  - Optimizer: `Adam(lr=0.001)`
  - Early stopping: `patience=10`
  - Lưu checkpoint tốt nhất vào `models/checkpoints/lstm_best.pt`.

### Bước 4: Đánh giá & So sánh (Task T24)
- Đo lường trên tập Test đã đóng băng:
  - Macro-F1 (Chỉ số chính của đề tài).
  - Per-class Precision, Recall, F1.
  - Ma trận nhầm lẫn (Confusion Matrix).
- So sánh kết quả với **Baseline Heuristic M0** hiện có trong `src/models/baseline_heuristic.py`.

---

## 3. Lệnh tạo Experiment để ghi nhận kết quả (Reproducibility)
Mỗi lần Tài chạy thử nghiệm chính thức, hãy tạo log bằng lệnh:

```powershell
python scripts/new_experiment.py --name lstm-baseline --seed 42 --dataset-version v0.1 --split-version split-v0.1
```
