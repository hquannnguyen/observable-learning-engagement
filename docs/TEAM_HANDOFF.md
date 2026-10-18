# CVF25PRJ10 — Biên bản Bàn giao Dữ liệu & Tiến độ Nhóm

> **Người bàn giao:** Vũ Văn Phong (Data & Annotation Lead - B23DCPT250)  
> **Ngày bàn giao:** 2026-10-18  
> **Phiên bản dữ liệu:** `dataset_version: v0.1` | `session_id: session_001`  
> **Trạng thái Pilot Quality Gate:** `DECISION: GO` (Đạt chuẩn)

---

## 1. Tổng hợp các Artifacts đã hoàn thành

| STT | Tên file / Artifact | Đường dẫn trong Repo | Dung lượng | Ý nghĩa & Mục đích |
|:---:|---|---|---:|---|
| 1 | **Video Pilot (10 phút)** | `data/raw/session_001/cam_A_pilot.mp4` | 948 MB | Video chuẩn 1080p, bao quát 24 sinh viên phòng máy, dùng làm tập dữ liệu chuẩn. |
| 2 | **Metadata buổi học** | `data/raw/session_001/metadata.json` | 664 B | Timeline các pha `lecture`, `individual_work`, cấu hình camera. Đã validate `[OK]`. |
| 3 | **Tọa độ Pose & Tracking** | `data/poses/cam_A_pilot_poses.parquet` | 1.4 MB | 4.375 lượt phát hiện 17 keypoints và ByteTrack ID của sinh viên. |
| 4 | **Báo cáo Chất lượng Pilot** | `reports/pilot_quality.md` | 1.8 KB | Đánh giá góc quay: Visible 70.4%, Upper Body 0.742, Quyết định GO. |
| 5 | **Bộ dữ liệu Gán nhãn (Weak Labels)** | `data/annotations/session_001_weak_labels.csv` | 73 KB | 427 cửa sổ (4s/2s), 100% hợp lệ (`VALID`: 380, `OCCLUDED`: 47). |
| 6 | **Dashboard Trực quan** | `app/streamlit_app.py` | 20 KB | Streamlit app hoàn chỉnh: Preview ẩn danh, Timeline, Heatmap 3x3, Xuất CSV. |

---

## 2. Phân công bàn giao cho từng thành viên

### 2.1 Bàn giao cho bạn NGUYỄN HỮU QUÂN (Pose, Tracking & Preprocessing)
* **Nhiệm vụ tiếp theo:**
  - Hoàn thiện module tiền xử lý lọc khớp và chuẩn hóa tọa độ (Task T13, T14).
  - Trích xuất đặc trưng tư thế & chuyển động `src/preprocessing/features.py` (Task T15).
  - Cắt sliding window chuẩn (Task T17).
* **Các file Quân cần lấy:**
  1. `data/poses/cam_A_pilot_poses.parquet`
  2. `data/raw/session_001/cam_A_pilot.mp4`
  3. `reports/pilot_quality.md`
* **Lệnh chạy tiếp theo của Quân:**
  ```powershell
  python scripts/extract_pose.py --video data/raw/session_001/cam_A_pilot.mp4 --sample-fps 10
  ```

---

### 2.2 Bàn giao cho bạn VŨ ĐỨC TÀI (Temporal Models & Training Lead)
* **Nhiệm vụ tiếp theo:**
  - Chia tập Train / Val / Test đóng băng không rò rỉ (Task T18).
  - Thiết kế và huấn luyện mô hình **Customized LSTM M2** (Task T22) và **TCN M3** (Task T25).
  - Đánh giá Macro-F1, Confusion Matrix so sánh với Baseline Heuristic M0 (Task T24).
* **Các file Tài cần lấy:**
  1. `data/annotations/session_001_weak_labels.csv` (File nhãn đã gán chuẩn 3 class).
  2. `data/poses/cam_A_pilot_poses.parquet` (Dữ liệu khớp xương để build PyTorch Dataset).
  3. `configs/baseline.yaml` (Cấu hình siêu tham số: LSTM hidden 128, 2 layers, lr 0.001).
  4. `notebooks/03_train_lstm_colab.ipynb` (Notebook huấn luyện mẫu trên Google Colab GPU).

---

### 2.3 Bàn giao cho bạn NGUYỄN DŨNG TIẾN (Streamlit, Anonymization & Reporting)
* **Nhiệm vụ tiếp theo:**
  - Chạy thử nghiệm và làm đẹp giao diện Streamlit (Task T35, T36, T37).
  - Kiểm thử cơ chế ẩn danh (Head blur, Skeleton, Anonymous Track ID) (Task T33).
  - Chuẩn bị tính năng tải báo cáo `results.csv`, `timeline.csv`, `zone_summary.csv` phục vụ demo (Task T38, T40).
* **Các file Tiến cần lấy:**
  1. `app/streamlit_app.py`
  2. `data/raw/session_001/cam_A_pilot.mp4`
  3. `reports/pilot_quality.md`
* **Lệnh khởi động giao diện của Tiến:**
  ```powershell
  python -m streamlit run app/streamlit_app.py
  ```

---

*Biên bản được xác lập tự động nhằm đảm bảo tính đồng bộ và tái lập theo quy chuẩn AGENTS.md.*
