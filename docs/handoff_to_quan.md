# Hướng dẫn Bàn giao Dữ liệu: Phong -> Quân (Tasks T06 & T07)

> **Người gửi:** Vũ Văn Phong (Data & Annotation Lead)  
> **Người nhận:** Nguyễn Hữu Quân (Pose, Tracking & Preprocessing Lead)  
> **Dữ liệu bàn giao:** `session_001` (Video pilot 10 phút + metadata)

---

## 1. Vị trí đặt file sau khi nhận

Khi Quân tải 2 file về máy, hãy tạo thư mục và đặt đúng vị trí theo quy chuẩn của repo:

```text
observable-learning-engagement/
└── data/
    └── raw/
        └── session_001/
            ├── cam_A_pilot.mp4   # File video pilot 10 phút
            └── metadata.json     # File thông tin buổi học
```

---

## 2. Kiểm tra môi trường chạy

Mở Terminal trong thư mục repo và cài đặt các thư viện (nếu chưa cài):

```powershell
pip install -r requirements.txt
```

---

## 3. Lệnh chạy trích xuất Pose & Tracking (Task T06 & T07)

Quân chạy lệnh sau để tự động chạy **YOLOv8-Pose** kết hợp **ByteTrack**:

```powershell
python scripts/extract_pose.py --video data/raw/session_001/cam_A_pilot.mp4 --sample-fps 10
```

- **Tần số trích mẫu:** 10 FPS (chuẩn baseline của đề tài).
- **Kết quả sinh ra:** File parquet lưu tại `data/poses/cam_A_pilot_poses.parquet`.

*(Tùy chọn: Nếu Quân muốn chạy trên GPU Google Colab cho nhanh, Quân có thể mở notebook `notebooks/01_pose_extraction_colab.ipynb`)*.

---

## 4. Kiểm tra chất lượng góc quay (Task T08 — Pilot Quality Review)

Quân mở giao diện trực quan để xem khung xương và kiểm tra xem có bị màn hình laptop che mất tay hoặc bị nhảy ID sinh viên không:

```powershell
python -m streamlit run app/streamlit_app.py
```

- Chọn nguồn video: `data/raw/session_001/cam_A_pilot.mp4`.
- Xem tab **🎥 Xem trước Ẩn danh** để đánh giá góc quay đạt tiêu chuẩn (**GO**) hay cần điều chỉnh (**ADJUST CAMERA**).

---

## 5. Bàn giao lại kết quả cho Phong

Sau khi chạy xong lệnh ở Mục 3, Quân gửi lại cho Phong file:
📁 **`data/poses/cam_A_pilot_poses.parquet`**

*(Phong sẽ dùng file này để sinh template CSV và tiến hành gán nhãn hành vi tham gia học tập theo Task T11)*.
