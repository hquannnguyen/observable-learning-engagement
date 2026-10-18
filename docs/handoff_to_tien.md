# Hướng dẫn Kỹ thuật: Bàn giao Dữ liệu & UI cho Bạn NGUYỄN DŨNG TIẾN

> **Người gửi:** Vũ Văn Phong (Data Lead)  
> **Người nhận:** Nguyễn Dũng Tiến (Streamlit, Anonymization & Reporting Lead - B23DCPT306)  
> **Nhiệm vụ trọng tâm:** Task T32 (Zone Analysis), T33/T36 (Anonymization), T35-T38 (Dashboard & Export), T40 (Demo QA)

---

## 1. Dữ liệu & Code Tiến nhận được (Sau khi `git pull origin main`)

1. **Giao diện Dashboard hoàn chỉnh:** `app/streamlit_app.py`.
2. **Module Ẩn danh hóa bảo mật:** `src/pose/anonymizer.py` (Làm mờ mặt, vẽ khung xương).
3. **Module Tổng hợp số liệu:** `src/reporting/aggregator.py` (Sinh dữ liệu timeline và ma trận vùng 3x3).
4. **Báo cáo chất lượng góc quay:** `reports/pilot_quality.md`.
5. **Video pilot thực tế để chạy demo:** `cam_A_pilot.mp4` (Tải từ link Drive nhóm).

---

## 2. Các bước Tiến cần làm để hoàn thành nhiệm vụ

### Bước 1: Khởi động và kiểm thử giao diện Dashboard (Task T35)
Mở Terminal và chạy:

```powershell
python -m streamlit run app/streamlit_app.py
```

- Kiểm tra tính năng kéo thả file video và tính năng nhập đường dẫn trực tiếp trên máy.
- Chạy thử nghiệm phân tích đoạn 30 giây hoặc 1 phút để kiểm tra tính mượt mà của thanh tiến trình.

---

### Bước 2: Kiểm định tính năng Ẩn danh bảo mật (Task T33 & T36)
- Vào tab **🎥 Xem trước Ẩn danh** trên giao diện:
  - Kiểm tra xem khuôn mặt của tất cả các bạn sinh viên đã được làm mờ (Head Blur) tự động chưa.
  - Kiểm tra xem ID hiển thị có đúng định dạng ẩn danh `Track #ID` (không được hiển thị tên/MSSV thật).
  - Khung xương đổi màu theo trạng thái (Xanh lá: HIGH, Vàng: MODERATE, Xám: UNCERTAIN/LOW).

---

### Bước 3: Đánh giá Phân bố Vùng lớp học 3x3 & Biến thiên Thời gian (Task T32 & T37)
- Vào tab **📈 Biểu đồ Thời gian:** Kiểm tra đồ thị diện tích thể hiện tỷ lệ % các mức độ tham gia thay đổi theo từng giây.
- Vào tab **🏫 Vùng Lớp học (3x3):** 
  - Đảm bảo ma trận 9 ô (hàng Trước, Giữa, Sau $\times$ Trái, Giữa, Phải) luôn hiển thị đầy đủ số lượng mẫu $N$ và tỷ lệ % theo đúng quy chuẩn khoa học.

---

### Bước 4: Kiểm tra tải báo cáo phục vụ bảo vệ đồ án (Task T38)
- Vào tab **💾 Xuất Dữ liệu**, tải thử 3 file:
  - `results.csv`: Dữ liệu chi tiết từng window 4s.
  - `timeline.csv`: Số liệu chuỗi thời gian.
  - `zone_summary.csv`: Bảng tổng hợp theo hàng ghế.
- Đảm bảo các file tải về mở được bằng Excel, không bị lỗi font Tiếng Việt và khớp dữ liệu với màn hình.

---

### Bước 5: Chuẩn bị Kịch bản Demo thực tế (Task T40)
- Tiến chuẩn bị kịch bản trình chiếu khoảng 3–5 phút cho giảng viên / hội đồng xem:
  1. Giới thiệu đề tài: Nhấn mạnh phân tích hành vi quan sát được, không suy diễn cảm xúc.
  2. Bấm chạy phân tích video thực tế trên Streamlit.
  3. Trình diễn video ẩn danh bảo vệ quyền riêng tư.
  4. Trình diễn đồ thị tham gia theo thời gian và bản đồ vùng lớp học 3x3.
  5. Tải file báo cáo kết quả tổng kết buổi học.
