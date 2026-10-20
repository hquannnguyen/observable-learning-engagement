# BÁO CÁO KỸ THUẬT CHI TIẾT THEO LUỒNG CÔNG VIỆC TỪNG THÀNH VIÊN
### DỰ ÁN: PHÂN TÍCH MỨC ĐỘ THAM GIA HỌC TẬP TỪ TÍN HIỆU HÀNH VI QUAN SÁT ĐƯỢC
**Mã đề tài:** CVFP26BCPPRJC012 — **Nhóm thực hiện:** CVF25PRJ10  
**Ngày lập báo cáo:** 20/10/2026 — **Phiên bản:** v2.0 (Toàn diện từ Dataset Split đến Đánh giá mô hình)

---

## MỤC LỤC
1. [TỔNG QUAN LUỒNG KỸ THUẬT TOÀN HỆ THỐNG](#1-tổng-quan-luồng-kỹ-thuật-toàn-hệ-thống)
2. [GIAI ĐOẠN 1: VŨ VĂN PHONG — DỮ LIỆU, GÁN NHÃN & PHÂN CHIA SESSION](#2-giai-đoạn-1-vũ-văn-phong--dữ-liệu-gán-nhãn--phân-chia-session)
3. [GIAI ĐOẠN 2: NGUYỄN HỮU QUÂN — THỊ GIÁC, POSE, TIỀN XỬ LÝ & ĐẶC TRƯNG](#3-giai-đoạn-2-nguyễn-hữu-quân--thị-giác-pose-tiền-xử-lý--đặc-trưng)
4. [GIAI ĐOẠN 3: VŨ ĐỨC TÀI — MÔ HÌNH HÓA CHUỖI THỜI GIAN, HUẤN LUYỆN & EVAL](#4-giai-đoạn-3-vũ-đức-tài--mô-hình-hóa-chuỗi-thời-gian-huấn-luyện--eval)
5. [GIAI ĐOẠN 4: NGUYỄN DŨNG TIẾN — STREAMLIT DASHBOARD, HEATMAP & BÁO CÁO](#5-giai-đoạn-4-nguyễn-dũng-tiến--streamlit-dashboard-heatmap--báo-cáo)
6. [TỔNG HỢP SỐ LIỆU ĐỊNH LƯỢNG & SẢN PHẨM BÀN GIAO](#6-tổng-hợp-số-liệu-định-lượng--sản-phẩm-bàn-giao)

---

## 1. TỔNG QUAN LUỒNG KỸ THUẬT TOÀN HỆ THỐNG

Hệ thống được thiết kế theo kiến trúc đường ống tuần tự (Sequential End-to-End Pipeline), đảm bảo dữ liệu đầu ra của phân hệ trước là đầu vào chuẩn xác, có thể kiểm chứng của phân hệ tiếp theo:

```text
[11 Video Lớp Học 1080p]
       │
       ▼ (Giai đoạn 1 - Vũ Văn Phong)
[Tập nhãn hợp đồng 8.532 dòng] + [Phân chia Session độc lập (Group Split)]
       │
       ▼ (Giai đoạn 2 - Nguyễn Hữu Quân)
[YOLOv8-Pose 10 FPS] ──> [ByteTrack ID ẩn danh] ──> [Khử lỗi chi & lệch tốc độ]
       │
       ▼
[Chuẩn hóa cơ thể + Trích xuất 77 Đặc trưng] ──> [Cửa sổ trượt 4.0s (Tensor [7259, 40, 77])]
       │
       ▼ (Giai đoạn 3 - Vũ Đức Tài)
[Mạng nơ-ron Temporal LSTM (2 lớp)] ──> [Hàm mất mát Class-weighted CrossEntropy]
       │
       ▼
[Cổng không chắc chắn (Uncertainty Gate)] ──> [Đánh giá Test cô lập: F1 94.77%]
       │
       ▼ (Giai đoạn 4 - Nguyễn Dũng Tiến)
[Streamlit Web Dashboard: Video Preview + Timeline + Heatmap 3x3 + Export CSV]
```

---

## 2. GIAI ĐOẠN 1: VŨ VĂN PHONG — DỮ LIỆU, GÁN NHÃN & PHÂN CHIA SESSION

### 2.1. Mục tiêu và Nguyên tắc đạo đức dữ liệu
Xây dựng tập dữ liệu video và nhãn gán hợp đồng phục vụ bài toán nghiên cứu. Tuân thủ nghiêm ngặt ranh giới học thuật tại `docs/scope.md`: **Chỉ ghi nhận các tín hiệu hành vi quan sát được**, tuyệt đối không gán nhãn suy diễn trạng thái tâm lý, cảm xúc hay định kiến đạo đức (không dùng từ "chán nản", "lười biếng").

### 2.2. Dữ liệu đầu vào
* 11 video quay thực tế trong các giảng đường: `class_1` đến `class_5`, `session_001` đến `session_006` (độ phân giải Full HD 1080p, tốc độ 30 FPS, góc máy nhìn bao quát).
* Phiên quay dài tiêu biểu: `session_001` dài 15 phút.

### 2.3. Quy trình công nghệ và Triển khai chi tiết
1. **Thiết lập chuẩn gán nhãn CVAT (`docs/cvat_labels_spec.json`):**
   - Xây dựng đặc tả JSON chuẩn tích hợp vào công cụ CVAT (Computer Vision Annotation Tool).
   - Thiết lập cấu trúc nhãn đa tầng:
     * **3 Canonical Labels:** `HIGH_OBSERVED_PARTICIPATION`, `MODERATE_OBSERVED_PARTICIPATION`, `LOW_OBSERVED_PARTICIPATION`.
     * **Thuộc tính hành vi nguyên tử (Atomic Behaviors):** Hướng nhìn đầu (`head_direction`), hành động tay (`hand_action`), tư thế thân (`torso_pose`).
     * **Trạng thái chất lượng:** `VALID`, `OCCLUDED`, `TRANSITION`, `INVALID`.
2. **Sinh tập nhãn hợp đồng toàn diện (`scripts/generate_annotations.py`):**
   - Kết hợp nhãn người gán thủ công và hệ thống sinh nhãn tự động từ luật quan sát động học yếu (`src/preprocessing/weak_labels.py`).
   - Tạo ra file nhãn tổng thể: `data/annotations/annotations_all_v0.1.csv` gồm **8.532 dòng nhãn**.
   - Phân định minh bạch nguồn gốc nhãn qua cột `label_source`: 1.726 nhãn người thật (`HUMAN`) và 5.887 nhãn luật yếu (`WEAK_RULE` đi kèm phiên bản luật `weak-v0.1`).
3. **Kiểm định chất lượng gán nhãn (QA - `scripts/dataset_qa.py`):**
   - Thực nghiệm gán nhãn độc lập song song giữa hai người (Phong và Tài) trên cùng tập video pilot (`pilot_double_annotator_phong.csv` và `pilot_double_annotator_tai.csv`).
   - Đánh giá chỉ số độ tin cậy đồng thuận: **Cohen's Kappa đạt 1.0**, xác nhận 100% mẫu tuân thủ đúng quy tắc hướng dẫn.
4. **Quy trình phân chia dữ liệu độc lập theo Session (Group Splits):**
   - *Nguyên tắc cốt lõi:* Tuyệt đối không cắt ngẫu nhiên các khung hình của cùng một buổi học vào cả Train và Test để tránh rò rỉ dữ liệu (Zero Data Leakage).
   - *Kết quả phân bổ (`data/splits/`):*
     * **Tập Train (`train_sessions.txt`):** 7 sessions (`class_1`, `class_2`, `class_4`, `session_001`, `session_002`, `session_004`, `session_005`) $\to$ **6.951 cửa sổ trượt**.
     * **Tập Validation (`val_sessions.txt`):** 2 sessions (`class_3`, `session_003`) $\to$ **167 cửa sổ trượt**.
     * **Tập Test (`test_sessions.txt`):** 2 sessions (`class_5`, `session_006`) $\to$ **141 cửa sổ trượt**.

---

## 3. GIAI ĐOẠN 2: NGUYỄN HỮU QUÂN — THỊ GIÁC, POSE, TIỀN XỬ LÝ & ĐẶC TRƯNG

### 3.1. Mục tiêu
Chuyển hóa dữ liệu video thô thành chuỗi tọa độ khung xương bám sát chuyển động cơ thể, làm sạch nhiễu che khuất, tính toán 77 đặc trưng không gian - thời gian và đóng gói thành tensor cửa sổ trượt.

### 3.2. Quy trình công nghệ và Thuật toán chi tiết
1. **Trích xuất tư thế người (Pose Extraction - `src/pose/extractor.py`):**
   - Sử dụng mạng nơ-ron `YOLOv8-Pose` (weights: `yolov8n-pose.pt`, kích thước ảnh `imgsz=640`).
   - Lấy mẫu khung hình ở tốc độ **10 FPS** từ video gốc 30 FPS (`frame_step = 3`).
   - Định vị hộp bao và trích xuất 17 điểm khớp xương giải phẫu chuẩn COCO: 0-Nose, 1/2-Eyes, 3/4-Ears, 5/6-Shoulders, 7/8-Elbows, 9/10-Wrists, 11/12-Hips, 13/14-Knees, 15/16-Ankles.
   - Thử nghiệm mô hình độ nét cao: Chạy thực nghiệm `yolo11m-pose` tại `imgsz=1280` (`configs/experiments/pose_yolo11m_hd.yaml`), tăng số lượng người phát hiện ở góc xa từ 10 lên 14 người (+40% Recall).
2. **Theo vết đa đối tượng (Multi-Object Tracking - `src/pose/tracker.py`):**
   - Tích hợp bộ theo vết `ByteTrack` dựa trên Kalman Filter và ma trận IoU.
   - Duy trì ổn định mã định danh ẩn danh (`track_id`) qua thời gian, khắc phục hiện tượng nhảy ID khi học sinh tạm thời bị che khuất nhẹ.
3. **Xử lý toàn bộ Session 15 phút (Phương án B):**
   - Chạy nền toàn bộ video dài 15 phút của `session_001`: Xử lý thành công **9.000 khung hình**, ghi nhận **138.290 lượt phát hiện tư thế**.
4. **Khắc phục hai lỗi trực quan nghiêm trọng (`src/pose/visualization.py`):**
   - *Khắc phục lỗi lệch tốc độ khung xương 3x:* Trước đây video 30 FPS được lấy mẫu 10 FPS nhưng module vẽ lại truy vấn theo số thứ tự khung hình gốc, dẫn đến khung xương chạy nhanh gấp 3 lần chuyển động người thật. Đã viết lại cơ chế đồng bộ frame mapping chính xác 100%.
   - *Khử lỗi biến dạng chi giải phẫu:* Khi học sinh bị bàn ghế che khuất phần dưới, thuật toán nối khớp có xu hướng kéo dãn xương sang bạn ngồi cạnh. Đã bổ sung thuật toán kiểm tra ngưỡng khoảng cách chi:
     $$\text{Distance}(\text{joint}_1, \text{joint}_2) \le 1.6 \times \max(\text{Width}_{\text{box}}, \text{Height}_{\text{box}}) + 30\text{px}$$
     Nếu vượt quá, đường vẽ chi sẽ tự động ngắt, loại bỏ triệt để hiện tượng khung xương "nhảy loạn xạ".
5. **Lọc chất lượng & Chuẩn hóa tọa độ (`src/preprocessing/cleaner.py`):**
   - Nội suy tuyến tính các khoảng khuyết ngắn hạn ($\le 3$ frames), giữ nguyên mặt nạ khuyết tật (`missing_mask`) để không tạo tọa độ giả $(0, 0)$.
   - Chuẩn hóa gốc tọa độ về trung điểm hai vai: $\mathbf{O}_{\text{shoulder}} = \frac{1}{2}(\mathbf{K}_{\text{left\_shoulder}} + \mathbf{K}_{\text{right\_shoulder}})$.
   - Co giãn kích thước theo độ dài thân người (Torso distance: khoảng cách từ vai đến hông).
6. **Kỹ nghệ 77 đặc trưng không gian - thời gian (`src/preprocessing/features.py`):**
   Mỗi khung hình của một học sinh được biểu diễn bằng vector 77 chiều:
   - **Tọa độ chuẩn hóa (34 chiều):** $17 \text{ khớp} \times (x, y)$.
   - **Vận tốc và gia tốc động học (34 chiều):** Đạo hàm theo thời gian $\Delta x / \Delta t$ tại nhịp lấy mẫu $dt = 0.1\text{s}$.
   - **Đặc trưng hình học hành vi (6 chiều):**
     * Độ dốc vai (`shoulder_slope`): $\arctan2(\Delta y_{\text{shoulder}}, \Delta x_{\text{shoulder}})$.
     * Hướng quay đầu gián tiếp (`head_proxy_x`, `head_proxy_y`): Độ lệch giữa tọa độ mũi và trung điểm vai.
     * Độ giơ tay trái/phải (`left_hand_raised`, `right_hand_raised`): $\max(0, y_{\text{shoulder}} - y_{\text{wrist}})$.
     * Độ gục đầu (`head_down`): $\max(0, y_{\text{nose}} - y_{\text{mid\_shoulder}})$.
   - **Năng lượng chuyển động (1 chiều):** RMS Velocity toàn thân $\sqrt{\frac{1}{34} \sum v^2}$.
   - **Chỉ số chất lượng (2 chiều):** Tỷ lệ nhìn rõ (`visible_ratio`) và độ tin cậy trung bình (`mean_conf`).
7. **Đóng gói Cửa sổ trượt chuỗi thời gian (`scripts/build_dataset.py`):**
   - Cắt chuỗi frame liên tục của từng track thành các cửa sổ 4.0 giây (40 frames tại 10 FPS), bước trượt (stride) 2.0 giây (20 frames).
   - Ghép với nhãn hợp đồng của Phong để tạo thành tập tensor dữ liệu hoàn chỉnh:
     $$\mathbf{X} \in \mathbb{R}^{7259 \times 40 \times 77}$$
     Lưu tại `data/features/window_features.npz` và `data/features/participation_windows.parquet`.

---

## 4. GIAI ĐOẠN 3: VŨ ĐỨC TÀI — MÔ HÌNH HÓA CHUỖI THỜI GIAN, HUẤN LUYỆN & EVAL

### 4.1. Mục tiêu
Thiết kế kiến trúc học sâu mô hình hóa diễn biến hành vi theo thời gian, giải quyết bài toán mất cân bằng lớp và tích hợp Cổng không chắc chắn (Uncertainty Gate) đảm bảo độ tin cậy khoa học.

### 4.2. Kiến trúc mạng nơ-ron chuỗi thời gian Temporal LSTM (M2)
Được hiện thực hóa trong [`src/models/lstm.py`](file:///d:/Th%E1%BB%8B%20gi%C3%A1c%20m%C3%A1y%20t%C3%ADnh/observable-learning-engagement/src/models/lstm.py):
* **Đầu vào (Input):** Tensor kích thước $[\text{Batch Size}, 40, 77]$.
* **Tầng trích xuất chuỗi thời gian:** 2 tầng LSTM xếp chồng (Stacked LSTM):
  - Kích thước trạng thái ẩn (`hidden_size`): 128.
  - Dropout điều quy hóa: 0.30 giữa các tầng để chống học vẹt (overfitting).
* **Tầng phân loại (Classification Head):** Lấy trạng thái ẩn ở bước thời gian cuối cùng $h_{40}$ $\to$ Lớp Fully Connected $\to$ Hàm kích hoạt Softmax xuất ra 3 xác suất:
  $$\mathbf{P} = [P_{\text{HIGH}}, P_{\text{MODERATE}}, P_{\text{LOW}}] \quad \text{với} \quad \sum P_i = 1.0$$

### 4.3. Quá trình huấn luyện tối ưu (`scripts/train.py`)
* **Xử lý mất cân bằng lớp học đường (Class Imbalance):**
  Số lượng mẫu lớp `MODERATE` và `HIGH` nhiều hơn đáng kể so với lớp `LOW`. Tài áp dụng hàm mất mát `CrossEntropyLoss` với trọng số nghịch đảo tần suất:
  $$\mathbf{w}_{\text{class}} = \frac{N_{\text{total}}}{C \times N_{\text{class}}} \implies \mathbf{w} \approx [1.00, 0.72, 1.90]$$
* **Chiến lược tối ưu:**
  - Thuật toán tối ưu: `AdamW` (learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$).
  - Kích thước batch: 64.
  - Cơ chế Early Stopping: Theo dõi chỉ số `val_macro_f1` trên tập Validation độc lập (kiên nhẫn 5 epochs).
  - Kết quả huấn luyện: Mô hình đạt điểm cực đại tại **Epoch 11** với `val_macro_f1` đạt **93.69%**. Trọng số tối ưu được lưu tại: `models/checkpoints/lstm_baseline.pt`.

### 4.4. Cổng không chắc chắn (Uncertainty Gate - `src/uncertainty/gate.py`)
Để hệ thống không đưa ra phán đoán bừa bãi khi dữ liệu đầu vào mập mờ, Tài thiết kế cơ chế Cổng kép hai tầng:
1. **Tầng 1 (Chất lượng quan sát):** Kiểm tra nếu tỷ lệ nhìn rõ của học sinh $\text{visible\_ratio} < 0.60$ hoặc độ tin cậy khớp xương $< 0.30 \implies$ Trả về ngay trạng thái **`UNCERTAIN`**.
2. **Tầng 2 (Độ tự tin mô hình):** Kiểm tra xác suất cực đại $\max(\mathbf{P}) < 0.60 \implies$ Mô hình đang phân vân lưỡng lự $\implies$ Trả về trạng thái **`UNCERTAIN`**.
3. **Chấp nhận kết quả:** Chỉ khi vượt qua cả hai điều kiện trên, hệ thống mới trả về nhãn có xác suất cao nhất.

### 4.5. Kết quả kiểm thử độc lập trên tập Test cô lập (`scripts/evaluate.py`)
Đánh giá trên 2 session cô lập hoàn toàn (`class_5`, `session_006`):
* **Độ bao phủ (Coverage):** **95.90%** (chỉ 4.10% trường hợp mập mờ chuyển sang `UNCERTAIN`).
* **Macro-F1 (Accepted):** Đạt **94.77%** (tăng +2.56% so với dự đoán thô ban đầu).
* **Balanced Accuracy:** Đạt **96.40%**.
* **Overall Accuracy:** Đạt **94.87%**.
* **Độ chính xác từng lớp:**
  - `HIGH_OBSERVED_PARTICIPATION`: Precision 100.0%, Recall 92.54%, **F1 96.12%**.
  - `MODERATE_OBSERVED_PARTICIPATION`: Precision 85.29%, Recall 96.67%, **F1 90.62%**.
  - `LOW_OBSERVED_PARTICIPATION`: Precision 95.24%, Recall 100.0%, **F1 97.56%**.
* **Ma trận nhầm lẫn:** Tuyệt đối không có sai sót giữa hai cực `HIGH` và `LOW` (0 mẫu).

---

## 5. GIAI ĐOẠN 4: NGUYỄN DŨNG TIẾN — STREAMLIT DASHBOARD, HEATMAP & BÁO CÁO

### 5.1. Mục tiêu
Xây dựng giao diện ứng dụng web tương tác hoàn chỉnh, cho phép người dùng (giáo viên, nhà nghiên cứu) xem trực quan hóa video ẩn danh, biểu đồ tham gia của cả lớp theo dòng thời gian và phân tích sự chênh lệch theo vị trí chỗ ngồi.

### 5.2. Các phân hệ chức năng trên giao diện (`app/streamlit_app.py`)
1. **Phân hệ Tải video & Trực quan hóa khung xương ẩn danh:**
   - Cho phép tải video mới hoặc chọn các phiên quay demo sẵn có.
   - Trình phát video overlay hiển thị khung xương 17 khớp, hộp bao và `track_id` ẩn danh (ví dụ `track_01`, `track_02`), tuân thủ bảo mật thông tin cá nhân.
2. **Biểu đồ Dòng thời gian lớp học (Classroom Timeline Distribution):**
   - Biểu đồ diện tích/cột xếp chồng thể hiện biến động tỷ lệ % học sinh đạt mức `HIGH`, `MODERATE`, `LOW` và `UNCERTAIN` theo từng phút của buổi học.
   - Giúp giáo viên nắm bắt các mốc thời gian lớp sôi nổi (thảo luận, phát biểu) hoặc thời điểm lớp chùng xuống để điều chỉnh phương pháp sư phạm.
3. **Bản đồ nhiệt phân vùng lớp học $3 \times 3$ (Spatial Zone Heatmap):**
   - Trực quan hóa mức độ tham gia trung bình theo tọa độ chỗ ngồi:
     * Hàng: Bàn đầu (`Front`), Giữa lớp (`Middle`), Bàn cuối (`Back`).
     * Dãy: Trái (`Left`), Giữa (`Center`), Phải (`Right`).
   - Phân tích tương quan giữa khoảng cách tới bảng và mức độ chú ý quan sát được.
4. **Module Xuất dữ liệu báo cáo (Data Export - `src/reporting/exporter.py`):**
   - Nút bấm tải trọn gói các tệp báo cáo phục vụ lưu trữ học vụ:
     * `results.csv`: Kết quả dự đoán chi tiết của từng học sinh theo từng cửa sổ 4 giây.
     * `timeline.csv`: Thống kê tổng hợp cả lớp theo từng phút.
     * `zone_summary.csv`: Thống kê trung bình theo 9 khu vực phòng học.

---

## 6. TỔNG HỢP SỐ LIỆU ĐỊNH LƯỢNG & SẢN PHẨM BÀN GIAO

### 6.1. Bảng số liệu kỹ thuật tổng hợp

| Hạng mục dữ liệu / Thuật toán | Con số thực tế đã đạt được | Đơn vị / Trạng thái |
|:---|:---:|:---:|
| **Tổng số video thực tế** | 11 | Video Full HD 1080p |
| **Số frame của session dài (`session_001`)** | 9.000 | Khung hình (15 phút @ 10 FPS) |
| **Tổng số lượt phát hiện tư thế (`session_001`)** | 138.290 | Lượt người |
| **Tổng số dòng nhãn hợp đồng chuẩn** | **8.532** | Dòng nhãn (`annotations_all_v0.1.csv`) |
| **Độ đồng thuận người gán nhãn (QA)** | **1.0** | Cohen's Kappa |
| **Tổng số cửa sổ trượt (Sliding Windows)** | **7.259** | Cửa sổ 4.0 giây (40 frames) |
| **Kích thước ma trận tensor đầu vào** | **`[7259, 40, 77]`** | Tensor dữ liệu nén |
| **Phân chia mẫu: Train / Val / Test** | **6.951 / 167 / 141** | Cửa sổ (Group Split chống rò rỉ) |
| **Độ chính xác Macro-F1 (Accepted)** | **94.77%** | Tập Test cô lập |
| **Độ bao phủ Cổng không chắc chắn (Coverage)** | **95.90%** | Tỷ lệ mẫu tin cậy cao |
| **Tỷ lệ chuyển sang UNCERTAIN** | **4.10%** | Giữ an toàn dữ liệu mập mờ |
| **Số lượng test tự động vượt qua** | **67 / 67** | **100% Passed** |

### 6.2. Danh mục Artifacts đã hoàn thành và lưu trữ trong mã nguồn

1. `data/annotations/annotations_all_v0.1.csv` — File nhãn tổng thể 8.532 dòng.
2. `data/splits/` — Danh sách phân chia session độc lập (`train_sessions.txt`, `val_sessions.txt`, `test_sessions.txt`, `split_metadata.json`).
3. `data/features/window_features.npz` — Tensor chuỗi thời gian `[7259, 40, 77]`.
4. `models/checkpoints/lstm_baseline.pt` — Checkpoint trọng số mô hình LSTM tối ưu.
5. `reports/experiments/lstm_baseline/evaluation_report.json` — Báo cáo thực nghiệm kiểm thử độc lập.
6. `data/poses/session_001_cam_A_pose_overlay.mp4` — Video demo overlay khung xương 15 phút (599 MB).
7. `tests/` — 67 bài kiểm thử tự động toàn diện.
