# HƯỚNG DẪN GÁN NHÃN ĐỘC LẬP CHO PHONG VÀ TÀI (REVIEW_V03)

**Dự án:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được (CVF25PRJ10)  
**Cập nhật:** 24/11/2026  
**Người thực hiện:**  
* **Vũ Văn Phong** (`ANN_PHONG` - Người gán nhãn B)  
* **Vũ Đức Tài** (`ANN_TAI` - Người gán nhãn A)  
**Mục tiêu:** Hoàn thiện 600 cửa sổ độc lập (Double Annotation) phục vụ kiểm định chỉ số đồng thuận Cohen's Kappa ($\kappa \ge 0.70$) và đóng băng tập Ground Truth chuẩn.

---

## 1. NGUYÊN NHÂN LỖI MÀN HÌNH ĐEN TRÊN TOOL STUDIO V2

Khi mở `AnnotationStudio_V2_Standalone.exe`, nếu màn hình video bị đen xì và báo *"Hãy chọn Video & dữ liệu ở Bước 1 để bắt đầu gán nhãn"*, nguyên nhân gồm 2 lỗi cấu hình sau:

1. **Nạp nhầm "File Nhãn" vào làm "File Pose":**
   * Trong ảnh ở Bước 1 hiện: `[✓] Đã nạp file Pose: classes_train_labels.csv`.
   * `classes_train_labels.csv` là **file nhãn (Labels)** chứa danh sách mã cửa sổ (`window_id`), **hoàn toàn không chứa tọa độ khớp xương (Pose keypoints)**.
   * Tool của Tiến cần file tọa độ 17 khớp xương COCO (`kpt_00_x, kpt_00_y...`) nằm trong thư mục [`data/poses/`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/) (có đuôi **`_cam_A_poses.csv`**). Nếu nạp file nhãn làm file pose, tool không tìm thấy khớp xương $\rightarrow$ Màn hình đen xì.
2. **File gộp `classes_train_labels.csv` chứa tới 5 video khác nhau:**
   * File cũ gộp 7 mẫu của cả 5 video: `class_1.mp4`, `class_2.mp4`, `class_3.mp4`, `class_4.mp4`, `class_5.mp4`.
   * Nếu chọn video `class_1.mp4` (chỉ dài 15s) thì không thể hiển thị học sinh của các lớp khác.
   * **Giải pháp:** Đã tách 7 mẫu này thành từng file độc lập tương ứng $1-1$ với từng video (`class_1_cam_A_labels.csv` đến `class_5_cam_A_labels.csv`).

---

## 2. BẢNG ĐỐI CHIẾU CHUẨN XÁC 100% (VIDEO $\leftrightarrow$ FILE POSE $\leftrightarrow$ FILE NHÃN)

> [!IMPORTANT]
> **Quy tắc phân biệt bắt buộc khi mở Tool:**
> * **File POSE (Khung xương):** Luôn chọn trong thư mục [`data/poses/`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/) và có đuôi **`_cam_A_poses.csv`**. *(Tool chỉ đọc file `.csv`, không chọn file `.parquet`)*.
> * **File NHÃN (Danh sách duyệt):** Luôn chọn trong thư mục [`data/annotations/review_v03_studio/phong/`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/annotations/review_v03_studio/phong/) hoặc `tai/` (có đuôi **`_cam_A_labels.csv`**).

| Thứ tự | Session ID | Tập chia | Số mẫu | 1. Video ở Bước 1 | 2. File Pose thật sự (Bấm `[🦴 Đổi Pose]`) | 3. File Nhãn mở ở Bước 2 (`[📂 Mở CSV]`) |
|:---:|---|---|---:|---|---|---|
| **P0** | **`session_005`** | **TEST** | **244** | `data/raw/session_005/cam_A.mp4` | [`data/poses/session_005_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/session_005_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/session_005_cam_A_labels.csv` |
| **P1** | **`session_006`** | **TEST** | **56** | `data/raw/session_006/cam_A.mp4` | [`data/poses/session_006_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/session_006_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/session_006_cam_A_labels.csv` |
| **P2** | **`session_003`** | **VAL** | **54** | `data/raw/session_003/cam_A.mp4` | [`data/poses/session_003_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/session_003_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/session_003_cam_A_labels.csv` |
| **P3** | **`session_004`** | **VAL** | **46** | `data/raw/session_004/cam_A.mp4` | [`data/poses/session_004_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/session_004_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/session_004_cam_A_labels.csv` |
| **P4** | **`session_001`** | **TRAIN** | **98** | `data/raw/session_001/cam_A.mp4` | [`data/poses/session_001_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/session_001_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/session_001_cam_A_labels.csv` |
| **P5** | **`session_002`** | **TRAIN** | **95** | `data/raw/session_002/cam_A.mp4` | [`data/poses/session_002_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/session_002_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/session_002_cam_A_labels.csv` |
| **P6.1**| **`class_1`** | **TRAIN** | **2** | `data/raw/class_1.mp4` | [`data/poses/class_1_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/class_1_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/class_1_cam_A_labels.csv` |
| **P6.2**| **`class_2`** | **TRAIN** | **1** | `data/raw/class_2.mp4` | [`data/poses/class_2_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/class_2_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/class_2_cam_A_labels.csv` |
| **P6.3**| **`class_3`** | **TRAIN** | **2** | `data/raw/class_3.mp4` | [`data/poses/class_3_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/class_3_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/class_3_cam_A_labels.csv` |
| **P6.4**| **`class_4`** | **TRAIN** | **1** | `data/raw/class_4.mp4` | [`data/poses/class_4_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/class_4_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/class_4_cam_A_labels.csv` |
| **P6.5**| **`class_5`** | **TRAIN** | **1** | `data/raw/class_5.mp4` | [`data/poses/class_5_cam_A_poses.csv`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/poses/class_5_cam_A_poses.csv) | `.../review_v03_studio/<phong_hoac_tai>/class_5_cam_A_labels.csv` |

---

## 3. THAO TÁC TRÊN `AnnotationStudio_V2_Standalone.exe` (CHUẨN KHÔNG LỖI)

1. **Khởi động:** Mở file `D:\Downloads\AnnotationStudio_V2_Standalone.exe`.
2. **Ở Bước 1:**
   * Bấm nút **`[📂 Chọn Video...]`** $\rightarrow$ Chọn video cần gán (cột 1 của bảng trên, ví dụ `data/raw/session_005/cam_A.mp4` hoặc `data/raw/class_1.mp4`).
   * **Không bấm nút "Cắt Cửa Sổ 4s"** (vì các mốc thời gian đã nằm sẵn trong CSV).
3. **Chuyển sang Bước 2:**
   * **Nạp Pose:** Bấm nút **`[🦴 Đổi Pose]`** ở góc trên bên phải $\rightarrow$ Trỏ tới đúng file pose trong thư mục `data/poses/` (cột 2 của bảng trên, ví dụ `data/poses/session_005_cam_A_poses.csv`).
   * **Nạp Nhãn:** Bấm nút **`[📂 Mở CSV]`** $\rightarrow$ Trỏ tới đúng file nhãn của bạn trong `data/annotations/review_v03_studio/phong/` (cột 3 của bảng trên, ví dụ `session_005_cam_A_labels.csv`).
4. **Gán nhãn siêu tốc:**
   * Bấm **`1`**: Gán **`CAO (HIGH)`**
   * Bấm **`2`**: Gán **`VỪA (MODERATE)`**
   * Bấm **`3`**: Gán **`THẤP (LOW)`**
   * Bấm **`4`**: Gán **`BỊ CHE (OCCLUDED)`** (khi bị che khuất hoặc 5 khung hình diễn biến bên dưới có $\ge 3$ ô bị đen)
   * Nhấn phím **`Space`** (hoặc bấm `[✅ DUYỆT CẢ LỚP & SANG MỐC TIẾP]`) để chốt và nhảy sang mốc 4s tiếp theo.
5. **Lưu file:** Bấm **`[💾 Lưu File CSV]`** để lưu đè lên file CSV của session đó.

---

## 4. PHƯƠNG ÁN DỰ PHÒNG: GÁN NHÃN QUA WEB STREAMLIT (`app/annotation_app.py`)

Nếu không muốn phải chọn video và pose thủ công, hai bạn có thể sử dụng ứng dụng web Streamlit nội bộ đã được cấu hình tự động $100\%$:

```powershell
python -m streamlit run app/annotation_app.py
```

Trong giao diện trình duyệt:
1. Bạn Phong chọn **Phong** (`annotator_b.csv`), bạn Tài chọn **Tài** (`annotator_a.csv`).
2. Chọn bộ cửa sổ: **600 cửa sổ · phiếu độc lập Phong và Tài**.
3. Hệ thống tự động đọc đúng video và crop đúng khung xương theo từng dòng `window_id`, không cần người gán phải chọn file video hay pose.
4. Xem video clip, chọn nhãn `HIGH / MODERATE / LOW / OCCLUDED` và bấm **Lưu**.

---

## 5. BỘ LỆNH ĐỒNG BỘ & KIỂM ĐỊNH TỰ ĐỘNG

Sau khi gán xong một session trên Studio V2, chạy các lệnh sau tại thư mục gốc dự án:

```powershell
# 1. Xem tiến độ % hoàn thành của Phong và Tài
python scripts/manage_studio_review.py --status

# 2. Tự động gom dữ liệu đã lưu vào Master Dataset
python scripts/manage_studio_review.py --merge

# 3. Đo lường chỉ số đồng thuận Cohen's Kappa giữa Phong và Tài
python scripts/dataset_qa.py --annotator-a data/annotations/review_v03/annotator_tai.csv --annotator-b data/annotations/review_v03/annotator_phong.csv --min-kappa 0.70
```
