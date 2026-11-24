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

## 4. GÁN NHÃN TRÊN WEB STREAMLIT (`app/annotation_app.py`) — KHUYÊN DÙNG NHẤT

> [!TIP]
> **Giao diện Web Studio V2 hoàn toàn mới (Chạy mượt trong trình duyệt):**
> * Không cần cài đặt phần mềm ngoài, không cần `ffmpeg` (dùng thuần OpenCV giải mã trực tiếp).
> * **Có sẵn menu chọn nhanh từng Session / Class**, tự động nối đúng video và file pose $1-1$, không sợ chọn nhầm file.
> * **Chức năng BỎ CHE MẶT:** Checkbox `[x] Bỏ che mặt (Xem rõ mặt)` được **bật mặc định**, giúp nhìn rõ 100% mắt, nét mặt và hướng quay đầu của sinh viên để đánh giá chuẩn xác nhất.
> * **Gán siêu tốc 1-click:** Bấm nút `[🟢 CAO]`, `[🟡 VỪA]`, `[🔴 THẤP]`, `[⚪ BỊ CHE]` là tự lưu và tự nhảy sang sinh viên tiếp theo (chế độ `[x] Tự nhảy SV`). Không cần gõ tay bất kỳ ký tự nào!

### Cách chạy:
Mở Terminal tại thư mục dự án và chạy:
```powershell
python -m streamlit run app/annotation_app.py
```

### Các bước thao tác trên Web:
1. **Chọn người gán:** Tích chọn **Phong** hoặc **Tài** ở đầu trang.
2. **Tab 1: 🚀 Giao diện Studio (Kiểu Tiến - Chọn Nhanh Không Điền Lâu):**
   * **Chế độ 1: 🎯 Gán 600 Cửa Sổ Chuẩn (Review V0.3 - Mặc định & Khuyên dùng):**
     * Hệ thống **tự động nạp sẵn 600 cửa sổ chuẩn** của Phong hoặc Tài.
     * Có bộ lọc cực nhanh: `Chưa gán nhãn`, `Tập TEST (300 mẫu)`, `Tập VAL (100 mẫu)`, `Tập TRAIN (200 mẫu)`, hoặc lọc theo từng session.
     * Màn hình trái: Tự động trỏ đúng video và crop đúng sinh viên đang gán (viền trắng dày + tag vàng nổi bật), kèm dải 5 ảnh crop 4s có khung xương vàng.
     * Màn hình phải: Bấm 1 trong 4 nút `[CAO]`, `[VỪA]`, `[THẤP]`, `[BỊ CHE]`. Bật `[x] Tự nhảy cửa sổ kế` thì tool sẽ tự lưu và nhảy sang cửa sổ chưa gán tiếp theo ngay lập tức.
     * Tự động đồng bộ dữ liệu vào cả `review_v03_independent_20261124`, `review_v03`, và `review_v03_studio`!
   * **Chế độ 2: 👥 Duyệt Theo Mốc 4s Toàn Lớp (Theo Session / Cắt Đoạn):**
     * **BƯỚC 1:** Chọn phiên học trong dropdown (ví dụ `session_005 (TEST · 0s-60s)` hoặc `class_1`), bấm `[🚀 Cắt Cửa Sổ 4s & Vào Gán Nhãn -> Sang BƯỚC 2]`.
     * **BƯỚC 2:** Hiện toàn cảnh lớp học và duyệt theo từng mốc 4s, có nút `[✓ DUYỆT CẢ LỚP Ở MỐC NÀY & SANG MỐC TIẾP]`, gán đồng loạt `Tất cả CAO`, `Tất cả VỪA`, `Tất cả THẤP`.
     * **BƯỚC 3:** Kiểm tra phân bố và xuất file CSV.
   * **Chức năng BỎ CHE MẶT:** Checkbox `[x] Bỏ che mặt (Xem rõ mặt)` ở thanh trên cùng (mặc định luôn BẬT) giúp nhìn rõ 100% mắt và biểu cảm khuôn mặt.

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

## 6. Chế độ Studio 600 sau cập nhật

Chọn tab Studio và chế độ **Gán 600 Cửa Sổ Chuẩn**. Phiếu 600 của Phong/Tài được lưu riêng, không tự đồng bộ qua phiếu master hoặc Studio cũ. Trước khi lưu VALID, xem clip cả cửa sổ, xác nhận đúng người, chọn hành vi, xác định phase kèm căn cứ và bốn khoảng bằng chứng phù hợp mức được chọn. Có thể dùng ảnh tĩnh để hỗ trợ, không thay thế clip. Nếu bị che hoặc lỗi track, chọn trạng thái phù hợp và ghi lý do. Theo yêu cầu người dùng, bỏ che mặt mặc định trong Studio 600; bỏ chọn checkbox để bật làm mờ lại. Các nút là nút bấm chuột, không công bố phím tắt. Không gán hàng loạt cho toàn track. Khuyến nghị mỗi người chỉ mở một phiên ứng dụng cho phiếu của mình để tránh hai lần lưu đồng thời.

Các chế độ Studio theo session khác vẫn là luồng cũ và chưa được kiểm định lại theo quy trình 600 ở đây. Nhãn cũ và phần đồng bộ giữa các phiếu phải rà soát riêng; không chạy các script cũ để ghi đè dữ liệu.

### Thay đổi thao tác Studio theo yêu cầu người dùng

Chế độ Studio 600 đã bỏ bốn ô bằng chứng theo khoảng và điều kiện suy ra nhãn từ bốn khoảng. Người gán đánh giá toàn cửa sổ theo `docs/annotation_guide.md` v0.2. Vẫn cần clip, hành vi, căn cứ pha và xác nhận đã xem để lưu VALID. Nhãn mới ghi `annotation_guide_version=v0.2`, phương pháp `WHOLE_WINDOW_HUMAN_REVIEW`; nhãn cũ giữ nguyên phiên bản, cần rà soát nếu gộp với v0.3-pilot. Tab rubric riêng vẫn là tiêu chí thí điểm bốn khoảng. Nội dung này thay thế yêu cầu bốn khoảng của chế độ Studio ở các phần trên.
