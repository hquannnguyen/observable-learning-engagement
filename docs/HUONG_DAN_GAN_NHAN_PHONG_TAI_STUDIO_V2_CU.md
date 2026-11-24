# HƯỚNG DẪN GÁN NHÃN ĐỘC LẬP CHO PHONG VÀ TÀI (REVIEW_V03)
**Dự án:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được (CVF25PRJ10)  
**Công cụ thực hiện:** `AnnotationStudio_V2_Standalone.exe` (Tool Studio V2 của Tiến & Phong)  
**Đối tượng áp dụng:**  
- **Vũ Văn Phong** (`ANN_PHONG` - Người gán nhãn B)  
- **Vũ Đức Tài** (`ANN_TAI` - Người gán nhãn A)  
**Mục tiêu:** Hoàn thiện 600 cửa sổ đại diện (Double Annotation) để nghiệm thu chỉ số đồng thuận Cohen's Kappa ($\kappa \ge 0.70$) và đóng băng tập Ground Truth chuẩn cho huấn luyện & đánh giá mô hình.

---

## 1. NGUYÊN TẮC GÁN NHÃN CỐT LÕI (BẮT BUỘC TUÂN THỦ)

1. **Gán nhãn hoàn toàn độc lập:** Phong và Tài không trao đổi kết quả trong quá trình gán để đảm bảo độ tin cậy khoa học khi đo lường độ đồng thuận (Inter-Annotator Agreement).
2. **Quan sát hành vi biểu hiện, không suy diễn tâm lý:**
   - Hệ thống đánh giá **mức độ tham gia học tập quan sát được** (*observable learning participation*).
   - Tuyệt đối **không** đánh giá suy diễn chủ quan như: *"học sinh lười"*, *"học sinh chán nản"*, *"mất tập trung"*, *"buồn ngủ"*.
3. **Quy tắc 3 mức nhãn chuẩn:**
   - **`HIGH` (CAO - Phím 1):** Sinh viên chủ động tương tác: nhìn chăm chú bảng/giảng viên, giơ tay phát biểu, ghi chép bài, đọc sách theo yêu cầu, thảo luận nhóm sôi nổi.
   - **`MODERATE` (VỪA - Phím 2):** Ngồi ổn định, tư thế trung tính, nghe giảng nhưng không ghi chép hoặc cử động nhẹ nhàng, tư thế cơ bản hướng về lớp học.
   - **`LOW` (THẤP - Phím 3):** Gục đầu xuống bàn kéo dài, quay hẳn người ra sau/ra cửa sổ không liên quan bài học, dùng điện thoại giấu dưới bàn, không tham gia hoạt động lớp.
4. **Quy tắc xử lý che khuất và đứt gãy pose (`OCCLUDED` - Phím 4):**
   - Nếu sinh viên bị bạn phía trước che khuất phần lớn thân trên.
   - **Đặc biệt lưu ý:** Nếu dải **"5 khung hình diễn biến"** ở góc dưới bên trái có **từ 3 đến 4 ô bị đen hoàn toàn** (nghĩa là tracker bị mất dấu $> 50\%$ thời lượng cửa sổ 4s, dưới 20 frame pose thật) $\rightarrow$ **Bắt buộc bấm `[Phím 4] BỊ CHE (OCCLUDED)`**. Không gán nhãn HIGH/MODERATE/LOW cho trường hợp này để tránh làm hỏng mô hình temporal LSTM/TCN!

---

## 2. PHÂN CHIA THƯ MỤC & TỆP DỮ LIỆU CỦA MỖI BẠN

Toàn bộ 600 cửa sổ đã được chia sẵn thành từng video nguồn trong 2 thư mục riêng biệt:
- **Thư mục của bạn PHONG:** [`data/annotations/review_v03_studio/phong/`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/annotations/review_v03_studio/phong/)
- **Thư mục của bạn TÀI:** [`data/annotations/review_v03_studio/tai/`](file:///d:/Thị%20giác%20máy%20tính/observable-learning-engagement/data/annotations/review_v03_studio/tai/)

### Bảng ghép cặp Video nguồn $\leftrightarrow$ File CSV tương ứng cần mở:

| Ưu tiên | Session ID | Tập chia | Số mẫu | Đường dẫn Video ở BƯỚC 1 | File CSV cần mở ở BƯỚC 2 (Bấm `[📂 Mở CSV]`) | File Pose tương ứng (nếu cần `[🦴 Đổi Pose]`) |
|:---:|---|---|---:|---|---|---|
| **P0** | **`session_005`** | **TEST** | **244** | `data/raw/session_005/cam_A.mp4` | `.../review_v03_studio/<phong_hoac_tai>/session_005_cam_A_labels.csv` | `data/poses/session_005_cam_A_poses.parquet` |
| **P1** | **`session_006`** | **TEST** | **56** | `data/raw/session_006/cam_A.mp4` | `.../review_v03_studio/<phong_hoac_tai>/session_006_cam_A_labels.csv` | `data/poses/session_006_cam_A_poses.parquet` |
| **P2** | **`session_003`** | **VAL** | **54** | `data/raw/session_003/cam_A.mp4` | `.../review_v03_studio/<phong_hoac_tai>/session_003_cam_A_labels.csv` | `data/poses/session_003_cam_A_poses.parquet` |
| **P3** | **`session_004`** | **VAL** | **46** | `data/raw/session_004/cam_A.mp4` | `.../review_v03_studio/<phong_hoac_tai>/session_004_cam_A_labels.csv` | `data/poses/session_004_cam_A_poses.parquet` |
| **P4** | **`session_001`** | **TRAIN** | **98** | `data/raw/session_001/cam_A.mp4` | `.../review_v03_studio/<phong_hoac_tai>/session_001_cam_A_labels.csv` | `data/poses/session_001_cam_A_poses.parquet` |
| **P5** | **`session_002`** | **TRAIN** | **95** | `data/raw/session_002/cam_A.mp4` | `.../review_v03_studio/<phong_hoac_tai>/session_002_cam_A_labels.csv` | `data/poses/session_002_cam_A_poses.parquet` |
| **P6** | **`classes`** | **TRAIN** | **7** | `data/raw/class_1.mp4` | `.../review_v03_studio/<phong_hoac_tai>/classes_train_labels.csv` | `data/poses/class_1_cam_A_poses.parquet` |

---

## 3. QUY TRÌNH THAO TÁC TRÊN `AnnotationStudio_V2_Standalone.exe`

### Bước 1: Mở Tool và Chọn Video
1. Khởi chạy file: `D:\Downloads\AnnotationStudio_V2_Standalone.exe`.
2. Tại giao diện tab **`BƯỚC 1: 📽 Chọn Video & Cắt Đoạn`**:
   - Bấm nút **`[📂 Chọn Video...]`** $\rightarrow$ Tìm đến video cần gán theo bảng trên (ví dụ: `data/raw/session_005/cam_A.mp4`).
   - **Lưu ý quan trọng:** Không cần bấm "Cắt đoạn" hay "Tạo mốc mới", vì danh sách cửa sổ đã được tính toán sẵn trong file CSV.

### Bước 2: Nạp Danh Sách Cửa Sổ (Mở CSV)
1. Dùng chuột chuyển sang tab **`BƯỚC 2: 👥 Gán & Duyệt Nhiều Sinh Viên`**.
2. Nhìn lên góc trên bên phải thanh điều hướng, bấm nút xanh **`[📂 Mở CSV]`**.
3. Chọn đúng file CSV của bạn trong thư mục `phong` (hoặc `tai`).
   - *Ví dụ đối với Phong khi gán session 5:* Mở file `data/annotations/review_v03_studio/phong/session_005_cam_A_labels.csv`.
4. Giao diện sẽ hiển thị ngay ảnh toàn cảnh lớp học, vị trí các sinh viên kèm khung nhận diện và khung xương (pose skeleton).
   - *Mẹo:* Nếu chưa thấy khung xương, bấm nút **`[🦴 Đổi Pose]`** và chọn file pose `.parquet` (hoặc `.csv`) tương ứng tại thư mục `data/poses/`.

### Bước 3: Thao Tác Gán Nhãn Siêu Tốc
Mỗi mốc 4 giây hiển thị toàn bộ sinh viên trong lớp học tại thời điểm đó:
1. **Chọn sinh viên:**
   - Click chuột trực tiếp vào khung của sinh viên trên ảnh toàn cảnh lớp học.
   - Hoặc click chuột phải vào sinh viên để đổi màu nhanh: `Xanh (Cao)` $\rightarrow$ `Vàng (Vừa)` $\rightarrow$ `Đỏ (Thấp)` $\rightarrow$ `Xám (Bị che)`.
2. **Quan sát diễn biến thời gian:**
   - Nhìn vào dải **5 khung hình diễn biến 4 giây** ở góc dưới bên trái để thấy rõ chuyển động cử chỉ của sinh viên đó trong suốt 4 giây.
3. **Bấm phím tắt:**
   - Bấm **`1`**: Gán **`CAO (HIGH)`**
   - Bấm **`2`**: Gán **`VỪA (MODERATE)`**
   - Bấm **`3`**: Gán **`THẤP (LOW)`**
   - Bấm **`4`**: Gán **`BỊ CHE (OCCLUDED)`**
4. **Duyệt cả mốc thời gian:**
   - Sau khi đã duyệt hết các sinh viên trong mốc đó, nhấn phím **`Space`** (hoặc bấm nút lớn màu xanh lá **`[✅ DUYỆT CẢ LỚP Ở MỐC NÀY & SANG MỐC TIẾP]`**).
   - Tool sẽ tự động chuyển sang mốc 4 giây tiếp theo.

### Bước 4: Lưu Kết Quả
- Khi hoàn thành hết một session, hoặc khi muốn tạm nghỉ giải lao:
- Bấm nút **`[💾 Lưu File CSV]`** (hoặc sang tab BƯỚC 3 bấm Xuất CSV) để lưu đè lên file CSV bạn vừa mở.

---

## 4. XỬ LÝ CÁC TÌNH HUỐNG THỰC TẾ HAY GẶP

1. **Khung hình diễn biến bị đen 3 đến 4 ô:**
   - *Hiện tượng:* Ở dải 5 ảnh bên dưới, ô 1-2 hiện ảnh nhưng ô 3-4-5 đen ngòm.
   - *Nguyên nhân:* Sinh viên bị mất dấu tracking hoặc cúi đầu quá sâu che khuất camera.
   - *Cách xử lý:* Bấm ngay **`[Phím 4] BỊ CHE (OCCLUDED)`**. Không gán nhãn hành vi.
2. **Lệch vị trí giữa khung ảnh lớn và ảnh crop diễn biến:**
   - *Hiện tượng:* Khung chọn trên ảnh lớn nằm ở bạn hàng 2, nhưng ảnh crop bên dưới lại là bạn hàng 1 (do tracker bị nhảy ID).
   - *Cách xử lý:* Luôn nhìn vào **ảnh crop thực tế ở dải 5 ảnh bên dưới** để chấm điểm cho đúng bạn sinh viên đó.
3. **Sinh viên chỉ quay đi thảo luận bài trong giờ discussion:**
   - *Pha thảo luận (discussion):* Quay sang nói chuyện với bạn bên cạnh/bàn sau về bài học được tính là **`HIGH`** hoặc **`MODERATE`**.
   - *Pha giảng bài (lecture):* Quay hẳn người ra sau nói chuyện riêng trong khi giảng viên đang giảng trên bảng thì tính là **`LOW`**.

---

## 5. LỆNH KIỂM TRA TIẾN ĐỘ & HỢP NHẤT DỮ LIỆU TỰ ĐỘNG

Mở PowerShell tại thư mục gốc của dự án (`observable-learning-engagement`) và chạy các lệnh tiện ích sau:

### 1. Xem bảng tiến độ gán nhãn thời gian thực:
```powershell
python scripts/manage_studio_review.py --status
```
*Lệnh này sẽ in ra chi tiết: Mỗi session Phong và Tài đã gán được bao nhiêu cửa sổ, còn lại bao nhiêu, tỷ lệ % hoàn thành toàn dự án.*

### 2. Tự động hợp nhất dữ liệu vào Master Dataset:
```powershell
python scripts/manage_studio_review.py --merge
```
*Chạy lệnh này sau khi bạn lưu file CSV trong Tool. Hệ thống sẽ tự động đồng bộ kết quả vào `data/annotations/review_v03/annotator_phong.csv` và `annotator_tai.csv`.*

### 3. Kiểm định độ đồng thuận (Cohen's Kappa):
```powershell
python scripts/dataset_qa.py --annotator-a data/annotations/review_v03/annotator_tai.csv --annotator-b data/annotations/review_v03/annotator_phong.csv --min-kappa 0.70
```
*Lệnh này sẽ tự động tính toán ma trận nhầm lẫn và chỉ số Cohen's Kappa giữa Phong và Tài, kiểm tra xem đã đủ điều kiện nghiệm thu hay chưa.*
