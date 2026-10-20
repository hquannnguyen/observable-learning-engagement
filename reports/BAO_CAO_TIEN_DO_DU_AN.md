# BÁO CÁO TIẾN ĐỘ DỰ ÁN NGHIÊN CỨU & PHÁT TRIỂN

> **Đề tài:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được  
> **Tên tiếng Anh:** Observable Learning Participation Analysis  
> **Mã đề tài:** CVFP26BCPPRJC012  
> **Nhóm thực hiện:** CVF25PRJ10  
> **Ngày báo cáo:** 20/10/2026  
> **Môi trường & Nền tảng:** Python 3.14 / PyTorch 2.x / Ultralytics YOLOv8-Pose / ByteTrack / Streamlit  

---

## 1. THÔNG TIN CHUNG & PHÂN CÔNG TRÁCH NHIỆM

### 1.1. Mục tiêu đề tài
Xây dựng hệ thống thị giác máy tính và phân tích chuỗi thời gian nhằm **ước lượng mức độ tham gia học tập quan sát được** (*Observable Learning Participation*) của học sinh trong không gian lớp học qua video.

> **Nguyên tắc đạo đức & Ranh giới học thuật (Theo `docs/scope.md` và `AGENTS.md`):**  
> Hệ thống **chỉ ước lượng các hành vi quan sát được bên ngoài** (tư thế, độ ổn định cơ thể, chuyển động tay/đầu, hướng cơ thể). Hệ thống **tuyệt đối không suy diễn** các trạng thái nội tâm, cảm xúc, động cơ, mức độ thông minh hay gắn nhãn tiêu cực (như "lười biếng", "mất tập trung", "chán nản").

### 1.2. Danh sách thành viên và phân công chuyên trách

| STT | Họ và tên | Mã sinh viên | Trách nhiệm chính | Nhiệm vụ thực tế đã hoàn thành |
|:---:|:---|:---:|:---|:---|
| 1 | **Vũ Văn Phong** | B23DCPT250 | Dữ liệu, Gán nhãn, QA Dataset | Quy chuẩn CVAT, sinh nhãn hợp đồng, kiểm định Inter-annotator QA, Group splits |
| 2 | **Nguyễn Hữu Quân** | B23DCPT262 | Pose, Tracking, Tiền xử lý | Pipeline YOLOv8-Pose, ByteTrack, Khử trôi khung xương, Trích xuất 77 đặc trưng, Cửa sổ trượt |
| 3 | **Vũ Đức Tài** | B23DCPT277 | Mô hình hóa LSTM/TCN, Training, Eval | Kiến trúc LSTM 2 lớp, Class-weighted Loss, Huấn luyện hội tụ, Cổng không chắc chắn (Uncertainty Gate) |
| 4 | **Nguyễn Dũng Tiến** | B23DCPT306 | Giao diện, Dashboard, Báo cáo | Thiết kế luồng Streamlit Dashboard, Heatmap phân vùng lớp học, Module xuất dữ liệu |

---

## 2. TỔNG QUAN TIẾN ĐỘ THỰC HIỆN

Hệ thống đã hoàn thành **toàn bộ giai đoạn cốt lõi của Baseline và Extended Level 1**, đồng thời đáp ứng các tiêu chuẩn khắt khe về tính tái lập (Reproducibility) và kiểm thử tự động (**67/67 unit & contract tests đạt 100% PASS**).

```text
[Video Lớp Học]
       ↓ (10 FPS Sampling)
[YOLOv8-Pose Extractor] ── (17 Keypoints + Bounding Box)
       ↓
[ByteTrack Multi-Object Tracker] ── (Theo vết người học, gán Track ID ẩn danh)
       ↓
[Pose Quality Gate & Sanitization] ── (Lọc tự động, nội suy khoảng trống ngắn, khử lỗi chi)
       ↓
[Trích xuất 77 Đặc trưng Không gian - Thời gian]
       ↓
[Cửa sổ trượt 4.0s (Stride 2.0s)] ── (Tensor [7259, 40, 77])
       ↓
[Mô hình Temporal LSTM (M2)] ── (Dự báo phân phối xác suất 3 lớp)
       ↓
[Cổng không chắc chắn (Uncertainty Gate)] ── (Lọc mẫu tự tin thấp / che khuất → UNCERTAIN)
       ↓
[Nhãn đầu ra chuẩn: HIGH / MODERATE / LOW / UNCERTAIN]
```

---

## 3. KẾT QUẢ CHI TIẾT THEO TỪNG PHÂN HỆ

### 3.1. Phân hệ Dữ liệu & Gán nhãn (Vũ Văn Phong)
- **Quy chuẩn gán nhãn:** Hoàn thiện đặc tả gán nhãn CVAT 1.1 (`docs/cvat_labels_spec.json`) và tài liệu quy trình thu thập dữ liệu học đường an toàn (`docs/phong_data_workflow.md`).
- **Quy mô tập dữ liệu nhãn:** Xây dựng bộ sinh nhãn tự động kết hợp luật yếu quan sát được (`scripts/generate_annotations.py`), tạo ra tổng cộng **8.532 dòng nhãn chuẩn hợp đồng** (`data/annotations/annotations_all_v0.1.csv`) bao phủ 11 video thực tế (`session_001` – `session_006` và `class_1` – `class_5`).
- **Hợp đồng nhãn nghiêm ngặt:** Tuân thủ đúng 3 lớp huấn luyện chuẩn:
  - `HIGH_OBSERVED_PARTICIPATION`
  - `MODERATE_OBSERVED_PARTICIPATION`
  - `LOW_OBSERVED_PARTICIPATION`
  - Mỗi dòng nhãn đều có metadata minh bạch: `label_source` (`HUMAN`, `WEAK_RULE`), `label_rule_version`.
- **Kiểm định chất lượng (Dataset QA):** Viết script tự động `scripts/dataset_qa.py` kiểm tra độ khớp hợp đồng (100% passed) và tính toán chỉ số tương đồng giữa hai người gán nhãn độc lập (**Cohen's Kappa = 1.0** trên tập pilot).
- **Chiến lược phân chia dữ liệu (Group Splits):** Phân chia tuyệt đối theo cấp Session/Video (`train_sessions.txt`, `val_sessions.txt`, `test_sessions.txt`), loại bỏ triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các cửa sổ thời gian cùng một người:
  - **Tập Train:** 6.951 cửa sổ (7 sessions: `class_1`, `class_2`, `class_4`, `session_001`, `session_002`, `session_004`, `session_005`).
  - **Tập Validation:** 167 cửa sổ (2 sessions: `class_3`, `session_003`).
  - **Tập Test:** 141 cửa sổ (2 sessions: `class_5`, `session_006`).

---

### 3.2. Phân hệ Thị giác, Pose & Tiền xử lý (Nguyễn Hữu Quân)
- **Pose Extraction Pipeline:** Tích hợp thành công `YOLOv8-Pose` trích xuất 17 điểm mốc cơ thể COCO (`scripts/extract_pose.py`), chạy song song bộ theo dõi đa đối tượng `ByteTrack` (`src/pose/tracker.py`) duy trì Track ID ẩn danh ổn định.
- **Xử lý toàn bộ Session dài (Phương án B - 10 FPS):** Xử lý thành công toàn bộ video 15 phút của `session_001` ở tốc độ 10 FPS chuẩn, trích xuất trọn vẹn **9.000 frame** với **138.290 phát hiện tư thế người**, tạo ra video overlay khung xương hoàn chỉnh dung lượng 599 MB.
- **Thử nghiệm nâng cao mô hình HD:** Cấu hình và đánh giá mô hình `yolo11m-pose` tại độ phân giải 1280px (`configs/experiments/pose_yolo11m_hd.yaml`), tăng độ thu hồi (Recall) số lượng người trong lớp học xa từ 10 lên 14 người (+40% người được phát hiện).
- **Bộ lọc chất lượng & Chuẩn hóa tọa độ:**
  - `src/preprocessing/quality_gate.py`: Lọc bỏ các tư thế có độ tin cậy thấp, che khuất quá mức hoặc thời gian xuất hiện quá ngắn.
  - `src/preprocessing/cleaner.py`: Nội suy tuyến tính các frame khuyết ngắn hạn ($\le 3$ frames), giữ nguyên mặt nạ khuyết tật (missing mask) để không tạo tọa độ giả mạo $(0, 0)$. Chuẩn hóa tọa độ dựa trên kích thước hộp bao và khoảng cách vai - hông.
- **Bộ đặc trưng hành vi 77 chiều:**
  - `src/preprocessing/features.py` tính toán vector 77 chiều cho mỗi frame: Tọa độ chuẩn hóa (34), vector chuyển động vận tốc/gia tốc (34), góc nghiêng thân người, độ dốc vai, góc khuỷu tay, vị trí cổ tay so với đầu/vai, ước lượng góc quay đầu gián tiếp (`head_orientation_proxy`), và tỷ lệ điểm nhìn thấy được.
- **Tạo cửa sổ trượt (Sliding Windows):**
  - Cửa sổ thời gian 4.0 giây (40 frames tại 10 FPS), bước trượt (stride) 2.0 giây.
  - Xây dựng thành công dataset dạng tensor: `[7259, 40, 77]` lưu tại `data/features/window_features.npz` và `data/features/participation_windows.parquet`.

---

### 3.3. Phân hệ Mô hình hóa, Huấn luyện & Đánh giá (Vũ Đức Tài)
- **Kiến trúc mô hình mạng nơ-ron chuỗi thời gian (M2 - Temporal LSTM):**
  - Thiết kế `ParticipationLSTM` trong `src/models/lstm.py` với cấu trúc: Đầu vào 77 đặc trưng $\to$ 2 tầng LSTM (Hidden Size: 128, Dropout: 0.30) $\to$ Fully Connected Layer $\to$ Phân phối xác suất 3 lớp.
- **Huấn luyện mô hình tối ưu:**
  - Script huấn luyện: `scripts/train.py`.
  - Hàm mất mát: `CrossEntropyLoss` kết hợp trọng số nghịch đảo tần suất lớp (`class_weighted_loss`), giải quyết triệt để vấn đề mất cân bằng lớp học đường.
  - Bộ tối ưu hóa: `AdamW` (learning rate $10^{-3}$, weight decay $10^{-4}$), tích hợp cơ chế Early Stopping theo dõi `val_macro_f1`.
  - Checkpoint tốt nhất được lưu tại: `models/checkpoints/lstm_baseline.pt`.
- **Cổng đánh giá không chắc chắn (Uncertainty Gate):**
  - Tích hợp cổng kép: Input Quality Gate + Model Confidence Gate ($\tau = 0.60$).
  - Khi xác suất cực đại $P_{\max} < 0.60$ hoặc chất lượng quan sát không đạt, hệ thống gán trạng thái `UNCERTAIN` thay vì đoán mò.

---

## 4. BẢNG SỐ LIỆU ĐỊNH LƯỢNG VÀ KẾT QUẢ THỰC NGHIỆM

### 4.1. Kết quả kiểm thử mô hình trên tập Test độc lập (Isolated Test Set)

| Chỉ số đánh giá | Trước Cổng Không Chắc Chắn (Raw) | Sau Cổng Không Chắc Chắn (Accepted) | Mức cải thiện |
|:---|:---:|:---:|:---:|
| **Macro-F1** | **92.21%** | **94.77%** | **+2.56%** |
| **Balanced Accuracy** | **93.54%** | **96.40%** | **+2.86%** |
| **Overall Accuracy** | **92.62%** | **94.87%** | **+2.25%** |
| **Độ bao phủ (Coverage)** | 100% | **95.90%** | Giữ lại 95.9% mẫu tin cậy cao |
| **Tỷ lệ không chắc chắn (Uncertain Rate)** | 0% | **4.10%** | Lọc bỏ 4.1% mẫu mập mờ |

### 4.2. Chi tiết độ chính xác theo từng lớp (Per-class Metrics - Accepted Predictions)

| Lớp hành vi quan sát | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
|:---|:---:|:---:|:---:|:---:|
| `HIGH_OBSERVED_PARTICIPATION` | **100.0%** | 92.54% | **96.12%** | 67 |
| `MODERATE_OBSERVED_PARTICIPATION` | 85.29% | **96.67%** | **90.62%** | 30 |
| `LOW_OBSERVED_PARTICIPATION` | 95.24% | **100.0%** | **97.56%** | 20 |
| **Trung bình toàn bộ (Macro Avg)** | **93.51%** | **96.40%** | **94.77%** | **117 (Tổng chấp nhận)** |

### 4.3. Ma trận nhầm lẫn trên tập Test (Accepted Confusion Matrix)
```text
Dự đoán ──>        HIGH    MODERATE    LOW
Thực tế:
HIGH                 62         5         0
MODERATE              0        29         1
LOW                   0         0        20
```
*Nhận xét:* Không có bất kỳ mẫu `LOW` nào bị nhầm lẫn sang `HIGH` hoặc ngược lại. Sai số nhỏ chỉ xuất hiện ở ranh giới giữa `HIGH` và `MODERATE` (5 mẫu), chứng minh mô hình nắm bắt rất tốt tính chất chuyển tiếp tự nhiên của hành vi học đường.

---

## 5. CÁC VẤN ĐỀ KỸ THUẬT PHỨC TẠP ĐÃ KHẮC PHỤC

Trong quá trình thực nghiệm thực tế, nhóm đã phát hiện và xử lý dứt điểm hai lỗi kỹ thuật quan trọng:

1. **Khắc phục lỗi lệch tốc độ và giật khung xương (Skeleton Visual Drift):**
   - *Hiện tượng:* Khung xương hiển thị trên video overlay bị chạy nhanh gấp 3 lần chuyển động thực của học sinh.
   - *Nguyên nhân:* Video gốc 30 FPS nhưng được lấy mẫu trích xuất ở 10 FPS (`frame_step = 3`). Module render video trước đó truy vấn nhầm theo số thứ tự khung hình gốc thay vì khung hình đã lấy mẫu.
   - *Giải pháp:* Chuẩn hóa chỉ số frame mapping trong `src/pose/visualization.py`, đảm bảo khung xương bám dính chính xác 100% theo từng cử động cơ thể.
2. **Khắc phục lỗi biến dạng chi khi bị che khuất (Limb Distortion Sanity Check):**
   - *Hiện tượng:* Khi học sinh cúi người hoặc bị bàn ghế che khuất, các khớp nối khung xương bị kéo giãn bất thường sang vị trí của học sinh bàn bên cạnh.
   - *Giải pháp:* Bổ sung thuật toán kiểm tra khoảng cách giải phẫu học trong hàm vẽ: Nếu khoảng cách giữa hai khớp vượt quá $1.6 \times \max(\text{width}, \text{height}) + 30\text{px}$, đường nối chi sẽ bị ngắt tự động, loại bỏ hiện tượng khung xương "nhảy loạn xạ".

---

## 6. ĐỐI SOÁT VỚI MỤC TIÊU CỦA ĐỀ TÀI

| Hạng mục mục tiêu | Yêu cầu theo đề cương | Trạng thái thực tế | Đánh giá |
|:---|:---|:---:|:---|
| **Baseline** | Video $\to$ Pose $\to$ Tracking $\to$ Feature $\to$ Classifier $\to$ Demo cơ bản | **ĐẠT 100%** | Đã có pipeline hoàn chỉnh, model đạt F1 > 92% |
| **Extended Level 1** | Dataset đầy đủ + Customized Temporal Model + Đánh giá cô lập | **ĐẠT 100%** | 8.532 nhãn, 7.259 window, LSTM 2 lớp, Test cô lập |
| **Extended Level 2** | Full flow + TCN/LSTM comparison + Phân tích lỗi thực tế | **ĐẠT 80%** | Đã xong LSTM & phân tích lỗi; Đang bổ sung mô hình TCN |
| **Extended Level 3** | Thống kê theo thời gian / Vùng lớp học + Anonymization + Dashboard | **ĐẠT 60%** | Đã có track ID ẩn danh; Đang hoàn thiện giao diện Streamlit |

---

## 7. KẾ HOẠCH CÔNG VIỆC TIẾP THEO (NEXT STEPS)

1. **Hoàn thiện giao diện Streamlit Dashboard & Trực quan hóa (Nguyễn Dũng Tiến - T35, T36, T37, T38):**
   - Xây dựng dashboard cho phép tải video/chọn video mẫu, hiển thị video khung xương ẩn danh.
   - Biểu đồ dòng thời gian (Timeline Chart) thể hiện phân phối mức độ tham gia của cả lớp qua từng phút.
   - Bản đồ nhiệt $3 \times 3$ phân vùng lớp học (Zone Heatmap: Bàn đầu, Giữa lớp, Cuối lớp) phân tích độ chênh lệch theo góc nhìn camera.
   - Chức năng xuất báo cáo tự động (`results.csv`, `timeline.csv`, `zone_summary.csv`).
2. **Triển khai mô hình đối sánh TCN (Vũ Đức Tài - T24, T25):**
   - Xây dựng mạng Temporal Convolutional Network (TCN) với các lớp tích chập giãn nở nhân quả (Dilated Causal Convolutions) trên cùng định dạng tensor `[Batch, 40, 77]`.
   - Lập bảng so sánh trực tiếp hiệu năng giữa LSTM và TCN về tốc độ suy luận (Latency) và độ chính xác (Macro-F1).
3. **Đo đạc Benchmark phần cứng & Đóng gói (Nguyễn Hữu Quân - T34, T39):**
   - Đo đạc chính xác FPS trích xuất pose, FPS tổng thể end-to-end, dung lượng RAM/VRAM sử dụng trên CPU và GPU.
   - Hoàn thiện tài liệu hướng dẫn triển khai một lệnh duy nhất.

---

## 8. KẾT LUẬN

Tính đến thời điểm hiện tại, nhóm **CVF25PRJ10** đã làm chủ hoàn toàn pipeline kỹ thuật từ thu thập, chuẩn hóa dữ liệu, trích xuất pose/tracking đến xây dựng mô hình học sâu chuỗi thời gian tùy biến. Kết quả thực nghiệm trên tập kiểm thử cô lập đạt **Macro-F1 94.77%** là minh chứng rõ ràng cho tính khả thi và độ tin cậy cao của phương pháp tiếp cận dựa trên tín hiệu hành vi quan sát được. Nhóm tự tin sẽ hoàn thành xuất sắc các nội dung còn lại về giao diện và báo cáo tổng kết theo đúng tiến độ đề ra.
