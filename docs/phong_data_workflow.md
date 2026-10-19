# Hướng Dẫn Quy Trình Dữ Liệu Chi Tiết — Vũ Văn Phong
**Dự án:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được (CVF25PRJ10)  
**Phụ trách:** Vũ Văn Phong (Data Collection / Annotation / Dataset QA)  
**Phối hợp:** Vũ Đức Tài (Double Annotation & Evaluation), Nguyễn Hữu Quân (Pose & Tracking)

---

## Sơ đồ luồng công việc tổng thể

```text
       VIDEO GỐC (IMG_3354.MOV - 79 phút)
                       │
                       ▼
            CHIA SESSION / CẮT VIDEO
         (scripts/split_video.py -> H.264)
                       │
                       ▼
                 GÁN METADATA
            (metadata.json cho từng session)
                       │
                       ▼
           XÂY DỰNG ANNOTATION GUIDE
         (docs/annotation_guide.md v0.2)
                       │
                       ▼
              ANNOTATION TRÊN CVAT
         (docs/cvat_labels_spec.json)
                       │
                       ▼
           DOUBLE ANNOTATION (15–20%)
         (Phong + Tài gán độc lập session_001)
                       │
                       ▼
             REVIEW / ADJUDICATION
        (scripts/dataset_qa.py -> Cohen's Kappa)
                       │
                       ▼
         BÀN GIAO DATASET v1 CHO QUÂN + TÀI
```

---

## BƯỚC 1: CHIA SESSION / CẮT VIDEO

### 1. Mục tiêu
Video gốc `IMG_3354.MOV` có thời lượng gần 80 phút, định dạng HEVC (H.265) từ iPhone. Định dạng này **không tương thích** với trình duyệt web và CVAT (gây giật lag/màn hình đen khi gán nhãn). Ta cần:
- Cắt thành các session 15 phút chuẩn theo protocol.
- Chuyển mã sang chuẩn **H.264 MP4 (+faststart)** để tải tức thì trên CVAT.

### 2. Cấu trúc thư mục sau khi cắt
```text
data/raw/
├── IMG_3354.MOV               # Video gốc 80 phút (không xóa, không push git)
├── session_001/
│   ├── cam_A.mp4              # 00:00:00 -> 00:15:00 (Pilot subset)
│   └── metadata.json
├── session_002/
│   ├── cam_A.mp4              # 00:15:00 -> 00:30:00
│   └── metadata.json
├── session_003/
│   ├── cam_A.mp4              # 00:30:00 -> 00:45:00
│   └── metadata.json
├── session_004/
│   ├── cam_A.mp4              # 00:45:00 -> 01:00:00
│   └── metadata.json
├── session_005/
│   ├── cam_A.mp4              # 01:00:00 -> 01:15:00
│   └── metadata.json
└── session_006/
    ├── cam_A.mp4              # 01:15:00 -> 01:19:10
    └── metadata.json
```

### 3. Lệnh thực hiện
Script `scripts/split_video.py` tự động xử lý toàn bộ quá trình:
```powershell
python scripts/split_video.py --video data/raw/IMG_3354.MOV --duration 15m
```

---

## BƯỚC 2: GÁN METADATA (`metadata.json`)

Mỗi thư mục `session_XXX` bắt buộc phải có một file `metadata.json` đúng chuẩn `docs/dataset_schema.md`:

```json
{
  "session_id": "session_001",
  "date": "2026-10-07",
  "camera_id": "A",
  "fps": 30,
  "resolution": "1920x1080",
  "camera_view": "front_oblique",
  "participants": 8,
  "class_phase": [
    {
      "name": "lecture",
      "start_ms": 0,
      "end_ms": 900000
    }
  ],
  "lighting_condition": "normal",
  "consent_confirmed": true,
  "notes": "Session 001 - Pilot testing subset recorded in classroom."
}
```

*Lưu ý cho Phong:* Nếu trong 15 phút có đổi hoạt động (ví dụ 10 phút đầu giảng bài, 5 phút sau làm bài tập), Phong chỉ cần mở file `metadata.json` và sửa lại timeline `class_phase`:
```json
"class_phase": [
  {"name": "lecture", "start_ms": 0, "end_ms": 600000},
  {"name": "individual_work", "start_ms": 600000, "end_ms": 900000}
]
```

---

## BƯỚC 3: XÂY DỰNG ANNOTATION GUIDE & KHUNG NHÃN

### 1. Ba nhãn mức độ tham gia (Canonical Labels)
Chỉ được dùng 3 nhãn sau, **không suy diễn tâm lý/cảm xúc**:
1. **`HIGH_OBSERVED_PARTICIPATION`**:
   - Hành vi phù hợp cao với pha hoạt động: hướng mắt/thân lên bảng trong `lecture`; đọc/viết liên tục trong `individual_work`; quay sang trao đổi, thảo luận tích cực trong `discussion`.
2. **`MODERATE_OBSERVED_PARTICIPATION`**:
   - Có biểu hiện tham gia nhưng không liên tục, tư thế thụ động, ít cử động nhưng không làm việc riêng hoặc gây gián đoạn.
3. **`LOW_OBSERVED_PARTICIPATION`**:
   - Hành vi ngược lại pha hoạt động trong phần lớn thời gian: quay lưng lên bảng, nằm gục xuống bàn trong giờ giảng, bấm điện thoại giấu dưới bàn, không tham gia khi nhóm thảo luận.

### 2. Trạng thái chất lượng (Annotation States)
- `VALID`: Rõ ràng, thấy đủ người, đủ điều kiện gán 1 trong 3 nhãn trên.
- `OCCLUDED`: Bị che khuất trên 40% cơ thể (ví dụ người khác đi ngang qua che mất). Để trống nhãn hành vi.
- `TRANSITION`: Đang chuyển đổi pha hoạt động (thầy cô thu bài, học sinh đổi chỗ). Để trống nhãn hành vi.
- `INVALID`: Học sinh đi ra ngoài khung hình camera, lỗi mờ nhòe. Để trống nhãn hành vi.

---

## BƯỚC 4: HƯỚNG DẪN ANNOTATION TRÊN CVAT

### 1. Khởi tạo Task trên CVAT
1. Truy cập CVAT (trên trình duyệt, ví dụ https://app.cvat.ai hoặc máy local).
2. Nhấn **Create a new task**.
3. Đặt tên: `CVF25_session_001_cam_A`.
4. Tại mục **Constructor / Labels**:
   - Chọn chuyển sang tab **Raw**.
   - Dán toàn bộ nội dung trong file [`docs/cvat_labels_spec.json`](../docs/cvat_labels_spec.json) vào.
   - CVAT sẽ tự động tạo đủ 3 nhãn kèm 7 thuộc tính (`atomic_behavior`, `class_phase`, `zone_id`, `annotation_state`, `occlusion_level`, `annotation_confidence`, `notes`).
5. Tại mục **Files**: Chọn tải lên file `data/raw/session_001/cam_A.mp4`.
6. Nhấn **Submit & Open**.

### 2. Cách gán nhãn cho từng học sinh
- Chọn công cụ **Draw new rectangle** (chọn chế độ **Track** để giữ nguyên ID của học sinh qua các frame).
- Vẽ bounding box bao quanh học sinh (từ đầu đến thân trên/vị trí ngồi).
- Chọn nhãn tương ứng (`HIGH`, `MODERATE`, `LOW`).
- Chọn thuộc tính phù hợp (`atomic_behavior`, `zone_id` ví dụ `F-C`, `M-L`).
- Kéo timeline: Nếu học sinh đổi hành vi ở frame nào, chỉ cần đổi nhãn tại frame đó; CVAT sẽ tự động phân đoạn track.

### 3. Xuất nhãn sau khi gán
- Nhấn **Export dataset** -> Chọn định dạng **CVAT for video 1.1** hoặc định dạng CSV.
- Lưu file kết quả về: `data/annotations/session_001_phong.csv`.

---

## BƯỚC 5: DOUBLE ANNOTATION 15–20%

Theo quy định bắt buộc của đồ án (Task T12):
- **Phạm vi double annotation:** Chọn `session_001` (15 phút / 79 phút gốc $\approx 19\%$).
- **Phân công:**
  - **Phong:** Gán toàn bộ `session_001` -> file `data/annotations/session_001_phong.csv`.
  - **Tài:** Gán độc lập `session_001` trên CVAT (không xem bài của Phong) -> file `data/annotations/session_001_tai.csv`.

---

## BƯỚC 6: REVIEW, ADJUDICATION VÀ KIỂM TRA CHẤT LƯỢNG (QA)

### 1. Kiểm tra tính hợp lệ của file nhãn
Chạy script QA để phát hiện lỗi định dạng, thiếu trường hoặc sai nhãn:
```powershell
python scripts/dataset_qa.py --validate data/annotations/session_001_phong.csv
```

### 2. Đo mức độ đồng thuận giữa Phong và Tài (Cohen's Kappa)
```powershell
python scripts/dataset_qa.py `
  --annotator-a data/annotations/session_001_phong.csv `
  --annotator-b data/annotations/session_001_tai.csv `
  --adjudication-out data/annotations/session_001_disagreements.csv
```

- **Mục tiêu:** Hệ số **Cohen's Kappa $\kappa \ge 0.60$** (đạt mức đồng thuận Substantial).
- **Adjudication:** Mở file `session_001_disagreements.csv` chứa các đoạn Phong và Tài chấm khác nhau:
  - Cả hai ngồi lại cùng xem video tại các đoạn đó.
  - Thống nhất nhãn cuối cùng (`label_source = "ADJUDICATED"`).
  - Cập nhật lại Annotation Guide nếu phát hiện trường hợp mơ hồ cần làm rõ.

---

## BƯỚC 7: BÀN GIAO CHO QUÂN VÀ TÀI

Sau khi hoàn thành Bước 6:
1. Đã có các session video sạch tại `data/raw/session_XXX/`.
2. Đã có file `metadata.json` được kiểm duyệt.
3. Đã có nhãn Pilot được đồng thuận cao tại `data/annotations/session_001_adjudicated.csv`.
4. Phong tiếp tục gán các session còn lại (`session_002` đến `session_006`) dựa trên guideline đã được chuẩn hóa.
5. Quân chạy trích xuất Pose/Tracking (`scripts/extract_pose.py`) và Tài chạy huấn luyện LSTM/TCN (`scripts/train.py`).
