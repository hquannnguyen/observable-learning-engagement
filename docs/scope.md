# Scope & Terminology Specification — v1.0

## 1. Phạm vi bài toán

Hệ thống nhận **chuỗi video** và sử dụng các tín hiệu hành vi **có thể quan sát được** — chủ yếu là pose, chuyển động và diễn biến theo thời gian — để phân loại trạng thái tham gia học tập ở mức khái quát.

### Trong phạm vi
- Person pose estimation.
- Multi-object tracking bằng ID ẩn danh.
- Pose/motion feature extraction.
- Temporal modeling bằng LSTM/TCN.
- Classification HIGH/MODERATE/LOW.
- Uncertainty gate.
- Tổng hợp theo thời gian và vùng lớp học.
- Ẩn danh khi demo/report.

### Ngoài phạm vi baseline
- Face recognition để xác định danh tính.
- Suy luận cảm xúc như vui/buồn/chán.
- Suy luận động cơ, ý định hoặc mức độ hiểu bài.
- Gaze estimation chính xác nếu không có mô hình/ground truth gaze riêng.
- Chấm điểm hay xếp hạng cá nhân dựa trên video.

## 2. Input

### Video
- Định dạng ưu tiên: MP4.
- Camera cố định.
- Khuyến nghị 1080p, 25–30 FPS.
- Xử lý nội bộ có thể sampling 5–10 FPS.

### Context metadata
- `session_id`
- `camera_id`
- `class_phase`: `lecture`, `individual_work`, `discussion`, ...
- `zone_id`: Front/Middle/Back × Left/Center/Right

## 3. Output chính thức

### 3.1. HIGH_OBSERVED_PARTICIPATION
Cửa sổ thời gian có nhiều tín hiệu quan sát phù hợp rõ với hoạt động lớp hiện tại và chất lượng quan sát đủ.

### 3.2. MODERATE_OBSERVED_PARTICIPATION
Có một số tín hiệu phù hợp nhưng yếu, ít hoạt động hoặc không liên tục.

### 3.3. LOW_OBSERVED_PARTICIPATION
Trong một cửa sổ đủ dài có nhiều tín hiệu quan sát không phù hợp với hoạt động hiện tại. Không gán nguyên nhân tâm lý.

### 3.4. UNCERTAIN
Không đủ căn cứ để đưa ra một trong ba trạng thái trên do che khuất, thiếu keypoint, track ngắn, chuyển trạng thái hoặc model confidence thấp.

## 4. Training target vs system output

```text
Training target:
  HIGH_OBSERVED_PARTICIPATION / MODERATE_OBSERVED_PARTICIPATION / LOW_OBSERVED_PARTICIPATION

Quality state:
  VALID / OCCLUDED / TRANSITION / INVALID

Canonical system/API output:
  HIGH_OBSERVED_PARTICIPATION / MODERATE_OBSERVED_PARTICIPATION / LOW_OBSERVED_PARTICIPATION / UNCERTAIN

Display output:
  HIGH / MODERATE / LOW / UNCERTAIN
```

`UNCERTAIN` không được coi là hành vi/thái độ thứ tư của người học. Đây là **trạng thái bất định của hệ thống**.

## 5. Quy tắc ngôn ngữ

### Cho phép
- "Hệ thống ước lượng mức độ tham gia quan sát được."
- "Pose quality thấp tại vùng cuối lớp."
- "Cửa sổ được gắn UNCERTAIN do thiếu keypoint."
- "head-orientation proxy"

### Không cho phép
- "Sinh viên A đang chán."
- "Sinh viên B không chú ý."
- "Sinh viên C lười."
- "Model biết sinh viên đang nghĩ gì."

## 6. Customized model

Customized model là **mô hình temporal LSTM/TCN do nhóm tự thiết kế cấu trúc đầu ra, huấn luyện và đánh giá trên dataset của đề tài**.

YOLOv8-Pose không được tuyên bố là model nhóm tự huấn luyện nếu chỉ sử dụng pretrained weights.

## 7. Definition of Done cho scope

Scope được coi là đóng băng v1.0 khi:
1. Cả nhóm chấp nhận input/output nêu trên.
2. README dùng thống nhất terminology.
3. Annotation guide không chứa emotion/intention labels.
4. UI/report không diễn đạt `LOW` thành kết luận tâm lý.
5. Mọi thay đổi scope sau đó được ghi vào changelog hoặc commit.
