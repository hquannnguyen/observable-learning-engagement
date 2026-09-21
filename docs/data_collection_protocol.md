# Data Collection Protocol — CVF25PRJ10

**Task:** T03  
**Status:** Draft v0.1 — phải được nhóm review trước khi quay dataset chính.  
**Mục tiêu:** Chuẩn hóa cách thu video để dữ liệu đủ dùng cho pose/tracking/temporal modeling, đồng thời giảm rủi ro quyền riêng tư.

## 1. Phạm vi thu thập

Hệ thống chỉ phân tích **mức độ tham gia học tập quan sát được** từ tư thế/chuyển động theo thời gian. Không thu dữ liệu để nhận dạng danh tính, suy luận cảm xúc, động cơ hay ý định.

### Pilot bắt buộc trước dataset chính
- 6–10 người tham gia tự nguyện.
- 10–15 phút.
- Ít nhất 3 pha: `lecture`, `individual_work`, `discussion`.
- Một camera cố định là tối thiểu; camera B chỉ dùng nếu có điều kiện để đánh giá viewpoint bias.
- Chỉ khi pilot đạt Pose/Tracking Quality Gate mới mở rộng thu dataset chính.

## 2. Thiết lập camera

| Hạng mục | Thiết lập khuyến nghị | Lý do |
|---|---|---|
| Resolution | 1920×1080 | Giữ đủ chi tiết keypoint ở xa |
| FPS quay | 25–30 FPS | Video nguồn ổn định; pipeline có thể sampling xuống |
| Sample FPS baseline | 10 FPS | Theo `configs/baseline.yaml` |
| Camera | Cố định/tripod | Giảm rung và thay đổi scale |
| Vị trí | Cao hơn tầm đầu, chéo trước/lệch nhẹ | Thấy thân trên và tay của nhiều người |
| Zoom | Cố định trong session | Tránh scale thay đổi không kiểm soát |
| Âm thanh | Tắt nếu không cần | Giảm dữ liệu riêng tư |
| Ánh sáng | Bình thường; ghi chú điều kiện | Phục vụ quality/bias analysis |

## 3. Zone lớp học

Dùng lưới 3×3:

```text
FRONT:   F-L | F-C | F-R
MIDDLE:  M-L | M-C | M-R
BACK:    B-L | B-C | B-R
```

Mỗi session nên đổi vị trí người tham gia khi khả thi để giảm confounding giữa **người** và **zone**.

## 4. Metadata bắt buộc

Mỗi session phải có `metadata.json` theo `docs/dataset_schema.md` và ví dụ tại `data/sample/metadata.example.json`.

Tối thiểu:
- `session_id`
- `date`
- `camera_id`
- `fps`
- `resolution`
- `camera_view`
- `participants`
- `class_phase`
- `lighting_condition`
- `consent_confirmed`
- `notes`

## 5. Quy trình trước khi quay

1. Giải thích mục đích đề tài và loại dữ liệu được ghi.
2. Xác nhận người tham gia đồng ý theo quy trình của môn/đơn vị.
3. Không đưa người không đồng ý vào khung hình; nếu không thể đảm bảo, không sử dụng đoạn đó.
4. Kiểm tra camera, dung lượng, pin/nguồn, khung hình, ánh sáng.
5. Xác định zone và phase timeline dự kiến.
6. Quay test 30–60 giây và kiểm tra nhanh pose ở vị trí gần/giữa/xa.

## 6. Quy trình trong khi quay

- Không thay đổi camera/zoom giữa session nếu không có lý do và không ghi log.
- Ghi mốc thời gian khi chuyển phase hoạt động.
- Nếu camera bị dịch/chặn, ghi chú thời điểm.
- Không yêu cầu người tham gia diễn cảm xúc hoặc hành vi mang tính tâm lý; chỉ tổ chức hoạt động học có thể quan sát.

## 7. Quy trình sau khi quay

1. Đặt file theo convention:

```text
data/raw/session_001/cam_A.mp4
/data/raw/session_001/metadata.json
```

2. Kiểm tra video mở được, duration hợp lệ, metadata đầy đủ.
3. Lưu raw video ở nơi hạn chế truy cập; **không commit raw video lên GitHub public**.
4. Chạy pilot pose/tracking quality trước khi thu thêm dữ liệu lớn.
5. Sau khi có pose/features, ưu tiên dùng dữ liệu dẫn xuất cho phát triển thường ngày.

## 8. Pilot Acceptance Gate

Pilot chỉ được coi là đạt khi:
- camera thấy thân trên + tay của phần lớn vị trí mục tiêu;
- output pose có đúng bbox + 17 keypoints + confidence;
- `visible_ratio`/mean confidence được đo cho Front/Middle/Back;
- tracking có thể giữ ID tương đối ổn định trên đoạn kiểm tra;
- không có lỗi metadata nghiêm trọng;
- nhóm có quyết định rõ: **GO / ADJUST CAMERA / RE-RECORD**.

Không đặt trước một con số chất lượng “đẹp” nếu chưa đo pilot. Giá trị threshold dùng để lọc đầu vào lấy từ config và sẽ được tune trên validation.

## 9. Checklist T03

- [ ] Mục đích ghi hình đã được thông báo.
- [ ] Có xác nhận đồng thuận phù hợp.
- [ ] Không thu mapping track ID ↔ tên/MSSV nếu không cần.
- [ ] Camera nhìn thấy phần thân cần thiết.
- [ ] Có zone layout.
- [ ] Có phase timeline.
- [ ] Có metadata template.
- [ ] Có nơi lưu raw video hạn chế truy cập.
- [ ] Raw video không được public trong repo.
- [ ] Có quy trình anonymization cho demo/report.

## 10. Owner / Review

- Owner: **Vũ Văn Phong**
- Hỗ trợ: **Nguyễn Dũng Tiến**
- Review kỹ thuật pose/camera: **Nguyễn Hữu Quân**
