# Privacy & Consent Guide — CVF25PRJ10

Tài liệu này là hướng dẫn kỹ thuật/quy trình nội bộ cho bài tập lớn; không thay thế quy định của PTIT hoặc yêu cầu pháp lý/đạo đức áp dụng tại nơi thu dữ liệu.

## 1. Privacy-by-design

- Không dùng face recognition.
- Không cần tên/MSSV trong pipeline phân tích.
- `track_id` chỉ là ID ẩn danh theo video/session.
- Không public raw video có người thật lên GitHub.
- Ưu tiên dùng pose/features dẫn xuất sau khi hoàn tất extraction.
- Demo/report ưu tiên skeleton-only; nếu dùng frame thật phải làm mờ vùng đầu phù hợp.

## 2. Trước khi ghi hình

Người tham gia cần được thông báo tối thiểu:
- mục đích nghiên cứu/bài tập;
- loại dữ liệu được thu (video, pose/keypoints, metadata);
- mục đích sử dụng (phát triển, đánh giá, demo/báo cáo);
- ai trong nhóm/giảng viên có thể truy cập;
- raw video có/không được dùng trong demo công khai;
- cách yêu cầu loại bỏ dữ liệu theo quy trình mà nhóm/đơn vị cho phép.

Nếu quay trong lớp học thật, phải tuân thủ yêu cầu của giảng viên/đơn vị trước khi ghi.

## 3. Access policy

| Loại dữ liệu | Truy cập | Public repo |
|---|---|---|
| Raw video | Nhóm được phép + người hướng dẫn khi cần | **Không** |
| Metadata không nhận dạng | Nhóm | Có thể, sau review |
| Pose/keypoints | Nhóm | Chỉ nếu không chứa mapping danh tính và được phép |
| Features/windows | Nhóm | Có thể sau review |
| Model checkpoint | Nhóm/Drive | Không bắt buộc commit |
| Aggregate report | Nhóm/giảng viên | Có thể nếu đã ẩn danh |

## 4. Public demo

Các mode ưu tiên:
1. **Skeleton-only** + anonymous track ID.
2. Blur/pixelate head ROI + bbox/skeleton.
3. Không hiển thị tên, MSSV hoặc mapping danh tính.

Không hiển thị câu kiểu “sinh viên X mất tập trung”. Report theo thời gian/zone hoặc anonymous track và luôn nêu giới hạn suy luận.

## 5. Data retention

Nhóm phải chốt trước dataset chính:
- nơi lưu raw video;
- ai có quyền truy cập;
- mốc xóa/archival sau khi môn học kết thúc nếu được yêu cầu;
- cách lưu backup.

Các trường này nên nằm trong project log hoặc tài liệu quản lý dữ liệu nội bộ, không đưa secret/access token vào repo.

## 6. Consent checklist

- [ ] Người tham gia được thông báo mục đích.
- [ ] Có xác nhận đồng thuận phù hợp với bối cảnh.
- [ ] Không quay người từ chối tham gia.
- [ ] Không tạo mapping danh tính không cần thiết.
- [ ] Biết nơi raw video được lưu.
- [ ] Biết demo sẽ ẩn danh như thế nào.
- [ ] Có owner chịu trách nhiệm data/privacy.
