# Kiểm tra nhãn người — 24/11/2026

## Phạm vi và kết luận

Đọc trực tiếp hai file người dùng cung cấp, đối chiếu với pose YOLO11 và bảng cửa sổ hiện có. Không thay đổi file nguồn, không cập nhật tensor, không tự xác nhận nội dung hành vi trong video.

| Chỉ tiêu | class_1 | session_006 |
|---|---:|---:|
| Tổng dòng | 84 | 308 |
| VALID / HUMAN | 83 | 248 |
| OCCLUDED, nhãn và nguồn để trống đúng contract | 1 | 60 |
| HIGH | 46 | 21 |
| MODERATE | 25 | 195 |
| LOW | 12 | 32 |
| Khoảng thời gian được ghi (ms) | 0–14900 | 0–60000 |
| Track ID khác nhau | 14 | 22 |
| Lỗi state/label/source/rule theo validator | 0 | 0 |
| Lỗi thời gian/confidence hoặc khóa trùng | 0 | 0 |

Các dòng đều dài 4000 ms. class_1_labels.csv và class_1_labels_final.csv giống nhau từng byte: không phải hai bộ nhãn độc lập. Mỗi bộ chỉ có mã annotator HUMAN_ANNOTATOR; chưa có dữ liệu để đo đồng thuận/Kappa. Nhãn ghi guide v0.2; không tự chuyển thành v0.3-pilot.

## Các điểm cần xử lý trước khi tích hợp

1. **Định danh:** class_1 ghi session_id=raw, camera_id=class_1; pose hiện có ghi class_1/cam_A. session_006 ghi camera_id=A; pose ghi cam_A. Đây là khác biệt kỹ thuật cần ánh xạ có ghi nhận; track ID xuất hiện trong pose không chứng minh chắc chắn cùng người nếu annotation dùng một lần tracking khác.
2. **Thiếu window_id:** cả hai file sẽ bị loader nhãn người hiện tại trong scripts/build_dataset.py bỏ qua vì thiếu trường này. Không chỉ chép file vào thư mục human rồi chạy train.
3. **Khác cửa sổ:** không dòng nào khớp hoàn toàn cả start/end với cửa sổ hiện có, ngay cả sau ánh xạ định danh. Ví dụ annotation 0–4000, tensor có 0–3900 (40 mẫu @10 FPS). Không sửa timestamp bằng cách trừ 100 ms hàng loạt.
4. **Đối chiếu gần, chỉ phục vụ kiểm tra:** yêu cầu cùng track và chênh lệch từng đầu mút <=150 ms tìm được 70/84 cửa sổ class_1, 197/308 session_006; không có window đích trùng trong nhóm ứng viên này. Đây chưa phải phép ghép đã phê duyệt. Các dòng còn lại gồm lệch lưới cửa sổ, đoạn cuối hoặc khác quá trình xử lý track; cần rà soát hoặc dựng tensor theo chính khoảng được gán.
5. **Quality và nhãn người:** cả 70 ứng viên gần của class_1 đang INVALID trong tensor cũ; 69 trong đó được người gán VALID. Cần xác minh phase lecture và tách nguyên nhân UNKNOWN_PHASE khỏi lỗi chất lượng pose, rồi dựng lại artifact có version. Với session_006 có 185 annotation VALID tìm được ứng viên gần, trong đó 158 có quality VALID hiện tại. Không dùng nhãn người tự động bỏ qua quality gate.
6. **Pose trong khoảng:** mọi track đều tồn tại và mọi dòng có ít nhất một mẫu pose trong thời gian gán. class_1 có 27–40 mẫu/khoảng; session_006 có 18 dòng dưới 20 mẫu/khoảng, trong đó 13 dòng annotation VALID. Cần xem lại đúng người và khả năng che khuất/gap; số mẫu thấp chưa tự chứng minh nhãn người sai. CSV alignment ghi các dòng cụ thể.
7. **Phase mâu thuẫn:** session_006 track 2, 0–4000 ms ghi lecture; cùng khoảng các track khác ghi discussion. Metadata cũ ghi discussion toàn video nhưng chưa được xác minh độc lập, do đó cần xem ngữ cảnh để sửa có bằng chứng; không mặc định metadata là đúng.
8. **Zone:** một số track có nhiều zone. Ví dụ class_1 track 1 chuyển M-C sang F-C sau 2 giây; session_006 track 2 chuyển F-R sang M-C ở giây 12. Có thể do di chuyển, đổi ID hoặc gán zone; phải xem video trước khi báo cáo theo vùng.

## Sử dụng tiếp theo

- Giữ nguyên hai file nguồn và SHA-256 đã lưu trong audit.json.
- Xác nhận annotation được tạo từ chính bộ track YOLO11 hiện tại; kiểm tra các trường hợp phase/zone/pose thưa bằng video.
- Tạo bản chuẩn hóa riêng có định danh canonical, window_id và báo cáo ghép. Với cửa sổ không trùng, chọn quy tắc overlap đã kiểm chứng hoặc dựng đặc trưng theo khoảng annotation, không ghép tùy tiện.
- Nhãn class_1 thuộc Train; session_006 thuộc Test theo split-v0.1. Không chuyển session_006 sang Train. 32 nhãn LOW ở đây là LOW do người gán trong một phút, chưa phải LOW đã tích hợp vào test tensor.
- Sau khi thống nhất hướng dẫn, gán độc lập một phần để QA/Kappa và phân xử bất đồng. Lưu người gán thực tế bằng mã riêng; mã HUMAN_ANNOTATOR hiện tại không đủ phân biệt hai người.
- Chỉ đánh giá mô hình sau khi chốt quy trình và subset. Một phút đầu của một session không đại diện cho toàn bộ Test; vẫn cần session_005 và các khoảng khác nếu mục tiêu là đánh giá trên hai session.

## Artifacts

- audit.json: thống kê và SHA-256 nguồn.
- class_1_alignment.csv, session_006_alignment.csv: đối chiếu từng dòng, dòng CSV nguồn đánh số từ 2.
- *_contract_issues.csv: rỗng ngoài header vì không có lỗi contract nhãn.
- Script tái chạy từ repository root: `python reports/audit_manual_annotations_20261124.py`.

Kiểm tra dữ liệu chạy thành công. Không chạy lại unit tests vì không sửa logic ứng dụng; kết quả này xác nhận cấu trúc và mức tương thích dữ liệu, không xác nhận độ đúng hành vi hoặc độ ổn định ID qua xem video.
