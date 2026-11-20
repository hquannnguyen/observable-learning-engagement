# Kết quả triển khai bổ sung ngày 09 tháng 11 năm 2026

Đã sửa các vấn đề có thể xử lý từ mã nguồn và dữ liệu hiện có. Chưa có người hoàn tất annotation nên chưa có accuracy/F1 độc lập, chưa train lại trên nhãn người và chưa xác nhận nghiệm thu đầy đủ Extended Level 1 hoặc Level 3.

## 1 Các thay đổi đã thực hiện

- Context phase được đọc theo timeline metadata. Metadata chưa xác minh được giữ riêng ở declared_class_phase, không mặc định thành lecture và không sinh nhãn weak mặc định.
- Script cắt video không tự luân phiên phase và không tự ghi đã có consent. Metadata mới ghi source recording và khoảng cắt để hỗ trợ kiểm tra nguồn.
- Window được tạo riêng theo session/camera/track, bỏ cửa sổ qua frame mất/gap, nhận phase và zone theo window. Builder từ chối pose trùng khóa và giữ nguyên file split.
- FPS/window/stride/ngưỡng track đi từ cấu hình; inference chuyển YAML pose tới extractor. Metadata ghi model pose, imgsz, tracker, thiết bị và trạng thái preview.
- Train ghi snapshot hiệu lực và từ chối split trống nhãn. TCN khôi phục đúng cấu hình lồng; chuẩn hóa pose được vector hóa và đối chiếu với đầu ra cũ trên 400 dòng thực.
- Template nhãn người đủ schema, lấy được window không nhãn; n-test=0 lấy toàn bộ test. QA từ chối phiếu trống khi yêu cầu HUMAN, phát hiện khóa trùng. Có app gán nhãn độc lập và CLI đánh giá human tách khỏi weak.
- Báo cáo vị trí tách human metrics và weak agreement, kèm N/chất lượng/coverage. Không gọi heatmap là bằng chứng bias.
- App chọn được YOLOv8n/YOLO11m, tạo preview làm mờ, hỗ trợ H.264 và hiển thị báo cáo vị trí. HUD lấy tên model thực dùng.
- Đính chính log Kappa synthetic và AdamW; giữ các số liệu model cũ dưới phạm vi weak-label lịch sử.

## 2 Dataset chờ review mới

Đường dẫn: `data/features_context_v0_2/`. Version: `v0.2-context-review`. Tensor `[27116,40,77]`.

| Split | Số cửa sổ |
|---|---:|
| Train | 11.188 |
| Validation | 10.013 |
| Test | 5.915 |
| Tổng | 27.116 |

Ba file split giữ nguyên SHA256 so với split-v0.1. Số window giảm so với 32.959 baseline do loại cửa sổ mất tính liên tục. 24.324 cửa sổ không được sinh nhãn do thiếu phase đã xác minh; 2.792 cửa sổ có visibility thấp. Tất cả target để trống (-1 trong NPZ). Không phải dữ liệu bị ép thành LOW/UNCERTAIN training class.

Metadata cũ có 10.433 cửa sổ khai báo individual_work, 9.881 lecture, 6.549 discussion, 253 không có metadata. Các giá trị này chưa được coi là pha thực tế vì script cũ sinh luân phiên. Cần người xem lại timeline. Mẫu review 600 cửa sổ nằm tại `data/annotations/human_review_v0_2/`.

Baseline `data/features/` và các checkpoint cũ vẫn giữ nguyên. Không dùng checkpoint cũ như thể đã train trên version mới.

## 3 Demo hai video do người dùng cung cấp

Hai video được sao chép local, không thêm vào split train/test cũ. Bộ trích pose thực dùng: `yolo11m-pose.pt`, imgsz 1280, confidence 0.35, ByteTrack. Bộ phân loại: `20261025-162855-lstm-full-v0-1.pt` (input 77).

| Video nguồn | Thời lượng nguồn | Frame lấy mẫu | Window | Accepted | Uncertain | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| 106705-673786412_medium.mp4 | 21,14 giây | 212 | 107 | 74 | 33 | 69,16% |
| 215475_medium.mp4 | 10,24 giây | 103 | 23 | 12 | 11 | 52,17% |

Đầu ra ở `reports/inference/external_106705_yolo11/` và `reports/inference/external_215475_yolo11/`: predictions, inference metadata, báo cáo vị trí, bundle CSV/JSON/Markdown và preview H.264. Không có ground truth để tính độ chính xác. Hai run CPU chạy đồng thời với tác vụ khác, có bước preview; thời gian trong metadata không phải benchmark cô lập và không so sánh trực tiếp với benchmark YOLOv8 cũ.

Preview được kiểm tra lại để mở rộng vùng làm mờ từ dải mắt ra toàn vùng đầu/mặt và sửa HUD từ tên YOLOv8 cố định sang YOLO11m thực dùng. Lượt bổ sung privacy dùng cùng YOLO11m tại imgsz 640 để phát hiện lại vùng đầu trên video nguồn; không thay đổi prediction/classifier của lượt chính imgsz 1280. Detector vẫn có thể bỏ sót nên chưa tuyên bố ẩn danh tuyệt đối.

## 4 Bộ gán nhãn và đánh giá

Bộ `data/annotations/external_review_20261109/` có 130 window, frozen_predictions.csv, manifest/hash nguồn và hai phiếu trống riêng cho annotator_a/annotator_b. Không có nhãn mô hình trong phiếu. Không có HUMAN giả hoặc điểm Kappa mới được tự sinh.

Chạy `python -m streamlit run app/annotation_app.py --server.port 8502`. Quân chọn tên Quân (phiếu A), Phong chọn Phong (phiếu B). Người gán xem đúng track/time, chọn pha và trạng thái thực tế rồi tự lưu nhãn. Tool hiển thị tiến độ, tiếp tục cửa sổ chưa gán, giữ lại mức che khuất và cho tải CSV. Đã sửa lỗi pandas 3 khi gán số 0.8 vào cột confidence ban đầu trống. Kiểm thử AppTest nhấn lưu, mở lại và bảo toàn phiếu còn lại trên dữ liệu tạm cho cả hai người; toàn bộ suite đạt 122 passed. Khi hoàn tất, thực hiện QA, phân xử và `scripts/evaluate_human.py` theo `docs/HUMAN_REVIEW_RUNBOOK.md`.

Đợt cập nhật tiếp theo cùng ngày: tool mở mặc định mẫu 600 cửa sổ từ video dataset cũ, giữ riêng 130 cửa sổ hai video ngoài. Clip từ video gốc tô khung vàng, làm tối người khác và phóng to đúng track; không dùng overlay cũ chỉ dài 15 giây cho cửa sổ muộn hơn. Thử trên đoạn 18,002–21,902 giây của session_005: đủ 40 frame, đúng track ở 40 frame. Bộ tiêu chí v0.3-pilot yêu cầu bốn khoảng bằng chứng, căn cứ nhiệm vụ, hành vi và xác nhận; ngưỡng 3/4 là đề xuất riêng cần pilot, chưa được kiểm chứng. Toàn bộ suite đạt 133 passed; kiểm tra trình duyệt phát clip H.264 4 giây, 1280×592, không tràn ngang mobile. Không tự tạo nhãn người hoặc số Kappa.

## 5 Kiểm chứng và phần còn thiếu

Kiểm thử đã chạy lại đầy đủ; dashboard và app gán nhãn kiểm tra bằng Streamlit AppTest không có exception. Bằng chứng số test cuối cùng được ghi trong PROJECT_STATUS và log triển khai.

Còn cần người thật: xác minh timeline, gán độc lập, phân xử, kiểm tra nguồn recording/người giữa các split và đánh giá theo vị trí. Chưa có benchmark video dài/GPU hoặc nghiệm thu cuối. Gói local review không chứa video gốc và không phải tuyên bố dataset đã được nghiệm thu.
