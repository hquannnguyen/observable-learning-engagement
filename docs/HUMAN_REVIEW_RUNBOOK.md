# Gán nhãn độc lập và hoàn thiện đánh giá

Ngày 09/11/2026. Các công cụ dưới đây không tự tạo nhãn HUMAN.

## 1 Dữ liệu và trạng thái

Dataset `data/features/` và các checkpoint ngày 25/10 là baseline lịch sử, giữ nguyên để đối chiếu. Các điểm số ở đó là weak-label agreement.

Dataset mới `data/features_context_v0_2/` được tái tạo từ pose gốc, giữ nguyên ba file split. Nó loại cửa sổ bắc qua frame mất, tách camera và lấy phase theo khoảng thời gian. Metadata cắt video cũ tự luân phiên phase nên chưa được xác minh. Mặc định phase hiệu lực của những khai báo này là UNKNOWN; không sinh nhãn weak khi chưa biết phase. Đây là bộ dữ liệu chờ review, chưa phải bộ train đã nghiệm thu.

Không bật `phase_verified=true` cho cả session chỉ vì một vài cửa sổ trông giống một hoạt động. Người review phải kiểm tra timeline thực tế, ghi các khoảng không chồng lấn và người/ngày review. Không đổi split sau khi xem kết quả.

## 2 Hai video ngoài bộ train

Hai file do người dùng cung cấp ngày 09/11 đã được sao chép local vào:

- `data/raw/external_106705-673786412_medium/cam_A.mp4`
- `data/raw/external_215475_medium/cam_A.mp4`

Chúng không có trong danh sách session train/val/test cũ. Đây là bằng chứng chạy pipeline trên file mới; chưa chứng minh độc lập về người, bối cảnh hoặc quyền công bố. Metadata lưu hash nguồn và để trạng thái chưa xác minh. Không đưa video gốc vào Git hoặc gói chia sẻ công khai.

Luồng demo sử dụng `configs/experiments/external_demo_yolo11m.yaml`: YOLO11m-Pose, imgsz 1280, confidence 0.35, tracking ByteTrack, checkpoint LSTM full v0.1. Không huấn luyện lại hoặc chọn threshold theo các video này.

## 3 Gán nhãn bằng giao diện

Từ thư mục gốc dự án:

```powershell
python -m streamlit run app/annotation_app.py --server.port 8502
```

Mở http://127.0.0.1:8502, chọn đúng tên ở **Bạn là người gán nào?**, rồi chọn **Bộ video cần gán**:

- **Dataset cũ · bộ gán nhãn chính** (mặc định): `human_review_v0_2`, 600 cửa sổ mỗi người từ 11 session/video của dataset gốc, gồm 200 train / 100 val / 300 test. Đây là mẫu review từ bộ cửa sổ đã sửa context, không phải toàn bộ 32.959 cửa sổ baseline. Phiếu ẩn nhãn weak cũ để người gán xác minh độc lập; không thay đổi các split hoặc dữ liệu baseline.
- **Hai video mới · đánh giá bổ sung**: `external_review_20261109`, 130 cửa sổ mỗi người từ hai video người dùng cung cấp. Dùng kiểm tra trên dữ liệu mới, không thay thế bộ cũ.

Trong mỗi bộ, tên người gán được ánh xạ riêng:

- **Quân** → `annotator_a.csv`; khi lưu ghi `annotator_id=Nguyễn Hữu Quân`.
- **Phong** → `annotator_b.csv`; khi lưu ghi `annotator_id=Vũ Văn Phong`.

Đơn vị gán là **một người (track ID) × một cửa sổ khoảng 4 giây**. Ví dụ `track 7, 10–13,9 giây` chỉ gán người mang ID 7 trong khoảng đó, không gán chung cho cả lớp. Cùng một người có thể có nhiều cửa sổ và nhãn thay đổi theo thời gian. Số cửa sổ không phải số người; ID chỉ có ý nghĩa trong session/camera tương ứng.

Tool chỉ hiển thị phiếu của người đang chọn, không hiển thị kết quả mô hình. Nhãn đã lưu ở một bộ được giữ nguyên khi chuyển sang bộ khác. Lựa chọn tên phục vụ ghi nhận người thao tác, không phải cơ chế đăng nhập hoặc phân quyền.

App mở tại cửa sổ đầu tiên chưa gán (ưu tiên train/validation để thử tiêu chí), hiển thị tiến độ và đánh dấu ✓ các cửa sổ đã lưu. Dùng **Lưu và đến cửa sổ chưa gán tiếp** để làm liên tục, hoặc **Lưu tại cửa sổ này** để sửa và giữ nguyên vị trí. Có nút trước/tiếp và đến cửa sổ chưa gán. Chuyển cửa sổ mà chưa bấm lưu sẽ bỏ thay đổi chưa gửi. Chỉ thao tác lưu hợp lệ mới ghi tên người gán; dòng VALID có nguồn HUMAN. Các dòng khác được giữ nguyên.

Clip gán nhãn được cắt đúng khoảng từ video gốc, không dùng các overlay cũ có thể chỉ dài 15 giây. Người cần xem có **khung vàng**, sáng hơn người xung quanh; bên phải phóng to cùng người đó. Khi track không xuất hiện, clip báo không nhìn thấy và không bám theo người khác để đoán. Clip lặp khoảng 4 giây, có thời gian gốc ở chân hình. Nếu không định vị được, không gán mức tham gia. Tọa độ lấy từ `data/poses/*_poses.csv` hoặc `poses.csv` cạnh preview của lượt inference.

Hai lượt inference cũ chưa lưu tọa độ được khôi phục bằng `scripts/prepare_annotation_tracks.py --inference-dir ...`, kiểm tra đúng số detection và toàn bộ khóa window/track/time với prediction đã đóng băng trước khi xuất file. Hai bộ đã khôi phục lần lượt khớp 107 và 23 window; audit `.annotation_provenance.json` nằm cạnh CSV tọa độ. Không thay đổi prediction hoặc nhãn. Lượt inference mới tự lưu `poses.csv` để không phải khôi phục lại.

Đọc [tiêu chí thí điểm v0.3](ANNOTATION_RUBRIC_PILOT.md) và mở bảng ví dụ ngay trong tool. Ghi quan sát ở bốn khoảng nhỏ trước, sau đó ghi pha và căn cứ xác định nhiệm vụ, hành vi cụ thể, mức tham gia khớp bảng, độ tin cậy và xác nhận đã xem đúng người. Tool lưu bằng chứng vào `annotation_evidence`, căn cứ nhiệm vụ vào `phase_evidence`, đánh dấu `annotation_guide_version=v0.3-pilot`. Không tự nâng nhãn v0.2 thành v0.3. Bảng 3/4 là đề xuất vận hành để pilot, chưa phải tiêu chuẩn đã được kiểm chứng; cần thống nhất bằng nhóm luyện tập/pilot train/validation trước khi gán test chính thức. Không dùng điểm mô hình để chọn tiêu chí.

Bật **Xem thêm video gốc toàn cảnh để xác định nhiệm vụ (chỉ dùng local)** nếu cần ngữ cảnh trước/sau; video này chưa làm mờ. Chọn OCCLUDED/TRANSITION/INVALID khi che khuất, chuyển hoạt động, lỗi ID hoặc chưa đủ cơ sở (kể cả chưa biết nhiệm vụ). Không ép nhãn; UNKNOWN không phải pha đã xác minh.

Phiếu được lưu trên máy đang chạy Streamlit và giữ lại khi mở lại tool. Bấm **Tải CSV phiếu của tôi** để lấy bản sao bàn giao. Quân và Phong có thể dùng hai tab riêng, mỗi tab chọn đúng một tên; không so sánh nhãn trước khi cả hai hoàn tất. Trên máy khác, chạy tool cùng mã nguồn, video, file tọa độ trong `data/poses` và bộ phiếu tương ứng rồi bàn giao CSV riêng. Gói review không chứa video gốc hoặc toàn bộ tracking gốc. Không cần mở server ra Internet.

Lỗi pandas 3 khi ghi độ tin cậy `0.8` vào cột chuỗi đã được sửa: loader giữ các khóa dưới dạng văn bản, khai báo confidence dạng số nullable, rồi lưu theo `window_id` với thay thế file tạm. Mức che khuất được khôi phục đúng khi mở lại cửa sổ.

Nếu dùng CSV, giữ nguyên window_id, session/camera/track và timestamp. Điền đầy đủ schema trong template, không đổi khóa. HUMAN/ADJUDICATED để trống label_rule_version. Với trạng thái khác VALID, để trống participation_label và label_source.

## 4 Kiểm tra và phân xử

```powershell
python scripts/dataset_qa.py --validate data/annotations/external_review_20261109/annotator_a.csv --require-human
python scripts/dataset_qa.py --validate data/annotations/external_review_20261109/annotator_b.csv --require-human
python scripts/dataset_qa.py --annotator-a data/annotations/external_review_20261109/annotator_a.csv --annotator-b data/annotations/external_review_20261109/annotator_b.csv --strict-human --min-kappa 0.60 --report-json reports/human_qa_20261109.json --adjudication-out data/annotations/external_review_20261109/disagreements.csv
```

Phiếu trống sẽ bị từ chối. Script strict kiểm tra schema và nguồn nhãn, từ chối khóa trùng và đánh dấu hai file giống nhau theo khóa/nhãn để cần kiểm tra tính độc lập. Ngưỡng 0.60 là tiêu chí review đã chọn, không phải kết quả đã đạt. Số mẫu nhỏ không bảo đảm một sai số thống kê cố định cho Macro-F1.

Người phân xử xem lại bất đồng và tạo `adjudicated.csv` có nguồn ADJUDICATED cho các nhãn VALID. Không sinh file này bằng cách sao chép nhãn mô hình. Lưu hash file, người phân xử và ngày đóng băng.

## 5 Đánh giá trên nhãn người

Chuẩn bị file gộp `predictions.csv` của hai video trong thư mục review bằng công cụ bàn giao. Khi đã có nhãn hoàn tất:

```powershell
python scripts/evaluate_human.py --predictions data/annotations/external_review_20261109/frozen_predictions.csv --annotations data/annotations/external_review_20261109/adjudicated.csv --out-dir reports/human_evaluation_20261109
```

Lệnh kiểm tra schema, nguồn nhãn, window ID và các khóa thời gian. Nó không đánh giá phiếu trống hoặc nhãn WEAK_RULE như ground truth người. Đầu ra có metric raw/accepted, coverage, hash đầu vào và bảng theo session/zone. Các nhóm ít mẫu được cảnh báo; không kết luận bias nhân quả hoặc xếp hạng người học từ bảng này.

Gán nhãn độc lập tập đánh giá không yêu cầu train lại trước. Nếu sau đó dùng nhãn train để huấn luyện mới, giữ test đóng băng và tạo experiment ID/version mới. Không gọi run cũ là run đã train trên dataset v0.2.

## 6 Demo và bàn giao

```powershell
python -m streamlit run app/streamlit_app.py
```

Chọn bộ trích tư thế và checkpoint, upload video, chạy phân tích rồi xem preview/summary. Video mới tự tạo preview làm mờ vùng đầu detection. Vẫn cần kiểm tra người bị detector bỏ sót trước khi chia sẻ; blur theo detection không bảo đảm xóa hết khả năng nhận diện.

Gói `releases/cvf25prj10-review-20261109.zip` là gói review local, không phải tuyên bố nghiệm thu cuối. Manifest nêu rõ phần còn thiếu. Video gốc và lịch sử Git không được đưa vào gói. Giữ nguyên các dependency và đường dẫn nguồn được ghi trong manifest để tái lập.
