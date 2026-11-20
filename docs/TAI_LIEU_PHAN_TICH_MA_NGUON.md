# Phân tích mã nguồn và quy trình triển khai hệ thống tham gia học tập quan sát được

**Dự án:** Observable Learning Engagement · CVF25PRJ10 · CVFP26BCPPRJC012  
**Ngày phân tích:** 09/11/2026  
**Thư mục dự án:** `D:\ThiGiacMayTinh\observable-learning-engagement`

Hệ thống xử lý video lớp học để ước lượng mức độ tham gia học tập quan sát được qua tư thế cơ thể và chuyển động. Phần cốt lõi đã có mã nguồn: trích xuất pose bằng **YOLOv8-Pose hoặc YOLO11-Pose**, theo dõi bằng ByteTrack, tạo đặc trưng theo thời gian, phân loại bằng LSTM hoặc TCN, lọc dự đoán thiếu tin cậy và trình bày kết quả trên Streamlit. Baseline mặc định là YOLOv8n-Pose; nhánh YOLO11m-Pose độ phân giải 1280 đã được tích hợp vào code, có trọng số và artifact overlay riêng trong thư mục dự án.

Tài liệu dành cho thành viên cần hiểu, thuyết trình, vận hành hoặc phát triển dự án. Mỗi giai đoạn trình bày mục đích, dữ liệu vào và ra, cách thực hiện, công nghệ và điểm cần kiểm tra. Các ký hiệu `[Sxx]` dẫn tới danh mục mã nguồn ở cuối tài liệu.

**Kết luận về mức độ hoàn thiện:** hệ thống đã có luồng suy luận trên video mới và các artifact thí nghiệm. Tuy nhiên, bộ đánh giá hiện hành dùng nhãn sinh bằng quy tắc. Các điểm Macro-F1 hiện tại phản ánh mức đồng thuận với nhãn quy tắc, chưa chứng minh độ chính xác trên nhãn người đánh giá độc lập. Bảng trạng thái dự án được cập nhật ngày 25/10/2026; các kết quả trích dẫn dưới đây là artifact đã lưu của lần đó.


**Cập nhật triển khai ngày 09/11/2026:** đã sửa context phase, cửa sổ qua gap, cấu hình suy luận, checkpoint TCN, snapshot train và công cụ nhãn người. Dataset chờ review mới có **27.116 cửa sổ**, được lưu riêng ở `data/features_context_v0_2/`; không sinh nhãn từ phase chưa xác minh. Tool gán nhãn mặc định dùng mẫu **600 cửa sổ theo từng người từ video dataset cũ**. Hai video mới do người dùng cung cấp là bộ bổ sung đã chạy YOLO11m-Pose + LSTM: tổng **130 cửa sổ**. Tool đánh dấu khung vàng/phóng to đúng người và yêu cầu ghi bằng chứng theo tiêu chí **v0.3-pilot** đang chờ thử nghiệm/thống nhất. Kiểm thử toàn bộ: **133 passed**. Chưa có nhãn HUMAN hoàn tất, nên chưa tính accuracy/F1 trên người thật hoặc train lại trên nhãn người. Hướng dẫn thực hiện tại `docs/HUMAN_REVIEW_RUNBOOK.md` và `docs/ANNOTATION_RUBRIC_PILOT.md`.

## 1 Tổng quan bài toán và kiến trúc

### 1.1 Hệ thống giải quyết vấn đề gì

Một ảnh đơn chỉ ghi nhận tư thế tại một thời điểm. Dự án dùng chuỗi quan sát để mô tả diễn biến trong một khoảng thời gian: vị trí các khớp, chuyển động tay, độ nghiêng vai, vị trí đầu so với vai và chất lượng quan sát. Mô hình temporal tổng hợp những tín hiệu này thành một trong ba mức tham gia quan sát được.

Đơn vị dự đoán là **một cửa sổ thời gian của một track**, không phải toàn bộ người học trong cả buổi. Track ID là mã theo dõi tạm thời trong video, không phải định danh cá nhân. Hệ thống không đo trực tiếp cảm xúc, động cơ, mức hiểu bài hoặc sự chú ý nội tại. Hình học đầu từ pose 2D chỉ là chỉ dấu gần đúng, không phải phép đo hướng nhìn chính xác.

### 1.2 Hợp đồng nhãn

| Chỉ số mô hình | Giá trị lưu trong dữ liệu và kết quả | Hiển thị |
|---|---|---|
| 0 | `HIGH_OBSERVED_PARTICIPATION` | HIGH |
| 1 | `MODERATE_OBSERVED_PARTICIPATION` | MODERATE |
| 2 | `LOW_OBSERVED_PARTICIPATION` | LOW |
| Không có | `UNCERTAIN` | UNCERTAIN |

`UNCERTAIN` do tầng kiểm tra chất lượng và độ tin cậy tạo ra; mô hình chính chỉ học ba lớp. Nhãn LOW không đủ căn cứ để kết luận người học lười, chán hoặc không chú ý. Quy ước nhãn nằm tập trung trong `src/contracts/labels.py`, gồm mapping chỉ số, nhãn hiển thị và kiểm tra nguồn gốc nhãn. [S01]

Nguồn nhãn gồm `HUMAN` do người gán, `WEAK_RULE` do quy tắc sinh và `ADJUDICATED` sau phân xử. Nhãn quy tắc bắt buộc có `label_rule_version`. Trạng thái chất lượng gồm `VALID`, `OCCLUDED`, `TRANSITION`, `INVALID`; chỉ trạng thái VALID được mang nhãn huấn luyện.

### 1.3 Hai luồng hoạt động

```text
LUỒNG XÂY DỰNG MÔ HÌNH
Video được phép sử dụng + metadata
  -> Trích pose và tracking
  -> Làm sạch và chuẩn hóa
  -> Đặc trưng và cửa sổ thời gian
  -> Gán nhãn + chia theo session
  -> Huấn luyện LSTM hoặc TCN
  -> Checkpoint + đánh giá + nhật ký thí nghiệm

LUỒNG SUY LUẬN
Video mới + checkpoint
  -> Trích pose và tracking
  -> Gán zone + làm sạch
  -> Đặc trưng và cửa sổ tương thích checkpoint
  -> Xác suất ba lớp
  -> Uncertainty gate
  -> Kết quả từng cửa sổ
  -> Tổng hợp theo thời gian và vùng
  -> Dashboard + CSV + JSON + báo cáo Markdown
```

YOLOv8-Pose và YOLO11-Pose là các lựa chọn cho bộ trích xuất pose pretrained. Phần mô hình nhóm thiết kế và huấn luyện là bộ phân loại temporal LSTM/TCN. Không có bước fine-tune YOLO trong các script huấn luyện hiện tại. Tên module và một số docstring còn ghi YOLOv8 nhưng implementation nạp model theo đường dẫn trọng số, không cố định chỉ YOLOv8. [S02, S08, S09, S10, S26]

### 1.4 Cấu trúc thư mục và trách nhiệm

| Thư mục | Trách nhiệm |
|---|---|
| `configs/` | Cấu hình baseline, biến thể thí nghiệm, lưới zone |
| `data/raw/` | Video và metadata phiên quay |
| `data/poses/` | Pose, track, video overlay |
| `data/annotations/` | Nhãn quy tắc, template nhãn người, dữ liệu QA |
| `data/features/` | Metadata cửa sổ và tensor NPZ |
| `data/splits/` | Danh sách session cố định cho train, val, test |
| `src/contracts/` | Hợp đồng nhãn và nguồn nhãn |
| `src/pose/` | Trích pose, thống kê track, trực quan hóa |
| `src/preprocessing/` | Chất lượng, làm sạch, đặc trưng, cửa sổ, weak label |
| `src/models/` | LSTM, TCN, factory tạo và khôi phục mô hình |
| `src/training/` | Dataset, vòng huấn luyện và các metric |
| `src/uncertainty/` | Kiểm tra chất lượng và độ tin cậy |
| `src/reporting/` | Tổng hợp và xuất báo cáo |
| `scripts/` | CLI điều phối từng công đoạn |
| `app/` | Giao diện Streamlit |
| `models/checkpoints/` | Checkpoint temporal |
| `reports/experiments/` | Snapshot cấu hình, metadata, kết quả từng thí nghiệm |
| `tests/` | Kiểm thử hợp đồng và các thành phần |
| `notebooks/` | Điểm vào Colab dạng orchestration/template |

## 2 Công nghệ sử dụng và vai trò

Các phiên bản dưới đây lấy từ `requirements.txt` trong repository; đây là phiên bản được dự án khai báo, không phải kết quả cài mới trong lần viết tài liệu. [S18]

| Công nghệ | Phiên bản khai báo | Vai trò thực tế |
|---|---|---|
| Python | Tài liệu môi trường ghi 3.14.7; CI cấu hình 3.12 | Ngôn ngữ triển khai |
| Ultralytics | 8.4.173 | Nạp YOLO pose và tích hợp tracker |
| YOLOv8n Pose | `yolov8n-pose.pt` | Baseline trích 17 keypoint |
| YOLO11m Pose | `yolo11m-pose.pt` | Nhánh pose dùng cấu hình `imgsz=1280`, confidence 0,35; có artifact overlay |
| YOLOv8m Pose | `yolov8m-pose.pt` | Trọng số có trong thư mục, có thể chọn qua `--model`; chưa có metric so sánh riêng được truy vết |
| ByteTrack | `bytetrack.yaml` | Baseline duy trì ID giữa các frame |
| BoT-SORT | `botsort.yaml` | Phương án tracker so sánh trong benchmark |
| OpenCV | 5.0.0.93 | Đọc video, lấy FPS, vẽ skeleton, blur, ghi video |
| PyTorch | 2.14.1 | LSTM, Conv1d, loss, optimizer, checkpoint |
| torchvision | 0.29.1 | Dependency được khai báo; không phải lõi bộ phân loại temporal |
| NumPy | 2.5.3 | Tensor đặc trưng, mask, phép tính số và NPZ |
| pandas | 3.0.6 | Bảng pose, annotation, groupby, join và CSV |
| PyArrow | 26.0.0 | Engine Parquet |
| PyYAML | 6.0.3 | Đọc cấu hình YAML |
| Streamlit | 1.66.1 | Dashboard và upload video |
| Plotly | 7.1.0 | Biểu đồ timeline và heatmap |
| Matplotlib | 3.11.2 | Dependency phục vụ trực quan hóa trong môi trường dự án |
| pytest | 8.4.2 | Kiểm thử tự động |
| scikit-learn | `>=1.5,<2` | Dependency khai báo cho ML; M1 chưa triển khai |
| Git và GitHub Actions | Workflow trong `.github/workflows/ci.yml` | Truy vết commit và cấu hình kiểm thử CI |

Các metric phân loại và Cohen’s kappa được tính trực tiếp trong mã bằng NumPy/Python; không nên mô tả rằng toàn bộ phần đánh giá đang gọi scikit-learn. YOLO11m-Pose đã có đường thực thi trong extractor/visualization, file trọng số local và overlay mang tên nhánh YOLO11. Chưa có bảng metric pose đối chứng đủ căn cứ để kết luận nhánh này tốt hơn YOLOv8n-Pose. [S26]

### 2.1 Vai trò của YOLOv8 và YOLO11 trong repository

| Thành phần | YOLOv8n Pose mặc định | YOLO11m Pose độ phân giải cao |
|---|---|---|
| Cấu hình | `configs/baseline.yaml` | `configs/experiments/pose_yolo11m_hd.yaml` |
| Trọng số | `yolov8n-pose.pt` | `yolo11m-pose.pt` |
| Confidence trong YAML | 0,30 | 0,35 |
| Kích thước suy luận | Baseline không đặt `imgsz`; báo cáo kỹ thuật ghi 640 | YAML đặt `imgsz=1280` |
| Sampling | 10 FPS | 10 FPS |
| Tracker | ByteTrack | ByteTrack |
| Đầu ra dùng chung | Bbox và 17 keypoint COCO | Bbox và 17 keypoint COCO |
| Dấu vết nhánh YOLO11 | Không áp dụng | Có `data/poses/class_1_yolo11m_hd_overlay.mp4` |

Nhánh YOLO11m-HD hướng tới kiểm tra việc trích pose ở người nhỏ hoặc xa bằng model khác và kích thước ảnh suy luận lớn hơn. Đây là mục tiêu của cấu hình, chưa phải kết quả recall/F1 đã đo. `imgsz=1280` là tham số tiền xử lý/suy luận của Ultralytics, không có nghĩa video nguồn bị đổi thành video 1280×1280 lưu trên đĩa.

Hai lựa chọn pose đi vào cùng hợp đồng bbox/keypoint, rồi dùng chung tracking, cleaning, feature và temporal classifier. Thay model pose không thay LSTM thành YOLO11 và cũng không làm YOLO tự dự đoán ba mức tham gia học tập. Nếu đổi bộ trích pose cho dữ liệu temporal, cần ghi lại nguồn model, kiểm tra chất lượng và đánh giá lại tác động tới phân phối feature.

**Truy vết kết quả:** overlay riêng là artifact của nhánh YOLO11 trong dự án; tên file không đủ để xác minh toàn bộ tham số lúc chạy. CSV pose và metadata do extractor trả về hiện chưa ghi `model_name`/`imgsz`; snapshot LSTM/TCN lưu tên pose trong config cũng chưa chứng minh model đã dùng để tạo từng CSV. Vì vậy chưa gán các điểm Macro-F1 hiện có thành kết quả của YOLO11 nếu không có nhật ký trích xuất tương ứng. [S02, S26]

## 3 Giai đoạn 1 Chốt phạm vi và chuẩn bị môi trường

### Mục tiêu và đầu ra

Giai đoạn này thống nhất ý nghĩa đầu ra, cấu trúc dữ liệu và cách tái lập thí nghiệm. Đầu vào là phạm vi đề tài; đầu ra là hợp đồng nhãn, cấu hình baseline, môi trường chạy và quy ước lưu artifact.

### Cách thực hiện

1. Đọc `docs/scope.md` để thống nhất thuật ngữ và giới hạn suy luận.
2. Dùng nhãn canonical từ `src/contracts/labels.py`, giữ thứ tự HIGH, MODERATE, LOW nhất quán ở dataset, logits, softmax và báo cáo.
3. Tạo môi trường Python riêng và cài dependency từ `requirements.txt`.
4. Kiểm tra skeleton repository bằng `scripts/check_setup.py`.
5. Dùng `configs/baseline.yaml` làm mốc; tạo file trong `configs/experiments/` cho biến thể.

Baseline hiện đặt sampling 10 FPS, pose confidence 0,30, visibility 0,60, track tối thiểu 2 giây, cửa sổ 4 giây, stride 2 giây, LSTM hidden 128, 2 layer, dropout 0,30, batch 32, learning rate 0,001 và ngưỡng xác suất 0,60. Đây là các giá trị ban đầu của mã, chưa phải bộ tham số tối ưu đã được chứng minh.

### Công nghệ và kiểm tra

Python venv cô lập dependency; YAML tách tham số khỏi code; Git ghi nhận phiên bản mã; JSON/CSV lưu metadata và kết quả. Các file cấu hình thí nghiệm được đọc như YAML thông thường. Trường `parent_config` mang tính mô tả; không có cơ chế kế thừa và gộp cấu hình tự động trong các loader đã kiểm tra.

Điều kiện chuyển bước: repository đủ thư mục, nhãn thống nhất và môi trường import được dependency cần thiết. Việc `check_setup.py` chạy thành công không chứng minh mô hình đã học tốt hoặc dữ liệu đã đạt yêu cầu khoa học. [S01, S17, S18]

## 4 Giai đoạn 2 Thu thập dữ liệu và xây dựng nhãn

### 4.1 Chuẩn bị video và metadata

Đầu vào là video lớp học được phép sử dụng. Protocol khuyến nghị camera cố định, độ phân giải 1920×1080, nguồn 25–30 FPS và ghi các pha `lecture`, `individual_work`, `discussion`. Đây là khuyến nghị thu thập, không phải bộ kiểm tra bắt buộc đã tự động hóa toàn bộ trong extractor.

Mỗi session cần `session_id`, camera, thời điểm quay, FPS, resolution, góc máy, số người, timeline pha hoạt động, ánh sáng, xác nhận đồng thuận và ghi chú. Tổ chức dự kiến:

```text
data/raw/session_001/cam_A.mp4
data/raw/session_001/metadata.json
```

Nên quay pilot ngắn để kiểm tra thân trên, bàn tay, che khuất và chất lượng tại các vùng gần/xa. Video gốc không được đưa công khai lên repository; các bước phát triển thường ngày ưu tiên pose và feature. Protocol thu thập hiện còn được ghi là DRAFT trong bảng trạng thái. [S19, S20]

### 4.2 Nhãn quy tắc

`src/preprocessing/weak_labels.py` sinh nhãn từ feature cửa sổ và `class_phase`, đánh dấu `WEAK_RULE`, phiên bản `weak-v0.1`. Quy tắc được áp dụng theo thứ tự, nên quy tắc gặp trước có thể quyết định kết quả. [S07]

| Trường hợp | Điều kiện chính trong code | Nhãn quy tắc |
|---|---|---|
| Chất lượng cửa sổ không VALID | Kiểm tra trước các quy tắc hành vi | Không gán nhãn |
| Tay cao hơn vai | Feature tay trái hoặc phải > 0,05 ở ít nhất 5 bước | HIGH |
| Làm việc cá nhân | `head_down > 0,02` ở >50% bước và motion trung bình >0,01 | HIGH |
| Làm việc cá nhân với vị trí đầu thấp kéo dài | `head_down > 0,30` ở >80% bước và motion <0,005 | LOW |
| Giảng bài với vị trí đầu thấp | `head_down > 0,25` ở >70% bước | LOW |
| Giảng bài với độ lệch ngang đầu lớn | `abs(head_proxy_x) > 0,45` ở >60% bước | LOW |
| Giảng bài với đầu ít lệch ngang và ít thấp | Hai điều kiện hình học đạt >70% bước | HIGH nếu motion >0,015, còn lại MODERATE |
| Thảo luận | Motion trung bình >0,02 | HIGH |
| Thảo luận có motion rất thấp và đầu thấp | Motion <0,005 và `head_down >0,20` ở >50% bước | LOW |
| Không khớp nhánh cụ thể | Quy tắc mặc định | MODERATE |

Ở 10 FPS, 5 bước tương đương khoảng 0,5 giây. Các ngưỡng này thuộc không gian tọa độ đã chuẩn hóa, không phải số pixel hoặc góc nhìn thực. Nhánh tay giơ trong implementation chạy trước kiểm tra phase, dù chú thích chỉ nhắc lecture/discussion. Không được diễn giải các tên rule nội bộ thành chẩn đoán trạng thái tâm lý.

`scripts/generate_annotations.py` sinh annotation quy tắc từ pose. `build_dataset.py` có thể tự gọi weak rule trên tensor cửa sổ; không bắt buộc chạy script sinh annotation trước rồi mới build dataset. Nhãn người được builder ghép bằng `window_id`, không tự động biến mọi annotation theo khoảng thời gian thành nhãn cửa sổ.

### 4.3 Nhãn do người đánh giá

Quy trình dự kiến là lấy mẫu cửa sổ, tạo template trống, hai người xem video độc lập, kiểm tra schema, tính đồng thuận và phân xử bất đồng. Template bỏ nhãn quy tắc ở mức từng dòng để giảm ảnh hưởng tới người gán. Manifest vẫn lưu số lượng tổng hợp theo nguồn mẫu.

```powershell
python scripts/sample_annotation_subset.py --windows data/features/participation_windows.csv --out-dir data/annotations/human --n-test 200 --n-train 150 --n-val 50
```

Sau khi hai người hoàn thành `ann_a.csv` và `ann_b.csv`, kiểm tra bằng:

```powershell
python scripts/dataset_qa.py --annotator-a data/annotations/human/ann_a.csv --annotator-b data/annotations/human/ann_b.csv --strict-human --min-kappa 0.60 --report-json data/annotations/human/qa_report.json
```

Cohen’s kappa được tính theo `kappa = (p_o - p_e) / (1 - p_e)`, với `p_o` là tỷ lệ đồng thuận quan sát và `p_e` là đồng thuận kỳ vọng từ phân phối nhãn. Ngưỡng 0,60 trong lệnh là tiêu chí QA được chọn, không phải kết quả đã đo. Chế độ strict từ chối nguồn nhãn weak và file giống nhau theo khóa/nhãn, nhằm phát hiện những bản sao không phải gán độc lập. [S15]

**Công cụ đã sửa ngày 09/11:** template có đủ cột schema, lấy mẫu được cả window chưa có nhãn/che khuất; `--n-test 0` lấy toàn bộ test. QA từ chối phiếu trống khi yêu cầu nhãn người và từ chối khóa trùng. App `app/annotation_app.py` chọn Quân (phiếu A) / Phong (phiếu B), có tiến độ, lưu tiếp, xuất CSV. Mặc định mở mẫu 600 cửa sổ từ video gốc dataset cũ (200 train/100 val/300 test); 130 cửa sổ hai video mới là bộ bổ sung. Một nhãn áp dụng cho một track trong một khoảng thời gian, không phải cả lớp. Clip cắt từ video gốc có khung vàng và phóng to đúng người; không dùng overlay cũ có thể chỉ dài 15 giây. Tiêu chí v0.3-pilot yêu cầu ghi bốn khoảng bằng chứng, căn cứ nhiệm vụ, hành vi và xác nhận; nhãn phải khớp bảng quy đổi đề xuất. Ngưỡng 3/4 chưa được kiểm chứng, cần pilot/thống nhất trên train/validation trước khi chốt hướng dẫn và gán test. Chỉ ghi tên khi người gán lưu; HUMAN dành cho nhãn VALID do họ xác nhận. Đã sửa lỗi pandas 3 với confidence 0.8 và giữ mức che khuất khi mở lại. [S27]

Đầu ra đạt yêu cầu của giai đoạn này phải có nhãn người thật và báo cáo QA độc lập. Repository hiện chưa có bằng chứng đáp ứng điều kiện đó; nhãn ví dụ/template không được tính là ground truth thực nghiệm.

## 5 Giai đoạn 3 Lấy mẫu video trích pose và theo dõi người

### 5.1 Đầu vào và đầu ra

Đầu vào là đường dẫn video, session/camera, cấu hình pose và tùy chọn tracker. Đầu ra là bảng: một dòng cho một người hoặc track tại một frame được lấy mẫu; kèm metadata về FPS, resolution, số frame và tổng detection.

Một dòng pose có thông tin session, camera, `frame_idx`, `timestamp_ms`, `track_id`, bbox, bbox confidence, 17 bộ `(x, y, confidence)`, tỷ lệ keypoint thấy được, confidence trung bình và zone. Danh sách cột thực tế của extractor có 64 cột. [S02]

### 5.2 Lấy mẫu theo thời gian

OpenCV mở video bằng `VideoCapture`. Code đọc FPS nguồn, tổng frame, kích thước; nếu FPS không hợp lệ thì dùng fallback 30. Mỗi frame nguồn có thời gian ước tính `raw_frame_idx × 1000 / original_fps`. Khi tới mốc lấy mẫu tiếp theo, frame được đưa vào YOLO; mốc tăng `1000 / sample_fps` ms.

Với nguồn 30 FPS và mục tiêu 10 FPS, thông thường khoảng 3 frame nguồn lấy một frame. Code vẫn giải mã lần lượt các frame, nên sampling giảm số lượt inference, không bỏ hoàn toàn chi phí đọc video. `frame_idx` là chỉ số frame đã lấy mẫu; `timestamp_ms` dựa trên vị trí frame nguồn. `--max-frames` giới hạn frame sampled, không phải frame nguồn.

### 5.3 Trích 17 keypoint bằng YOLOv8 hoặc YOLO11

YOLO trả bbox và keypoint theo thứ tự COCO: mũi; hai mắt; hai tai; hai vai; hai khuỷu; hai cổ tay; hai hông; hai gối; hai cổ chân. Khi một điểm có confidence dưới ngưỡng hoặc tọa độ gốc `(0,0)`, extractor lưu tọa độ NaN và giữ confidence, tránh coi điểm thiếu là một quan sát thật.

Hàm `extract_poses_from_video` gọi `YOLO(eff_model_name)` của Ultralytics. `eff_model_name` lấy từ `model_name` nếu có, còn lại từ YAML. `eff_imgsz` cũng ưu tiên tham số trực tiếp rồi mới tới YAML. Hai giá trị này đi qua cùng hàm trích frame; `imgsz` được chuyển vào cả `model.predict` và `model.track`. Do đó, YOLO11-Pose đã được tích hợp qua cơ chế chọn trọng số và kích thước suy luận, dù docstring của extractor vẫn còn tên YOLOv8. [S02, S26]

```text
visible_ratio = số keypoint hợp lệ / 17
mean_keypoint_conf = tổng confidence của 17 keypoint / 17
```

Ngưỡng `confidence_threshold` được truyền cho YOLO dưới tham số `conf`, đồng thời dùng để xét keypoint thấy được. Vì vậy cấu hình hiện đang dùng cùng một giá trị cho confidence detection và visibility keypoint; không có hai tham số độc lập trong hàm này.

### 5.4 Tracking

Khi bật `--track`, extractor gọi `model.track(frame, persist=True, tracker=...)`; nếu không bật thì gọi `model.predict`. ByteTrack được cung cấp qua Ultralytics, không được viết lại từ đầu trong repository. `persist=True` giữ trạng thái giữa các frame liên tiếp của lần xử lý.

`src/pose/tracker.py` tính tổng frame, thời điểm đầu/cuối, thời lượng và chất lượng trung bình của từng track, rồi hỗ trợ lọc track quá ngắn. Thời lượng hiện là `(end_ms - start_ms) / 1000`. ID có thể đổi khi người bị che hoặc mất detection; chưa có cơ chế xác minh danh tính xuyên session. [S03]

### 5.5 Ví dụ thực hiện

```powershell
python scripts/extract_pose.py --video data/raw/class_1.mp4 --session class_1 --camera cam_A --output data/poses/class_1_cam_A_poses.csv --track --sample-fps 10 --device cpu
```

Trích pose bằng YOLO11m-HD theo YAML có sẵn, lưu output riêng để không ghi đè baseline:

```powershell
python scripts/extract_pose.py --video data/raw/class_1.mp4 --session class_1 --camera cam_A --config configs/experiments/pose_yolo11m_hd.yaml --output data/poses_yolo11/class_1_cam_A_poses.csv --track --device cpu
```

Lệnh này lấy `yolo11m-pose.pt`, `imgsz=1280`, confidence 0,35 và sampling 10 FPS từ YAML. Có thể chọn trực tiếp bằng `--model yolo11m-pose.pt --imgsz 1280 --conf 0.35 --sample-fps 10`. Tham số trực tiếp được ưu tiên hơn YAML trong nhánh xử lý video đơn.

Để trích lại bằng YOLO11 và tạo overlay qua entry point visualization:

```powershell
python scripts/visualize_poses.py --video data/raw/class_1.mp4 --model yolo11m-pose.pt --imgsz 1280 --conf 0.35 --sample-fps 10 --track --blur-face --output data/poses_yolo11/class_1_yolo11m_hd_overlay.mp4
```

Trong `process_single`, nếu có `--model`, script chạy trích xuất mới kể cả khi tìm thấy pose CSV; nếu không chỉ định model và có CSV thì có thể dùng lại pose đã lưu. Script visualization không nhận `--config`, nên các giá trị 0,35 và 1280 cần truyền trực tiếp khi muốn khớp nhánh HD. [S26]

Nhánh batch của `extract_pose.py` nhận model/imgsz từ YAML, nhưng chưa chuyển đầy đủ override CLI `--model` và `--imgsz` như nhánh video đơn. Luồng `predict_video.py` đã truyền YAML pose tới extractor; app có lựa chọn YOLOv8n/YOLO11m. Metadata suy luận ghi model, imgsz và tracker thực dùng. Nhánh batch CLI vẫn cần đối chiếu override trực tiếp với cấu hình.

Điều kiện chuyển bước: có track ID sử dụng được, timestamp tăng hợp lý, bbox hợp lệ và thống kê chất lượng cho từng vùng. Chưa có ground truth tracker thì không kết luận ByteTrack/BoT-SORT tốt hơn về ID switch chỉ từ FPS.

## 6 Giai đoạn 4 Kiểm tra chất lượng làm sạch và chuẩn hóa

### 6.1 Hai mức kiểm tra chất lượng

`evaluate_pose_quality` phân loại từng dòng pose. Ưu tiên kiểm tra track ngắn, tiếp theo confidence thấp, rồi visibility thấp. Track ngắn hoặc confidence thấp dẫn tới INVALID; thiếu visibility dẫn tới OCCLUDED; còn lại VALID. Ngưỡng baseline tương ứng là 2 giây, 0,30 và 0,60. [S04]

Khi tạo cửa sổ, builder kiểm tra **trung bình** visibility và pose confidence trên cửa sổ. Cửa sổ dưới ngưỡng mang trạng thái INVALID và lý do `LOW_VISIBILITY` hoặc `LOW_KEYPOINT_CONF`. Đây là phép kiểm tra khác với trạng thái từng frame; không nên nói rằng mọi frame không VALID đều bị xóa trước khi tạo chuỗi.

### 6.2 Các bước làm sạch thực tế

`clean_and_normalize_poses` thực hiện theo thứ tự: lọc track ngắn → nội suy tọa độ thiếu → làm mượt → chuẩn hóa → tùy chọn điền thiếu còn lại. [S05]

1. **Lọc track:** chỉ giữ track đạt thời lượng tối thiểu.
2. **Nội suy:** xử lý từng track theo thứ tự frame, nội suy tuyến tính các đoạn NaN ngắn; mặc định tối đa 3 dòng liên tiếp. Đoạn thiếu dài hơn được giữ NaN ở bước nội suy.
3. **Làm mượt:** rolling mean có tâm, mặc định cửa sổ 3; có nhánh Gaussian tùy chọn. Mục đích là giảm rung keypoint giữa các quan sát.
4. **Chuẩn hóa:** thêm các cột `norm_kpt_XX_x/y`, giữ lại tọa độ và confidence gốc.
5. **Điền tiếp nếu được yêu cầu:** forward/backward fill theo track, mặc định tắt.

Nội suy dùng `limit_direction="both"`, nên các đoạn ngắn ở đầu/cuối cũng có thể được điền. Rolling mean có thể tiếp tục tạo giá trị tại một số vị trí trước đó còn NaN. Mask v2 được tính sau bước làm sạch từ tọa độ chuẩn hóa còn hữu hạn; đó không phải mask nguyên gốc của từng keypoint trước nội suy.

### 6.3 Chuẩn hóa torso

Khi thấy hai vai, tâm chuẩn hóa là trung điểm vai. Khi thiếu vai, dùng tâm bbox. Nếu thấy hai hông thì scale là độ dài từ tâm vai tới trung điểm hông, chặn dưới 20 pixel. Khi thiếu hông do bàn che, fallback là giá trị lớn nhất của `1,5 × khoảng cách vai`, `0,5 × chiều cao bbox` và 20.

```text
x_normalized = (x - center_x) / scale
y_normalized = (y - center_y) / scale
```

Chuẩn hóa giảm ảnh hưởng vị trí ảnh và kích thước người trong khung. Nó không khôi phục pose 3D hoặc loại bỏ toàn bộ sai khác góc máy. Chế độ `bbox` thay thế dùng góc trên trái bbox làm gốc và chia theo chiều rộng/cao bbox.

### 6.4 Thực hiện và điểm cần kiểm tra

```powershell
python scripts/clean_poses.py --input data/poses/class_1_cam_A_poses.csv --output data/poses_cleaned/class_1_cam_A_poses_cleaned.csv --max-gap-frames 3 --smooth-window 3 --norm-method torso --report reports/class_1_track_quality.csv
```

Cleaner CLI, dataset builder và video inference hiện đều lấy thời lượng track tối thiểu từ cấu hình. Hàm chuẩn hóa được vector hóa; đã đối chiếu đầu ra torso/bbox với implementation cũ trên 400 dòng pose thực, đồng thời chạy lại kiểm thử.

Builder làm sạch riêng từng session/camera, tạo cửa sổ theo `(session_id, camera_id, track_id)` và từ chối pose trùng khóa. Cửa sổ có bước frame thiếu hoặc gap timestamp quá lớn được bỏ qua; chưa có cơ chế tái tạo timestep trống. [S27]

## 7 Giai đoạn 5 Xây dựng đặc trưng tư thế và chuyển động

### 7.1 Đầu vào và representation

`compute_track_features` nhận bảng của một track và trả ma trận `[T,F]` kiểu float32. Mặc định `posefeat-v1` có 77 chiều; `posefeat-v2` thêm 17 mask, thành 94 chiều. Các feature có thứ tự cố định để weak rule và checkpoint sử dụng đúng cùng ý nghĩa cột. [S06]

| Nhóm | Số chiều | Vị trí trong v1 | Cách tạo |
|---|---|---|---|
| Tọa độ chuẩn hóa | 34 | 0–33 | 17 điểm × 2 tọa độ |
| Vận tốc | 34 | 34–67 | Hiệu tọa độ giữa hai dòng chia `dt` |
| Hình học | 6 | 68–73 | Độ nghiêng vai, độ lệch mũi, tay cao, đầu thấp |
| Năng lượng chuyển động | 1 | 74 | RMS vận tốc trên 34 tọa độ |
| Chất lượng | 2 | 75–76 | Visibility và confidence trung bình |
| Mask chỉ có ở v2 | 17 | 77–93 | Cặp tọa độ còn hữu hạn trước zero-fill feature |

### 7.2 Chi tiết công thức

Vận tốc dùng `dt = 1 / sample_fps`, nên `v[t] = (p[t] - p[t-1]) / dt`. Dòng đầu được chép vận tốc dòng thứ hai nếu có. Năng lượng chuyển động là căn bậc hai của trung bình bình phương 34 thành phần vận tốc.

Sáu feature hình học gồm:

- `shoulder_slope = atan2(y_vai_phải - y_vai_trái, x_vai_phải - x_vai_trái + 1e-6)`.
- `head_proxy_x` và `head_proxy_y`: độ lệch mũi so với trung điểm vai.
- `left_hand_raised` và `right_hand_raised`: phần dương của tọa độ y vai trừ y cổ tay tương ứng.
- `head_down`: phần dương của y mũi trừ y trung điểm vai.

Trong tọa độ ảnh, y tăng từ trên xuống dưới. Vì vậy cổ tay cao hơn vai sẽ có y nhỏ hơn, tạo feature tay giơ dương. Đây là quan hệ hình học; code không xác định động tác có đúng với nhiệm vụ học hay không bằng một bộ nhận dạng hoạt động riêng.

### 7.3 Xử lý thiếu và giới hạn

Trước khi đưa vào mô hình, NaN ở tọa độ được thay 0. v1 không có mask riêng, nên tọa độ thiếu có thể khó phân biệt với tọa độ thật bằng 0. v2 thêm mask lấy từ trạng thái hữu hạn của tọa độ sau làm sạch. Cuối cùng toàn bộ tensor được ép hữu hạn: NaN → 0, +Inf → 1, -Inf → -1.

Vận tốc vẫn được tính giữa các dòng kế tiếp với dt cố định. Nếu track mất frame, khoảng thời gian thực có thể lớn hơn dt; nếu điểm chuyển từ thiếu sang có sau zero-fill, có thể tạo vận tốc giả. Đây là hạn chế của cách tính hiện tại và là lý do cần kiểm tra continuity/mask trước khi mở rộng mô hình.

Các feature như gia tốc, góc khuỷu đầy đủ hoặc độ ổn định tư thế được nhắc trong danh sách ý tưởng của dự án, nhưng không phải các kênh bổ sung đang có trong vector 77/94 chiều. Không nên trình bày chúng như đã triển khai.

## 8 Giai đoạn 6 Tạo cửa sổ gán nhãn và chia dữ liệu

### 8.1 Sliding window

`build_track_windows` cắt chuỗi theo **số dòng**. Với sampling 10 FPS, window 4 giây và stride 2 giây, `W=40`, `S=20`. Các đoạn lần lượt là dòng `[0:40]`, `[20:60]`, `[40:80]`. Một track có T dòng tạo `floor((T-W)/S)+1` cửa sổ nếu T≥W. [S07]

Ví dụ track có 100 dòng liên tục tạo 4 cửa sổ, mỗi tensor `[40,77]` hoặc `[40,94]`. Mỗi cửa sổ có `window_id`, session, camera, track, timestamp đầu/cuối, phase, zone, version, chất lượng, nguồn nhãn và split. Tên ID có dạng `session_cam_A_trk001_w0000`.

Ở 10 FPS liên tục, 40 quan sát từ t=0 đến t=3.900 ms có khoảng chênh timestamp là 3,9 giây, dù cấu hình gọi là cửa sổ 4 giây. Bản sửa từ chối cửa sổ bắc qua frame mất hoặc gap timestamp lớn. Phase được xét trên từng window; cửa sổ giao hai phase được đánh TRANSITION. Zone dùng mode trong window. Hệ thống chưa nội suy lại lưới thời gian đều.

### 8.2 Điều phối dataset

`scripts/build_dataset.py` đọc pose, làm sạch nếu chưa có cột chuẩn hóa, gộp dữ liệu, đọc split cố định, tạo window theo track, gán nhãn theo mode và xuất metadata/tensor. [S07]

Ba mode là `weak`, `human`, `hybrid`. Mode weak gọi rule; mode human giữ các window ghép được với HUMAN/ADJUDICATED; mode hybrid dùng nhãn người khi có, giữ weak ở phần còn lại. Khi cùng `window_id` có nhãn phân xử, loader ưu tiên ADJUDICATED hơn HUMAN. Mode hybrid không tự tạo một human test set độc lập.

```powershell
python scripts/build_dataset.py --poses-dir data/poses --output-dir data/features --splits-dir data/splits --labels weak --feature-version v1 --config configs/baseline.yaml
```

Để tạo v2, dùng output riêng như `data/features_v2` và truyền `--feature-version v2`. Khi train v2 phải trỏ đúng NPZ v2; chọn cấu hình có tên v2 không tự chuyển tensor v1 thành v2.

### 8.3 Chia train validation test

Builder ưu tiên đọc cả ba file `train_sessions.txt`, `val_sessions.txt`, `test_sessions.txt`. Nếu chỉ có một phần file thì báo lỗi. Validator từ chối session trùng trong một split, giao giữa các split, session không có pose hoặc session pose chưa nằm trong split. SHA-256 từng file được lưu để truy vết.

Nếu chưa có file nào, fallback sắp xếp session và chia theo thứ tự, dành khoảng 15% cho val và test, yêu cầu ít nhất 3 session. Đây là deterministic split; tham số seed không làm ngẫu nhiên hóa fallback. Khi đã freeze, không tùy ý đổi test để cải thiện điểm.

Split baseline v0.1 đã lưu để đối chiếu:

| Split | Session | Tổng window | Có nhãn |
|---|---|---|---|
| Train | class_1 đến class_5, session_001, session_002 | 13.481 | 11.629 |
| Validation | session_003, session_004 | 12.011 | 10.466 |
| Test | session_005, session_006 | 7.467 | 6.518 |
| Tổng | 11 recording/session ID | 32.959 | 28.613 |

Các con số lấy từ `data/features/split_report.json`. Tensor tổng có shape `[32959,40,77]`. 4.346 window chưa có nhãn được mã hóa -1. Trong test có 949 window không nhãn, gồm các cửa sổ chất lượng không đạt. Split theo session ngăn rò rỉ window cùng recording; vẫn cần kiểm tra các session có trùng nội dung hoặc cùng người/ngày quay khi đánh giá khả năng khái quát. [S21]

Dataset chờ review **v0.2-context-review** có 27.116 window: train 11.188, validation 10.013, test 5.915. Cùng file split với v0.1; số window giảm do kiểm tra gap. Tất cả phase hiệu lực đang UNKNOWN vì metadata chưa được người xác minh. Các khai báo cũ vẫn nằm trong declared_class_phase và audit; không chuyển thành nhãn thật. Bộ này chưa có target huấn luyện, cần hoàn tất review trước khi train. [S27]

### 8.4 File đầu ra

`participation_windows.csv/parquet` lưu metadata; `window_features.npz` lưu `X_train/y_train`, `X_val/y_val`, `X_test/y_test`, tensor tổng, nhãn tổng, các mảng window ID và feature schema. `split_report.json` lưu số lượng, nguồn nhãn, trạng thái chất lượng và hash split.

Builder hiện lấy FPS, window, stride và ngưỡng track từ YAML nếu không có override CLI. Dataset version có thể đặt bằng `--dataset-version`. Feature version vẫn chọn tường minh qua CLI. Dataset mới lưu cấu hình hiệu lực và audit metadata; ba file split đóng băng không bị sửa.

Nên để một bản pose chuẩn cho mỗi recording trong thư mục đầu vào. Builder từ chối khóa pose trùng để tránh đọc lặp raw/cleaned/tracked của cùng recording.

## 9 Giai đoạn 7 Thiết kế và huấn luyện mô hình temporal

### 9.1 Dataset và dữ liệu batch

`WindowFeatureDataset` nhận `[N,T,F]` và nhãn số. Nếu có nhãn, các mẫu -1 bị loại; feature chuyển float32, nhãn int64. DataLoader trộn train và giữ thứ tự val/test. Class weight được tính chỉ từ train: `w_c = N / (C × max(n_c,1))`. [S08]

Đầu vào baseline một batch là `[32,40,77]`. Mô hình trả logits `[32,3]`; softmax chuyển logits thành xác suất theo thứ tự canonical. Model factory chọn LSTM hoặc TCN từ `model.type` hoặc override CLI.

### 9.2 Kiến trúc LSTM thực tế

`ParticipationLSTM` có LSTM 2 tầng, hidden 128, `batch_first=True`, dropout 0,30, mặc định một chiều. Sau LSTM, code lấy trung bình output theo thời gian và output timestep cuối, rồi cộng trung bình hai vector:

```text
X [B,40,77]
  -> LSTM output [B,40,128]
  -> 0,5 × (mean theo thời gian + output bước cuối) [B,128]
  -> Dropout
  -> Linear 128 -> 64
  -> ReLU -> Dropout
  -> Linear 64 -> 3
  -> logits [B,3]
```

Class có tham số hỗ trợ bidirectional, nhưng factory hiện không truyền lựa chọn này từ cấu hình. Một chú thích nói head có layer norm; cấu trúc thực tế **không có LayerNorm**. Khi trình bày kiến trúc nên dựa vào các layer được tạo trong code. [S09]

### 9.3 Kiến trúc TCN thực tế

`ParticipationTCN` nhận cùng tensor với LSTM, chuyển thành `[B,F,T]`, dùng Conv1d 1×1 chiếu F về 64 channel. Bốn block có kernel 3 và dilation 1,2,4,8. Mỗi block gồm convolution, cắt phần padding bên phải, BatchNorm1d, ReLU, Dropout và cộng residual.

```text
X [B,40,77] -> transpose [B,77,40]
  -> Conv1d 1x1 [B,64,40]
  -> 4 block dilation 1,2,4,8
  -> concat(mean theo thời gian, bước cuối) [B,128]
  -> Linear 128 -> 64 -> ReLU -> Dropout
  -> Linear 64 -> 3
```

Khác LSTM, TCN **nối** hai vector pooling thay vì lấy trung bình. Với một convolution mỗi block, receptive field convolution của chuỗi block này là `1 + (3-1) × (1+2+4+8) = 31` bước. Đây là tính toán từ kiến trúc; không phải một kết quả thực nghiệm độc lập. Global mean vẫn tổng hợp các output trên cả cửa sổ. [S10]

Padding/cropping tạo tính causal cho convolution. Tuy nhiên BatchNorm trong chế độ train dùng thống kê trên chiều thời gian, và bộ phân loại cần cả cửa sổ; chưa đủ căn cứ để mô tả toàn bộ hệ thống như bộ dự đoán streaming causal ở từng frame.

### 9.4 Vòng huấn luyện

`scripts/train.py` đặt seed cho random, NumPy, PyTorch; tự chọn CUDA nếu có và không chỉ định thiết bị. Sau đó dùng CrossEntropyLoss có class weight và Adam. Mỗi batch thực hiện zero grad → forward → loss → backward → optimizer step. Mỗi epoch đánh giá validation; checkpoint tốt nhất được chọn theo Macro-F1 validation. [S08, S11]

Nếu validation không cải thiện trong `--patience` epoch thì dừng sớm. Default CLI là 30 epoch tối đa, patience 10. Sau train, script nạp lại trọng số tốt nhất và đánh giá test. Không có scheduler, augmentation chuỗi hoặc hiệu chỉnh xác suất được triển khai trong vòng train này. Seed giúp tái lập nhưng chưa bảo đảm bitwise deterministic trên mọi GPU.

```powershell
python scripts/train.py --features data/features/window_features.npz --config configs/baseline.yaml --name lstm_baseline --epochs 30 --patience 10 --device cpu
python scripts/train.py --features data/features/window_features.npz --config configs/experiments/tcn_baseline.yaml --name tcn_baseline --epochs 30 --patience 10 --device cpu
```

### 9.5 Artifact và khả năng truy vết

Mỗi run tạo experiment ID, metadata, snapshot YAML, metrics JSON và training history CSV; checkpoint `.pt` nằm trong `models/checkpoints/`. Checkpoint lưu model state, optimizer state, epoch tốt nhất, loại mô hình, input size và một số tham số kiến trúc. Metadata bổ sung hash checkpoint, feature schema, phân phối lớp và override.

`new_experiment.py` ghi seed, git commit, dataset/split version và hash snapshot. Bản sửa đồng bộ model type, batch size, learning rate, patience và feature/dataset version hiệu lực vào snapshot. Patience lấy từ YAML nếu CLI không override. Train từ chối split không có nhãn, thay vì tạo kết quả từ phiếu trống. [S11, S17]

Factory đã sửa để khôi phục cấu hình TCN lồng trong checkpoint; kiểm thử round-trip với hidden size, kernel và số level khác mặc định đã pass. Các checkpoint baseline cũ vẫn được hỗ trợ. [S27]

## 10 Giai đoạn 8 Đánh giá và diễn giải kết quả

### 10.1 Ghép tensor với metadata

`scripts/evaluate.py` đọc checkpoint, tensor test và metadata window. Script yêu cầu các mảng `window_ids_test`, kiểm tra độ dài, kiểm tra ID trùng/thiếu và đối chiếu nhãn tensor với nhãn trong bảng bằng **window_id**. Cách này tránh ghép nhầm khi CSV thay đổi thứ tự. Các mẫu y=-1 được loại trước suy luận và tính metric. [S12]

```powershell
python scripts/evaluate.py --experiment-dir reports/experiments/20261025-162855-lstm-full-v0-1 --features data/features/window_features.npz --windows-meta data/features/participation_windows.csv --config configs/baseline.yaml
```

Lệnh trên minh họa đánh giá lại một experiment có sẵn và sẽ ghi lại artifact đánh giá của experiment đó. Khi làm thí nghiệm mới, thay bằng experiment ID mới; không ghi đè kết quả đã chốt để thử tham số.

### 10.2 Các metric được tính

Từ confusion matrix, mỗi lớp có precision `TP/(TP+FP)`, recall `TP/(TP+FN)` và F1 là trung bình điều hòa của precision/recall. Macro-F1 là trung bình F1 ba lớp, giúp thấy rõ hơn chất lượng lớp ít mẫu. Balanced accuracy là trung bình recall ba lớp; accuracy là tỷ lệ đúng toàn bộ mẫu được đánh giá. Khi mẫu số bằng 0, code trả 0 cho metric lớp tương ứng.

Script lưu hai bộ kết quả: `raw_metrics` từ argmax trên toàn bộ mẫu có nhãn; `accepted_metrics` chỉ trên mẫu qua gate. Đồng thời lưu coverage, uncertain rate, lý do gate, nguồn nhãn và số mẫu theo session. Accepted Macro-F1 luôn phải đi cùng coverage vì có thể tăng khi từ chối các mẫu khó.

### 10.3 Kết quả hiện hành trong artifact

Hai thí nghiệm full ngày 25/10/2026 cùng dùng split-v0.1 và feature v1. Mỗi thí nghiệm có 7.467 window test, 6.518 window có nhãn WEAK_RULE để đánh giá, loại 949 window không nhãn. [S22, S23]

| Chỉ số | LSTM full | TCN full |
|---|---|---|
| Experiment ID | `20261025-162855-lstm-full-v0-1` | `20261025-163801-tcn-full-v0-1` |
| Raw Macro-F1 | 0,9098 | 0,9027 |
| Raw balanced accuracy | 0,9107 | 0,9070 |
| Raw accuracy so với nhãn quy tắc | 0,9228 | 0,9170 |
| Accepted Macro-F1 | 0,9151 | 0,9198 |
| Accepted window | 6.423 | 6.254 |
| Coverage trên 6.518 mẫu có nhãn | 98,54% | 95,95% |
| Bị từ chối trong tập có nhãn | 95 | 264 |
| Nguồn nhãn đánh giá | WEAK_RULE | WEAK_RULE |

Trong artifact này, LSTM có raw Macro-F1 và coverage cao hơn; TCN có accepted Macro-F1 cao hơn nhưng từ chối nhiều mẫu hơn. Chưa có human test set hoặc nhiều lần chạy độc lập để kết luận mô hình nào tốt hơn khi triển khai thực tế. Các experiment cũ chỉ dùng đoạn 15 giây không được thay thế cho hai run full trong báo cáo hiện hành.

**Mẫu số cần hiểu đúng:** coverage 98,54% của LSTM là `6423/6518`, không phải `6423/7467`. Do các window không nhãn bị loại trước gate, coverage đánh giá không phản ánh toàn bộ mức bao phủ của mọi window từ video. Nếu tính số accepted đánh giá trên toàn bộ 7.467 window thì khoảng 86,02%, nhưng đây là tỷ lệ suy ra khác mẫu số, không phải field coverage được script báo cáo.

### 10.4 File kết quả

`evaluation_report.json` là tổng hợp metric; `predictions.csv` là dấu vết từng window: true label, raw label, gated label, p_max, lý do gate, chất lượng và nguồn nhãn. `metric_scope` tự phân biệt weak, human hoặc mixed từ nguồn nhãn. Hệ thống hiện chưa có calibration ECE, temperature scaling, thí nghiệm M0/M1 và ablation hoàn chỉnh.

## 11 Giai đoạn 9 Xử lý bất định trong dự đoán

### Mục tiêu

Tầng bất định tránh ép mô hình trả một nhãn hành vi khi quan sát kém hoặc xác suất thấp. `explain_uncertainty_gate` độc lập với LSTM/TCN: đầu vào gồm ba xác suất, visibility, confidence và ngưỡng; đầu ra là nhãn canonical cùng lý do. [S13]

### Thứ tự quyết định

1. Kiểm tra vector xác suất có đúng 3 phần tử, hữu hạn, trong [0,1], tổng gần 1 và đúng thứ tự lớp.
2. Nếu visibility <0,60: trả UNCERTAIN, lý do LOW_VISIBILITY.
3. Nếu mean keypoint confidence <0,30: trả UNCERTAIN, lý do LOW_KEYPOINT_CONF.
4. Nếu xác suất lớn nhất <0,60: trả UNCERTAIN, lý do LOW_MODEL_CONF.
5. Còn lại: lấy argmax, lý do ACCEPTED.

Ví dụ xác suất `[0,72; 0,20; 0,08]`, visibility 0,80 và confidence 0,65 được chấp nhận HIGH. Giữ nguyên xác suất nhưng visibility 0,45 sẽ thành UNCERTAIN. Xác suất `[0,42; 0,38; 0,20]` vẫn thành UNCERTAIN dù pose đủ tốt. Đúng bằng ngưỡng được chấp nhận nếu các điều kiện khác đạt vì code dùng phép so sánh `<`.

`coverage = accepted / số mẫu được đưa qua gate`; uncertain rate là phần còn lại. Softmax 0,60 không mặc nhiên đồng nghĩa xác suất đúng được hiệu chỉnh là 60%. Các ngưỡng cần chọn trên validation và giữ nguyên khi đánh giá test. Hiện code có gate và lý do rõ ràng, chưa có bằng chứng tune/calibration hệ thống.

## 12 Giai đoạn 10 Tổng hợp theo thời gian và vùng lớp

### 12.1 Gán zone hình học

`scripts/assign_zones.py` lấy tâm bbox, chuẩn hóa theo miền bbox quan sát trong recording rồi chia lưới 3×3. Cột là LEFT/CENTER/RIGHT; theo cấu hình mặc định, phía trên ảnh là BACK, phía dưới là FRONT. Tên zone có dạng `BACK-LEFT`. Đây là quy ước cần đối chiếu vị trí camera, không phải phép suy ra tự động hướng bảng. [S14]

```text
x_norm = (bbox_center_x - min_bbox_x1) / observed_width
y_norm = (bbox_center_y - min_bbox_y1) / observed_height
```

Miền quan sát phụ thuộc các detection trong video; gán zone hình học chưa phải sơ đồ ghế thật và không có hiệu chỉnh phối cảnh/homography. Khi đám đông hoặc camera thay đổi, cùng một chỗ vật lý có thể nhận zone khác.

```powershell
python scripts/assign_zones.py --poses-dir data/poses --output-dir data/poses_zones --config configs/zones/grid_3x3.yaml
```

Sau đó build dataset từ `data/poses_zones` để window có zone. Window hiện dùng zone phổ biến nhất trong cửa sổ; lưới vẫn là quy ước hình học, không tự xác minh hướng bảng/camera.

### 12.2 Tổng hợp timeline

`aggregate_timeline` nhóm theo `(session_id, bucket thời gian)`, mặc định bucket 60 giây. Bucket lấy từ `t_start_ms`, không trải một window sang nhiều bucket theo số giây chồng lấn. Việc giữ session riêng tránh gộp phút 0 của các video khác nhau. [S14]

Mỗi nhóm có số window, số HIGH/MODERATE/LOW/UNCERTAIN, tỷ lệ và chất lượng trung bình. Các trường `pct_*` được lưu dạng tỷ lệ 0–1 dù có tiền tố pct. Khi hiển thị phần trăm cần nhân 100.

### 12.3 Tổng hợp vùng và ý nghĩa thống kê

`aggregate_zones` nhóm theo zone, lưu cùng hệ đếm và N. Nếu không có zone thật nào thì trả bảng rỗng. Hàm có thể gộp nhiều session nếu đầu vào chứa nhiều session, nên cần lọc trước khi muốn kết luận cho một lớp hoặc một buổi cụ thể.

Các window chồng lấn và một track xuất hiện nhiều lần. Do đó, `n_windows` không phải số người duy nhất, tỷ lệ HIGH không phải tỷ lệ học sinh và cũng chưa phải tỷ lệ thời gian chính xác. Một track bị tách thành nhiều ID cũng không tương ứng nhiều người thật. Heatmap hiện là phân phối kết quả, chưa phải bản đồ F1 hoặc bằng chứng về bias. Muốn đánh giá sai lệch vị trí cần nhãn người, N đủ lớn, chất lượng pose, mức che khuất và metric riêng từng zone.

## 13 Giai đoạn 11 Suy luận video mới dashboard và xuất báo cáo

### 13.1 Luồng video mới

`scripts/predict_video.py` nhận video và checkpoint, kiểm tra file, tạo ngưỡng từ YAML, trích pose có tracking, gán zone nếu được cấu hình, làm sạch, nạp mô hình, tạo feature theo input size và chạy gate. Nó không cần split train/test và không tự tạo ground truth. [S16]

Checkpoint input size 94 kích hoạt feature có mask; 77 dùng v1. Code kiểm tra chiều feature trước khi dự đoán. Mô hình temporal dùng thiết bị chọn bởi `--device` (mặc định CPU). Suy luận chia batch bằng `--batch-size` để hạn chế bộ nhớ activation; pipeline vẫn giữ tensor feature trong RAM nên chưa phải streaming hoàn toàn.

```powershell
python scripts/predict_video.py --video data/raw/class_1.mp4 --checkpoint models/checkpoints/20261025-162855-lstm-full-v0-1.pt --out-dir reports/inference/class_1_demo --max-frames 150 --device cpu
```

Đầu ra gồm `predictions.csv`, `inference_meta.json` và thư mục export. Khi không có nhãn, summary chỉ báo phân phối, chất lượng, coverage và cảnh báo không có ground truth; không tính accuracy. `inference_meta.json` ghi thời gian, số frame, số window accepted/uncertain, FPS và thời gian xử lý trên mỗi phút video.

**Cấu hình đã nối xuyên suốt:** YAML pose được truyền tới extractor. Hai video mới ngày 09/11 đã chạy với yolo11m-pose.pt, imgsz 1280 và checkpoint LSTM full v0.1. Có 107 và 23 window; accepted lần lượt 74 và 12. Đây là inference không có ground truth. Các run CPU có tải đồng thời và bước xử lý preview nên không dùng làm benchmark cô lập. [S27]

### 13.2 Giao diện Streamlit

`app/streamlit_app.py` đọc artifact bằng pandas/JSON, cache danh sách và nội dung, dùng Plotly để vẽ. Logic train không nằm trong UI. App có sáu tab: [S16]

| Tab | Chức năng hiện có |
|---|---|
| Tổng quan | Raw/accepted Macro-F1, coverage, N và metadata |
| Timeline | Phân phối theo phút, lọc session, biểu đồ cột |
| Vùng lớp | Bảng N và heatmap 3×3, cảnh báo N<20 |
| Preview ẩn danh | Chọn và phát video overlay đã có trong `data/poses/` |
| Export | Tạo bundle, đổi bucket và tải file |
| Video mới | Upload, chọn checkpoint, giới hạn sampled frame, gọi pipeline |

Tab video mới lưu file local trong `data/raw/uploads/`, gọi `predict_video.py` bằng `subprocess.run`, chờ hoàn tất rồi hiển thị bảng và summary. Default demo là 150 sampled frame. Giao diện hiện có spinner, chưa có tiến độ từng công đoạn hoặc hàng đợi tác vụ nền. App cũng yêu cầu có experiment metadata ngay khi khởi động; nếu không có thì `st.stop()` trước phần upload.

```powershell
python -m streamlit run app/streamlit_app.py
```

### 13.3 Xuất báo cáo

`export_bundle` kiểm tra các cột bắt buộc rồi ghi năm artifact. [S14]

| File | Nội dung |
|---|---|
| `results.csv` | Bảng kết quả từng window |
| `timeline.csv` | Tổng hợp theo session và bucket |
| `zone_summary.csv` | Tổng hợp theo zone |
| `summary.json` | N, coverage, uncertain rate, nguồn nhãn, cảnh báo |
| `summary_report.md` | Báo cáo dễ đọc với bảng timeline và zone |

Ví dụ xuất từ experiment có sẵn:

```powershell
python scripts/export_report.py --experiment-dir reports/experiments/20261025-162855-lstm-full-v0-1 --out-dir reports/export_demo --bucket-seconds 60
```

Summary có thể tính accuracy trên accepted nếu có `true_label`, nhưng giữ tên `accepted_accuracy_vs_labeled_source` để không đồng nhất nhãn weak với nhãn người. Khi không có ground truth, giá trị này là null.

### 13.4 Overlay và giới hạn ẩn danh

`src/pose/visualization.py` vẽ skeleton, bbox và anonymous track ID lên video, có hàm `apply_face_blur`. Vùng blur lấy từ keypoint mặt và biên mở rộng; nếu thiếu thì fallback phần trên khoảng 28% bbox. Code có đường tạo MP4 bằng OpenCV và chuyển mã nếu công cụ cần thiết sẵn có.

```powershell
python scripts/visualize_poses.py --all --blur-face
```

Anonymous ID không tự xóa khuôn mặt trên video nền. Các hàm render tổng quát vẫn có tùy chọn blur; luồng video mới bật làm mờ mặc định, tạo head_blurred_preview.mp4 và dùng H.264 khi dependency FFmpeg sẵn có. Vùng làm mờ đã mở rộng ra toàn đầu/mặt. App chỉ liệt kê preview đã làm mờ; vẫn cần kiểm tra người bị detector bỏ sót trước khi công bố. HUD nhận tên model thực dùng, không cố định YOLOv8. [S02, S16, S20]

## 14 Giai đoạn 12 Kiểm thử benchmark và đóng gói

### 14.1 Kiểm thử phần mềm

Repository có test cho mapping nhãn, tính nhất quán cấu hình, schema dữ liệu, nguồn nhãn, pose thiếu, cleaner, feature/mask, cửa sổ, split leakage, forward LSTM/TCN, checkpoint, một bước train/evaluate, ghép metadata bằng ID, gate, zone, aggregation/export và bootstrap notebook. [S24]

Lệnh chạy theo hướng dẫn dự án:

```powershell
python scripts/check_setup.py
python -m pytest -q --basetemp=D:/ThiGiacMayTinh/.pytest_tmp
```

`--basetemp` phải trỏ tới thư mục tạm dành riêng cho pytest, không chứa dữ liệu cần giữ vì pytest có thể dọn nội dung. Đặt ngoài repository giúp test tìm root notebook không nhận nhầm thư mục tạm là repo.

Đã chạy lại ngày 09/11/2026: 133 test pass; dashboard và app gán nhãn đều AppTest 0 exception. Có thêm kiểm thử phase/timeline, gap, nhãn người, cấu hình suy luận, snapshot và TCN non-default. Kiểm thử tool gồm nhấn lưu trên phiếu tạm cho cả Quân/Phong, số 0.8, mở lại mức che khuất, phiếu độc lập khi đổi tên/bộ dữ liệu, bằng chứng bắt buộc, nhãn khớp tiêu chí, khung đúng track, không dùng tọa độ cũ khi mất người, seek và cache clip. Trình duyệt mở/phát clip H.264 4 giây ở dataset cũ, mobile không tràn ngang. CI cấu hình chạy Ubuntu/Python 3.12, cài requirements, kiểm tra skeleton rồi pytest. Có workflow không đồng nghĩa lần chạy CI mới nhất đã thành công.

### 14.2 Đo hiệu năng

`scripts/benchmark.py` đo pose/tracker, tốc độ temporal model và tùy chọn end-to-end. Benchmark model dùng tensor ngẫu nhiên batch 122 window, warm-up rồi 3 trial ×100 lượt, lấy trial nhanh nhất và chia cho 122 để báo ms/window. Đó là tốc độ trung bình quy đổi theo batch, không phải latency độc lập của một request đơn. [S25]

```powershell
python scripts/benchmark.py --video data/raw/class_1.mp4 --pose-frames 60 --trackers bytetrack.yaml,botsort.yaml --max-frames 150 --end-to-end --checkpoint models/checkpoints/20261025-162855-lstm-full-v0-1.pt --out reports/benchmark_demo.md
```

Artifact benchmark hiện lưu môi trường Windows 10 AMD64, Python 3.14.7, PyTorch 2.14.1+cpu. Chưa ghi đủ model CPU/RAM vật lý để tái lập phần cứng chính xác. Kết quả lưu như sau:

| Phép đo | Kết quả lưu | Điều kiện |
|---|---|---|
| ByteTrack pose | 4,33 sampled FPS | 60 sampled frame, class_1 |
| BoT-SORT pose | 4,06 sampled FPS | Cùng 60 frame |
| LSTM full | 0,260 ms/window quy đổi | Batch 122, best trial |
| TCN full | 0,140 ms/window quy đổi | Cùng kiểu benchmark |
| LSTM tham số | 246.531 | input 77, cấu hình full baseline |
| TCN tham số | 63.363 | Cấu hình full baseline |
| End-to-end | 6,26 sampled FPS | 150 sampled frame, 66 window |
| Thời gian nội bộ E2E | 23,955 giây | Đo trong script suy luận |
| Thời gian xử lý mỗi phút video | 95,82 giây/phút | Quy đổi từ video ngắn |
| Thời gian subprocess bên ngoài | 35,283 giây | Có thêm overhead khởi động |

Pose FPS riêng và E2E dùng phạm vi/warm-up khác nhau, không dùng chúng để kết luận rằng thêm mô hình làm hệ thống nhanh hơn. Chỉ số `peak_python_ram_mb` dùng tracemalloc, không đo toàn bộ RAM của tiến trình, native tensors hay VRAM. ID switches và fragmentation đang N/A vì chưa có nhãn tracking. [S25]

Tốc độ temporal rất nhỏ so với trích pose trong các artifact này; đây là dấu hiệu pose là phần chi phí lớn trên CPU. Chưa đủ bằng chứng để cam kết real time, GPU performance hoặc thời gian video dài. Khi benchmark mới cần ghi phần cứng, phiên bản, chiều dài video, sampled FPS, batch size, warm-up, số lần lặp và độ phân tán.

### 14.3 Đóng gói thực nghiệm

Gói bàn giao cần có code, requirements, config, hướng dẫn, split đã freeze, schema feature, metadata thí nghiệm, checkpoint được lưu đúng nơi và báo cáo giới hạn. Với Colab, notebook chỉ nên mount dữ liệu phù hợp và gọi `src/`/`scripts/`; không sao chép thành một codebase khác. Các notebook hiện được mô tả là template, chưa phải bằng chứng đã chạy huấn luyện hoàn chỉnh trên Colab.

Task demo cuối và đóng gói cuối còn NOT STARTED trong bảng trạng thái. Muốn nghiệm thu cần test với video độc lập, hoàn tất human annotation, kiểm tra ẩn danh và đo hiệu năng đúng điều kiện triển khai.

## 15 Hướng dẫn chạy dự án theo từng nhu cầu

### 15.1 Cài môi trường local

Chạy từ repository root. Các lệnh là hướng dẫn vận hành, có thể cài dependency và tạo artifact khi người vận hành thực hiện.

```powershell
Set-Location D:/ThiGiacMayTinh/observable-learning-engagement
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_setup.py
```

Nếu PowerShell không cho activate theo chính sách máy, có thể gọi trực tiếp `.venv/Scripts/python.exe` cho từng lệnh. Với GPU, cần bản PyTorch phù hợp môi trường GPU và ghi lại cấu hình hiệu lực; không lấy benchmark CPU hiện có làm kết quả GPU.

### 15.2 Xem kết quả đã có

Khi đã có `metadata.json`, `evaluation_report.json`, `predictions.csv` và checkpoint tương ứng, chỉ cần chạy Streamlit. Không bắt buộc huấn luyện lại. Các tab overview/timeline/zone dùng artifact lưu; tab video mới cần checkpoint tồn tại trên đĩa.

### 15.3 Tạo bộ dữ liệu và huấn luyện mới

Thực hiện lần lượt: kiểm tra video/metadata → trích pose có `--track` cho từng video → giữ một bản pose chuẩn → gán zone nếu cần → freeze split → build feature v1/v2 → kiểm tra nguồn nhãn và số lượng → train → evaluate → export. Lệnh cụ thể nằm tại các giai đoạn 3–11.

Các file split hiện hành có đủ 11 session. Nếu chỉ trích một video rồi build với split cũ, validator sẽ báo thiếu session; đó là hành vi đúng. Muốn làm dataset thử riêng cần thư mục pose và split thử riêng, có ít nhất 3 session để tạo train/val/test; không sửa split chính chỉ để chạy thử.

Huấn luyện với tensor v2 phải dùng NPZ v2 và cấu hình v2. Mô hình input 77 không sử dụng trực tiếp tensor 94 chiều. Các model factory không tự ánh xạ feature bằng tên khi schema thay đổi.

### 15.4 Phân tích video mới

Dùng tab Video mới hoặc CLI `predict_video.py` với checkpoint đã chọn. Bắt đầu bằng khoảng 150 sampled frame để kiểm tra luồng; bỏ `--max-frames` ở CLI nếu muốn xử lý hết video. Chất lượng đầu ra phụ thuộc track có đủ 40 quan sát và chất lượng pose; tăng giới hạn frame không bảo đảm mọi người đều có window hợp lệ.

Pipeline hiện hướng tới xử lý file video. Mặc dù sơ đồ mục tiêu có camera, chưa có entry point live webcam hoàn chỉnh trong app/predict script đã kiểm tra. Không trình bày tính năng camera thời gian thực như đã bàn giao.

## 16 Trạng thái hiện tại và hướng phát triển

### 16.1 Phần đã có triển khai

Theo `docs/PROJECT_STATUS.md`, các phần cốt lõi đã có code/artifact gồm pose, tracking, quality gate, cleaning, feature v1/v2, windows và frozen split, LSTM/TCN, train/evaluate, uncertainty, dashboard và export. Upload video đã gọi được full inference. Các bằng chứng về độ đúng phần mềm và độ đúng khoa học cần đọc riêng: test phần mềm không thay thế nhãn người.

Ở tầng pose, repository có cả baseline YOLOv8n và nhánh YOLO11m-HD, cùng trọng số YOLOv8m có thể chọn qua CLI. Nhánh YOLO11 có cấu hình, đường chạy và artifact overlay riêng; chưa có đánh giá đối chứng theo extractor hoặc liên kết đầy đủ từ model pose tới từng dataset temporal. [S26]

### 16.2 Phần còn thiếu

| Hạng mục | Tình trạng lưu | Việc cần thực hiện |
|---|---|---|
| Nhãn người và double annotation | Chưa có human test set | Gán độc lập, kiểm tra schema, QA và phân xử |
| Thu thập dữ liệu chính | PARTIAL | Xác minh đủ thời lượng, người, phase và metadata |
| M0 rule baseline và M1 ML | NOT STARTED | Xây baseline so sánh độc lập trên cùng split |
| Representation/motion/window ablation | NOT STARTED | Thiết kế biến thể, chọn bằng validation |
| Calibration | NOT STARTED | Reliability, ECE, temperature scaling hoặc phương pháp phù hợp |
| Phân tích F1/bias theo zone | PARTIAL | Bổ sung ground truth, N, pose quality và occlusion |
| Tracker quality | PARTIAL | Gán ground truth track và đo ID switch/fragmentation |
| Ẩn danh demo | PARTIAL | Bật blur phù hợp và kiểm tra trực quan |
| Benchmark dài và GPU | PARTIAL | Đo đúng video/hardware triển khai, nhiều lần lặp |
| Demo cuối và package cuối | NOT STARTED | Kiểm tra độc lập và đóng gói bàn giao |

### 16.3 Thứ tự thực hiện đề xuất

Trước hết cần hoàn thiện nhãn người, vì đây là điều kiện để đánh giá chất lượng độc lập. Song song có thể xử lý các khác biệt cấu hình và khóa dữ liệu đã nêu ở giai đoạn 4–7, bảo đảm pipeline sử dụng cùng sample rate, feature schema và kiến trúc checkpoint. Sau đó huấn luyện lại theo split cố định, chạy baseline/ablation, chọn ngưỡng trên validation và đánh giá test một lần theo protocol đã thống nhất.

Bước cuối là chuẩn hóa demo ẩn danh, benchmark trên video dài và đóng gói môi trường. Những thay đổi này nên tách thành nhiệm vụ có artifact kiểm chứng, không chỉ cập nhật trạng thái trong báo cáo. Phần này là hướng thực hiện tiếp theo, không phải danh sách chức năng đã có.

## 17 Các điểm đã sửa và phần cần hoàn tất

Ngày 09/11 đã sửa: không mặc định thiếu phase là lecture; không tự luân phiên phase khi cắt video; không tự xác nhận consent; áp dụng timeline theo window; tách camera; bỏ window qua gap; từ chối pose trùng; nối YAML pose tới inference; thêm batching và thiết bị temporal; sửa checkpoint TCN, snapshot hiệu lực và lỗi biến sess khi save-viz; sửa template/QA; thêm báo cáo vị trí tách human/weak; bật preview làm mờ trong luồng video mới và bổ sung codec H.264. [S27]

Phần chưa thể hoàn tất tự động: người thật xác minh phase và gán nhãn độc lập, phân xử, đánh giá accuracy/F1 và sai lệch vị trí trên nhãn đó. Các run v0.1 giữ nguyên ý nghĩa weak-label; chưa train lại trên dataset mới đang chờ review. Chưa đo video dài/GPU, chưa có calibration/ablation đầy đủ. Tính độc lập theo người/buổi quay vẫn cần kiểm tra nguồn; tên thư mục session không chứng minh được điều đó.

App gán nhãn: `python -m streamlit run app/annotation_app.py --server.port 8502`, mở http://127.0.0.1:8502, chọn Quân/Phong để vào phiếu riêng. Hướng dẫn QA, đánh giá và bàn giao: `docs/HUMAN_REVIEW_RUNBOOK.md`.

## 18 Danh mục mã nguồn và bằng chứng

Các đường dẫn dưới đây tính từ thư mục gốc repository. Tên hàm giúp định vị nội dung cụ thể khi đọc code.

| Mã | File hoặc nhóm file | Nội dung đối chiếu |
|---|---|---|
| S01 | `src/contracts/labels.py`, `docs/scope.md` | Mapping nhãn, annotation state và provenance |
| S02 | `src/pose/extractor.py`, `scripts/extract_pose.py` | `extract_frame_poses`, `extract_poses_from_video`, CLI |
| S03 | `src/pose/tracker.py` | `compute_track_statistics`, `filter_short_tracks` |
| S04 | `src/preprocessing/quality_gate.py` | `evaluate_pose_quality` |
| S05 | `src/preprocessing/cleaner.py`, `scripts/clean_poses.py` | Interpolation, smoothing, normalization |
| S06 | `src/preprocessing/features.py`, `feature_contract.py` trong cùng thư mục | Feature 77/94, `validate_temporal_features` |
| S07 | `src/preprocessing/windows.py`, `weak_labels.py`, `scripts/build_dataset.py` | Window, weak rule, frozen split, human join |
| S08 | `src/training/dataset.py`, `trainer.py` | Dataset, class weight, train/evaluate loop, metric |
| S09 | `src/models/lstm.py` | `ParticipationLSTM` |
| S10 | `src/models/tcn.py`, `src/models/factory.py` | TCN, factory và checkpoint reconstruction |
| S11 | `scripts/train.py` | Huấn luyện, chọn checkpoint, metadata |
| S12 | `scripts/evaluate.py` | Join theo window ID, raw/accepted metric |
| S13 | `src/uncertainty/gate.py` | Gate và lý do từ chối |
| S14 | `scripts/assign_zones.py`, `configs/zones/grid_3x3.yaml`, `src/reporting/aggregation.py`, `exporter.py` | Zone, timeline, summary, export |
| S15 | `scripts/sample_annotation_subset.py`, `scripts/dataset_qa.py`, `docs/annotation_guide.md` | Human annotation, QA, Cohen’s kappa |
| S16 | `scripts/predict_video.py`, `app/streamlit_app.py`, `src/pose/visualization.py` | Inference, UI, blur, overlay |
| S17 | `scripts/new_experiment.py`, `docs/experiment_protocol.md` | Experiment ID, snapshot, hash và version |
| S18 | `requirements.txt`, `configs/baseline.yaml`, `docs/ENVIRONMENT.md` | Dependency và tham số baseline |
| S19 | `docs/data_collection_protocol.md`, `docs/dataset_schema.md` | Thu thập, schema và metadata |
| S20 | `docs/privacy_and_consent.md`, `docs/LIMITATIONS.md`, `docs/PROJECT_STATUS.md` | Quyền riêng tư, giới hạn và trạng thái |
| S21 | `data/features/split_report.json`, `data/splits/*_sessions.txt` | Số window, nguồn nhãn, split cố định |
| S22 | `reports/experiments/20261025-162855-lstm-full-v0-1/evaluation_report.json` | Metric LSTM full |
| S23 | `reports/experiments/20261025-163801-tcn-full-v0-1/evaluation_report.json` | Metric TCN full |
| S24 | `tests/`, `.github/workflows/ci.yml`, `docs/PROJECT_STATUS.md` | Nội dung test, workflow và kết quả xác minh lưu |
| S25 | `scripts/benchmark.py`, `reports/benchmark.json`, `reports/benchmark.md` | Phương pháp và số đo hiệu năng lưu |
| S26 | `configs/experiments/pose_yolo11m_hd.yaml`, `src/pose/extractor.py`, `scripts/extract_pose.py`, `scripts/visualize_poses.py`, `yolo11m-pose.pt`, `data/poses/class_1_yolo11m_hd_overlay.mp4` | Nhánh YOLO11m-HD, chọn model/imgsz, trọng số và artifact overlay local |
| S27 | `src/preprocessing/context.py`, `scripts/audit_dataset.py`, `scripts/evaluate_human.py`, `scripts/prepare_inference_annotations.py`, `src/reporting/bias_analysis.py`, `app/annotation_app.py`, `docs/HUMAN_REVIEW_RUNBOOK.md` | Đợt sửa 09/11, audit dataset và quy trình đánh giá độc lập |

## 19 Thuật ngữ dùng khi trình bày

| Thuật ngữ | Ý nghĩa trong dự án |
|---|---|
| Pose | Vị trí các điểm khớp cơ thể 2D và confidence |
| Keypoint | Một điểm như vai, khuỷu, cổ tay hoặc mũi |
| Track | Chuỗi detection được tracker ghép theo thời gian |
| Window | Đoạn gồm số bước cố định của một track |
| Temporal model | Mô hình học quan hệ theo thời gian trong window |
| Weak label | Nhãn tự sinh từ quy tắc, cần tách nguồn với nhãn người |
| Ground truth | Nhãn tham chiếu độc lập theo quy trình đã kiểm chứng |
| Leakage | Dữ liệu liên quan bị lọt giữa train và tập đánh giá |
| Macro-F1 | Trung bình F1 các lớp, mỗi lớp trọng số ngang nhau |
| Coverage | Tỷ lệ mẫu được chấp nhận trong tập đưa qua gate |
| Calibration | Mức phù hợp giữa xác suất mô hình và tần suất đúng |
| Ablation | Thử bỏ/đổi một thành phần để đo đóng góp của nó |
| Head orientation proxy | Chỉ dấu hình học đầu gần đúng, không phải gaze |
| Zone | Vùng hình học trên ảnh theo cấu hình lưới |

Một cách mô tả ngắn khi thuyết trình: hệ thống chuyển video thành chuỗi tư thế của các track ẩn danh, chuẩn hóa và trích đặc trưng để LSTM/TCN ước lượng ba mức tham gia quan sát được trong từng cửa sổ; tầng bất định giữ lại các dự đoán đủ căn cứ, sau đó tổng hợp theo thời gian và vùng. Kết quả thực nghiệm hiện tại là mức đồng thuận với nhãn quy tắc, cần nhãn người độc lập để đánh giá chất lượng hành vi.
