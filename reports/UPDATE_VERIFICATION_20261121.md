# Đánh giá và cập nhật theo kế hoạch đính kèm

Ngày 21/11/2026. Đối chiếu `detailed_update_plan (1).md` và `relabel_videos_app (1).py` với nhánh `feature/ver2`, bắt đầu từ commit `6f7a3c9`. Các sửa đổi dưới đây ở working tree.

## Đánh giá kế hoạch

Nhánh hiện tại đã chứa `origin/main` đang lưu local, dependency đã ghim phiên bản và có TCN, dashboard, reporting. Baseline kiểm thử trước cập nhật đạt 148 passed. `python -m pip check` báo không có dependency hỏng. Chưa fetch remote trong lượt này.

Script sinh nhãn đã dùng WEAK_RULE. App đính kèm vẫn có lỗi ghi đè dữ liệu nguồn, trộn đánh giá người gán, tự điền nhãn cũ và tìm sai đường dẫn video/pose. Các phần này được cập nhật theo hợp đồng repository.

## Các thay đổi

| Phần | Kết quả |
|---|---|
| Re-label | Phiếu riêng ANN_PHONG, ANN_TAI và ADJUDICATOR, lưu nguyên tử trong `data/annotations/relabel_reviews/<source>/`. Ưu tiên confidence ≤0.80 hoặc PEER_ORIENTED. Khóa cửa sổ được tạo ổn định từ session/camera/track/time. |
| Quan sát | Clip đầy đủ, bốn ảnh đại diện, bbox và skeleton đúng track, lọc session/camera. Nhãn VALID cần clip, xác nhận, phase và bằng chứng. App này dùng guide v0.2, app annotation hiện có tiếp tục dùng rubric pilot riêng. |
| Dữ liệu thiếu | Độ dài gap được ghi ở chính các mẫu thực sự được nội suy. Smoothing giữ NaN ở các khoảng chưa điền. |
| Quality gate | Tính visibility thân trên từ tọa độ quan sát trước cleaning. Config YOLO11 HD bật lựa chọn dùng thân trên xuyên window và uncertainty gate. Baseline chính thức tiếp tục dùng toàn thân cho window/inference. |
| Nối track | Cleaner, builder và inference nhận tùy chọn `tracking.merge_max_gap_seconds`. Mặc định chưa bật heuristic nối ID, cần kiểm chứng trên tracking ground truth. |
| Pose CSV inference | Sửa FPS từ config, nhóm cleaning theo recording/camera, sửa lỗi log khi đầu vào CSV, chỉ cho chọn một loại input. |
| Report | Export shareable loại window_id chứa track ID, track_id, bbox và cột tự do bằng whitelist. Predictions nội bộ giữ khóa phục vụ kiểm toán. |
| Dashboard | Upload có thư mục riêng theo lượt chạy, timeline/zone/export xem được lượt suy luận mới. Ngưỡng mặc định lấy từ calibration checkpoint, override bằng lựa chọn rõ trong UI. |
| BiLSTM | CLI train hỗ trợ bilstm và checkpoint lưu bidirectional để dựng lại đúng kiến trúc. |
| Calibration | `--calibrate` fit temperature bằng validation NLL, lưu temperature và threshold trong checkpoint. Chọn threshold theo validation accuracy dưới ràng buộc coverage mặc định 0.90. Test chỉ dùng để báo cáo. Evaluation và inference dùng lại cùng temperature/threshold. |
| Coverage | Curve evaluation tính cả mẫu bị từ chối chất lượng. Accuracy ở mức coverage bằng 0 để trống. |

Coverage dùng để chọn threshold trong train hiện chỉ xét model confidence. Curve evaluation kết hợp quality gate. Đây là hai mẫu số cần đọc đúng trong artifact. Calibration thực tế vẫn cần validation labels đã nghiệm thu.

## Kiểm tra thực hiện

- Lần kiểm tra cuối `python -m pytest -q -p no:cacheprovider` đạt **152 passed in 23.66s**, gồm 148 bài hiện có và 4 bài hồi quy về gap, export, coverage và phiếu legacy.
- Streamlit AppTest đã mở clip từ video thực, nhấn lưu HUMAN trên phiếu tạm, đọc lại CSV, mở phiếu ANN_TAI riêng và kiểm tra byte nguồn không đổi. Dashboard AppTest không có exception. Một lượt điều khiển AppTest đổi reviewer ngay sau submit gặp lỗi serialize option, lượt kiểm tra lại bằng session riêng cho từng reviewer đã đạt. Chưa xác minh thao tác đổi reviewer bằng trình duyệt thật.
- Smoke BiLSTM với tensor tổng hợp 12 mẫu/split, một epoch trên CPU đã đi qua train, fit temperature trên validation, checkpoint reload và evaluate. NLL validation sau fit không tăng. Smoke này kiểm tra phần mềm, không cung cấp metric nghiên cứu.
- Lệnh inference từ `data/poses/class_1_cam_A_poses.csv` với checkpoint LSTM full cũ trả exit code 0, xuất 54 window, 51 accepted và 3 uncertain. Export không có track_id/window_id. Đây là inference không ground truth.
- `compileall` và `git diff --check` dùng kiểm tra syntax và patch.

## Đối chiếu hai cấu hình pose

Đã đo ngày 20/11 trên 60 frame lấy mẫu đầu tiên của mỗi video, 10 FPS mục tiêu, CPU, PyTorch 2.14.1+cpu. Thời gian bao gồm nạp/trích pose, chưa phải benchmark dài hoặc steady-state. Cấu hình YOLOv8n và YOLO11m khác model, resolution và confidence. Đánh giá sự khác biệt áp dụng cho toàn cấu hình.

| Video | Cấu hình | Detection | Mean visible ratio | Mean confidence | FPS |
|---|---|---:|---:|---:|---:|
| class_1 | YOLOv8n baseline | 627 | 0.7921 | 0.6523 | 6.59 |
| class_1 | YOLO11m HD 1280 | 830 | 0.7064 | 0.6416 | 0.66 |
| class_3 | YOLOv8n baseline | 748 | 0.8434 | 0.6947 | 7.59 |
| class_3 | YOLO11m HD 1280 | 752 | 0.7526 | 0.6792 | 0.62 |

Số liệu thô lưu ở `reports/pose_ab_20261120.json`. Mean confidence/visibility giữa hai nhóm detection không phải phép so cùng người. Cần nhãn missed detection/keypoint theo vùng để kết luận chất lượng. Kết quả ngắn hiện chưa đủ cơ sở chốt YOLO11m làm baseline chính thức.

## Cách dùng

```powershell
python -m streamlit run app/relabel_app.py --server.port 8502
python -m streamlit run app/streamlit_app.py
python -m pytest -q -p no:cacheprovider
python scripts/train.py --model bilstm --features <dataset-da-nghiem-thu.npz> --config configs/baseline.yaml --name bilstm-calibrated --calibrate
```

Mỗi reviewer dùng phiếu riêng và gán các cửa sổ tương ứng để tính đồng thuận. QA và phân xử theo `docs/HUMAN_REVIEW_RUNBOOK.md`. Chỉ gộp các file đã QA/phân xử và ghi version dataset. Việc tạo `annotations_all_v0.2.csv`, báo Kappa và train chính thức còn phụ thuộc nhãn người thật. Không tăng label schema version chỉ vì đổi nguồn nhãn nếu enum vẫn giữ nguyên.

Chưa nghiệm thu hiệu quả trên human test, độ đúng nối track, calibration trên human validation, benchmark GPU/video dài hoặc bỏ sót người trong preview. Chưa có căn cứ chuyển các task nghiên cứu tương ứng sang DONE.
