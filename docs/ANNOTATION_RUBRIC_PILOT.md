# Tiêu chí gán nhãn hành vi — v0.3-pilot

Trạng thái: **đề xuất thí điểm, chờ Quân/Phong và người phụ trách thống nhất**. Không phải thang đo đã được kiểm chứng. Chưa có số liệu đồng thuận mới.

## Đơn vị và đối tượng

Một nhãn áp dụng cho một `session_id + camera_id + track_id + t_start_ms + t_end_ms`, thường dài khoảng 4 giây. Khung vàng và phần phóng to trong tool chỉ đúng người cần xem. Không gán chung cho cả lớp, không coi track ID là danh tính xuyên video. Nếu tracker đổi sang người khác, chọn INVALID và ghi lỗi ID.

## Cơ sở tham khảo và giới hạn

[BROMP — nhóm tác giả, University of Pennsylvania](https://learninganalytics.upenn.edu/ryanbaker/bromp.html) và [manual 2.0](https://learninganalytics.upenn.edu/ryanbaker/BROMP.pdf) là tài liệu tham khảo về quan sát theo cá nhân, quy trình huấn luyện người gán và kiểm tra độ đồng thuận. Dự án chỉ tham khảo nguyên tắc quan sát hành vi; không triển khai hoặc tuyên bố chứng nhận BROMP, không dùng các nhãn cảm xúc của protocol đó.

**Bốn khoảng thời gian, ngưỡng 3/4 và ánh xạ ba mức dưới đây là lựa chọn vận hành đề xuất riêng của dự án. BROMP không xác nhận các ngưỡng này.** Cần thử nghiệm, phân xử bất đồng và rà soát tính phù hợp của cấu trúc nhãn trước khi đóng băng phiên bản chính thức. Độ đồng thuận cao cũng không tự chứng minh đo được chú ý, hiểu bài hoặc tham gia bên trong.

## Bước 1: xác minh nhiệm vụ thực tế

Xem thêm ngữ cảnh ngoài đoạn cần gán, đối chiếu lịch hoạt động hoặc người cung cấp video nếu có. Ghi bằng chứng của pha lecture/individual_work/discussion. Không dùng phase tự sinh trong metadata cũ làm căn cứ. Khi không xác định được nhiệm vụ, giữ UNKNOWN và không gán một trong ba mức tham gia.

| Pha đã biết | Ví dụ bằng chứng phù hợp | Trường hợp không được suy diễn |
|---|---|---|
| Giảng bài | Thân/đầu chủ yếu hướng vùng giảng dạy; ghi chép gắn với hoạt động đang diễn ra | Chỉ nhìn thẳng không chứng minh hiểu bài; cúi đầu không mặc định không phù hợp |
| Làm bài cá nhân | Nhìn được tài liệu bài tập; tay viết hoặc thao tác đúng nhiệm vụ đã biết | Chỉ đầu cúi, không thấy tài liệu/tay: chưa rõ |
| Thảo luận | Hướng về nhóm kèm cử chỉ/trao đổi phù hợp hoạt động đã xác minh | Chỉ quay sang bạn không đủ kết luận đang tham gia thảo luận |

Hành vi **không phù hợp** phải có bằng chứng cụ thể mâu thuẫn với nhiệm vụ đã biết, thí dụ đang làm một hoạt động khác mà ngữ cảnh xác nhận không thuộc nhiệm vụ. Không mặc định dùng điện thoại, ngồi yên, quay đầu hoặc cúi đầu là không phù hợp. Nếu không nhìn rõ đang làm gì hoặc không biết yêu cầu bài học, ghi chưa rõ.

## Bước 2: ghi bằng chứng trong bốn khoảng

Chia cửa sổ thành bốn khoảng gần bằng nhau, xem cả diễn biến trong mỗi khoảng và đánh dấu một trong:

- ALIGNED: thấy hành vi rõ phù hợp với nhiệm vụ thực tế.
- CONTRADICTORY: thấy hành vi rõ không phù hợp với nhiệm vụ thực tế, có mô tả cụ thể để người khác kiểm tra.
- AMBIGUOUS: thấy người nhưng chưa xác định được hành vi có phù hợp không.
- UNOBSERVABLE: che khuất hoặc không xác định được đúng người.

Nếu hành vi đổi trong một khoảng và không thể chọn rõ, dùng AMBIGUOUS. Không suy ra từ mặt/cảm xúc hoặc chỉ một frame; video làm mờ vẫn có thể không đủ bằng chứng. Phân biệt **chưa rõ** với **không phù hợp**.

## Bước 3: áp dụng bảng quy đổi thử nghiệm

| Bằng chứng của bốn khoảng | Mức được phép chọn nếu pha đã xác minh và trạng thái VALID |
|---|---|
| Ít nhất 3 ALIGNED | HIGH_OBSERVED_PARTICIPATION |
| Ít nhất 3 CONTRADICTORY | LOW_OBSERVED_PARTICIPATION |
| Ít nhất 3 khoảng rõ (ALIGNED/CONTRADICTORY), nhưng không bên nào đạt 3 | MODERATE_OBSERVED_PARTICIPATION |
| Ít hơn 3 khoảng rõ | Không đủ bằng chứng; không gán một trong ba mức |

MODERATE ở phiên bản thí điểm này là trường hợp có bằng chứng hành vi xen kẽ, không phải nhãn thay thế cho thiếu dữ liệu. Đây là cách vận hành hẹp hơn hướng dẫn v0.2; không gộp nhãn hai phiên bản mà không rà soát. Cần đánh giá liệu hướng dẫn có phân biệt được ba mức trong bối cảnh video thực tế hay không.

Nếu thiếu bằng chứng vì che khuất, chọn OCCLUDED; nếu đổi pha/trạng thái rõ trong cửa sổ, chọn TRANSITION; nếu lỗi ID, ngoài khung, không biết nhiệm vụ hoặc dữ liệu không đủ vì nguyên nhân khác, chọn INVALID và ghi nguyên nhân. Các trạng thái này không có participation_label/label_source. Không dùng UNCERTAIN làm training class.

Tool yêu cầu mô tả hành vi, căn cứ xác định pha, đánh dấu bốn khoảng và xác nhận đã xem đúng người trước khi lưu VALID. Nhãn chọn thủ công phải khớp bảng quy đổi. `annotation_evidence` lưu JSON bằng chứng và xác nhận; `phase_evidence` lưu căn cứ nhiệm vụ; `annotation_guide_version=v0.3-pilot`. Đây là ánh xạ từ quan sát do người dùng nhập và xác nhận, không phải gợi ý từ mô hình.

## Bước 4: pilot và đóng băng hướng dẫn

1. Quân và Phong thống nhất cách hiểu hướng dẫn bằng một nhóm ví dụ luyện tập, không dùng nhóm này để báo điểm đánh giá độc lập.
2. Chọn một nhóm pilot riêng, thí dụ 30–50 cửa sổ có nhiều pha, vị trí, mức che khuất và người. Số lượng này là đề xuất thực hành, không phải bảo đảm thống kê.
3. Mỗi người gán độc lập, không xem phiếu đối phương hoặc dự đoán mô hình. So sánh cả trạng thái, pha, bốn khoảng bằng chứng và nhãn.
4. Chạy QA/Kappa, xem tỷ lệ không gán được và các nhóm bất đồng. Không chỉ tính Kappa trên một nhóm dễ rồi tuyên bố toàn bộ dataset đáng tin cậy. Kappa ≥0.60 là ngưỡng review đang dùng trong dự án, không phải kết quả đã đạt và không thay thế kiểm tra validity.
5. Phân xử từng bất đồng bằng video và căn cứ; sửa hướng dẫn, tăng version khi cần. Nếu sửa tiêu chí, gán lại các mẫu liên quan, ghi lịch sử. Không chỉnh nhãn chỉ vì dự đoán mô hình khác.
6. Đóng băng hướng dẫn và tập test trước đánh giá cuối. Giữ nguyên split; không dùng test để chọn tiêu chí nhằm tăng điểm mô hình. Ưu tiên chốt hướng dẫn trên nhóm luyện tập/pilot từ train/validation trước khi gán test chính thức.
