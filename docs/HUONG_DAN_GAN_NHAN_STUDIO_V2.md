# Hướng dẫn gán nhãn Studio V2 — Quy tắc thống nhất toàn nhóm

> Áp dụng cho: `AnnotationAppDesktop.exe` (Studio + Rubric), `AnnotationStudio_V2_Standalone.exe` và bản web `annotation_app.py`.
> Mục tiêu: gán nhãn **nhất quán giữa Train / Val / Test** và **giữa các annotator** để không tụt Cohen's Kappa.

---

## 1. Nguyên tắc gán nhãn

1. **Nhìn toàn bộ cửa sổ (4 giây), không nhìn 1 ảnh đơn lẻ.**
   Phải xem hết vòng lặp video hoặc dải 5 ảnh diễn biến rồi mới bấm lưu.
2. **Chỉ mô tả hành vi quan sát được, KHÔNG suy diễn tâm lý.**
   Không gán theo cảm giác "nhìn có vẻ chán", "đang mất tập trung", "lười", "ngủ" nếu không có tư thế/hành vi vật lý tương ứng rõ ràng.
3. **Xem đúng người trong khung vàng (track_id).**
   Không bị phân tâm bởi bạn ngồi cạnh.
4. **Đánh giá theo đúng pha học (`class_phase`).**
   Cúi đầu viết bài trong giờ giảng khác với gục đầu nằm lên bàn; quay sang bạn trong giờ thảo luận khác với quay xuống nói chuyện trong giờ nghe giảng.
5. **Không ép nhãn khi không nhìn rõ.**
   Thà chọn **BỊ CHE (OCCLUDED)** hoặc **INVALID** còn hơn đoán mò làm nhiễu dữ liệu huấn luyện/kiểm thử.

---

## 2. Quy trình 4 bước cho mỗi cửa sổ 4 giây

### Bước 1: Kiểm tra chất lượng hiển thị (`annotation_state`)

| Lựa chọn | Khi nào chọn |
|---|---|
| **[4] BỊ CHE (OCCLUDED)** *(không chọn CAO/VỪA/THẤP)* | **> 40% thời lượng** cửa sổ (từ **2/5 ảnh diễn biến** trở lên) người này bị bàn ghế/người khác che mất vùng **đầu–vai–tay**, hoặc quá mờ, không đủ nhìn rõ hướng người & hành vi. |
| **INVALID** *(ô Trạng thái)* | Nhảy Track ID sang người khác giữa clip; người đi ra ngoài khung hình (OUT_OF_VIEW); video lỗi đứng hình; hoặc không xác định được lớp đang học pha gì. |
| **TRANSITION** *(ô Trạng thái)* | Cửa sổ cắt đúng lúc **đổi pha hoạt động** của lớp (ví dụ đang nghe giảng chuyển sang đứng dậy/dọn vở thảo luận nhóm) khiến không có trạng thái ổn định. |
| **VALID** | Nhìn rõ người trong **≥ 60%** cửa sổ (từ **3/5 ảnh diễn biến** trở lên) → đi tiếp Bước 2–4. |

### Bước 2: Xác định Pha học (`class_phase`) & Căn cứ

| Pha | Ý nghĩa |
|---|---|
| `lecture` (Giảng bài) | Giáo viên đang giảng/trình bày trước lớp (bộ video lớp học của nhóm hầu hết là pha này). |
| `individual_work` (Làm bài cá nhân) | Lớp đang tự làm bài tập, đọc tài liệu, gõ/viết độc lập. |
| `discussion` (Thảo luận nhóm) | Lớp đang trao đổi theo cặp hoặc nhóm. |

> ⚠️ Nếu không biết pha nào → giữ **UNKNOWN** và chuyển trạng thái về **INVALID**.
> Tuyệt đối không gán CAO/VỪA/THẤP khi pha chưa rõ.

### Bước 3: Tích chọn Hành vi quan sát được (`atomic_behavior` — bắt buộc ≥ 1 ô)

Chọn đúng các hành vi nhìn thấy trong **phần lớn cửa sổ**:

- [ ] Nhìn bảng / Giảng viên (`BOARD_OR_INSTRUCTOR_FACING`)
- [ ] Đọc sách / Viết / Gõ phím (`READING_WRITING`)
- [ ] Giơ tay phát biểu (`HAND_RAISED`)
- [ ] Quay sang thảo luận nhóm (`PEER_ORIENTED`)
- [ ] Cúi gục đầu kéo dài (`HEAD_DOWN_EXTENDED`)
- [ ] Xoay người hướng khác (`TURNING_AWAY`)
- [ ] Bị che khuất khớp xương (`OCCLUDED_KEYPOINTS`)
- [ ] Không rõ hành vi (`UNCLEAR`)

### Bước 4: Chọn Mức tham gia quan sát được (`participation_label`)

Chia nhẩm cửa sổ 4 giây thành **4 phần** (mỗi phần ~1 giây) để đếm tỷ lệ thời gian:

| Nút bấm | Nhãn lưu hệ thống | Tiêu chuẩn áp dụng (theo từng pha) |
|---|---|---|
| **[1] CAO** | `HIGH_OBSERVED_PARTICIPATION` | Hành vi phù hợp rõ ràng chiếm **≥ 3/4 thời lượng (~3–4 giây)**:<br>• **lecture:** Đầu & thân hướng ổn định về bảng/giảng viên; hoặc vừa nhìn bảng vừa ghi chép/đọc tài liệu; hoặc giơ tay phát biểu.<br>• **individual_work:** Tư thế đọc/viết/gõ phím rõ ràng, hướng vào tài liệu/máy tính ≥ 3s.<br>• **discussion:** Hướng thân/đầu về nhóm kèm cử chỉ trao đổi rõ ≥ 3s. |
| **[2] VỪA** | `MODERATE_OBSERVED_PARTICIPATION` | Tín hiệu **xen kẽ** hoặc phù hợp mức yếu (khoảng **1/2 đến < 3/4** cửa sổ):<br>• Hướng bảng/vở khoảng 2 giây rồi quay đi/cúi nghỉ nhẹ ~2 giây (xen kẽ, chưa bên nào chiếm trọn ≥ 3s).<br>• Ngồi hướng về phía trước nhưng tư thế chùng/ngả nhẹ, đầu hơi lệch/dao động, không thấy cử chỉ tích cực (ghi chép/hướng thẳng dứt khoát) và cũng không quay hẳn đi hay gục đầu.<br>• **Lưu ý:** VỪA dùng cho tư thế trung tính/xen kẽ **có quan sát được**, KHÔNG dùng thay thế cho BỊ CHE. |
| **[3] THẤP** | `LOW_OBSERVED_PARTICIPATION` | Hành vi **KHÔNG phù hợp** với pha học chiếm **≥ 3/4 thời lượng (~3–4 giây)** (vẫn nhìn rõ):<br>• **lecture:** Cúi gục hẳn đầu xuống bàn/khoanh tay gục đầu ≥ 3s (`HEAD_DOWN_EXTENDED`); hoặc xoay người/ngoảnh hẳn đầu sang hướng khác ≥ 3s (`TURNING_AWAY`); hoặc quay hẳn sang nói chuyện/nhìn bạn ≥ 3s (`PEER_ORIENTED` sai pha); hoặc dùng điện thoại/làm việc riêng rõ rệt.<br>• **individual_work:** Gục đầu bất động lên bàn ≥ 3s, hoặc quay hẳn người đi không nhìn bài.<br>• **discussion:** Tách khỏi nhóm, gục đầu hoặc quay hướng khác hoàn toàn ≥ 3s. |

---

## 3. Bảng xử lý 6 tình huống dễ gán lệch nhất *(Cả nhóm PHẢI thống nhất)*

Đây là các ca hay gây tụt Cohen's Kappa và lệch nhãn giữa tập Train và Test:

### 3.1. Ngồi yên, mặt/thân hướng thẳng về bảng suốt 4s (không viết, không giơ tay)

- Trong pha **lecture** → xếp **[1] CAO** (Nhìn bảng / Giảng viên).
- **Không hạ** xuống VỪA hay THẤP chỉ vì "ngồi im không chép bài".
- **Bắt buộc áp dụng đồng nhất ở cả Train, Val và Test.**

### 3.2. Cúi đầu xuống bàn

| Quan sát | Nhãn |
|---|---|
| Thấy tay cầm bút / lật sách / gõ bàn phím, hoặc cổ cúi góc vừa để nhìn vở | **[1] CAO** hoặc **[2] VỪA** (Đọc sách / Viết / Gõ phím) |
| Đầu gục sát mặt bàn / úp mặt vào khuỷu tay / gục cổ bất động kéo dài ≥ 3s | **[3] THẤP** (Cúi gục đầu kéo dài) |
| Chỉ cúi thoáng 1–2s rồi ngẩng lên nhìn bảng | **[2] VỪA** |
| Bị bàn/bạn phía trước che khuất mất phần lớn đầu–tay nên không phân biệt được đọc vở hay gục ngủ | **[4] BỊ CHE (OCCLUDED)** |

### 3.3. Quay sang nhìn bạn trong pha lecture

| Quan sát | Nhãn |
|---|---|
| Liếc nhanh **< 1.5s** rồi quay ngay về bảng | **[1] CAO** (hoặc VỪA nếu phần còn lại không hướng bảng rõ) |
| Quay khoảng **~2s** rồi quay lại bảng | **[2] VỪA** |
| Quay hẳn sang bạn **≥ 3s** | **[3] THẤP** (Quay sang thảo luận nhóm + Xoay người hướng khác) |

### 3.4. Ngồi bàn cuối, hình nhỏ/hơi nhòe

- Vẫn phân biệt được **hướng thân, hướng vai và hướng đầu** quay về đâu → gán **VALID** bình thường.
- Nhòe/khuất không nhận ra đang hướng về bảng hay cúi gục → **[4] BỊ CHE (OCCLUDED)**.

### 3.5. Chống tay lên cằm / nghiêng đầu nhìn bảng

- Hướng nhìn và thân vẫn hướng bảng **ổn định ≥ 3s** → vẫn là **[1] CAO** (hoặc **[2] VỪA** nếu tư thế ngả/lơ đãng đổi hướng nhiều lần).
- **Chống cằm KHÔNG tự động là THẤP.**

### 3.6. Gán vị trí ghế (`zone_id`)

- Một `track_id` trong cùng 1 video thường **ngồi cố định 1 chỗ**: `F-L / F-C / F-R`, `M-L / M-C / M-R`, `B-L / B-C / B-R`.
- Chọn **1 lần chuẩn ngay cửa sổ đầu tiên** của track đó để các cửa sổ sau giữ nhất quán.

---

*Phiên bản quy tắc: v0.2 · Nhóm CVF25PRJ10 · Cập nhật: 2026-11-26*
