# Annotation Guide v0.2 — CVF25PRJ10

**Task:** T04  
**Mục tiêu:** Để hai annotator độc lập có thể xem cùng một cửa sổ video và áp dụng cùng quy tắc mà không suy luận cảm xúc/ý định.

## 1. Nguyên tắc bắt buộc

1. Gán nhãn từ **hành vi quan sát được trong cả cửa sổ**, không từ một frame đơn.
2. Luôn xét `class_phase` (`lecture`, `individual_work`, `discussion`, ...).
3. Không dùng các từ “lười”, “buồn chán”, “không có động lực”, “mất tập trung” như ground truth.
4. Khi bằng chứng không đủ, dùng quality/annotation state phù hợp thay vì ép một trong ba training label.
5. `UNCERTAIN` là **system output**, không phải class hành vi thứ tư để temporal classifier học.
6. `participation_label` lưu bằng canonical value đầy đủ; không lưu `HIGH/MODERATE/LOW` trong dataset.
7. Mỗi VALID label phải ghi `label_source` để phân biệt HUMAN / WEAK_RULE / ADJUDICATED.

## 2. Model training classes — canonical stored values

### `HIGH_OBSERVED_PARTICIPATION`
Cửa sổ có nhiều tín hiệu quan sát phù hợp rõ với hoạt động học hiện tại và chất lượng quan sát đủ.

Ví dụ:
- `lecture`: thân/đầu chủ yếu hướng khu vực giảng dạy, tư thế tham gia ổn định;
- `individual_work`: đọc/viết rõ, chuyển động tay phù hợp nhiệm vụ;
- `discussion`: hướng về nhóm, có thay đổi tư thế/cử chỉ tương tác phù hợp.

### `MODERATE_OBSERVED_PARTICIPATION`
Có một số tín hiệu phù hợp nhưng yếu, không liên tục hoặc ít hoạt động; không đủ mạnh để xếp HIGH và cũng không có bằng chứng kéo dài để xếp LOW.

### `LOW_OBSERVED_PARTICIPATION`
Trong phần lớn cửa sổ có nhiều tín hiệu quan sát **không phù hợp với hoạt động hiện tại**, với chất lượng quan sát đủ để đưa ra nhãn. Không gán nguyên nhân tâm lý.

## 3. Annotation / quality states

- `VALID`: có đủ bằng chứng để dùng một trong 3 canonical training label.
- `OCCLUDED`: bị che đáng kể.
- `TRANSITION`: cửa sổ cắt qua thay đổi hoạt động/trạng thái rõ.
- `INVALID`: ngoài khung, lỗi video, không đủ dữ liệu.

Nếu state không phải VALID thì `participation_label`, `label_source`, `label_rule_version` đều để trống.

## 4. Label provenance

- `HUMAN`: người gán trực tiếp; `label_rule_version` trống.
- `WEAK_RULE`: rule/labeling function sinh nhãn; phải ghi `label_rule_version` như `weak-v0.1`.
- `ADJUDICATED`: nhãn sau review bất đồng/weak label; `label_rule_version` trống.

Mọi row ghi `annotation_guide_version=v0.2`. Weak label không được dùng làm ground truth test chính thức nếu chưa được adjudicate theo protocol.

## 5. Atomic observable behaviors

| Code | Mô tả | Không được suy luận |
|---|---|---|
| `BOARD_OR_INSTRUCTOR_FACING` | Thân/đầu chủ yếu hướng vùng giảng dạy | Không đồng nghĩa “đang chú ý” |
| `READING_WRITING` | Tư thế/chuyển động phù hợp đọc hoặc viết | Head-down không phải LOW mặc định |
| `HAND_RAISED` | Tay cao hơn vai trong một khoảng | Không tự động = HIGH toàn cửa sổ |
| `PEER_ORIENTED` | Hướng về người bên cạnh/nhóm | Có thể đúng trong discussion |
| `HEAD_DOWN` | Đầu ở vị trí thấp kéo dài | Không suy luận ngủ/mất tập trung |
| `BODY_TURNED` | Thân quay khỏi hướng tham chiếu | Ý nghĩa phụ thuộc phase |
| `OCCLUDED` | Không thấy đủ keypoint | Dùng quality state |
| `OUT_OF_VIEW` | Ra khỏi vùng quan sát | Không dự đoán |
| `UNKNOWN` | Không đủ căn cứ | Không ép nhãn |

Có thể lưu nhiều atomic behavior bằng dấu `;` trong CSV.

## 6. Window annotation

Baseline:
- window: 4 giây;
- stride: 2 giây;
- annotator phải xem toàn bộ cửa sổ;
- ghi `class_phase`, `zone_id`, `occlusion_level`, `annotation_confidence`.

Nếu > khoảng 40% cửa sổ bị che hoặc không quan sát được đủ tư thế, ưu tiên state `OCCLUDED/INVALID`; ngưỡng cuối cùng phải được review sau pilot.

## 7. Decision flow

```text
Có đủ dữ liệu quan sát?
 ├─ Không → OCCLUDED / INVALID, không label
 └─ Có
     ↓
Cửa sổ đang chuyển phase/trạng thái rõ?
 ├─ Có → TRANSITION, không label
 └─ Không
     ↓
Xét hành vi + class_phase
     ↓
Canonical HIGH / MODERATE / LOW label
     ↓
Ghi label_source (+ rule version nếu WEAK_RULE)
```

## 8. Các trường hợp dễ gán sai

- **Cúi đầu:** trong `individual_work` + dấu hiệu đọc/viết có thể HIGH/MODERATE; không tự động LOW.
- **Quay sang bạn:** trong `discussion` có thể HIGH/MODERATE; trong `lecture` phải xem tỷ lệ thời gian/context.
- **Ngồi yên:** không tự động HIGH hay LOW; có thể MODERATE nếu phù hợp phase nhưng tín hiệu yếu.

## 9. Double annotation QA

- Hai annotator gán độc lập ít nhất 15–20% subset.
- Báo Cohen’s kappa cho categorical label.
- Review disagreement và cập nhật guide trước annotation toàn bộ.
- Đóng băng guide version trước khi tạo test set chính thức.
- Không sửa test label vì model dự đoán khác; chỉ sửa khi có lỗi annotation có bằng chứng và phải log.

## 10. File format / source of truth

- Schema: `docs/dataset_schema.md`.
- Template: `data/sample/annotation.example.csv`.
- Enum/mapping code: `src/contracts/labels.py`.

## 11. Owner / Review

- Owner: **Vũ Văn Phong**
- Review: **Vũ Đức Tài**
- Cả nhóm phải thống nhất v1 trước annotation chính thức.
