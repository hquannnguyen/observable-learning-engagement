# Dataset Schema — CVF25PRJ10

Tài liệu này là **contract chính thức** cho metadata, annotation, pose và window dataset. Không thay đổi cột/enum bắt buộc mà không tăng schema/version và cập nhật consumer + test tương ứng.

## 1. Canonical label contract

### 1.1 Giá trị lưu trong dataset/model

`participation_label` **chỉ** được lưu bằng ba canonical value sau:

```text
HIGH_OBSERVED_PARTICIPATION
MODERATE_OBSERVED_PARTICIPATION
LOW_OBSERVED_PARTICIPATION
```

Không lưu `HIGH`, `MODERATE`, `LOW` vào annotation/window dataset. Ba tên ngắn chỉ dùng ở UI/report.

### 1.2 Mapping model/index/UI

| Canonical stored/model label | Model index | Display/output label |
|---|---:|---|
| `HIGH_OBSERVED_PARTICIPATION` | 0 | `HIGH` |
| `MODERATE_OBSERVED_PARTICIPATION` | 1 | `MODERATE` |
| `LOW_OBSERVED_PARTICIPATION` | 2 | `LOW` |
| — | — | `UNCERTAIN` |

`UNCERTAIN` không phải training label. Nó được tạo bởi quality/model-confidence gate. Khi export kết quả hệ thống nên giữ **cả** `canonical_label` và `display_label` nếu cần trao đổi máy-máy; với `UNCERTAIN`, `canonical_label=UNCERTAIN`.

Source of truth trong code: `src/contracts/labels.py`.

## 2. Session layout

```text
data/
├── raw/session_001/
│   ├── cam_A.mp4
│   └── metadata.json
├── annotations/
├── poses/
├── features/
└── splits/
```

Raw video không được commit lên repo public.

## 3. `metadata.json`

| Field | Type | Ví dụ |
|---|---|---|
| `session_id` | string | `S001` |
| `date` | `YYYY-MM-DD` | `2026-09-21` |
| `camera_id` | string | `A` |
| `fps` | number | `30` |
| `resolution` | string | `1920x1080` |
| `camera_view` | enum/string | `front_oblique` |
| `participants` | integer | `8` |
| `class_phase` | array | timeline phase |
| `lighting_condition` | string | `normal` |
| `consent_confirmed` | boolean | `true` |
| `notes` | string | ghi chú |

Xem `data/sample/metadata.example.json`.

## 4. Annotation CSV

Canonical columns:

```text
session_id,camera_id,track_id,t_start_ms,t_end_ms,class_phase,
atomic_behavior,participation_label,annotation_state,
label_source,label_rule_version,annotation_guide_version,
annotation_confidence,occlusion_level,zone_id,annotator_id,notes
```

### 4.1 State-label rule

- `annotation_state=VALID` → `participation_label` bắt buộc là một trong 3 canonical label.
- `OCCLUDED/TRANSITION/INVALID` → `participation_label` phải để trống.
- `annotation_confidence` thuộc `[0,1]`, là confidence của annotator/labeling process, **không phải model confidence**.
- `t_start_ms >= 0` và `t_end_ms > t_start_ms`.
- `track_id` là anonymous ID trong session, không phải danh tính thật.

### 4.2 Label provenance

`label_source` dùng để không trộn ground truth người gán với weak label:

| `label_source` | Ý nghĩa | `label_rule_version` |
|---|---|---|
| `HUMAN` | Annotator gán trực tiếp theo guide | để trống |
| `WEAK_RULE` | Labeling function/rule sinh weak label | **bắt buộc**, ví dụ `weak-v0.1` |
| `ADJUDICATED` | Nhãn sau khi review bất đồng/weak label | để trống |

Với row không `VALID`, `label_source` và `label_rule_version` để trống.

`annotation_guide_version` luôn ghi version guide đang dùng, ví dụ `v0.2`.

> Weak label **không tự động được coi là ground truth test set**. Test set chính thức phải dùng HUMAN/ADJUDICATED label theo protocol.

## 5. Pose/Tracking Parquet

Một row / track / sampled frame:

```text
session_id,camera_id,frame_idx,timestamp_ms,track_id,
bbox_x1,bbox_y1,bbox_x2,bbox_y2,bbox_conf,
kpt_00_x,kpt_00_y,kpt_00_conf,...,kpt_16_x,kpt_16_y,kpt_16_conf,
visible_ratio,mean_keypoint_conf,zone_id
```

Missing keypoint phải có mask/NaN hoặc representation rõ; **không dùng `(0,0)` như một keypoint thật**.

## 6. Feature Window Parquet

Tối thiểu:

```text
window_id,session_id,camera_id,track_id,t_start_ms,t_end_ms,
class_phase,zone_id,feature_schema_version,dataset_version,split_version,
label_schema_version,quality_state,mean_visible_ratio,mean_pose_confidence,
participation_label,label_source,label_rule_version,annotation_guide_version,split
```

Quy tắc label/state/provenance giống annotation CSV. Feature tensor có thể lưu trong parquet/list columns hoặc `.npy/.npz` + `feature_path`.

## 7. Versioning

Baseline version fields:

```text
dataset_version: v0.1
split_version: split-v0.1
label_schema_version: labels-v1
feature_schema_version: posefeat-v1
annotation_guide_version: v0.2
```

- `dataset_version`: nội dung dataset/annotation.
- `split_version`: phép chia train/val/test đã freeze.
- `label_schema_version`: enum/mapping/relationship của label.
- `feature_schema_version`: representation model input.
- Mọi thay đổi làm ảnh hưởng thí nghiệm phải tăng version thích hợp và ghi experiment metadata.

## 8. Splits

Không random frame/window cùng session vào nhiều split.

Ưu tiên:
1. group theo participant nếu có anonymous participant key an toàn xuyên session;
2. nếu không, group theo `session_id`/day.

`test` phải được đóng băng trước khi tuning model chính thức. Mỗi window phải lưu `split` và `split_version`; experiment phải lưu cùng `split_version` để tái lập.
