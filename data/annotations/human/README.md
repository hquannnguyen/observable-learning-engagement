# Human annotation workspace

Thư mục này chỉ chứa **nhãn do người gán**. Không đặt nhãn do
`scripts/generate_annotations.py` sinh ra ở đây.

## 1. File trong thư mục

| File | Vai trò |
|---|---|
| `annotation_template.csv` | Danh sách cửa sổ cần gán, cột nhãn để trống, **không** hiển thị weak label |
| `sampling_manifest.json` | Seed, số lượng, hash nguồn, phân bố weak label ở mức tổng hợp |
| `ann_phong.csv` | Bản gán độc lập của annotator 1 (tên file do nhóm đặt) |
| `ann_tai.csv` | Bản gán độc lập của annotator 2 |
| `adjudicated.csv` | Nhãn cuối sau khi xử lý bất đồng, `label_source=ADJUDICATED` |
| `qa_report.json` | Kết quả Cohen's Kappa + confusion matrix |

## 2. Quy trình

1. Mỗi annotator copy `annotation_template.csv` thành file riêng; **không sửa file template gốc**.
2. Xem video/pose overlay theo `video_path` và `t_start_ms`–`t_end_ms`; xem toàn bộ cửa sổ 4 giây.
3. Điền:
   - `participation_label`: một trong 3 canonical value đầy đủ;
   - `annotation_state`: `VALID` / `OCCLUDED` / `TRANSITION` / `INVALID`;
   - `label_source`: `HUMAN`;
   - `label_rule_version`: để trống với HUMAN;
   - `annotator_id`, `annotation_confidence` (0–1), `notes`.
4. Nếu state khác `VALID`: để trống `participation_label`, `label_source`, `label_rule_version`.
5. Hai annotator **không trao đổi nhãn** trong lúc gán.
6. Chạy QA:

```bash
python scripts/dataset_qa.py \
  --annotator-a data/annotations/human/ann_phong.csv \
  --annotator-b data/annotations/human/ann_tai.csv \
  --strict-human --min-kappa 0.60 \
  --report-json data/annotations/human/qa_report.json \
  --adjudication-out data/annotations/human/disagreements.csv
```

7. Review `disagreements.csv`, chốt `adjudicated.csv` với `label_source=ADJUDICATED`.
8. Đóng băng test set: ghi SHA256 của `adjudicated.csv` vào `test-human-v1.md` và không sửa nữa.

## 3. Cảnh báo

- Không copy file của người này sang người kia; `--strict-human` sẽ cảnh báo nếu hai file giống nhau từng dòng.
- Không dùng nhãn `WEAK_RULE` trong test set chính thức.
- Không sửa nhãn test vì model dự đoán khác; chỉ sửa khi có lỗi annotation và phải ghi log.
- Nếu chưa có nhãn người, mọi báo cáo phải ghi `metric_scope=weak_rule_labels`, không được gọi là accuracy trên người thật.
