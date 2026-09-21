# Experiment Logging Protocol — v1.1

Mục tiêu: mọi thí nghiệm quan trọng phải có thể truy vết và tái lập; **metadata phải khớp config snapshot**.

## Metadata tối thiểu

- `experiment_id`
- `created_at`
- `name`
- `seed`
- `git_commit`
- `dataset_version`
- `split_version`
- `source_config`
- `config_sha256`
- `notes`

## Quy tắc đặt ID

```text
YYYYMMDD-HHMMSS-<slug>
```

## Cấu trúc artifact

```text
reports/experiments/<experiment_id>/
├── metadata.json
├── config.yaml
├── metrics.json                 # tạo sau train/evaluate
├── confusion_matrix.png         # tạo sau evaluate
├── training_history.csv         # tạo sau train
└── notes.md
```

## Snapshot consistency

`config.yaml` là snapshot cấu hình **effective** của lần chạy. Nếu CLI truyền `--seed`, `--dataset-version` hoặc `--split-version`, script phải cập nhật chính snapshot trước khi ghi metadata. Vì vậy:

```text
metadata.seed == config.project.seed
metadata.dataset_version == config.data.dataset_version
metadata.split_version == config.data.split_version
```

`config_sha256` là SHA-256 của byte thực tế trong file `config.yaml` đã ghi, kể cả newline của hệ điều hành. Dùng giá trị này để phát hiện snapshot thay đổi sau khi experiment được tạo.

Seed phải là số nguyên trong `[0, 2**32)`. Không chấp nhận boolean, số thực hoặc chuỗi trong config/API; `--seed` trên CLI được parse thành số nguyên. Seed không hợp lệ bị từ chối trước khi tạo artifact.

## Failure behavior

- Config/YAML/required field được validate **trước** khi tạo experiment folder.
- Artifact được ghi vào thư mục tạm và chỉ rename thành experiment ID sau khi hoàn tất.
- Nếu lỗi giữa quá trình ghi, thư mục tạm được dọn; không để experiment rỗng/incomplete.

## Nguyên tắc

1. Không ghi đè experiment cũ.
2. Mỗi experiment dùng một config snapshot riêng.
3. Seed/dataset/split metadata phải khớp snapshot.
4. Ghi git commit; nếu chưa init Git thì `NO_GIT_COMMIT`, nhưng experiment chính thức nên chạy trong Git repo.
5. Không thay đổi test split sau khi xem metric chỉ để tăng kết quả.
6. Mọi artifact/metric báo cáo phải có experiment ID.

## CLI

Dùng baseline version trong config:

```bash
python scripts/new_experiment.py --name lstm-baseline
```

Hoặc override có snapshot đồng bộ:

```bash
python scripts/new_experiment.py   --name lstm-ablation   --seed 123   --dataset-version v1.0   --split-version split-v1.0
```
