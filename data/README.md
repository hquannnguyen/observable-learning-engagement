# Data Directory

Repo chỉ chứa **schema, sample và split definitions**. Raw video/private derived data không mặc định commit lên GitHub public.

```text
data/
├── raw/          # private raw video; ignored by git
├── sample/       # example metadata/annotation committed to repo
├── annotations/  # annotation outputs (review before publishing)
├── poses/        # generated pose/tracking data; ignored by git
├── features/     # generated feature/window data; ignored by git
└── splits/       # fixed split definitions; có thể version-control
```

Contracts:
- `docs/data_collection_protocol.md`
- `docs/annotation_guide.md`
- `docs/dataset_schema.md`
- `docs/privacy_and_consent.md`

Không lưu tên/MSSV trong data pipeline nếu không có nhu cầu hợp lệ và được phép.
