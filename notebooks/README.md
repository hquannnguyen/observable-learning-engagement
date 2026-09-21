# Google Colab Notebooks

Notebooks are **orchestration/demo entry points**. Core logic must remain in `src/` and be importable from local/Colab/server environments.

Planned notebooks:
- `00_environment_check.ipynb`
- `01_pose_extraction_colab.ipynb`
- `02_build_dataset_colab.ipynb`
- `03_train_lstm_colab.ipynb`
- `04_train_tcn_colab.ipynb`
- `05_evaluation_colab.ipynb`

Current state: templates only. A notebook name does **not** imply the related task is implemented.

Colab workflow:

```text
GitHub repo → Colab clone/open → install requirements
→ mount approved dataset/checkpoint storage
→ call modules under src/
→ save experiment artifacts
→ commit only code/config/lightweight reports back to Git
```

## Project root bootstrap

Mỗi notebook tìm repository root bằng nhiều marker (`AGENTS.md`, `configs/baseline.yaml`, `scripts/check_setup.py`, `src/`) thay vì chỉ kiểm tra `README.md`. Vì vậy mở notebook từ `notebooks/`, repository root hoặc `/content` sau khi clone đều không nhận nhầm `notebooks/` là root. Có thể đặt `CVF_PROJECT_ROOT=/content/<repo>` nếu Colab có nhiều repo.
