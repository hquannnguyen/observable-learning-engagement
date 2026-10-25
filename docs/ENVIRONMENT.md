# ENVIRONMENT — CVF25PRJ10

## 1. Môi trường đã kiểm chứng

| Thành phần | Giá trị |
|---|---|
| OS | Windows 10 (AMD64) |
| Python | 3.14.7 |
| PyTorch | 2.14.1+cpu |
| Ultralytics | 8.4.173 |
| OpenCV | 5.0.0.93 |
| NumPy / pandas | 2.5.3 / 3.0.6 |
| Streamlit / Plotly | 1.66.1 / 7.1.0 |
| pytest | 8.4.2 |
| GPU | Không dùng; toàn bộ benchmark là CPU |

`requirements.txt` pin các version trên (trừ scikit-learn, hiện chưa cài trong môi trường test).
Nếu dùng GPU, cài PyTorch bản CUDA tương ứng và ghi lại version khi benchmark.

## 2. Cài đặt

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Chạy kiểm thử

Trên Windows, nếu `%TEMP%` bị chặn quyền, dùng basetemp ngoài repo:

```bash
python -m pytest -q --basetemp=D:\ThiGiacMayTinh\.pytest_tmp
```

Kỳ vọng: toàn bộ test pass. Không dùng `--basetemp` trỏ vào trong repo vì test
`test_notebooks.py` kiểm tra root detection và sẽ tìm nhầm project root.

## 4. Chạy pipeline

```bash
python scripts/build_dataset.py
python scripts/train.py --name lstm_weaklabel_v0.1 --epochs 30
python scripts/evaluate.py --experiment-dir reports/experiments/<experiment_id>
python scripts/export_report.py --experiment-dir reports/experiments/<experiment_id>
streamlit run app/streamlit_app.py
```

## 5. Lưu ý

- Không commit raw video, checkpoint `.pt`, hoặc file lớn.
- `reports/experiments/*/` chỉ giữ metadata/metrics nhỏ; xem `.gitignore`.
- Mọi số benchmark phải ghi hardware + version; xem `reports/benchmark.md`.
- Chưa có human test set; metric hiện tại là weak-label agreement.
