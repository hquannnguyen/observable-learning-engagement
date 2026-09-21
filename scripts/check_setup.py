from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "AGENTS.md",
    "docs/scope.md",
    "docs/IMPLEMENTATION_PLAN.md",
    "docs/experiment_protocol.md",
    "docs/data_collection_protocol.md",
    "docs/annotation_guide.md",
    "docs/dataset_schema.md",
    "docs/privacy_and_consent.md",
    "configs/baseline.yaml",
    "configs/experiments",
    "data/README.md",
    "data/raw",
    "data/sample/metadata.example.json",
    "data/sample/annotation.example.csv",
    "data/sample/participation_windows.example.csv",
    "data/annotations",
    "data/poses",
    "data/features",
    "data/splits/README.md",
    "src/contracts/labels.py",
    "src/pose",
    "src/preprocessing",
    "src/preprocessing/feature_contract.py",
    "src/models",
    "src/training",
    "src/uncertainty",
    "src/uncertainty/gate.py",
    "src/reporting",
    "models/README.md",
    "models/checkpoints",
    "scripts/check_setup.py",
    "scripts/new_experiment.py",
    "app/streamlit_app.py",
    "notebooks/README.md",
    "notebooks/00_environment_check.ipynb",
    "notebooks/01_pose_extraction_colab.ipynb",
    "notebooks/02_build_dataset_colab.ipynb",
    "notebooks/03_train_lstm_colab.ipynb",
    "notebooks/04_train_tcn_colab.ipynb",
    "notebooks/05_evaluation_colab.ipynb",
    "reports/experiments",
    "tests/test_config.py",
    "tests/test_data_contract.py",
    "tests/test_feature_shapes.py",
    "tests/test_uncertainty_gate.py",
    "tests/test_experiment_creation.py",
    ".github/workflows/ci.yml",
    ".github/pull_request_template.md",
    "requirements.txt",
    ".gitignore",
]

missing = [p for p in REQUIRED if not (ROOT / p).exists()]

print("CVF25PRJ10 setup check")
print("Root:", ROOT)
if missing:
    print("FAILED - missing paths:")
    for p in missing:
        print(" -", p)
    sys.exit(1)

print(f"OK - {len(REQUIRED)} required paths found.")
print("Repository is ready for T03/T04 review and implementation work.")
