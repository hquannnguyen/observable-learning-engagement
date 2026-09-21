import datetime as dt
import json
import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.new_experiment import create_experiment


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "reports" / "experiments").mkdir(parents=True)
    config = {
        "project": {"id": "CVF25PRJ10", "seed": 42},
        "data": {"dataset_version": "v0.1", "split_version": "split-v0.1"},
        "model": {"classes": ["a", "b", "c"]},
    }
    (root / "configs" / "baseline.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return root


def test_experiment_snapshot_metadata_are_consistent(tmp_path):
    root = make_root(tmp_path)
    exp = create_experiment(
        name="test-run",
        config_path=Path("configs/baseline.yaml"),
        seed=123,
        dataset_version="v1.0",
        split_version="split-v2.0",
        root=root,
        now=dt.datetime(2026, 9, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
    )
    metadata = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
    snapshot = yaml.safe_load((exp / "config.yaml").read_text(encoding="utf-8"))
    assert metadata["seed"] == snapshot["project"]["seed"] == 123
    assert metadata["dataset_version"] == snapshot["data"]["dataset_version"] == "v1.0"
    assert metadata["split_version"] == snapshot["data"]["split_version"] == "split-v2.0"
    assert metadata["config_sha256"] == hashlib.sha256((exp / "config.yaml").read_bytes()).hexdigest()


def test_invalid_config_does_not_leave_experiment_folder(tmp_path):
    root = make_root(tmp_path)
    bad = root / "configs" / "bad.yaml"
    bad.write_text("project: [broken", encoding="utf-8")
    before = set((root / "reports" / "experiments").iterdir())
    with pytest.raises(ValueError, match="Invalid YAML"):
        create_experiment(name="bad", config_path=bad, root=root)
    after = set((root / "reports" / "experiments").iterdir())
    assert after == before


def test_missing_split_version_is_rejected_before_directory_creation(tmp_path):
    root = make_root(tmp_path)
    cfg = yaml.safe_load((root / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    del cfg["data"]["split_version"]
    path = root / "configs" / "missing-split.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="data.split_version"):
        create_experiment(name="missing-split", config_path=path, root=root)
    assert not any((root / "reports" / "experiments").iterdir())


@pytest.mark.parametrize("source,seed", [
    (source, seed)
    for source in ["config", "override"]
    for seed in [True, False, 1.9, "42", -1, 2**32, None]
    if source == "config" or seed is not None
])
def test_invalid_seed_rejected_before_artifacts(tmp_path, seed, source):
    root = make_root(tmp_path)
    path = root / "configs/baseline.yaml"
    kwargs = {}
    if source == "config":
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg["project"]["seed"] = seed
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    else:
        kwargs["seed"] = seed
    with pytest.raises(ValueError, match="seed must be an integer"):
        create_experiment(name="invalid-seed", config_path=path, root=root, **kwargs)
    assert not any((root / "reports/experiments").iterdir())


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_boundaries_and_source_unchanged(tmp_path, seed):
    root = make_root(tmp_path)
    path = root / "configs/baseline.yaml"
    before = path.read_bytes()
    exp = create_experiment(name="boundary", config_path=path, root=root, seed=seed)
    assert json.loads((exp / "metadata.json").read_text())["seed"] == seed
    assert path.read_bytes() == before
