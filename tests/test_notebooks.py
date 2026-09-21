import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_root_detection(path, tmp_path, monkeypatch):
    monkeypatch.delenv("CVF_PROJECT_ROOT", raising=False)
    namespace = {}
    source = "".join(cells(path)[2]["source"])
    exec(source.split("ROOT = find_project_root()")[0], namespace)
    find = namespace["find_project_root"]
    assert find(ROOT) == ROOT
    assert find(ROOT / "notebooks") == ROOT
    # An explicit start outside the real checkout models Colab's /content.
    def make_repo(name):
        repo = tmp_path / name
        for marker in ["AGENTS.md", "configs/baseline.yaml", "scripts/check_setup.py"]:
            marker_path = repo / marker
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()
        (repo / "src").mkdir()
        return repo
    repo = make_repo("clone")
    assert find(tmp_path) == repo
    make_repo("second")
    with pytest.raises(RuntimeError, match="Multiple"):
        find(tmp_path)
    monkeypatch.setenv("CVF_PROJECT_ROOT", str(repo))
    assert find(tmp_path) == repo
    monkeypatch.setenv("CVF_PROJECT_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not a valid"):
        find(tmp_path)


def test_environment_notebook_loads_actual_baseline():
    notebook = cells(ROOT / "notebooks/00_environment_check.ipynb")
    source = next("".join(c["source"]) for c in notebook if "yaml.safe_load" in "".join(c["source"]))
    namespace = {"ROOT": ROOT}
    exec(compile(source, "environment-config-cell", "exec"), namespace)
    expected = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    assert namespace["config"] == expected


def test_notebook_checks_use_kernel_and_fail_fast(monkeypatch):
    import subprocess
    import sys
    notebook = cells(ROOT / "notebooks/00_environment_check.ipynb")
    source = next("".join(c["source"]) for c in notebook if "# Repository contract checks" in "".join(c["source"]))
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    exec(source, {"ROOT": ROOT, "sys": sys})
    assert len(run.call_args_list) == 2
    for call in run.call_args_list:
        assert call.args[0][0] == sys.executable
        assert call.kwargs == {"cwd": ROOT, "check": True}
    run.reset_mock()
    run.side_effect = subprocess.CalledProcessError(1, "check_setup")
    with pytest.raises(subprocess.CalledProcessError):
        exec(source, {"ROOT": ROOT, "sys": sys})
    assert run.call_count == 1
