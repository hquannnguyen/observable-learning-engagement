from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "experiment"


def git_commit(root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "NO_GIT_COMMIT"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    if not isinstance(data.get("project"), dict):
        raise ValueError("Config requires project section")
    if "seed" not in data["project"]:
        raise ValueError("Config requires project.seed")
    if not isinstance(data.get("data"), dict):
        raise ValueError("Config requires data section")
    for key in ("dataset_version", "split_version"):
        if not data["data"].get(key):
            raise ValueError(f"Config requires data.{key}")
    return data


def build_snapshot(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    dataset_version: str | None = None,
    split_version: str | None = None,
) -> tuple[dict[str, Any], int, str, str]:
    snapshot = copy.deepcopy(config)
    effective_seed = seed if seed is not None else snapshot["project"]["seed"]
    effective_dataset_version = str(
        dataset_version
        if dataset_version is not None
        else snapshot["data"]["dataset_version"]
    )
    effective_split_version = str(
        split_version
        if split_version is not None
        else snapshot["data"]["split_version"]
    )

    if type(effective_seed) is not int or not 0 <= effective_seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not effective_dataset_version.strip():
        raise ValueError("dataset_version must not be blank")
    if not effective_split_version.strip():
        raise ValueError("split_version must not be blank")

    # Crucial: metadata and snapshot must describe the same run.
    snapshot["project"]["seed"] = effective_seed
    snapshot["data"]["dataset_version"] = effective_dataset_version
    snapshot["data"]["split_version"] = effective_split_version
    return (
        snapshot,
        effective_seed,
        effective_dataset_version,
        effective_split_version,
    )


def create_experiment(
    *,
    name: str,
    config_path: Path,
    seed: int | None = None,
    dataset_version: str | None = None,
    split_version: str | None = None,
    notes: str = "",
    root: Path = ROOT,
    now: dt.datetime | None = None,
) -> Path:
    """Create an experiment atomically after all config validation succeeds."""
    config_path = config_path if config_path.is_absolute() else root / config_path

    # Validate and resolve run metadata BEFORE creating any experiment directory.
    config = load_config(config_path)
    snapshot, eff_seed, eff_dataset, eff_split = build_snapshot(
        config,
        seed=seed,
        dataset_version=dataset_version,
        split_version=split_version,
    )

    now = now or dt.datetime.now().astimezone()
    experiment_id = f"{now:%Y%m%d-%H%M%S}-{slugify(name)}"
    experiments_root = root / "reports" / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    final_dir = experiments_root / experiment_id
    if final_dir.exists():
        raise FileExistsError(f"Experiment already exists: {final_dir}")

    config_text = yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True)

    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-experiment-", dir=experiments_root))
        (temp_dir / "config.yaml").write_text(config_text, encoding="utf-8")
        config_sha256 = hashlib.sha256((temp_dir / "config.yaml").read_bytes()).hexdigest()

        try:
            source_config = str(config_path.relative_to(root))
        except ValueError:
            source_config = str(config_path)

        metadata = {
            "experiment_id": experiment_id,
            "created_at": now.isoformat(),
            "name": name,
            "seed": eff_seed,
            "git_commit": git_commit(root),
            "dataset_version": eff_dataset,
            "split_version": eff_split,
            "source_config": source_config,
            "config_sha256": config_sha256,
            "notes": notes,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (temp_dir / "notes.md").write_text(
            f"# {experiment_id}\n\n{notes or 'Chưa có ghi chú.'}\n",
            encoding="utf-8",
        )

        # Atomic on the same filesystem: incomplete temp folders are never final runs.
        temp_dir.rename(final_dir)
    except Exception:
        if "temp_dir" in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible experiment folder")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override project.seed; the snapshot is updated to the same value.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Override data.dataset_version and snapshot it.",
    )
    parser.add_argument(
        "--split-version",
        default=None,
        help="Override data.split_version and snapshot it.",
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    exp_dir = create_experiment(
        name=args.name,
        config_path=Path(args.config),
        seed=args.seed,
        dataset_version=args.dataset_version,
        split_version=args.split_version,
        notes=args.notes,
    )
    metadata = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
    print("Created experiment:", metadata["experiment_id"])
    print("Artifact directory:", exp_dir)
    print("Seed:", metadata["seed"])
    print("Git commit:", metadata["git_commit"])
    print("Dataset version:", metadata["dataset_version"])
    print("Split version:", metadata["split_version"])


if __name__ == "__main__":
    main()
