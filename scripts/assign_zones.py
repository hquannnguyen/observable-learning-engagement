#!/usr/bin/env python3
"""Assign geometric 3x3 classroom zones to pose detections.

The grid is normalized to the observed bounding-box extent of each recording
(because raw pose CSVs do not store frame width/height). This is a geometric
assignment, not a verified seat map; reports must label it as such.

Usage:
    python scripts/assign_zones.py --poses-dir data/poses --output-dir data/poses_zones
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("assign_zones")


def load_grid(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "grid" not in cfg or "rows" not in cfg["grid"] or "cols" not in cfg["grid"]:
        raise ValueError(f"Invalid zone config: {config_path}")
    return cfg


def zone_for_point(x_norm: float, y_norm: float, cfg: Dict[str, Any]) -> str:
    row_id = None
    for row in cfg["grid"]["rows"]:
        if row["y_min"] <= y_norm < row["y_max"] or (
            y_norm == 1.0 and row["y_max"] == 1.0
        ):
            row_id = row["id"]
            break
    col_id = None
    for col in cfg["grid"]["cols"]:
        if col["x_min"] <= x_norm < col["x_max"] or (
            x_norm == 1.0 and col["x_max"] == 1.0
        ):
            col_id = col["id"]
            break
    row_id = row_id or cfg["grid"]["rows"][-1]["id"]
    col_id = col_id or cfg["grid"]["cols"][-1]["id"]
    return cfg.get("zone_id_format", "{row}-{col}").format(row=row_id, col=col_id)


def assign(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    required = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"pose data missing bbox columns: {sorted(missing)}")

    cx = (out["bbox_x1"] + out["bbox_x2"]) / 2.0
    cy = (out["bbox_y1"] + out["bbox_y2"]) / 2.0

    # Observed extent; fall back to 1.0 to avoid division by zero.
    x_min, x_max = float(out["bbox_x1"].min()), float(out["bbox_x2"].max())
    y_min, y_max = float(out["bbox_y1"].min()), float(out["bbox_y2"].max())
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)

    x_norm = ((cx - x_min) / width).clip(0.0, 1.0)
    y_norm = ((cy - y_min) / height).clip(0.0, 1.0)

    out["zone_id"] = [
        zone_for_point(float(x), float(y), cfg) for x, y in zip(x_norm, y_norm)
    ]
    out["zone_source"] = cfg.get("zone_source", "GRID_3X3")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign 3x3 geometric zones to pose files.")
    parser.add_argument("--poses-dir", default="data/poses")
    parser.add_argument("--output-dir", default="data/poses_zones")
    parser.add_argument("--config", default="configs/zones/grid_3x3.yaml")
    args = parser.parse_args()

    cfg = load_grid(Path(args.config))
    poses_dir = Path(args.poses_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(poses_dir.glob("*_poses.csv"))
    if not files:
        logger.error("No *_poses.csv found in %s", poses_dir)
        sys.exit(1)

    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        assigned = assign(df, cfg)
        out_path = out_dir / path.name
        assigned.to_csv(out_path, index=False)
        counts = assigned["zone_id"].value_counts().to_dict()
        logger.info("%s -> %s | %s", path.name, out_path.name, counts)


if __name__ == "__main__":
    main()
