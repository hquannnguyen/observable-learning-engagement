"""Tests for geometric 3x3 classroom zone assignment (remediation C1)."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.assign_zones import assign, load_grid, zone_for_point

CONFIG = Path("configs/zones/grid_3x3.yaml")


@pytest.fixture(scope="module")
def grid():
    return load_grid(CONFIG)


def test_zone_corner_mapping(grid):
    assert zone_for_point(0.1, 0.1, grid) == "BACK-LEFT"
    assert zone_for_point(0.5, 0.5, grid) == "MIDDLE-CENTER"
    assert zone_for_point(0.9, 0.9, grid) == "FRONT-RIGHT"


def test_assign_uses_bbox_center(grid):
    df = pd.DataFrame(
        {
            "bbox_x1": [0, 80],
            "bbox_y1": [0, 80],
            "bbox_x2": [10, 90],
            "bbox_y2": [10, 90],
        }
    )
    out = assign(df, grid)
    assert list(out["zone_id"]) == ["BACK-LEFT", "FRONT-RIGHT"]
    assert set(out["zone_source"]) == {"GRID_3X3"}


def test_assign_requires_bbox_columns(grid):
    with pytest.raises(ValueError, match="missing bbox"):
        assign(pd.DataFrame({"x": [1]}), grid)
