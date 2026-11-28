#!/usr/bin/env python3
"""Headless smoke test for DesktopStudioController (no Tkinter window needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.desktop_studio import DesktopStudioController

root = ROOT
ctrl = DesktopStudioController(root)
print("Controller created OK")

datasets = ctrl.set_person("Phong")
print(f"Phong datasets: {len(datasets)} found")
for d, lbl, p in datasets:
    print(f"  {d}: {p.name}")

if datasets:
    ctrl.load_dataset(datasets[0][2])
    print(f"Loaded df shape: {ctrl.df.shape}")
    done_cnt = int(ctrl.done_mask().sum())
    print(f"Done count: {done_cnt}/{len(ctrl.df)}")
    ctrl.apply_filter("Tất cả")
    print(f"Filtered indices: {len(ctrl.filtered_indices)}")
    row = ctrl.current_row()
    if row is not None:
        print(f"First row window_id: {row['window_id']}")
    sug = ctrl.load_suggestions()
    print(f"AI suggestions loaded: {len(sug)} entries")
    ctrl.set_person("Tài")
    datasets_tai = ctrl.set_person("Tài")
    print(f"Tài datasets: {len(datasets_tai)} found")
else:
    print("No datasets found for Phong — check data/annotations/ paths")

print("SMOKE TEST PASSED")
