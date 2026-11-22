#!/usr/bin/env python3
"""Verify spatial and temporal alignment between raw class videos, extracted poses, and rendered overlay videos."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
POSES_DIR = ROOT / "data" / "poses"


def verify_alignment():
    print("=" * 105)
    print(f"{'Class':<8} | {'Raw Res':<10} | {'Ov Res':<10} | {'Raw Sec':<8} | {'Ov Sec':<8} | {'Df Frames':<10} | {'Ov Frames':<10} | {'Spatial Fit':<11} | {'Status'}")
    print("=" * 105)

    all_ok = True

    for i in range(1, 8):
        c = f"class_{i}"
        raw_v = RAW_DIR / f"{c}.mp4"
        overlay_v = POSES_DIR / f"{c}_yolo11m_hd_overlay.mp4"
        parquet_f = POSES_DIR / f"{c}_cam_A_poses.parquet"

        if not raw_v.exists() or not overlay_v.exists() or not parquet_f.exists():
            print(f"{c:<8} | MISSING FILES")
            all_ok = False
            continue

        # 1. Raw video properties
        cap_raw = cv2.VideoCapture(str(raw_v))
        w_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_raw = cap_raw.get(cv2.CAP_PROP_FPS) or 30.0
        n_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        sec_raw = round(n_raw / fps_raw, 2)
        cap_raw.release()

        # 2. Overlay video properties
        cap_ov = cv2.VideoCapture(str(overlay_v))
        w_ov = int(cap_ov.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_ov = int(cap_ov.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_ov = cap_ov.get(cv2.CAP_PROP_FPS) or 10.0
        n_ov = int(cap_ov.get(cv2.CAP_PROP_FRAME_COUNT))
        sec_ov = round(n_ov / fps_ov, 2)
        cap_ov.release()

        # 3. Parquet pose properties
        df = pd.read_parquet(parquet_f)
        n_df_frames = df["frame_idx"].nunique()

        # Spatial fit: keypoints should fall within image canvas boundaries
        valid_x = df[[f"kpt_{k:02d}_x" for k in range(17)]].values
        valid_y = df[[f"kpt_{k:02d}_y" for k in range(17)]].values
        x_in_bounds = np.nanmin(valid_x) >= 0 and np.nanmax(valid_x) <= w_raw
        y_in_bounds = np.nanmin(valid_y) >= 0 and np.nanmax(valid_y) <= h_raw

        # Checks
        res_match = (w_raw == w_ov) and (h_raw == h_ov)
        frames_match = (n_ov == n_df_frames)
        dur_match = abs(sec_raw - sec_ov) <= 0.2
        spatial_ok = x_in_bounds and y_in_bounds

        is_match = res_match and frames_match and dur_match and spatial_ok
        if not is_match:
            all_ok = False

        status_str = "100% KHỚP" if is_match else "LỆCH"
        spatial_str = "Trong khung hình" if spatial_ok else "Tràn khung"
        print(f"{c:<8} | {w_raw}x{h_raw:<6} | {w_ov}x{h_ov:<6} | {sec_raw:<8} | {sec_ov:<8} | {n_df_frames:<10} | {n_ov:<10} | {spatial_str:<11} | {status_str}")

    print("=" * 105)
    if all_ok:
        print("KẾT QUẢ: TOÀN BỘ 7 VIDEO OVERLAY ĐÃ ĐƯỢC XÁC THỰC KHỚP HOÀN TOÀN CẢ VỀ KHÔNG GIAN VÀ THỜI GIAN!")
    else:
        print("CẢNH BÁO: Phát hiện video bị lệch.")


if __name__ == "__main__":
    verify_alignment()
