import cv2
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet("data/poses/class_3_cam_A_poses.parquet")

l_raised = (df["kpt_09_y"] < df["kpt_05_y"]) & (df["kpt_09_conf"] > 0.25)
r_raised = (df["kpt_10_y"] < df["kpt_06_y"]) & (df["kpt_10_conf"] > 0.25)
raised = df[l_raised | r_raised]

print(f"Total detections: {len(df)}")
print(f"Hand raised detections: {len(raised)}")
print("Frames with hand raised:", sorted(raised["frame_idx"].unique()))

for t_id, grp in raised.groupby("track_id"):
    print(f"Track {t_id}: {len(grp)} frames, frame range [{grp['frame_idx'].min()}, {grp['frame_idx'].max()}]")

# Let's inspect confidence of wrist keypoints around the hand-raise motion
print("\n--- Wrist confidence and detection around hand raises ---")
sample_track = raised["track_id"].iloc[0]
track_full = df[df["track_id"] == sample_track].sort_values("frame_idx")
print(f"Inspecting full trajectory of Track {sample_track}:")
for _, r in track_full.iterrows():
    f = int(r["frame_idx"])
    lw_y, lw_c = r["kpt_09_y"], r["kpt_09_conf"]
    rw_y, rw_c = r["kpt_10_y"], r["kpt_10_conf"]
    ls_y, rs_y = r["kpt_05_y"], r["kpt_06_y"]
    # Check if hand is near or above shoulder
    is_up = (lw_y < ls_y) or (rw_y < rs_y)
    if is_up or (rw_c > 0.2 and rw_y < rs_y + 50):
        print(f"  Frame {f:03d} (ts={r['timestamp_ms']:>5}ms): R_wrist_y={rw_y:.1f} vs R_sh_y={rs_y:.1f} (conf={rw_c:.2f}) | L_wrist_y={lw_y:.1f} (conf={lw_c:.2f}) | IS_UP={is_up}")
