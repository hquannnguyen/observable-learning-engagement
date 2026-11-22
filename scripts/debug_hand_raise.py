import cv2
from pathlib import Path
import pandas as pd

df = pd.read_parquet("data/poses/class_3_cam_A_poses.parquet")
raw_vid = Path("data/raw/class_3.mp4")
out_dir = Path("reports/debug_class_3_hand_raise")
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(raw_vid))
native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Track 7 bounding box in frame 41
track_7_f41 = df[(df["frame_idx"] == 41) & (df["track_id"] == 7)].iloc[0]
bx1, by1, bx2, by2 = int(track_7_f41["bbox_x1"]), int(track_7_f41["bbox_y1"]), int(track_7_f41["bbox_x2"]), int(track_7_f41["bbox_y2"])

# Expand crop to include raised hand above head
pad = 120
crop_y1 = max(0, by1 - pad)
crop_y2 = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), by2 + 40)
crop_x1 = max(0, bx1 - pad)
crop_x2 = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), bx2 + pad)

print(f"Tracking crop region for Track 7: x=[{crop_x1}, {crop_x2}], y=[{crop_y1}, {crop_y2}]")

# Read native frames 120 to 155 (around 4.0s - 5.2s)
cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
curr = 120
while curr <= 155:
    ret, frame = cap.read()
    if not ret:
        break
    # Save crop
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    cv2.imwrite(str(out_dir / f"native_frame_{curr:04d}.jpg"), crop)
    curr += 1

cap.release()
print(f"Saved native frames 120-155 to {out_dir}")
