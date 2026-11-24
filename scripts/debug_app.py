import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pose.annotation_preview import render_annotation_preview

csv_path = ROOT / "data/annotations/review_v03_independent_20261124/annotator_b.csv"
if not csv_path.exists():
    csv_path = ROOT / "data/annotations/review_v03/annotator_phong.csv"

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")

for idx in [0, 1, 5, 10]:
    row = df.loc[idx]
    v_raw = row.get("video_path", "")
    video = Path(v_raw) if Path(v_raw).is_absolute() else ROOT / v_raw
    pose_path = ROOT / "data/poses" / f"{row['session_id']}_{row['camera_id']}_poses.csv"
    
    print(f"\n--- Row {idx}: window_id={row['window_id']} ---")
    print(f"  session_id: {row['session_id']}, camera_id: {row['camera_id']}")
    print(f"  video: exists={video.is_file()}, path={video}")
    print(f"  pose:  exists={pose_path.is_file()}, path={pose_path}")

    if video.is_file() and pose_path.is_file():
        try:
            print("  Rendering preview...")
            focused, meta = render_annotation_preview(
                video, pose_path, row["track_id"],
                float(row["t_start_ms"]), float(row["t_end_ms"]),
                ROOT / "data/annotations/.preview_cache"
            )
            print(f"  Rendered preview OK: exists={focused.is_file()}, size={focused.stat().st_size} bytes, path={focused}")
        except Exception as e:
            print(f"  ERROR rendering preview: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
