"""CLI script to build window feature tensors and metadata (Task T15 & T17).

Usage:
    python scripts/build_features.py --poses data/poses/cam_A_pilot_poses.parquet --annotations data/annotations/session_001_weak_labels.csv --output-dir data/features
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.windows import build_window_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready feature tensors and windows metadata.")
    parser.add_argument("--poses", type=str, default="data/poses/cam_A_pilot_poses.parquet", help="Path to pose parquet")
    parser.add_argument("--annotations", type=str, default="data/annotations/session_001_weak_labels.csv", help="Path to annotations CSV")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory for features")
    parser.add_argument("--session-id", type=str, default="session_001", help="Session ID")
    parser.add_argument("--timesteps", type=int, default=40, help="Timesteps per window (default: 40)")

    args = parser.parse_args()

    build_window_dataset(
        poses_path=Path(args.poses),
        annotations_path=Path(args.annotations),
        output_dir=Path(args.output_dir),
        session_id=args.session_id,
        target_timesteps=args.timesteps,
    )


if __name__ == "__main__":
    main()
