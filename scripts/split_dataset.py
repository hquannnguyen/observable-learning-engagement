"""Group train/validation/test split by track_id without data leakage (Task T18).

Contract:
- Strictly isolates tracks between splits (0% participant/track leakage).
- Freezes split with fixed random seed (42).
- Tags windows with split_version: split-v0.1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def group_split_dataset(
    windows_csv_path: Path,
    features_npz_path: Path,
    splits_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    split_version: str = "split-v0.1",
) -> None:
    if not windows_csv_path.exists():
        raise FileNotFoundError(f"Windows CSV not found: {windows_csv_path}")

    df = pd.read_csv(windows_csv_path)
    print(f"[INFO] Loaded {len(df)} windows from {windows_csv_path}")

    # Group tracks
    unique_tracks = sorted(df["track_id"].unique())
    n_tracks = len(unique_tracks)
    print(f"[INFO] Total unique tracks to split: {n_tracks}")

    rng = np.random.RandomState(seed)
    shuffled_tracks = rng.permutation(unique_tracks)

    n_train = int(round(n_tracks * train_ratio))
    n_val = int(round(n_tracks * val_ratio))
    n_test = n_tracks - n_train - n_val

    train_tracks = set(shuffled_tracks[:n_train])
    val_tracks = set(shuffled_tracks[n_train:n_train + n_val])
    test_tracks = set(shuffled_tracks[n_train + n_val:])

    # Assign split to each row
    def assign_split(track_id: int) -> str:
        if track_id in train_tracks:
            return "train"
        elif track_id in val_tracks:
            return "val"
        else:
            return "test"

    df["split"] = df["track_id"].apply(assign_split)
    df["split_version"] = split_version

    # Overwrite windows CSV with updated splits
    df.to_csv(windows_csv_path, index=False, encoding="utf-8")

    # Update NPZ archive with split array
    if features_npz_path.exists():
        npz_data = np.load(features_npz_path)
        np.savez_compressed(
            features_npz_path,
            X=npz_data["X"],
            y=npz_data["y"],
            window_ids=npz_data["window_ids"],
            track_ids=npz_data["track_ids"],
            splits=df["split"].to_numpy(dtype=str),
        )

    # Save split definitions and summary
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {
        "split_version": split_version,
        "seed": seed,
        "total_windows": len(df),
        "total_tracks": n_tracks,
        "train": {
            "tracks": [int(t) for t in sorted(list(train_tracks))],
            "window_count": int((df["split"] == "train").sum()),
        },
        "val": {
            "tracks": [int(t) for t in sorted(list(val_tracks))],
            "window_count": int((df["split"] == "val").sum()),
        },
        "test": {
            "tracks": [int(t) for t in sorted(list(test_tracks))],
            "window_count": int((df["split"] == "test").sum()),
        },
    }

    summary_file = splits_dir / f"{split_version}_summary.json"
    summary_file.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    print(f"\n[OK] Group Split Completed (Seed {seed}):")
    print(f"  * TRAIN: {len(train_tracks)} tracks ({split_summary['train']['window_count']} windows)")
    print(f"  * VAL:   {len(val_tracks)} tracks ({split_summary['val']['window_count']} windows)")
    print(f"  * TEST:  {len(test_tracks)} tracks ({split_summary['test']['window_count']} windows)")
    print(f"[OK] Summary saved to: {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset by track_id without leakage (Task T18).")
    parser.add_argument("--windows", type=str, default="data/features/session_001_windows.csv", help="Windows CSV")
    parser.add_argument("--features", type=str, default="data/features/session_001_features.npz", help="Features NPZ")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Splits output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    group_split_dataset(
        windows_csv_path=Path(args.windows),
        features_npz_path=Path(args.features),
        splits_dir=Path(args.splits_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
