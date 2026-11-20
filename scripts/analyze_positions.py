"""Export positional quality/coverage and separately scoped classification metrics."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.bias_analysis import analyze_positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()
    table, report = analyze_positions(pd.read_csv(args.predictions), args.min_samples)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "position_metrics.csv", index=False)
    (out / "position_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{report['status']}: {len(table)} session/zone groups -> {out}")

if __name__ == "__main__":
    main()
