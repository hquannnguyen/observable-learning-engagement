"""Evaluate frozen window predictions against completed independent annotations."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dataset_qa import validate_annotation_dataframe
from src.contracts.labels import CANONICAL_TRAINING_LABELS, UNCERTAIN_LABEL
from src.reporting.bias_analysis import analyze_positions
from src.training.trainer import compute_multiclass_metrics

def join_human_annotations(predictions, annotations):
    valid, errors = validate_annotation_dataframe(annotations)
    if not valid:
        raise ValueError(f"Incomplete/invalid annotations: {errors[:5]}")
    if "window_id" not in annotations or annotations["window_id"].duplicated().any() or predictions["window_id"].duplicated().any():
        raise ValueError("Unique window_id required in annotations and predictions")
    labelled = annotations[annotations["annotation_state"] == "VALID"].copy()
    if labelled.empty or not labelled["label_source"].isin(["HUMAN", "ADJUDICATED"]).all():
        raise ValueError("Completed HUMAN/ADJUDICATED labels required; empty templates and weak labels are not ground truth")
    if not labelled["window_id"].isin(predictions["window_id"]).all():
        raise ValueError("Human labels refer to windows absent from these predictions")
    truth = labelled.set_index("window_id")
    result = predictions[predictions["window_id"].isin(truth.index)].copy()
    for key in ("session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms"):
        if key in result and not (result[key].astype(str).to_numpy() == result["window_id"].map(truth[key]).astype(str).to_numpy()).all():
            raise ValueError(f"Human/prediction {key} mismatch")
    result["true_label"] = result["window_id"].map(truth["participation_label"])
    result["label_source"] = result["window_id"].map(truth["label_source"])
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    pred_path, ann_path = Path(args.predictions), Path(args.annotations)
    predictions, annotations = pd.read_csv(pred_path), pd.read_csv(ann_path)
    result = join_human_annotations(predictions, annotations)
    labels = list(CANONICAL_TRAINING_LABELS)
    def metrics(frame, col):
        if frame.empty:
            return None
        return compute_multiclass_metrics(frame["true_label"].map(labels.index).to_numpy(), frame[col].map(labels.index).to_numpy())
    accepted = result[result["gated_label"] != UNCERTAIN_LABEL]
    report = {"metric_scope": "human_labels", "evaluated_windows": len(result),
              "coverage": len(accepted) / len(result), "raw_metrics": metrics(result, "pred_label"),
              "accepted_metrics": metrics(accepted, "gated_label"),
              "annotation_state_counts": annotations["annotation_state"].value_counts().to_dict(),
              "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
              "annotations_sha256": hashlib.sha256(ann_path.read_bytes()).hexdigest()}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "human_predictions.csv", index=False)
    (out / "human_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    positions, positional_report = analyze_positions(result)
    positions.to_csv(out / "position_metrics.csv", index=False)
    (out / "position_report.json").write_text(json.dumps(positional_report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Evaluated {len(result)} independently labelled windows -> {out}")

if __name__ == "__main__":
    main()
