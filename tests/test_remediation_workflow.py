"""Regression tests for real context, human-only evaluation and model provenance."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.preprocessing.context import attach_session_context
from src.preprocessing.windows import build_track_windows
from src.preprocessing.weak_labels import assign_weak_labels_to_windows
from src.reporting.bias_analysis import analyze_positions
from src.models.factory import build_model, model_from_checkpoint
from scripts.evaluate_human import join_human_annotations
from scripts.split_video import create_metadata_template
from scripts.dataset_qa import REQUIRED_COLUMNS
from tests.test_lstm_pipeline import make_dummy_track
from tests.test_reporting_export import make_predictions


def test_generated_phase_metadata_is_not_verified(tmp_path):
    session = tmp_path / "test_s"
    session.mkdir()
    (session / "metadata.json").write_text(json.dumps({"session_id": "test_s", "class_phase": [{"name": "discussion", "start_ms": 0, "end_ms": 10000}]}))
    result, audit = attach_session_context(make_dummy_track(), tmp_path)
    assert set(result.class_phase) == {"UNKNOWN"}
    assert set(result.declared_class_phase) == {"discussion"}
    assert audit[0]["status"] == "REVIEW_REQUIRED"
    records, tensors = build_track_windows(result)
    labelled = assign_weak_labels_to_windows(records, tensors)
    assert all(x["participation_label"] is None for x in labelled)


def test_phase_timeline_is_applied_per_window_and_transition_excluded(tmp_path):
    session = tmp_path / "test_s"
    session.mkdir()
    (session / "metadata.json").write_text(json.dumps({"session_id": "test_s", "phase_verified": True, "class_phase": [
        {"name": "lecture", "start_ms": 0, "end_ms": 4000},
        {"name": "discussion", "start_ms": 4000, "end_ms": 10000}]}))
    result, _ = attach_session_context(make_dummy_track(80), tmp_path)
    records, tensors = build_track_windows(result)
    assert [x["class_phase"] for x in records] == ["lecture", "TRANSITION", "discussion"]
    labelled = assign_weak_labels_to_windows(records, tensors)
    assert labelled[1]["quality_state"] == "TRANSITION"
    assert labelled[1]["participation_label"] is None


def test_windows_do_not_bridge_missing_frames_or_mix_cameras():
    track = make_dummy_track(60).drop(index=25)
    records, _ = build_track_windows(track)
    assert records == []
    track = make_dummy_track()
    track.loc[0, "camera_id"] = "cam_B"
    with pytest.raises(ValueError, match="camera_id"):
        build_track_windows(track)


def test_nondefault_tcn_checkpoint_round_trip():
    cfg = {"type": "tcn", "tcn": {"hidden_size": 24, "kernel_size": 5, "num_levels": 2, "dropout": 0.1}}
    model = build_model(cfg, 77).eval()
    loaded = model_from_checkpoint({"model_type": "tcn", "input_size": 77, "tcn": cfg["tcn"], "model_state_dict": model.state_dict()})
    x = torch.randn(2, 40, 77)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_split_video_does_not_invent_phase_or_consent():
    meta = create_metadata_template("s1", duration_s=10)
    assert meta["class_phase"][0]["name"] == "UNKNOWN"
    assert meta["phase_verified"] is False
    assert meta["consent_confirmed"] is False


def test_position_metrics_separate_weak_and_human():
    predictions = make_predictions()
    _, report = analyze_positions(predictions)
    assert report["status"] == "HUMAN_REVIEW_REQUIRED"
    assert all(x["human_raw_macro_f1"] is None for x in report["groups"])
    predictions["label_source"] = "HUMAN"
    _, human_report = analyze_positions(predictions)
    assert all(x["weak_agreement_macro_f1"] is None for x in human_report["groups"])
    assert human_report["groups"][0]["human_raw_macro_f1"] is not None


def test_annotation_sampler_all_test_includes_unlabelled_and_schema(tmp_path):
    windows = pd.DataFrame([{"window_id": f"w{i}", "session_id": "s1", "track_id": i,
        "split": "test", "participation_label": None, "quality_state": "INVALID",
        "t_start_ms": i*4000, "t_end_ms": i*4000+3900} for i in range(3)])
    source = tmp_path / "windows.csv"
    windows.to_csv(source, index=False)
    subprocess.run([sys.executable, "scripts/sample_annotation_subset.py", "--windows", str(source), "--out-dir", str(tmp_path / "human"), "--n-test", "0", "--n-train", "0", "--n-val", "0"], check=True, capture_output=True)
    template = pd.read_csv(tmp_path / "human/annotation_template.csv")
    assert len(template) == 3
    assert set(REQUIRED_COLUMNS) <= set(template.columns)
    assert template.participation_label.isna().all()


def test_human_evaluation_rejects_weak_and_blank_sources():
    pred = make_predictions()
    annotation = pd.DataFrame([{**{c: "" for c in REQUIRED_COLUMNS},
        "window_id": "w0", "session_id": "s1", "track_id": 1, "camera_id": "cam_A",
        "t_start_ms": 0, "t_end_ms": 3900, "annotation_state": "VALID", "annotation_confidence": 0.8,
        "participation_label": pred.true_label.iloc[0], "label_source": "WEAK_RULE", "label_rule_version": "weak-v0.1"}])
    with pytest.raises(ValueError, match="HUMAN"):
        join_human_annotations(pred, annotation)
    annotation["label_source"] = "HUMAN"
    annotation["label_rule_version"] = ""
    result = join_human_annotations(pred, annotation)
    assert len(result) == 1
    assert result.label_source.iloc[0] == "HUMAN"


def test_effective_config_is_snapshotted(tmp_path):
    import yaml
    from scripts.new_experiment import create_experiment
    original = {"project": {"seed": 42}, "data": {"dataset_version": "v0.1", "split_version": "s1"}, "model": {"type": "lstm"}}
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(original))
    effective = {**original, "model": {"type": "tcn"}, "training": {"batch_size": 7}}
    folder = create_experiment(name="effective", config_path=source, root=tmp_path, effective_config=effective)
    snapshot = yaml.safe_load((folder / "config.yaml").read_text())
    assert snapshot["model"]["type"] == "tcn"
    assert snapshot["training"]["batch_size"] == 7
    assert yaml.safe_load(source.read_text()) == original


def test_preview_sampling_matches_25fps_extraction(tmp_path):
    import cv2
    from src.pose.visualization import render_pose_overlay_video
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    output = render_pose_overlay_video(source, tmp_path / "preview.mp4", poses_df=pd.DataFrame(), sample_fps=10, blur_face=True, transcode_h264=False)
    cap = cv2.VideoCapture(str(output))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 4
    cap.release()


def test_head_blur_covers_lower_face_not_only_eye_strip():
    from src.pose.visualization import apply_face_blur
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    original = image.copy()
    points = np.zeros((17, 3))
    points[:3] = [[150, 80, .9], [135, 65, .9], [165, 65, .9]]
    apply_face_blur(image, (50, 20, 250, 290), points)
    assert image[105:115, 135:165].var() < original[105:115, 135:165].var() / 5


def test_hud_uses_effective_pose_model(monkeypatch):
    import cv2
    from src.pose.visualization import draw_hud_header
    seen = []
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args, **kwargs: seen.append(text))
    draw_hud_header(np.zeros((100, 800, 3), dtype=np.uint8), "external", 0, 0, 2, pipeline_label="YOLO11m-Pose + ByteTrack")
    assert "YOLO11m-Pose" in seen[0]
    assert "YOLOv8" not in seen[0]


def test_inference_passes_pose_config_and_uses_sample_rate(tmp_path, monkeypatch):
    import yaml
    import scripts.predict_video as prediction
    source, checkpoint_path = tmp_path / "video.mp4", tmp_path / "checkpoint.pt"
    source.write_bytes(b"fixture")
    checkpoint_path.write_bytes(b"fixture")
    config = tmp_path / "custom.yaml"
    config.write_text(yaml.safe_dump({"pose": {"model": "yolo11m-pose.pt"}, "sequence": {"window_seconds": 2, "stride_seconds": 1}}))
    track = make_dummy_track(60)
    track["timestamp_ms"] = track["frame_idx"] * 200
    seen = {}
    def extract(**kwargs):
        seen.update(kwargs)
        return track, {"sample_fps": 5, "sampled_frames": 60, "raw_frames": 300,
                       "total_detections": 60, "confidence_threshold": 0.3, "pose_model": "yolo11m-pose.pt"}
    monkeypatch.setattr(prediction, "extract_poses_from_video", extract)
    monkeypatch.setattr(prediction, "clean_and_normalize_poses", lambda poses, **kw: poses)
    model = build_model({"type": "lstm", "hidden_size": 16, "num_layers": 1}, 77)
    monkeypatch.setattr(prediction.torch, "load", lambda *a, **kw: {"model_type": "lstm", "hidden_size": 16, "num_layers": 1, "input_size": 77, "model_state_dict": model.state_dict()})
    out = tmp_path / "inference"
    monkeypatch.setattr(sys, "argv", ["predict_video.py", "--video", str(source), "--checkpoint", str(checkpoint_path), "--config", str(config), "--out-dir", str(out), "--no-zone", "--skip-preview"])
    prediction.main()
    assert seen["config_path"] == str(config)
    assert len(pd.read_csv(out / "predictions.csv")) == 11
    assert len(pd.read_csv(out / "poses.csv")) == len(track)
    assert json.loads((out / "inference_meta.json").read_text())["poses"] == str(out / "poses.csv")
    assert json.loads((out / "position_report.json").read_text())["status"] == "HUMAN_REVIEW_REQUIRED"


def test_weak_v02_rules_provenance_and_behavior():
    from src.contracts.labels import HIGH, LOW, MODERATE
    from src.preprocessing.weak_labels import apply_weak_label_rules

    base_rec = {"class_phase": "lecture", "quality_state": "VALID"}
    # Neutral upright posture with realistic YOLO11m jitter -> MODERATE (unlike v0.1 HIGH)
    tensor_mod = np.zeros((40, 94), dtype=np.float32)
    tensor_mod[:, 69] = 0.22
    tensor_mod[:, 70] = -0.25
    tensor_mod[:, 74] = 0.05
    lbl, rule = apply_weak_label_rules(base_rec, tensor_mod, rule_version="weak-v0.2")
    assert lbl == MODERATE
    assert rule == "v2_moderate_lecture"

    # Clear attentive forward posture in lecture -> HIGH
    tensor_high = np.zeros((40, 94), dtype=np.float32)
    tensor_high[:, 69] = 0.08
    tensor_high[:, 70] = -0.32
    tensor_high[:, 74] = 0.06
    lbl, rule = apply_weak_label_rules(base_rec, tensor_high, rule_version="weak-v0.2")
    assert lbl == HIGH
    assert rule == "v2_attentive_lecture"

    # Deep slumped head in individual_work with low wrist motion -> LOW
    tensor_low = np.zeros((40, 94), dtype=np.float32)
    tensor_low[:, 70] = 0.10
    tensor_low[:, 52:56] = 0.02
    lbl, rule = apply_weak_label_rules({"class_phase": "individual_work", "quality_state": "VALID"}, tensor_low, rule_version="weak-v0.2")
    assert lbl == LOW
    assert rule == "v2_head_down_individual"

    # Active reading/writing in individual_work -> HIGH
    tensor_write = np.zeros((40, 94), dtype=np.float32)
    tensor_write[:, 69] = 0.10
    tensor_write[:, 70] = -0.05
    tensor_write[:, 52:56] = 0.20
    labelled = assign_weak_labels_to_windows([{"class_phase": "individual_work", "quality_state": "VALID"}], tensor_write[None, ...], rule_version="weak-v0.2")
    assert labelled[0]["participation_label"] == HIGH
    assert labelled[0]["label_source"] == "WEAK_RULE"
    assert labelled[0]["label_rule_version"] == "weak-v0.2"

    with pytest.raises(ValueError, match="Unsupported weak rule version"):
        apply_weak_label_rules(base_rec, tensor_mod, rule_version="weak-v99")
