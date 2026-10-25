"""Unit tests for feature engineering, sliding windows, and LSTM model."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.lstm import ParticipationLSTM
from src.preprocessing.feature_contract import validate_temporal_features
from src.preprocessing.features import compute_track_features
from src.preprocessing.windows import build_track_windows
from src.training.dataset import WindowFeatureDataset
from src.training.trainer import compute_multiclass_metrics, evaluate_model, train_epoch


def make_dummy_track(num_frames=60, track_id=1, session_id="test_s"):
    """Create synthetic pose track DataFrame."""
    rows = []
    for f in range(num_frames):
        row = {
            "session_id": session_id,
            "camera_id": "cam_A",
            "frame_idx": f,
            "timestamp_ms": f * 100,
            "track_id": track_id,
            "visible_ratio": 0.85,
            "mean_keypoint_conf": 0.80,
            "class_phase": "lecture",
            "zone_id": "M-C",
        }
        for i in range(17):
            row[f"norm_kpt_{i:02d}_x"] = 0.1 * (i - 8)
            row[f"norm_kpt_{i:02d}_y"] = 0.05 * i
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_extraction_shape_and_finiteness():
    track_df = make_dummy_track(num_frames=50)
    features, names = compute_track_features(track_df, sample_fps=10.0)

    assert features.shape == (50, 77)
    assert len(names) == 77
    assert np.isfinite(features).all()
    assert not np.isnan(features).any()


def test_feature_v2_mask_marks_missing_joint():
    track_df = make_dummy_track(num_frames=50)
    # Mark nose (keypoint 0) missing at frame 10.
    track_df.loc[10, "norm_kpt_00_x"] = np.nan
    track_df.loc[10, "norm_kpt_00_y"] = np.nan

    features, names = compute_track_features(
        track_df, sample_fps=10.0, include_mask=True
    )

    assert features.shape == (50, 94)
    assert len(names) == 94
    assert names[-17:] == [f"kpt_mask_{i:02d}" for i in range(17)]
    mask = features[:, 77:]
    assert mask.shape == (50, 17)
    assert mask[10, 0] == 0.0
    assert mask[10, 1] == 1.0
    assert mask[11, 0] == 1.0
    assert np.isfinite(features).all()


def test_sliding_window_builder():
    track_df = make_dummy_track(num_frames=60)  # 6 seconds @ 10 FPS
    records, tensors = build_track_windows(
        track_df,
        window_seconds=4.0,  # 40 steps
        stride_seconds=2.0,  # 20 steps
        sample_fps=10.0,
    )

    # 60 frames with W=40, S=20 -> windows at start 0, 20 -> 2 windows
    assert len(records) == 2
    assert tensors.shape == (2, 40, 77)

    for w in tensors:
        validated = validate_temporal_features(w, expected_timesteps=40)
        assert validated.shape == (40, 77)


def test_sliding_window_v2_uses_configured_thresholds():
    track_df = make_dummy_track(num_frames=60)
    records, tensors = build_track_windows(
        track_df,
        include_mask=True,
        quality_thresholds={"min_visible_ratio": 0.99, "min_mean_keypoint_confidence": 0.99},
    )
    assert tensors.shape == (2, 40, 94)
    assert records[0]["feature_schema_version"] == "posefeat-v2"
    assert records[0]["quality_state"] == "INVALID"
    assert records[0]["quality_reason"] == "LOW_VISIBILITY"


def test_lstm_forward_and_probabilities():
    model = ParticipationLSTM(input_size=77, hidden_size=64, num_layers=2, num_classes=3)
    dummy_input = torch.randn(8, 40, 77)  # Batch=8, T=40, F=77

    logits = model(dummy_input)
    assert logits.shape == (8, 3)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (8, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_tcn_is_causal_and_probabilities_sum_to_one():
    from src.models.factory import build_model
    from src.models.tcn import ParticipationTCN

    model = ParticipationTCN(input_size=77, hidden_size=32, num_levels=3)
    x = torch.randn(4, 40, 77)
    logits = model(x)
    assert logits.shape == (4, 3)

    probs = model.predict_proba(x)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    # The convolution stack itself must be causal: perturbing future steps
    # cannot change earlier outputs.
    from src.models.tcn import CausalConvBlock

    block = CausalConvBlock(4, 4).eval()
    base = torch.randn(1, 4, 10)
    changed = base.clone()
    changed[..., 5:] += 1.0
    with torch.no_grad():
        assert torch.allclose(block(base)[..., :5], block(changed)[..., :5], atol=1e-5)

    tcn_cfg = {"type": "tcn", "num_classes": 3, "tcn": {"hidden_size": 32, "num_levels": 3}}
    built = build_model(tcn_cfg, input_size=77)
    assert isinstance(built, ParticipationTCN)


def test_model_from_checkpoint_loads_weights():
    from src.models.factory import model_from_checkpoint

    model = ParticipationLSTM(input_size=77, hidden_size=16, num_layers=1, num_classes=3)
    checkpoint = {
        "model_type": "LSTM",
        "input_size": 77,
        "hidden_size": 16,
        "num_layers": 1,
        "num_classes": 3,
        "model_state_dict": model.state_dict(),
    }
    loaded = model_from_checkpoint(checkpoint)
    model.eval()
    x = torch.randn(2, 40, 77)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_predict_dataframe_end_to_end():
    from scripts.predict_video import predict_dataframe
    from src.uncertainty.gate import UncertaintyThresholds

    track = make_dummy_track(num_frames=60)
    model = ParticipationLSTM(input_size=77, hidden_size=16, num_layers=1, num_classes=3)
    preds = predict_dataframe(
        track,
        model,
        input_size=77,
        thresholds=UncertaintyThresholds(min_probability=0.0),
        quality_thresholds={"min_visible_ratio": 0.6, "min_mean_keypoint_confidence": 0.3},
    )
    assert len(preds) == 2
    assert set(preds["gated_label"]) <= {
        "HIGH_OBSERVED_PARTICIPATION",
        "MODERATE_OBSERVED_PARTICIPATION",
        "LOW_OBSERVED_PARTICIPATION",
        "UNCERTAIN",
    }
    assert preds["true_label"].isna().all()


def test_training_and_evaluation_step():
    X = np.random.randn(20, 40, 77).astype(np.float32)
    y = np.random.randint(0, 3, size=(20,))

    ds = WindowFeatureDataset(X, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = ParticipationLSTM(input_size=77, hidden_size=32, num_layers=1, num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss, acc = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0

    val_loss, metrics = evaluate_model(model, loader, criterion, torch.device("cpu"))
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
