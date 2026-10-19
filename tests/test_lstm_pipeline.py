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


def test_lstm_forward_and_probabilities():
    model = ParticipationLSTM(input_size=77, hidden_size=64, num_layers=2, num_classes=3)
    dummy_input = torch.randn(8, 40, 77)  # Batch=8, T=40, F=77

    logits = model(dummy_input)
    assert logits.shape == (8, 3)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (8, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


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
