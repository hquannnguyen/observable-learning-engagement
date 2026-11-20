"""Unit and integration tests for TCN temporal model and uncertainty calibration.

Tests Phase 4 requirements:
- ParticipationTCN initialization, forward pass, output shape [Batch, Classes=3]
- predict_proba normalized sum == 1.0
- Causality of CausalConvBlock
- Optimization backward step
- ParticipationLSTM bidirectional vs unidirectional
- Factory instantiation
- TemperatureScaler, compute_ece, and compute_accuracy_coverage_curve
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.factory import build_model, model_from_checkpoint
from src.models.lstm import ParticipationLSTM
from src.models.tcn import CausalConvBlock, ParticipationTCN
from src.training.trainer import (
    TemperatureScaler,
    compute_accuracy_coverage_curve,
    compute_ece,
)


def test_tcn_initialization_and_forward_shape():
    model = ParticipationTCN(
        input_size=77,
        hidden_size=64,
        num_levels=4,
        kernel_size=3,
        num_classes=3,
        dropout=0.2,
    )
    dummy_input = torch.randn(8, 40, 77)
    logits = model(dummy_input)

    assert logits.shape == (8, 3)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_tcn_predict_proba():
    model = ParticipationTCN(input_size=77, hidden_size=32, num_levels=2, num_classes=3)
    dummy_input = torch.randn(4, 40, 77)
    probs = model.predict_proba(dummy_input)

    assert probs.shape == (4, 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_causal_conv_block_causality():
    # If we perturb x at timestep t_future, outputs at t < t_future must not change.
    block = CausalConvBlock(in_channels=16, out_channels=16, kernel_size=3, dilation=2, dropout=0.0)
    block.eval()

    T = 20
    x1 = torch.randn(2, 16, T)
    x2 = x1.clone()

    # Modify only the last 5 timesteps in x2
    perturb_start = 15
    x2[..., perturb_start:] += torch.randn_like(x2[..., perturb_start:])

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    # For t < perturb_start, out1 and out2 should be identical
    assert torch.allclose(out1[..., :perturb_start], out2[..., :perturb_start], atol=1e-5)


def test_tcn_backward_step():
    model = ParticipationTCN(input_size=77, hidden_size=32, num_levels=2, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 40, 77)
    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Verify gradients exist and are finite
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()

    optimizer.step()


def test_lstm_bidirectional_and_unidirectional():
    # Unidirectional LSTM
    lstm_uni = ParticipationLSTM(input_size=77, hidden_size=64, num_classes=3, bidirectional=False)
    x = torch.randn(4, 40, 77)
    logits_uni = lstm_uni(x)
    assert logits_uni.shape == (4, 3)
    assert lstm_uni.num_directions == 1

    # Bidirectional LSTM
    lstm_bi = ParticipationLSTM(input_size=77, hidden_size=64, num_classes=3, bidirectional=True)
    logits_bi = lstm_bi(x)
    assert logits_bi.shape == (4, 3)
    assert lstm_bi.num_directions == 2
    assert lstm_bi.head[0].in_features == 128  # hidden_size * 2


def test_model_factory_tcn_and_bilstm():
    # Factory with TCN
    tcn_cfg = {
        "type": "tcn",
        "tcn": {"hidden_size": 32, "num_levels": 3, "kernel_size": 3, "dropout": 0.1},
    }
    model_tcn = build_model(tcn_cfg, input_size=77)
    assert isinstance(model_tcn, ParticipationTCN)
    assert model_tcn.hidden_size == 32
    assert model_tcn.num_levels == 3

    # Factory with BiLSTM
    bilstm_cfg = {
        "type": "bilstm",
        "hidden_size": 64,
        "num_layers": 2,
    }
    model_bilstm = build_model(bilstm_cfg, input_size=77)
    assert isinstance(model_bilstm, ParticipationLSTM)
    assert model_bilstm.bidirectional is True


def test_temperature_scaler_and_calibration():
    scaler = TemperatureScaler(init_temperature=1.5)
    assert scaler.temperature.item() == pytest.approx(1.5, rel=1e-3)

    # Synthetic validation logits and targets
    val_logits = torch.tensor(
        [
            [3.0, 1.0, 0.5],
            [0.2, 2.5, 0.8],
            [0.1, 0.5, 2.8],
            [2.5, 0.5, 0.2],
            [0.1, 2.9, 0.4],
        ],
        dtype=torch.float32,
    )
    val_targets = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    opt_temp = scaler.fit(val_logits, val_targets)
    assert opt_temp > 0.0

    scaled = scaler(val_logits)
    assert scaled.shape == val_logits.shape

    probs = torch.softmax(scaled, dim=-1).detach().numpy()
    targets_np = val_targets.numpy()

    # Compute ECE
    ece = compute_ece(probs, targets_np, n_bins=5)
    assert 0.0 <= ece <= 1.0

    # Compute Accuracy vs Coverage curve
    curve = compute_accuracy_coverage_curve(probs, targets_np, thresholds=[0.5, 0.6, 0.7, 0.8])
    assert len(curve) == 4
    for idx, row in curve.iterrows():
        assert "threshold" in row
        assert "coverage" in row
        assert "accuracy" in row
        assert 0.0 <= row["coverage"] <= 1.0
