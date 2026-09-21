import numpy as np
import pytest

from src.preprocessing.feature_contract import validate_temporal_features


def test_temporal_tensor_contract_accepts_model_ready_array():
    features = np.zeros((40, 64), dtype=np.float32)  # 10 FPS × 4 s baseline
    validated = validate_temporal_features(features, expected_timesteps=40)
    assert validated.shape == (40, 64)
    assert validated.dtype == np.float32


def test_temporal_tensor_contract_rejects_wrong_timestep_count():
    with pytest.raises(ValueError, match="Expected 40 timesteps"):
        validate_temporal_features(np.zeros((39, 64)), expected_timesteps=40)


def test_temporal_tensor_contract_rejects_nan():
    features = np.zeros((40, 64), dtype=np.float32)
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_temporal_features(features)
