"""Model factory so training/evaluation select LSTM or TCN from config/checkpoint."""

from __future__ import annotations

from typing import Any, Dict

from src.models.lstm import ParticipationLSTM
from src.models.tcn import ParticipationTCN


def build_model(
    model_cfg: Dict[str, Any],
    input_size: int,
    model_type: str | None = None,
) -> Any:
    """Build a model from a config ``model`` section."""
    effective = (model_type or model_cfg.get("type", "lstm")).lower()
    num_classes = int(model_cfg.get("num_classes", 3))

    if effective == "lstm":
        return ParticipationLSTM(
            input_size=input_size,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.30)),
        )
    if effective == "tcn":
        tcn_cfg = model_cfg.get("tcn", {})
        return ParticipationTCN(
            input_size=input_size,
            hidden_size=int(tcn_cfg.get("hidden_size", 64)),
            num_levels=int(tcn_cfg.get("num_levels", 4)),
            kernel_size=int(tcn_cfg.get("kernel_size", 3)),
            num_classes=num_classes,
            dropout=float(tcn_cfg.get("dropout", 0.20)),
        )
    raise ValueError(f"Unknown model type: {effective!r}")


def model_from_checkpoint(checkpoint: Dict[str, Any]) -> Any:
    """Rebuild the architecture recorded in a checkpoint and load its weights."""
    model_type = str(checkpoint.get("model_type", "LSTM")).lower()
    num_classes = int(checkpoint.get("num_classes", 3))
    input_size = int(checkpoint.get("input_size", 77))

    if model_type == "lstm":
        model = ParticipationLSTM(
            input_size=input_size,
            hidden_size=int(checkpoint.get("hidden_size", 128)),
            num_layers=int(checkpoint.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(checkpoint.get("dropout", 0.30)),
        )
    elif model_type == "tcn":
        tcn_cfg = checkpoint.get("tcn", {})
        model = ParticipationTCN(
            input_size=input_size,
            hidden_size=int(tcn_cfg.get("hidden_size", checkpoint.get("hidden_size", 64))),
            num_levels=int(tcn_cfg.get("num_levels", checkpoint.get("num_levels", 4))),
            kernel_size=int(tcn_cfg.get("kernel_size", checkpoint.get("kernel_size", 3))),
            num_classes=num_classes,
            dropout=float(tcn_cfg.get("dropout", checkpoint.get("dropout", 0.20))),
        )
    else:
        raise ValueError(f"Unknown checkpoint model_type: {model_type!r}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("checkpoint missing model_state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model
