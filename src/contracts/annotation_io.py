"""Editable annotation CSVs, including pandas 3's strict string dtypes."""
from pathlib import Path
import tempfile

import pandas as pd

from src.contracts.labels import validate_annotation_label


EDITABLE_FIELDS = {
    "class_phase", "phase_verified", "phase_source", "annotation_state",
    "participation_label", "label_source", "label_rule_version", "annotator_id",
    "atomic_behavior", "annotation_confidence", "occlusion_level", "notes",
    "annotation_guide_version",
    "annotation_evidence", "phase_evidence",
}


def read_annotation_csv(path: Path) -> pd.DataFrame:
    # Preserve IDs as text; object columns accept empty template cells and bool/text edits.
    frame = pd.read_csv(path, dtype=object, keep_default_na=False)
    frame["annotation_confidence"] = pd.to_numeric(
        frame["annotation_confidence"].replace("", pd.NA), errors="raise"
    ).astype("Float64")
    if "window_id" not in frame or frame["window_id"].eq("").any() or frame["window_id"].duplicated().any():
        raise ValueError("Phiếu cần window_id duy nhất và không để trống.")
    return frame


def save_annotation_row(path: Path, window_id: str, updates: dict) -> pd.DataFrame:
    """Reload before editing one stable key; atomically replace only this person's file."""
    if set(updates) - EDITABLE_FIELDS:
        raise ValueError("Không được thay đổi khóa video/track/thời gian của cửa sổ.")
    validate_annotation_label(
        participation_label=updates["participation_label"],
        annotation_state=updates["annotation_state"],
        label_source=updates["label_source"],
        label_rule_version=updates["label_rule_version"],
    )
    if not updates["annotator_id"].strip():
        raise ValueError("Thiếu người gán nhãn.")
    if not 0 <= float(updates["annotation_confidence"]) <= 1:
        raise ValueError("Độ tin cậy phải nằm trong [0, 1].")
    frame = read_annotation_csv(path)
    matches = frame.index[frame["window_id"].eq(window_id)]
    if len(matches) != 1:
        raise ValueError("Không tìm thấy cửa sổ trong phiếu hiện tại; hãy tải lại trang.")
    for key, value in updates.items():
        if key not in frame:
            frame[key] = pd.Series("", index=frame.index, dtype=object)
        frame.at[matches[0], key] = value
    write_annotation_csv(path, frame)
    return frame


def write_annotation_csv(path: Path, frame: pd.DataFrame) -> None:
    """Publish a complete CSV with a same-directory atomic replacement."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
