import pandas as pd
import pytest

from app.studio_view import _save_600_window
from src.contracts.annotation_io import read_annotation_csv
from src.contracts.labels import HIGH


def test_studio_saves_only_requested_window_without_overwriting_other_edits(tmp_path):
    path = tmp_path / "annotator_a.csv"
    pd.DataFrame([
        {"window_id": "w1", "annotation_confidence": "", "notes": ""},
        {"window_id": "w2", "annotation_confidence": "", "notes": ""},
    ]).to_csv(path, index=False)
    stale = read_annotation_csv(path)
    external = pd.read_csv(path, keep_default_na=False)
    external.loc[1, "notes"] = "independent update"
    external.to_csv(path, index=False)
    args = dict(df_600=stale, row_idx=0, target_label=HIGH, target_state="VALID",
                behaviors="READING_WRITING", phase="lecture", zone="", occlusion="none",
                notes="reviewed", annotator_id="Tài", primary_file_path=path,
                person_name="Tài", phase_evidence="Giảng viên đang giảng",
                confirmed=True)
    _save_600_window(**args)
    result = read_annotation_csv(path)
    assert result.loc[1, "notes"] == "independent update"
    assert result.loc[0, "participation_label"] == HIGH
    assert result.loc[0, "annotation_guide_version"] == "v0.2"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _save_600_window(**{**args, "phase_evidence": ""})
    assert path.read_bytes() == before


def test_occluded_has_no_label_or_source(tmp_path):
    path = tmp_path / "annotator_b.csv"
    pd.DataFrame([{"window_id": "w1", "annotation_confidence": ""}]).to_csv(path, index=False)
    frame = read_annotation_csv(path)
    _save_600_window(frame, 0, "", "OCCLUDED", "OCCLUDED", "UNKNOWN", "", "heavy", "", "Phong", path, "Phong", "", False)
    result = read_annotation_csv(path).iloc[0]
    assert result["participation_label"] == result["label_source"] == ""
    assert not result["phase_verified"] or result["phase_verified"] == "False"
