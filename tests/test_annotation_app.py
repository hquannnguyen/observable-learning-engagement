"""Exercise real form submission against isolated CSV fixtures, never user labels."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from scripts.dataset_qa import REQUIRED_COLUMNS, validate_annotation_dataframe
from src.contracts.annotation_io import read_annotation_csv, save_annotation_row
from src.contracts.labels import HIGH

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def review(tmp_path):
    directory = tmp_path / "data/annotations/review"
    directory.mkdir(parents=True)
    rows = []
    for i in range(2):
        row = dict.fromkeys(REQUIRED_COLUMNS, "")
        row.update(window_id=f"w{i}", session_id="test_session", camera_id="cam_A",
                   track_id="007", t_start_ms=i * 2000, t_end_ms=i * 2000 + 3900,
                   class_phase="UNKNOWN", phase_verified="False", phase_source="UNKNOWN",
                   annotation_guide_version="v0.2", zone_id="FRONT", video_path="", pose_overlay_path="")
        rows.append(row)
    for suffix in ("a", "b"):
        pd.DataFrame(rows).to_csv(directory / f"annotator_{suffix}.csv", index=False)
    return tmp_path, directory


def updates(name="Nguyễn Hữu Quân", **extra):
    result = dict(class_phase="lecture", annotation_state="VALID", participation_label=HIGH,
                label_source="HUMAN", label_rule_version="", annotator_id=name,
                annotation_confidence=0.8, atomic_behavior="BOARD_OR_INSTRUCTOR_FACING",
                occlusion_level="partial", notes="Đã xem toàn bộ cửa sổ", phase_verified=True,
                phase_source="HUMAN_WINDOW_REVIEW")
    result.update(extra)
    return result


def app_for(root):
    source = (ROOT / "app/annotation_app.py").read_text(encoding="utf-8")
    source = source.replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(root)!r})")
    return AppTest.from_string(source, default_timeout=30).run()


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def test_blank_numeric_column_save_and_round_trip(review):
    _, directory = review
    path = directory / "annotator_a.csv"
    loaded = read_annotation_csv(path)
    assert loaded.annotation_confidence.isna().all()
    assert loaded.track_id.tolist() == ["007", "007"]
    saved = save_annotation_row(path, "w0", updates())
    assert saved.at[0, "annotation_confidence"] == 0.8
    reopened = read_annotation_csv(path)
    assert reopened.at[0, "annotation_confidence"] == 0.8
    assert pd.isna(reopened.at[1, "annotation_confidence"])
    assert reopened.at[0, "phase_verified"] == "True"
    assert validate_annotation_dataframe(reopened.iloc[[0]]) == (True, [])


def test_save_reloads_other_rows_and_preserves_other_person(review):
    _, directory = review
    a, b = directory / "annotator_a.csv", directory / "annotator_b.csv"
    untouched = b.read_bytes()
    save_annotation_row(a, "w1", updates(notes="Second"))
    save_annotation_row(a, "w0", updates())
    saved = read_annotation_csv(a)
    assert saved.at[1, "notes"] == "Second"
    assert saved.at[1, "annotation_confidence"] == 0.8
    assert b.read_bytes() == untouched


def test_protected_keys_rejected_without_writing(review):
    _, directory = review
    path = directory / "annotator_a.csv"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="khóa"):
        save_annotation_row(path, "w0", {**updates(), "track_id": "99"})
    assert path.read_bytes() == original


@pytest.mark.parametrize("person,suffix,other,full_name", [
    ("Quân", "a", "b", "Nguyễn Hữu Quân"),
    ("Phong", "b", "a", "Vũ Văn Phong"),
])
def test_person_form_save_next_and_reopen(review, person, suffix, other, full_name):
    root, directory = review
    other_path = directory / f"annotator_{other}.csv"
    original_other = other_path.read_bytes()
    app = app_for(root)
    assert not app.exception
    app.radio[0].set_value(person).run()
    for element in app.selectbox:
        if element.label.startswith("Khoảng "):
            element.select("ALIGNED")
    app.run()
    widget(app.selectbox, "Pha hoạt động thực tế").select("lecture")
    widget(app.selectbox, "Khả năng quan sát").select("VALID")
    widget(app.selectbox, "Mức tham gia quan sát được nếu VALID").select(HIGH)
    widget(app.selectbox, "Mức che khuất").select("partial")
    widget(app.text_input, "Hành vi quan sát được").set_value("BOARD_OR_INSTRUCTOR_FACING")
    widget(app.text_input, "Bằng chứng xác định nhiệm vụ / pha hoạt động").set_value("Thấy giáo viên đang giảng trước bảng")
    widget(app.checkbox, "Tôi đã xem hết đoạn, xác định đúng người và xác nhận bằng chứng theo bảng tiêu chí").check()
    widget(app.button, "Lưu và đến cửa sổ chưa gán tiếp").click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    saved = read_annotation_csv(directory / f"annotator_{suffix}.csv")
    assert saved.at[0, "annotator_id"] == full_name
    assert saved.at[0, "label_source"] == "HUMAN"
    assert saved.at[0, "annotation_confidence"] == 0.8
    assert saved.at[0, "annotation_guide_version"] == "v0.3-pilot"
    assert 'DIRECT_HUMAN_OBSERVATION' in saved.at[0, "annotation_evidence"]
    assert saved.at[1, "annotator_id"] == ""
    assert widget(app.selectbox, "Cửa sổ").value == 1
    assert other_path.read_bytes() == original_other
    widget(app.button, "← Cửa sổ trước").click().run()
    assert not app.exception
    assert widget(app.selectbox, "Mức che khuất").value == "partial"
    assert widget(app.selectbox, "Pha hoạt động thực tế").value == "lecture"
    assert app.slider[0].value == 0.8
    # A fresh browser session resumes at the first genuinely unfinished window.
    reopened = app_for(root)
    reopened.radio[0].set_value(person).run()
    assert not reopened.exception
    assert widget(reopened.selectbox, "Cửa sổ").value == 1


def test_invalid_form_does_not_write_and_occluded_clears_label(review):
    root, directory = review
    path = directory / "annotator_a.csv"
    original = path.read_bytes()
    app = app_for(root)
    app.radio[0].set_value("Quân").run()
    widget(app.selectbox, "Khả năng quan sát").select("VALID")
    widget(app.button, "Lưu tại cửa sổ này").click().run()
    assert not app.exception
    assert app.error
    assert path.read_bytes() == original
    widget(app.selectbox, "Khả năng quan sát").select("OCCLUDED")
    widget(app.selectbox, "Mức che khuất").select("heavy")
    widget(app.button, "Lưu tại cửa sổ này").click().run()
    assert not app.exception
    saved = read_annotation_csv(path)
    assert saved.at[0, "annotation_state"] == "OCCLUDED"
    assert saved.at[0, "label_source"] == ""
    assert saved.at[0, "participation_label"] == ""
    assert saved.at[0, "phase_source"] == "UNKNOWN"
    assert validate_annotation_dataframe(saved.iloc[[0]]) == (True, [])


def test_original_dataset_is_default_and_external_remains_separate(review):
    root, directory = review
    original = directory.parent / "human_review_v0_2"
    external = directory.parent / "external_review_20261109"
    for destination in (original, external):
        destination.mkdir()
        for suffix in ("a", "b"):
            frame = read_annotation_csv(directory / f"annotator_{suffix}.csv")
            frame["session_id"] = destination.name
            frame.to_csv(destination / f"annotator_{suffix}.csv", index=False)
    before = {p: p.read_bytes() for p in directory.parent.glob("**/annotator_*.csv")}
    app = app_for(root)
    app.radio[0].set_value("Quân").run()
    chooser = widget(app.selectbox, "Bộ video cần gán")
    assert chooser.value == str(original / "annotator_a.csv")
    assert chooser.options[0] == "Dataset cũ · bộ gán nhãn chính"
    chooser.select(str(external / "annotator_a.csv")).run()
    assert not app.exception
    assert widget(app.selectbox, "Cửa sổ").options[0].find("external_review_20261109") >= 0
    app.radio[0].set_value("Phong").run()
    assert not app.exception
    assert widget(app.selectbox, "Bộ video cần gán").value == str(original / "annotator_b.csv")
    assert all(p.read_bytes() == content for p, content in before.items())


def test_valid_requires_basis_confirmation_and_label_consistent_with_evidence(review):
    root, directory = review
    path = directory / "annotator_a.csv"
    original = path.read_bytes()
    app = app_for(root)
    app.radio[0].set_value("Quân").run()
    for element in app.selectbox:
        if element.label.startswith("Khoảng "):
            element.select("ALIGNED")
    app.run()
    widget(app.selectbox, "Pha hoạt động thực tế").select("lecture")
    widget(app.selectbox, "Khả năng quan sát").select("VALID")
    from src.contracts.labels import LOW
    widget(app.selectbox, "Mức tham gia quan sát được nếu VALID").select(LOW)
    widget(app.button, "Lưu tại cửa sổ này").click().run()
    assert not app.exception and app.error
    assert path.read_bytes() == original
    widget(app.text_input, "Hành vi quan sát được").set_value("BOARD_OR_INSTRUCTOR_FACING")
    widget(app.text_input, "Bằng chứng xác định nhiệm vụ / pha hoạt động").set_value("Giáo viên đang giảng trước bảng")
    widget(app.checkbox, "Tôi đã xem hết đoạn, xác định đúng người và xác nhận bằng chứng theo bảng tiêu chí").check()
    widget(app.button, "Lưu tại cửa sổ này").click().run()
    assert not app.exception and app.error
    assert "Bằng chứng chưa phù hợp" in app.error[0].value
    assert path.read_bytes() == original
