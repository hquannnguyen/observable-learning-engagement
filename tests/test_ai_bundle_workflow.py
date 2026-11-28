"""Tests for the hybrid Local AI + ChatGPT Plus / Gemini Pro pre-annotation workflow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import app.studio_view as studio_view
from src.contracts.annotation_io import read_annotation_csv
from src.contracts.labels import HIGH, MODERATE
from src.models.factory import build_model
from src.pose.ai_bundle import (
    build_web_llm_prompt,
    build_window_contact_sheet,
    load_suggestions_map,
    parse_and_validate_suggestions,
    propose_local_suggestion,
    propose_local_suggestions_from_features,
    save_suggestions_map,
    validate_window_source,
)


def _make_raw_pose_df(
    num_frames: int = 50,
    track_id: int = 7,
    session_id: str = "s_test",
    camera_id: str = "cam_A",
) -> pd.DataFrame:
    rows = []
    for f in range(num_frames):
        row = {
            "session_id": session_id,
            "camera_id": camera_id,
            "frame_idx": f,
            "timestamp_ms": f * 100,
            "track_id": track_id,
            "bbox_x1": 180.0,
            "bbox_y1": 80.0,
            "bbox_x2": 320.0,
            "bbox_y2": 320.0,
            "bbox_conf": 0.9,
            "visible_ratio": 0.88,
            "mean_keypoint_conf": 0.82,
            "zone_id": "M-C",
        }
        for i in range(17):
            row[f"kpt_{i:02d}_x"] = 250.0 + ((i % 3) - 1) * 15.0
            row[f"kpt_{i:02d}_y"] = 110.0 + i * 10.0
            row[f"kpt_{i:02d}_conf"] = 0.85
        rows.append(row)
    return pd.DataFrame(rows)


def _write_test_video(path: Path, n_frames: int = 50, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    assert writer.isOpened()
    try:
        for idx in range(n_frames):
            frame = np.full((360, 640, 3), 40 + (idx % 20), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_validate_window_source_and_contact_sheet(tmp_path):
    video_path = tmp_path / "cam_A.mp4"
    _write_test_video(video_path, n_frames=50, fps=10)
    poses = _make_raw_pose_df(50, track_id=7, session_id="s_test", camera_id="cam_A")

    # Valid alignment
    win_poses, audit = validate_window_source(video_path, poses, "s_test", "cam_A", "7", 0.0, 4000.0)
    assert audit["ok"] is True
    assert audit["detection_count"] >= 40
    assert not win_poses.empty

    # Mismatched session fails validation
    _, bad_audit = validate_window_source(video_path, poses, "wrong_session", "cam_A", "7", 0.0, 4000.0)
    assert bad_audit["ok"] is False

    # Contact sheet generation
    out_img = tmp_path / "bundles" / "w1.jpg"
    canvas, meta = build_window_contact_sheet(
        video_path,
        poses,
        {
            "window_id": "w1",
            "session_id": "s_test",
            "camera_id": "cam_A",
            "track_id": "7",
            "t_start_ms": 0,
            "t_end_ms": 4000,
            "split": "train",
            "zone_id": "M-C",
        },
        output_path=out_img,
        class_phase="lecture",
        phase_verified=True,
    )
    assert out_img.is_file()
    assert canvas.shape[1] == 1280
    assert meta["class_phase"] == "lecture"
    assert meta["phase_verified"] is True
    assert meta["pose_quality"]["ok"] is True


def test_local_proposer_and_web_llm_json_roundtrip(tmp_path):
    video_path = tmp_path / "cam_A.mp4"
    _write_test_video(video_path, n_frames=50, fps=10)
    poses = _make_raw_pose_df(50, track_id=3, session_id="s_test", camera_id="cam_A")

    model = build_model({"type": "lstm", "hidden_size": 16, "num_layers": 1}, input_size=94)
    row = {
        "window_id": "s_test_cam_A_trk003_w0000",
        "session_id": "s_test",
        "camera_id": "cam_A",
        "track_id": "3",
        "t_start_ms": 0,
        "t_end_ms": 4000,
    }
    sug = propose_local_suggestion(
        row, poses, video_path=video_path, model=model, input_size=94, class_phase="lecture", model_id="test-lstm"
    )
    assert sug["window_id"] == "s_test_cam_A_trk003_w0000"
    assert sug["suggested_state"] == "VALID"
    assert sug["suggestion_source"] == "LOCAL_AI_PROPOSER"
    assert sug["suggested_behaviors"]
    assert "Pha=lecture" in sug["rationale"]

    # Missing track triggers OCCLUDED / rejection reason
    missing_row = {**row, "window_id": "w_missing", "track_id": "999"}
    bad_sug = propose_local_suggestion(
        missing_row, poses, video_path=video_path, model=model, input_size=94, class_phase="lecture"
    )
    assert bad_sug["suggested_state"] in ("OCCLUDED", "INVALID")
    assert bad_sug["suggested_label"] == ""
    assert bad_sug["rejection_reason"]

    # Prompt generation + fenced ChatGPT/Gemini JSON parsing
    prompt = build_web_llm_prompt(
        [{"window_id": "w1", "track_id": "3", "class_phase": "lecture", "pose_quality": {"mean_visible_ratio": 0.9, "mean_pose_confidence": 0.8, "frame_coverage": 1.0}}]
    )
    assert "w1" in prompt and "lecture" in prompt

    chatgpt_reply = """Dưới đây là kết quả:
```json
[
  {
    "window_id": "w1",
    "state": "VALID",
    "label": "HIGH",
    "behaviors": ["HAND_RAISED", "BOARD_OR_INSTRUCTOR_FACING"],
    "rationale": "Sinh viên giơ tay phát biểu rõ ràng.",
    "rejection_reason": ""
  },
  {
    "window_id": "w2",
    "state": "OCCLUDED",
    "label": "",
    "behaviors": ["OCCLUDED_KEYPOINTS"],
    "rationale": "Bị người phía trước che khuất.",
    "rejection_reason": "Che khuất >50% cửa sổ"
  }
]
```"""
    parsed = parse_and_validate_suggestions(chatgpt_reply)
    assert parsed[0]["suggested_label"] == HIGH
    assert parsed[1]["suggested_state"] == "OCCLUDED"
    assert parsed[1]["suggested_label"] == ""

    sug_file = tmp_path / "ai_suggestions.json"
    save_suggestions_map(sug_file, parsed)
    assert sug_file.with_suffix(".csv").is_file()
    loaded = load_suggestions_map(sug_file)
    assert set(loaded) == {"w1", "w2"}


def test_studio_assisted_review_does_not_write_human_until_saved(tmp_path, monkeypatch):
    ind_dir = tmp_path / "data/annotations/review_v03_independent_20261124"
    ind_dir.mkdir(parents=True)
    row = {
        "window_id": "s_test_cam_A_trk003_w0000",
        "session_id": "s_test",
        "camera_id": "cam_A",
        "track_id": "3",
        "t_start_ms": "0",
        "t_end_ms": "4000",
        "split": "train",
        "class_phase": "UNKNOWN",
        "atomic_behavior": "",
        "participation_label": "",
        "annotation_state": "",
        "label_source": "",
        "label_rule_version": "",
        "annotation_guide_version": "v0.2",
        "annotation_confidence": "",
        "occlusion_level": "none",
        "zone_id": "",
        "annotator_id": "",
        "notes": "",
        "phase_verified": "False",
        "phase_source": "UNKNOWN",
        "phase_evidence": "",
        "annotation_evidence": "",
        "video_path": "",
    }
    csv_path = ind_dir / "annotator_a.csv"
    pd.DataFrame([row, {**row, "window_id": "second_window", "track_id": "4"}]).to_csv(csv_path, index=False)
    sug_path = ind_dir / "ai_bundles" / "ai_suggestions.json"
    save_suggestions_map(
        sug_path,
        [
            {
                "window_id": "s_test_cam_A_trk003_w0000",
                "suggested_state": "VALID",
                "suggested_label": MODERATE,
                "suggested_behaviors": ["BOARD_OR_INSTRUCTOR_FACING"],
                "suggested_phase": "lecture",
                "suggested_occlusion": "none",
                "rationale": "Ngồi hướng bảng ổn định.",
                "rejection_reason": "",
                "confidence": 0.82,
                "suggestion_source": "VISION_LLM_IMPORT",
            }
        ],
    )

    monkeypatch.setattr(studio_view, "ROOT", tmp_path)
    root = Path(__file__).resolve().parents[1]
    app_code = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import app.studio_view as sv\n"
        f"sv.render_studio_tab('Tài', 'Vũ Đức Tài', Path({str(csv_path)!r}))\n"
    )
    app = AppTest.from_string(app_code, default_timeout=30).run()
    assert not app.exception

    # Assisted mode ON by default: pre-fills UI draft, but CSV stays untouched!
    assert read_annotation_csv(csv_path).at[0, "annotator_id"] == ""
    assert any(b.label.startswith("📝 Điền gợi ý") for b in app.button)
    assert any("✓" in b.label and "VỪA (MODERATE)" in b.label for b in app.button)

    # Toggling Assisted Mode OFF clears the auto-applied draft back to blind mode.
    ai_chk = next(c for c in app.checkbox if "gợi ý AI hỗ trợ" in c.label)
    ai_chk.uncheck().run()
    assert not app.exception
    assert not any(b.label.startswith("📝 Điền gợi ý") for b in app.button)
    assert not any("✓" in b.label and "VỪA (MODERATE)" in b.label for b in app.button)

    # Toggling back ON and clicking "Điền gợi ý để sửa" -> still must NOT touch CSV!
    ai_chk = next(c for c in app.checkbox if "gợi ý AI hỗ trợ" in c.label)
    ai_chk.check().run()
    assert not app.exception
    apply_btn = next(b for b in app.button if b.label.startswith("📝 Điền gợi ý"))
    apply_btn.click().run()
    assert not app.exception
    after_apply = read_annotation_csv(csv_path)
    assert after_apply.at[0, "annotator_id"] == ""
    assert after_apply.at[0, "participation_label"] == ""

    # The human may correct a draft; approval must save that correction and advance.
    import json
    video = tmp_path / "data/raw/s_test/cam_A.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"preview-fixture")
    pose = tmp_path / "data/poses/s_test_cam_A_poses.csv"
    pose.parent.mkdir(parents=True)
    pose.write_text("track_id,timestamp_ms\n3,0\n")
    monkeypatch.setattr(studio_view, "load_pose_dataframe", lambda *a: pd.DataFrame())
    monkeypatch.setattr(studio_view, "extract_frame_at_ms", lambda *a: None)
    monkeypatch.setattr(studio_view, "render_annotation_preview", lambda *a, **kw: (video, {"target_visible_frames": 40, "frames": 40}))
    app.run()
    next(b for b in app.button if "THẤP (LOW)" in b.label).click().run()
    next(t for t in app.text_input if t.label.startswith("Căn cứ bổ sung")).set_value("Pha đã được người duyệt đối chiếu").run()
    next(b for b in app.button if b.label.startswith("✅ Duyệt lựa chọn")).click().run()
    assert not app.exception
    result = read_annotation_csv(csv_path)
    assert result.at[0, "participation_label"] == "LOW_OBSERVED_PARTICIPATION"
    assert result.at[0, "label_source"] == "HUMAN"
    evidence = json.loads(result.at[0, "annotation_evidence"])
    assert evidence["method"] == "AI_ASSISTED_HUMAN_REVIEW"
    assert evidence["blind_review"] is False
    assert evidence["decision"] == "EDITED"
    assert evidence["ai_suggestion"]["suggested_label"] == MODERATE
    assert result.at[1, "annotator_id"] == ""
    assert app.selectbox(key="select_600_window_pos").value == 1


def test_load_suggestions_map_handles_empty_files(tmp_path: Path):
    empty_json = tmp_path / "codex_suggestions.json"
    empty_json.write_text("", encoding="utf-8")
    assert load_suggestions_map(empty_json) == {}

    empty_csv = tmp_path / "codex_suggestions.csv"
    empty_csv.write_text("  \n", encoding="utf-8")
    assert load_suggestions_map(empty_json) == {}


def test_propose_local_suggestions_from_features(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    poses_dir = tmp_path / "poses"
    raw_dir.mkdir()
    poses_dir.mkdir()
    (raw_dir / "s_test").mkdir()
    video = raw_dir / "s_test" / "cam_A.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(50):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    (poses_dir / "s_test_cam_A_poses.csv").write_text("session_id,camera_id\ns_test,cam_A\n", encoding="utf-8")

    windows_df = pd.DataFrame([
        {
            "window_id": "w_valid",
            "session_id": "s_test",
            "camera_id": "cam_A",
            "track_id": "7",
            "t_start_ms": 0.0,
            "t_end_ms": 3900.0,
            "mean_visible_ratio": 0.85,
            "mean_upper_body_visible_ratio": 0.90,
            "mean_pose_confidence": 0.80,
        },
        {
            "window_id": "w_occluded",
            "session_id": "s_test",
            "camera_id": "cam_A",
            "track_id": "8",
            "t_start_ms": 0.0,
            "t_end_ms": 3900.0,
            "mean_visible_ratio": 0.25,
            "mean_upper_body_visible_ratio": 0.25,
            "mean_pose_confidence": 0.20,
        },
    ])
    feats = np.zeros((2, 40, 94), dtype=np.float32)
    feats[0, :, 77:94] = 1.0
    feats[1, :5, 77:94] = 1.0
    model = build_model({"type": "lstm", "hidden_size": 16, "num_layers": 1, "dropout": 0.0}, input_size=94)
    out = propose_local_suggestions_from_features(
        windows_df,
        feats,
        raw_dir=raw_dir,
        poses_dir=poses_dir,
        model=model,
        class_phase="lecture",
        model_id="test-batch",
    )
    assert len(out) == 2
    assert out[0]["suggested_state"] == "VALID"
    assert out[0]["suggestion_source"] == "LOCAL_AI_PROPOSER"
    assert out[1]["suggested_state"] == "OCCLUDED"
    assert out[1]["needs_vision_review"] is True


def test_full_split_studio_mode_saves_independently(tmp_path: Path, monkeypatch):
    from scripts.dataset_qa import REQUIRED_COLUMNS

    ind_dir = tmp_path / "data/annotations/review_v03_independent_20261124"
    train_dir = tmp_path / "data/annotations/train_full_ai_assist"
    val_dir = tmp_path / "data/annotations/val_full_ai_assist"
    ind_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    base = {c: "" for c in REQUIRED_COLUMNS}
    base.update({
        "session_id": "s_test",
        "camera_id": "cam_A",
        "track_id": "3",
        "t_start_ms": "0",
        "t_end_ms": "3900",
        "class_phase": "lecture",
        "declared_class_phase": "lecture",
        "phase_source": "METADATA_UNVERIFIED",
        "phase_verified": "False",
        "annotation_guide_version": "v0.2",
        "video_path": "data/raw/s_test/cam_A.mp4",
    })
    ind_csv = ind_dir / "annotator_a.csv"
    train_csv = train_dir / "annotator_a.csv"
    val_csv = val_dir / "annotator_a.csv"
    pd.DataFrame([{**base, "window_id": "ind_w0", "split": "test"}]).to_csv(ind_csv, index=False)
    pd.DataFrame([
        {**base, "window_id": "train_w0", "split": "train"},
        {**base, "window_id": "train_w1", "split": "train", "track_id": "4"},
    ]).to_csv(train_csv, index=False)
    pd.DataFrame([{**base, "window_id": "val_w0", "split": "val"}]).to_csv(val_csv, index=False)
    save_suggestions_map(
        train_dir / "ai_bundles" / "ai_suggestions.json",
        [
            {
                "window_id": "train_w0",
                "suggested_state": "VALID",
                "suggested_label": HIGH,
                "suggested_behaviors": ["READING_WRITING"],
                "suggested_phase": "lecture",
                "suggested_phase_evidence": "Quan sát thấy hoạt động trình bày bài học trước lớp",
                "suggested_occlusion": "none",
                "rationale": "Tư thế đọc/ghi ổn định.",
                "rejection_reason": "",
                "confidence": 0.91,
                "suggestion_source": "LOCAL_AI_PROPOSER",
            }
        ],
    )

    video = tmp_path / "data/raw/s_test/cam_A.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"preview-fixture")
    pose = tmp_path / "data/poses/s_test_cam_A_poses.csv"
    pose.parent.mkdir(parents=True)
    pose.write_text("track_id,timestamp_ms\n3,0\n")

    monkeypatch.setattr(studio_view, "ROOT", tmp_path)
    monkeypatch.setattr(studio_view, "load_pose_dataframe", lambda *a: pd.DataFrame())
    monkeypatch.setattr(studio_view, "extract_frame_at_ms", lambda *a: None)
    monkeypatch.setattr(studio_view, "render_annotation_preview", lambda *a, **kw: (video, {"target_visible_frames": 40, "frames": 40}))

    root = Path(__file__).resolve().parents[1]
    app_code = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import app.studio_view as sv\n"
        f"sv.render_studio_tab('Tài', 'Vũ Đức Tài', Path({str(ind_csv)!r}))\n"
    )
    app = AppTest.from_string(app_code, default_timeout=30).run()
    assert not app.exception

    # Switch to Mode 2 (full datasets)
    mode_radio = next(r for r in app.radio if r.key == "studio_mode_radio")
    full_opt = next(opt for opt in mode_radio.options if "Tập Đầy Đủ" in opt)
    mode_radio.set_value(full_opt).run()
    assert not app.exception

    # Train full set selected with AI Assisted ON by default; CSV stays untouched until approval.
    assert read_annotation_csv(train_csv).at[0, "annotator_id"] == ""
    assert any(b.label.startswith("📝 Điền gợi ý") for b in app.button)

    # Approve and save -> must write only to train_csv, leaving ind_csv untouched
    next(b for b in app.button if b.label.startswith("✅ Duyệt lựa chọn")).click().run()
    assert not app.exception
    train_after = read_annotation_csv(train_csv)
    assert train_after.at[0, "participation_label"] == HIGH
    assert train_after.at[0, "label_source"] == "HUMAN"
    assert train_after.at[1, "annotator_id"] == ""
    assert read_annotation_csv(ind_csv).at[0, "annotator_id"] == ""
    assert app.selectbox(key="select_full_train_full_ai_assist_window_pos").value == 1


def test_train_15s_span_fast_labeling(tmp_path: Path, monkeypatch):
    import json
    from scripts.dataset_qa import REQUIRED_COLUMNS

    train_dir = tmp_path / "data/annotations/train_full_ai_assist"
    train_dir.mkdir(parents=True)
    base = {c: "" for c in REQUIRED_COLUMNS}
    base.update({
        "session_id": "s_test",
        "camera_id": "cam_A",
        "track_id": "3",
        "class_phase": "lecture",
        "declared_class_phase": "lecture",
        "phase_source": "METADATA_UNVERIFIED",
        "phase_verified": "False",
        "annotation_guide_version": "v0.2",
        "video_path": "data/raw/s_test/cam_A.mp4",
    })
    rows = [
        {**base, "window_id": "w0", "split": "train", "t_start_ms": "0", "t_end_ms": "3900"},
        {**base, "window_id": "w1", "split": "train", "t_start_ms": "6000", "t_end_ms": "9900"},
        {**base, "window_id": "w2", "split": "train", "t_start_ms": "11000", "t_end_ms": "14900"},
        {**base, "window_id": "w3_after_15s", "split": "train", "t_start_ms": "14000", "t_end_ms": "17900"},
        {**base, "window_id": "w4_other_trk", "split": "train", "track_id": "9", "t_start_ms": "2000", "t_end_ms": "5900"},
    ]
    train_csv = train_dir / "annotator_a.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)

    df_loaded = read_annotation_csv(train_csv)
    done_mask = pd.Series(False, index=df_loaded.index)
    assert studio_view.find_next_track_window_indices(df_loaded, 0, done_mask, max_span_ms=15000.0) == [1, 2]
    # Non-train split must never allow 15s span propagation.
    df_test = df_loaded.copy()
    df_test["split"] = "test"
    assert studio_view.find_next_track_window_indices(df_test, 0, done_mask, max_span_ms=15000.0) == []

    video = tmp_path / "data/raw/s_test/cam_A.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"preview-fixture")
    pose = tmp_path / "data/poses/s_test_cam_A_poses.csv"
    pose.parent.mkdir(parents=True)
    pose.write_text("track_id,timestamp_ms\n3,0\n")

    monkeypatch.setattr(studio_view, "ROOT", tmp_path)
    monkeypatch.setattr(studio_view, "load_pose_dataframe", lambda *a: pd.DataFrame())
    monkeypatch.setattr(studio_view, "extract_frame_at_ms", lambda *a: None)
    monkeypatch.setattr(studio_view, "render_annotation_preview", lambda *a, **kw: (video, {"target_visible_frames": 40, "frames": 40}))

    root = Path(__file__).resolve().parents[1]
    app_code = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import app.studio_view as sv\n"
        f"sv.render_studio_tab('Tài', 'Vũ Đức Tài', Path({str(train_csv)!r}))\n"
    )
    app = AppTest.from_string(app_code, default_timeout=30).run()
    mode_radio = next(r for r in app.radio if r.key == "studio_mode_radio")
    full_opt = next(opt for opt in mode_radio.options if "Tập Đầy Đủ" in opt)
    mode_radio.set_value(full_opt).run()
    assert not app.exception

    next(b for b in app.button if "CAO (HIGH)" in b.label).click().run()
    next(c for c in app.checkbox if "Đọc sách / Viết" in c.label).check().run()
    next(t for t in app.text_input if t.label.startswith("Căn cứ bổ sung")).set_value("Giảng viên đang giảng bài").run()
    next(c for c in app.checkbox if c.label.startswith("Tôi đã xem toàn bộ clip")).check().run()

    btn_15s = next(b for b in app.button if "Gán nhanh 3 cửa sổ (~15s)" in b.label)
    btn_15s.click().run()
    assert not app.exception

    after = read_annotation_csv(train_csv)
    assert after.loc[0:2, "participation_label"].tolist() == [HIGH, HIGH, HIGH]
    assert after.loc[0:2, "label_source"].tolist() == ["HUMAN", "HUMAN", "HUMAN"]
    assert after.at[3, "annotator_id"] == ""
    assert after.at[4, "annotator_id"] == ""
    ev1 = json.loads(after.at[1, "annotation_evidence"])
    assert ev1["method"] == "TRACK_15S_PROPAGATED_HUMAN_REVIEW"
    assert ev1["propagated_from_window_id"] == "w0"
    assert app.selectbox(key="select_full_train_full_ai_assist_window_pos").value == 3

    # Stale string in window selector session_state must never raise TypeError.
    stale_app_code = (
        "import sys\nfrom pathlib import Path\nimport streamlit as st\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "st.session_state.setdefault('select_full_train_full_ai_assist_window_pos', 'stale-label-str')\n"
        "import app.studio_view as sv\n"
        f"sv.render_studio_tab('Tài', 'Vũ Đức Tài', Path({str(train_csv)!r}))\n"
    )
    stale_app = AppTest.from_string(stale_app_code, default_timeout=30).run()
    mode_r2 = next(r for r in stale_app.radio if r.key == "studio_mode_radio")
    mode_r2.set_value(next(opt for opt in mode_r2.options if "Tập Đầy Đủ" in opt)).run()
    assert not stale_app.exception
