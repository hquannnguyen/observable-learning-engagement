"""Review legacy annotation windows with separate output sheets per reviewer."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_qa import REQUIRED_COLUMNS, validate_annotation_dataframe
from src.contracts.annotation_io import read_annotation_csv, write_annotation_csv
from src.contracts.labels import ANNOTATION_STATES, CANONICAL_TRAINING_LABELS, to_display_label
from src.pose.annotation_preview import render_annotation_preview

KEYS = ["session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms"]
REVIEWERS = ("ANN_PHONG", "ANN_TAI", "ADJUDICATOR")
PHASES = ("UNKNOWN", "lecture", "individual_work", "discussion")


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REQUIRED_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"CSV thiếu cột: {', '.join(sorted(missing))}")
    if frame[KEYS].eq("").any().any() or frame.duplicated(KEYS).any():
        raise ValueError("CSV có khóa cửa sổ trống hoặc bị trùng.")
    frame["window_id"] = [
        "legacy_" + hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        for key in frame[KEYS].itertuples(index=False, name=None)
    ]
    return frame.set_index("window_id", drop=False)


def find_recording(directory: Path, session: str, camera: str) -> Path | None:
    if any(Path(part).name != part or part in {"", ".", ".."} for part in (session, camera)):
        return None
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        candidates = [directory / session / f"{camera}{ext}", directory / f"{session}_{camera}{ext}"]
        if camera == "cam_A":
            candidates.append(directory / f"{session}{ext}")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


@st.cache_data(show_spinner=False, max_entries=16)
def sample_images(clip: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(clip)
    images = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            return images
        for index in np.linspace(0, count - 1, 4, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return images


def save_review(path: Path, source: pd.Series, updates: dict) -> None:
    record = {key: source[key] for key in (*KEYS, "window_id", "zone_id")}
    record.update(updates)
    candidate = pd.DataFrame([record])
    valid, errors = validate_annotation_dataframe(candidate)
    if not valid:
        raise ValueError(" ".join(errors))
    if path.is_file():
        existing = read_annotation_csv(path)
        candidate = pd.concat(
            [existing[existing.window_id.ne(record["window_id"])], candidate], ignore_index=True
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_annotation_csv(path, candidate)


def main() -> None:
    st.set_page_config(page_title="Rà soát nhãn video", layout="wide")
    st.title("Rà soát nhãn video theo track")
    st.caption("Xem đúng người trong toàn bộ đoạn. Mỗi người lưu vào phiếu riêng. "
               "Dùng ANN_PHONG và ANN_TAI cho hai vòng đánh giá độc lập.")
    with st.sidebar:
        annotations = Path(st.text_input("Thư mục annotation nguồn", str(ROOT / "data/annotations")))
        poses = Path(st.text_input("Thư mục pose", str(ROOT / "data/poses")))
        videos = Path(st.text_input("Thư mục video gốc", str(ROOT / "data/raw")))
        reviewer = st.selectbox("Người gán nhãn", REVIEWERS, index=None)
        files = sorted(annotations.glob("*_annotations.csv"))
        if not files:
            st.info("Chưa có CSV annotation nguồn.")
            return
        source = st.selectbox("Tệp annotation nguồn", files, format_func=lambda path: path.name)
        priority_only = st.checkbox("Chỉ xem nhóm ưu tiên")
    if reviewer is None:
        return

    output = annotations / "relabel_reviews" / source.stem / f"{reviewer}.csv"
    try:
        frame = load_source(source)
        saved = read_annotation_csv(output).set_index("window_id") if output.is_file() else pd.DataFrame()
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Không đọc được phiếu: {exc}")
        return

    weak = frame.label_source.eq("WEAK_RULE")
    priority = weak & (
        pd.to_numeric(frame.annotation_confidence, errors="coerce").le(0.80)
        | frame.atomic_behavior.str.contains(r"(?:^|;)\s*PEER_ORIENTED\s*(?:;|$)", regex=True)
    )
    eligible = frame.index[priority if priority_only else weak | frame.label_source.eq("")]
    ordered = sorted(eligible, key=lambda key: (key in saved.index, not priority.at[key]))
    st.write(f"Đã lưu {len(saved)} cửa sổ trong phiếu của {reviewer}.")
    if output.is_file():
        st.download_button("Tải phiếu CSV", output.read_bytes(), file_name=output.name, mime="text/csv")
    if not ordered:
        st.info("Bộ lọc hiện tại không có cửa sổ.")
        return

    scope = f"{source.resolve()}:{reviewer}"
    pending = st.session_state.pop(f"next:{scope}", None)
    if pending in ordered:
        st.session_state[f"window:{scope}"] = pending
    key = st.selectbox(
        "Cửa sổ", ordered, key=f"window:{scope}",
        format_func=lambda item: (
            f"{'✓' if item in saved.index else '○'} {frame.at[item, 'track_id']} "
            f"| {frame.at[item, 't_start_ms']}–{frame.at[item, 't_end_ms']} ms"
        ),
    )
    row = frame.loc[key]
    previous = saved.loc[key] if key in saved.index else {}
    clip_ready = False
    try:
        start, end = int(row.t_start_ms), int(row.t_end_ms)
        track = int(row.track_id.removeprefix("track_"))
        if start < 0 or end <= start:
            raise ValueError("Khoảng thời gian không hợp lệ.")
        video = find_recording(videos, row.session_id, row.camera_id)
        pose_path = poses / f"{row.session_id}_{row.camera_id}_poses.csv"
        if not pose_path.is_file():
            pose_path = poses / f"{row.session_id}_poses.csv"
        if video is None or not pose_path.is_file():
            raise ValueError("Thiếu video hoặc CSV pose cho session và camera này.")
        clip, audit = render_annotation_preview(
            video, pose_path, track, start, end, annotations / ".preview_cache",
            recording=(row.session_id, row.camera_id), skeleton=True,
        )
        clip_ready = audit["target_visible_frames"] > 0
        st.video(str(clip), loop=True)
        for column, image in zip(st.columns(4), sample_images(str(clip))):
            column.image(image, width="stretch")
        if st.checkbox("Xem video gốc để xác minh ngữ cảnh"):
            st.video(str(video), start_time=max(0, start // 1000 - 5))
    except (ValueError, OSError, KeyError, RuntimeError, cv2.error) as exc:
        st.warning(f"Chưa xem được đúng người và khoảng thời gian: {exc}")

    def saved_choice(field, choices):
        current = previous.get(field)
        return choices.index(current) if current in choices else None

    with st.form(f"review:{scope}:{key}"):
        phase = st.selectbox("Pha hoạt động thực tế", PHASES, index=saved_choice("class_phase", PHASES))
        evidence = st.text_input("Căn cứ xác định pha", value=previous.get("phase_evidence", ""))
        state = st.selectbox("Trạng thái quan sát", ANNOTATION_STATES, index=saved_choice("annotation_state", ANNOTATION_STATES))
        label = st.selectbox("Mức tham gia nếu VALID", CANONICAL_TRAINING_LABELS,
                             index=saved_choice("participation_label", CANONICAL_TRAINING_LABELS),
                             format_func=to_display_label)
        behavior = st.text_input("Hành vi quan sát được", value=previous.get("atomic_behavior", ""))
        confidence = st.slider("Độ tin cậy của người gán", 0.0, 1.0, float(previous.get("annotation_confidence", 0.8)), 0.05)
        occlusion = st.selectbox("Mức che khuất", ("none", "partial", "heavy"), index=saved_choice("occlusion_level", ("none", "partial", "heavy")))
        notes = st.text_area("Ghi chú", value=previous.get("notes", ""))
        confirmed = st.checkbox("Tôi đã xem toàn bộ đoạn và xác định đúng người.")
        submitted = st.form_submit_button("Lưu vào phiếu của tôi và chuyển tiếp", type="primary")

    if submitted:
        if state is None or occlusion is None:
            st.error("Chọn trạng thái quan sát và mức che khuất.")
        elif state == "VALID" and (not clip_ready or not confirmed or phase in (None, "UNKNOWN") or
                                   not evidence.strip() or not behavior.strip() or label is None):
            st.error("VALID cần xem đúng track, xác nhận, xác định pha và ghi bằng chứng, hành vi cùng nhãn.")
        elif state != "VALID" and not notes.strip():
            st.error("Ghi lý do khi cửa sổ không đủ điều kiện gán nhãn.")
        else:
            try:
                save_review(output, row, {
                    "class_phase": phase or "UNKNOWN", "phase_evidence": evidence.strip(),
                    "annotation_state": state, "participation_label": label if state == "VALID" else "",
                    "label_source": ("ADJUDICATED" if reviewer == "ADJUDICATOR" else "HUMAN") if state == "VALID" else "",
                    "label_rule_version": "", "annotation_guide_version": "v0.2", "annotation_confidence": confidence,
                    "atomic_behavior": behavior.strip(), "occlusion_level": occlusion, "notes": notes.strip(),
                    "annotator_id": reviewer,
                })
            except (ValueError, OSError) as exc:
                st.error(f"Chưa lưu được phiếu: {exc}")
            else:
                following = ordered[ordered.index(key) + 1:] + ordered[:ordered.index(key)]
                st.session_state[f"next:{scope}"] = next((item for item in following if item not in saved.index), key)
                st.rerun()


if __name__ == "__main__":
    main()
