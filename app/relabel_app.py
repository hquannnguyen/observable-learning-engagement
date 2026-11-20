"""Interactive Video & Pose-Track Re-Labeling Tool for observable-learning-engagement.

Designed to drop into `app/relabel_app.py` (or `scripts/relabel_app.py`) on branch `feature/ver2`.
Enforces the 17-column annotation schema from `scripts/generate_annotations.py` and
validation rules from `src/contracts/labels.py`.

Run:
    streamlit run relabel_videos_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# --- Canonical Contracts (mirrors src/contracts/labels.py) ---
HIGH = "HIGH_OBSERVED_PARTICIPATION"
MODERATE = "MODERATE_OBSERVED_PARTICIPATION"
LOW = "LOW_OBSERVED_PARTICIPATION"
CANONICAL_TRAINING_LABELS = (HIGH, MODERATE, LOW)
ANNOTATION_STATES = ("VALID", "OCCLUDED", "TRANSITION", "INVALID")
LABEL_SOURCES = ("HUMAN", "ADJUDICATED", "WEAK_RULE")
ATOMIC_BEHAVIORS = (
    "BOARD_OR_INSTRUCTOR_FACING",
    "HAND_RAISED",
    "READING_WRITING",
    "HEAD_DOWN",
    "PEER_ORIENTED",
    "OCCLUDED",
)
CLASS_PHASES = ("lecture", "individual_work", "discussion")
OCCLUSION_LEVELS = ("low", "high")

ANNOTATION_COLUMNS = [
    "session_id",
    "camera_id",
    "track_id",
    "t_start_ms",
    "t_end_ms",
    "class_phase",
    "atomic_behavior",
    "participation_label",
    "annotation_state",
    "label_source",
    "label_rule_version",
    "annotation_guide_version",
    "annotation_confidence",
    "occlusion_level",
    "zone_id",
    "annotator_id",
    "notes",
]

# COCO 17-keypoint skeleton connections for visual overlay
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _normalize_track_id(val: object) -> str:
    s = str(val).strip()
    if s.startswith("track_"):
        return s
    try:
        return f"track_{int(float(s)):03d}"
    except ValueError:
        return s


def validate_and_enforce_contract(row: dict) -> dict:
    """Enforce `src/contracts/labels.py` rules before saving."""
    state = row["annotation_state"]
    if state == "VALID":
        if row["participation_label"] not in CANONICAL_TRAINING_LABELS:
            raise ValueError(f"VALID row requires canonical label, got {row['participation_label']}")
        if row["label_source"] not in ("HUMAN", "ADJUDICATED"):
            row["label_source"] = "HUMAN"
        row["label_rule_version"] = None
    else:
        row["participation_label"] = None
        row["label_source"] = None
        row["label_rule_version"] = None
    return row


def find_video_file(video_dir: Path, session_id: str) -> Optional[Path]:
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        candidate = video_dir / f"{session_id}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(video_dir.glob(f"*{session_id}*"))
    return matches[0] if matches else None


def draw_pose_overlay(frame_bgr: np.ndarray, pose_row: Optional[pd.Series], track_label: str) -> np.ndarray:
    img = frame_bgr.copy()
    if pose_row is None:
        return img

    # Draw bounding box if present
    bbox_cols = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    if all(c in pose_row.index and pd.notna(pose_row[c]) for c in bbox_cols):
        x1, y1, x2, y2 = (int(pose_row[c]) for c in bbox_cols)
        cv2.rectangle(img, (x1, y1), (x2, y2), (37, 99, 235), 2)
        cv2.putText(
            img,
            track_label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (37, 99, 235),
            2,
        )

    # Draw 17 COCO keypoints if present
    pts: dict[int, tuple[int, int]] = {}
    for idx in range(17):
        xk, yk, ck = f"kpt_{idx:02d}_x", f"kpt_{idx:02d}_y", f"kpt_{idx:02d}_conf"
        if xk in pose_row.index and yk in pose_row.index:
            x, y = pose_row[xk], pose_row[yk]
            conf = pose_row[ck] if ck in pose_row.index and pd.notna(pose_row[ck]) else 1.0
            if pd.notna(x) and pd.notna(y) and float(conf) >= 0.25 and (float(x) > 0 or float(y) > 0):
                pts[idx] = (int(x), int(y))

    for a, b in COCO_EDGES:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (16, 185, 129), 2)
    for pt in pts.values():
        cv2.circle(img, pt, 4, (245, 158, 11), -1)

    return img


def sample_window_frames(
    video_path: Path,
    t_start_ms: int,
    t_end_ms: int,
    poses_df: Optional[pd.DataFrame],
    track_id: str,
    num_frames: int = 4,
) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    timestamps = np.linspace(t_start_ms, max(t_start_ms + 1, t_end_ms), num=num_frames, dtype=int)
    frames: list[tuple[int, np.ndarray]] = []

    track_poses = None
    if poses_df is not None and not poses_df.empty:
        track_poses = poses_df[poses_df["track_id_norm"] == track_id]

    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok:
            continue

        pose_row = None
        if track_poses is not None and not track_poses.empty and "timestamp_ms" in track_poses.columns:
            nearest_idx = (track_poses["timestamp_ms"] - ts).abs().idxmin()
            if abs(float(track_poses.loc[nearest_idx, "timestamp_ms"]) - ts) <= 250:
                pose_row = track_poses.loc[nearest_idx]

        annotated = draw_pose_overlay(frame, pose_row, track_id)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frames.append((int(ts), rgb))

    cap.release()
    return frames


def main() -> None:
    st.set_page_config(page_title="Video Re-Labeling Tool", layout="wide")
    st.title("Công cụ gán nhãn lại Video & Pose Track")
    st.caption("Tương thích hợp đồng dữ liệu `feature/ver2` (`src/contracts/labels.py` & `scripts/generate_annotations.py`)")

    with st.sidebar:
        st.header("Cấu hình dữ liệu")
        annotations_dir = Path(st.text_input("Thư mục annotations", "data/annotations"))
        poses_dir = Path(st.text_input("Thư mục poses", "data/poses"))
        videos_dir = Path(st.text_input("Thư mục video gốc", "data/raw_videos"))
        annotator_id = st.text_input("Mã người gán nhãn (annotator_id)", "ANN_PHONG")
        guide_version = st.text_input("Phiên bản hướng dẫn", "v0.2")

        if not annotations_dir.exists():
            st.error(f"Không tìm thấy thư mục: {annotations_dir}")
            return

        ann_files = sorted(
            p for p in annotations_dir.glob("*_annotations.csv")
            if not p.name.startswith("pilot_")
        )
        if not ann_files:
            st.warning("Chưa có file `*_annotations.csv` trong thư mục đã chọn.")
            return

        selected_file = st.selectbox("Chọn file annotation", ann_files, format_func=lambda p: p.name)

    df = pd.read_csv(selected_file)
    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Sidebar filters
    with st.sidebar:
        st.divider()
        st.header("Bộ lọc đoạn cần kiểm tra")
        source_filter = st.multiselect(
            "Lọc theo nguồn nhãn hiện tại",
            options=["WEAK_RULE", "HUMAN", "ADJUDICATED", "NONE"],
            default=["WEAK_RULE", "NONE"],
        )
        state_filter = st.multiselect(
            "Lọc theo trạng thái",
            options=list(ANNOTATION_STATES),
            default=list(ANNOTATION_STATES),
        )
        max_conf = st.slider("Độ tin cậy tối đa", 0.0, 1.0, 1.0, 0.05)

    mask = df["annotation_state"].isin(state_filter)
    src_series = df["label_source"].fillna("NONE")
    mask &= src_series.isin(source_filter)
    if "annotation_confidence" in df.columns:
        conf_vals = pd.to_numeric(df["annotation_confidence"], errors="coerce").fillna(0.0)
        mask &= conf_vals <= max_conf

    filtered_indices = df.index[mask].tolist()
    total_rows = len(df)
    human_done = int(df["label_source"].isin(["HUMAN", "ADJUDICATED"]).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số cửa sổ", total_rows)
    m2.metric("Đã gán nhãn HUMAN / ADJUDICATED", f"{human_done} ({human_done / max(1, total_rows):.1%})")
    m3.metric("Đoạn khớp bộ lọc", len(filtered_indices))

    if not filtered_indices:
        st.success("Không còn đoạn nào khớp bộ lọc hiện tại.")
        return

    if "cursor" not in st.session_state:
        st.session_state.cursor = 0
    st.session_state.cursor = min(st.session_state.cursor, len(filtered_indices) - 1)

    nav_c1, nav_c2, nav_c3 = st.columns([1, 4, 1])
    with nav_c1:
        if st.button("← Đoạn trước", use_container_width=True) and st.session_state.cursor > 0:
            st.session_state.cursor -= 1
    with nav_c3:
        if st.button("Đoạn tiếp →", use_container_width=True) and st.session_state.cursor < len(filtered_indices) - 1:
            st.session_state.cursor += 1
    with nav_c2:
        st.session_state.cursor = st.number_input(
            "Chỉ mục đoạn đang xem",
            min_value=0,
            max_value=len(filtered_indices) - 1,
            value=st.session_state.cursor,
            step=1,
        )

    row_idx = filtered_indices[st.session_state.cursor]
    row = df.loc[row_idx]
    session_id = str(row["session_id"])
    track_id = _normalize_track_id(row["track_id"])
    t_start_ms = int(row["t_start_ms"])
    t_end_ms = int(row["t_end_ms"])

    st.subheader(
        f"Phiên `{session_id}` · `{track_id}` · "
        f"[{t_start_ms / 1000:.1f}s – {t_end_ms / 1000:.1f}s] "
        f"(Nguồn cũ: `{row['label_source']}`)"
    )

    # Load pose CSV & video frames
    poses_df = None
    pose_file = poses_dir / f"{session_id}_poses.csv"
    if pose_file.exists():
        poses_df = pd.read_csv(pose_file)
        if "track_id" in poses_df.columns:
            poses_df["track_id_norm"] = poses_df["track_id"].map(_normalize_track_id)

    video_path = find_video_file(videos_dir, session_id) if videos_dir.exists() else None
    if video_path is not None:
        frames = sample_window_frames(video_path, t_start_ms, t_end_ms, poses_df, track_id, num_frames=4)
        if frames:
            cols = st.columns(len(frames))
            for col, (ts, rgb) in zip(cols, frames):
                col.image(rgb, caption=f"{ts / 1000:.2f}s", use_container_width=True)
        else:
            st.warning("Không đọc được khung hình trong khoảng thời gian này.")
    else:
        st.info(f"Không tìm thấy video gốc cho `{session_id}` trong `{videos_dir}`. Đang hiển thị siêu dữ liệu để gán nhãn.")

    # Annotation form
    with st.form(key=f"relabel_form_{row_idx}"):
        c1, c2, c3 = st.columns(3)
        with c1:
            curr_state = row["annotation_state"] if row["annotation_state"] in ANNOTATION_STATES else "VALID"
            new_state = st.selectbox("annotation_state", ANNOTATION_STATES, index=ANNOTATION_STATES.index(curr_state))

            curr_label = row["participation_label"] if row["participation_label"] in CANONICAL_TRAINING_LABELS else MODERATE
            new_label = st.selectbox(
                "participation_label (chỉ dùng khi VALID)",
                CANONICAL_TRAINING_LABELS,
                index=CANONICAL_TRAINING_LABELS.index(curr_label),
            )

        with c2:
            curr_beh = row["atomic_behavior"] if row["atomic_behavior"] in ATOMIC_BEHAVIORS else ATOMIC_BEHAVIORS[0]
            new_beh = st.selectbox("atomic_behavior", ATOMIC_BEHAVIORS, index=ATOMIC_BEHAVIORS.index(curr_beh))

            curr_phase = row["class_phase"] if row["class_phase"] in CLASS_PHASES else "lecture"
            new_phase = st.selectbox("class_phase", CLASS_PHASES, index=CLASS_PHASES.index(curr_phase))

        with c3:
            curr_occ = row["occlusion_level"] if row["occlusion_level"] in OCCLUSION_LEVELS else "low"
            new_occ = st.selectbox("occlusion_level", OCCLUSION_LEVELS, index=OCCLUSION_LEVELS.index(curr_occ))

            new_source = st.selectbox("label_source", ("HUMAN", "ADJUDICATED"), index=0)
            new_conf = st.slider(
                "annotation_confidence",
                0.0,
                1.0,
                float(row["annotation_confidence"]) if pd.notna(row["annotation_confidence"]) else 1.0,
                0.05,
            )

        new_notes = st.text_input("notes", value="" if pd.isna(row["notes"]) else str(row["notes"]))
        save_and_next = st.form_submit_button("Lưu nhãn & Chuyển đoạn tiếp theo", type="primary")

        if save_and_next:
            updated = {
                "annotation_state": new_state,
                "participation_label": new_label,
                "atomic_behavior": new_beh,
                "class_phase": new_phase,
                "occlusion_level": new_occ,
                "label_source": new_source,
                "label_rule_version": None,
                "annotation_guide_version": guide_version,
                "annotation_confidence": round(float(new_conf), 2),
                "annotator_id": annotator_id,
                "notes": new_notes,
            }
            updated = validate_and_enforce_contract(updated)
            for k, v in updated.items():
                df.at[row_idx, k] = v

            df.to_csv(selected_file, index=False)
            st.success(f"Đã lưu dòng {row_idx} vào {selected_file.name}.")
            if st.session_state.cursor < len(filtered_indices) - 1:
                st.session_state.cursor += 1
            st.rerun()

    # Export merged dataset
    st.divider()
    if st.button("Gộp toàn bộ *_annotations.csv thành annotations_all_v0.2.csv"):
        merged = pd.concat([pd.read_csv(p) for p in ann_files], ignore_index=True)
        out_path = annotations_dir / "annotations_all_v0.2.csv"
        merged.to_csv(out_path, index=False)
        st.success(f"Đã xuất {len(merged)} dòng ra `{out_path}`.")


if __name__ == "__main__":
    main()
