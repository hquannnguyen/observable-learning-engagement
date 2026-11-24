"""Tien Studio V2 interface for Streamlit.

Rebuilds the interactive, fast multi-student annotation tool (matching AnnotationStudio_V2_Standalone.exe)
for the web, featuring:
1. Step 1: Select video & segment 4s windows with start/end preview.
2. Step 2: Full classroom view with color-coded bounding boxes + 5-keyframe cropped temporal progression strip.
3. Fast 1-click labeling (CAO, VỪA, THẤP, BỊ CHE), behavior checkboxes, and auto-advance.
4. Step 3: Validation, summary statistics, export, and master sync.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from scripts.dataset_qa import validate_annotation_dataframe
from src.contracts.annotation_io import read_annotation_csv, write_annotation_csv
from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    HIGH,
    LOW,
    MODERATE,
)
from src.pose.studio_renderer import (
    extract_frame_at_ms,
    render_classroom_scene,
    render_progression_strip,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Available sessions registry
SESSIONS_CONFIG = {
    "session_005 (TEST · 0s-60s)": {
        "session_id": "session_005",
        "video": "data/raw/session_005/cam_A.mp4",
        "poses": "data/poses/session_005_cam_A_poses.csv",
        "split": "TEST",
        "phase": "individual_work",
        "start_s": 0.0,
        "end_s": 60.0,
    },
    "session_006 (TEST · 0s-60s)": {
        "session_id": "session_006",
        "video": "data/raw/session_006/cam_A.mp4",
        "poses": "data/poses/session_006_cam_A_poses.csv",
        "split": "TEST",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 60.0,
    },
    "session_003 (VAL · 0s-60s)": {
        "session_id": "session_003",
        "video": "data/raw/session_003/cam_A.mp4",
        "poses": "data/poses/session_003_cam_A_poses.csv",
        "split": "VAL",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 60.0,
    },
    "session_004 (VAL · 0s-60s)": {
        "session_id": "session_004",
        "video": "data/raw/session_004/cam_A.mp4",
        "poses": "data/poses/session_004_cam_A_poses.csv",
        "split": "VAL",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 60.0,
    },
    "session_001 (TRAIN · 0s-60s)": {
        "session_id": "session_001",
        "video": "data/raw/session_001/cam_A.mp4",
        "poses": "data/poses/session_001_cam_A_poses.csv",
        "split": "TRAIN",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 60.0,
    },
    "session_002 (TRAIN · 0s-60s)": {
        "session_id": "session_002",
        "video": "data/raw/session_002/cam_A.mp4",
        "poses": "data/poses/session_002_cam_A_poses.csv",
        "split": "TRAIN",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 60.0,
    },
    "class_1 (TRAIN · 0s-15s)": {
        "session_id": "class_1",
        "video": "data/raw/class_1.mp4",
        "poses": "data/poses/class_1_cam_A_poses.csv",
        "split": "TRAIN",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 15.0,
    },
    "class_2 (TRAIN · 0s-15s)": {
        "session_id": "class_2",
        "video": "data/raw/class_2.mp4",
        "poses": "data/poses/class_2_cam_A_poses.csv",
        "split": "TRAIN",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 15.0,
    },
    "class_3 (TRAIN · 0s-15s)": {
        "session_id": "class_3",
        "video": "data/raw/class_3.mp4",
        "poses": "data/poses/class_3_cam_A_poses.csv",
        "split": "TRAIN",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 15.0,
    },
    "class_4 (TRAIN · 0s-15s)": {
        "session_id": "class_4",
        "video": "data/raw/class_4.mp4",
        "poses": "data/poses/class_4_cam_A_poses.csv",
        "split": "TRAIN",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 15.0,
    },
    "class_5 (TRAIN · 0s-15s)": {
        "session_id": "class_5",
        "video": "data/raw/class_5.mp4",
        "poses": "data/poses/class_5_cam_A_poses.csv",
        "split": "TRAIN",
        "phase": "lecture",
        "start_s": 0.0,
        "end_s": 15.0,
    },
}

BEHAVIOR_CHOICES = [
    ("Nhìn bảng / Giảng viên", "BOARD_OR_INSTRUCTOR_FACING"),
    ("Đọc sách / Viết / Gõ phím", "READING_WRITING"),
    ("Giơ tay phát biểu", "HAND_RAISED"),
    ("Quay sang thảo luận nhóm", "PEER_ORIENTED"),
    ("Cúi gục đầu kéo dài", "HEAD_DOWN_EXTENDED"),
    ("Xoay người hướng khác", "TURNING_AWAY"),
    ("Bị che khuất khớp xương", "OCCLUDED_KEYPOINTS"),
    ("Không rõ hành vi", "UNCLEAR"),
]

BEHAVIOR_NAME_TO_CODE = dict(BEHAVIOR_CHOICES)
BEHAVIOR_CODE_TO_NAME = {code: name for name, code in BEHAVIOR_CHOICES}


@st.cache_data(max_entries=10)
def load_pose_dataframe(pose_csv_path: str) -> pd.DataFrame:
    """Load and cache pose detections DataFrame."""
    p = Path(pose_csv_path)
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p)


def get_default_label_path(annotator_key: str, session_id: str) -> Path:
    """Return canonical review CSV path for this annotator and session."""
    p_name = "phong" if "phong" in annotator_key.lower() else "tai"
    studio_path = ROOT / "data/annotations/review_v03_studio" / p_name / f"{session_id}_cam_A_labels.csv"
    if studio_path.is_file():
        return studio_path
    ann_path = ROOT / "data/annotations" / f"{session_id}_cam_A_annotations.csv"
    if ann_path.is_file():
        return ann_path
    return studio_path


def render_studio_tab(person_name: str, annotator_full_name: str, master_csv_path: Path):
    """Render the full Tien Studio V2 workflow."""
    st.markdown(
        """
        <style>
        .stButton>button {
            border-radius: 6px;
            font-weight: bold;
        }
        .btn-high {
            background-color: #28a745 !important;
            color: white !important;
        }
        .btn-mod {
            background-color: #ffc107 !important;
            color: black !important;
        }
        .btn-low {
            background-color: #dc3545 !important;
            color: white !important;
        }
        .btn-occ {
            background-color: #6c757d !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Studio Steps Navigation
    step = st.radio(
        "Chuyển bước công việc:",
        [
            "BƯỚC 1: 📂 Chọn Video & Cắt Đoạn",
            "BƯỚC 2: 👥 Gán & Duyệt Nhiều Sinh Viên",
            "BƯỚC 3: 📊 Kiểm Định & Xuất File CSV",
        ],
        horizontal=True,
        key="studio_step_radio",
    )

    if step.startswith("BƯỚC 1"):
        _render_step_1(person_name, annotator_full_name)
    elif step.startswith("BƯỚC 2"):
        _render_step_2(person_name, annotator_full_name, master_csv_path)
    else:
        _render_step_3(person_name, annotator_full_name, master_csv_path)


# ----------------------------------------------------------------------
# BƯỚC 1: CHỌN VIDEO & CẮT ĐOẠN CẦN PHÂN TÍCH
# ----------------------------------------------------------------------
def _render_step_1(person_name: str, annotator_full_name: str):
    st.subheader("BƯỚC 1: CHỌN VIDEO & CẮT ĐOẠN CẦN PHÂN TÍCH")
    st.caption("Chọn nhanh phiên quay lớp học (Session / Class) đã được cấu hình sẵn, hoặc tùy chỉnh đường dẫn.")

    # Dropdown to pick preset session
    session_names = list(SESSIONS_CONFIG.keys())
    selected_preset = st.selectbox("Chọn Phiên Học Cần Gán (Khuyên dùng):", session_names, index=0)
    preset_cfg = SESSIONS_CONFIG[selected_preset]

    col_v, col_p = st.columns(2)
    video_path_str = col_v.text_input(
        "🎥 Video đầu vào (.mp4):",
        value=str(ROOT / preset_cfg["video"]),
        help="Đường dẫn file video mp4 trên máy",
    )
    pose_path_str = col_p.text_input(
        "🦴 File Pose (.csv):",
        value=str(ROOT / preset_cfg["poses"]),
        help="Đường dẫn file pose csv tương ứng",
    )

    # Label target path
    default_label_file = get_default_label_path(person_name, preset_cfg["session_id"])
    label_path_str = st.text_input(
        "📝 File nhãn lưu kết quả (.csv):",
        value=str(default_label_file),
        help="Nơi lưu nhãn gán của bạn",
    )

    # Time trimming & window controls
    t_cols = st.columns(4)
    start_s = t_cols[0].number_input("⏱️ Từ giây:", value=preset_cfg["start_s"], step=1.0, min_value=0.0)
    end_s = t_cols[1].number_input("⏱️ Đến giây:", value=preset_cfg["end_s"], step=1.0, min_value=1.0)
    win_s = t_cols[2].number_input("🪟 Cửa sổ (s):", value=4.0, step=1.0, min_value=2.0)
    stride_s = t_cols[3].number_input("📏 Bước trượt (s):", value=2.0, step=1.0, min_value=1.0)

    # Action buttons for Step 1
    act_col1, act_col2 = st.columns(2)
    show_preview = act_col1.button("🖼️ Xem Trước Khung Hình Đầu & Cuối", use_container_width=True)
    start_segment = act_col2.button("🚀 Cắt Cửa Sổ 4s & Vào Gán Nhãn -> Sang BƯỚC 2", type="primary", use_container_width=True)

    # Save settings to session_state
    st.session_state["studio_session_id"] = preset_cfg["session_id"]
    st.session_state["studio_video_path"] = video_path_str
    st.session_state["studio_pose_path"] = pose_path_str
    st.session_state["studio_label_path"] = label_path_str
    st.session_state["studio_phase"] = preset_cfg["phase"]
    st.session_state["studio_start_s"] = start_s
    st.session_state["studio_end_s"] = end_s
    st.session_state["studio_win_s"] = win_s
    st.session_state["studio_stride_s"] = stride_s

    # Start & End Preview
    if show_preview or "studio_preview_shown" in st.session_state:
        st.session_state["studio_preview_shown"] = True
        v_path = Path(video_path_str)
        if not v_path.is_file():
            st.error(f"Không tìm thấy file video: {v_path}")
        else:
            f_start = extract_frame_at_ms(v_path, start_s * 1000.0)
            f_end = extract_frame_at_ms(v_path, max(0.0, (end_s - 0.5) * 1000.0))

            p_col1, p_col2 = st.columns(2)
            if f_start is not None:
                # Add text overlay
                f_s_drawn = f_start.copy()
                cv2.putText(f_s_drawn, f"DIEM BAT DAU: {start_s:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 220, 50), 3)
                p_col1.image(f_s_drawn, caption=f"Khung hình bắt đầu: {start_s:.1f}s", use_column_width=True)
            else:
                p_col1.warning("Chưa đọc được khung hình bắt đầu.")

            if f_end is not None:
                f_e_drawn = f_end.copy()
                cv2.putText(f_e_drawn, f"DIEM KET THUC: {end_s:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 200, 30), 3)
                p_col2.image(f_e_drawn, caption=f"Khung hình kết thúc: {end_s:.1f}s", use_column_width=True)
            else:
                p_col2.warning("Chưa đọc được khung hình kết thúc.")

    if start_segment:
        # Load or create label dataframe
        l_path = Path(label_path_str)
        if l_path.is_file():
            df_lbl = read_annotation_csv(l_path)
            st.success(f"Đã nạp file nhãn với {len(df_lbl)} cửa sổ sẵn sàng!")
        else:
            # Create from poses
            p_path = Path(pose_path_str)
            if not p_path.is_file():
                st.error(f"Không tìm thấy file pose để tạo cửa sổ: {p_path}")
                return
            df_pose = load_pose_dataframe(str(p_path))
            df_lbl = _generate_windows_from_pose(
                df_pose, preset_cfg["session_id"], "cam_A", start_s, end_s, win_s, stride_s, preset_cfg["phase"]
            )
            l_path.parent.mkdir(parents=True, exist_ok=True)
            df_lbl.to_csv(l_path, index=False, encoding="utf-8")
            st.success(f"Đã tự động cắt & tạo {len(df_lbl)} cửa sổ 4s thành công!")

        st.session_state["studio_step_radio"] = "BƯỚC 2: 👥 Gán & Duyệt Nhiều Sinh Viên"
        st.rerun()


def _generate_windows_from_pose(
    df_pose: pd.DataFrame, session_id: str, camera_id: str,
    start_s: float, end_s: float, win_s: float, stride_s: float, class_phase: str
) -> pd.DataFrame:
    """Generate 4s sliding windows from pose dataframe."""
    rows = []
    t = start_s
    w_idx = 0
    while t + win_s <= end_s + 0.05:
        t0_ms = int(round(t * 1000.0))
        t1_ms = int(round((t + win_s) * 1000.0))
        sub = df_pose[(df_pose["timestamp_ms"] >= t0_ms) & (df_pose["timestamp_ms"] < t1_ms)]
        if not sub.empty:
            for track_id, grp in sub.groupby("track_id"):
                vis = float(grp["visible_ratio"].mean()) if "visible_ratio" in grp else 0.8
                zone = str(grp["zone_id"].iloc[0]) if "zone_id" in grp and pd.notna(grp["zone_id"].iloc[0]) else "M-C"
                wid = f"{session_id}_{camera_id}_trk{int(track_id):03d}_w{w_idx:04d}"
                rows.append({
                    "window_id": wid,
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "track_id": str(track_id),
                    "t_start_ms": t0_ms,
                    "t_end_ms": t1_ms,
                    "class_phase": class_phase,
                    "atomic_behavior": "",
                    "participation_label": "",
                    "annotation_state": "VALID" if vis >= 0.3 else "OCCLUDED",
                    "label_source": "",
                    "label_rule_version": "",
                    "annotation_guide_version": "v0.3-studio",
                    "annotation_confidence": 0.8,
                    "occlusion_level": "none" if vis >= 0.7 else ("partial" if vis >= 0.3 else "heavy"),
                    "zone_id": zone,
                    "annotator_id": "",
                    "notes": f"auto_vis_ratio={vis:.2f}",
                })
        t += stride_s
        w_idx += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# BƯỚC 2: GÁN & DUYỆT NHIỀU SINH VIÊN (CHI TIẾT & CỰC NHANH)
# ----------------------------------------------------------------------
def _render_step_2(person_name: str, annotator_full_name: str, master_csv_path: Path):
    video_path_str = st.session_state.get("studio_video_path")
    pose_path_str = st.session_state.get("studio_pose_path")
    label_path_str = st.session_state.get("studio_label_path")

    if not video_path_str or not label_path_str:
        st.info("Hãy chọn Video và nạp dữ liệu ở BƯỚC 1 trước.")
        return

    v_path = Path(video_path_str)
    p_path = Path(pose_path_str) if pose_path_str else None
    l_path = Path(label_path_str)

    if not l_path.is_file():
        st.error(f"Không tìm thấy file nhãn: {l_path}. Hãy quay lại Bước 1 để tạo hoặc chọn file.")
        return

    df_lbl = read_annotation_csv(l_path)
    if df_lbl.empty:
        st.warning("File nhãn rỗng.")
        return

    df_pose = load_pose_dataframe(str(p_path)) if p_path and p_path.is_file() else pd.DataFrame()

    # Pre-calculate time slices (Mốc 4s)
    # A slice is defined by unique (t_start_ms, t_end_ms)
    time_slices = df_lbl[["t_start_ms", "t_end_ms"]].drop_duplicates().sort_values("t_start_ms").reset_index(drop=True)
    total_slices = len(time_slices)

    # State for slice index
    slice_idx_key = "studio_slice_idx"
    if slice_idx_key not in st.session_state or st.session_state[slice_idx_key] >= total_slices:
        st.session_state[slice_idx_key] = 0

    cur_slice_idx = st.session_state[slice_idx_key]
    cur_t_start = int(time_slices.loc[cur_slice_idx, "t_start_ms"])
    cur_t_end = int(time_slices.loc[cur_slice_idx, "t_end_ms"])

    # Windows in current time slice
    slice_mask = (df_lbl["t_start_ms"].astype(int) == cur_t_start) & (df_lbl["t_end_ms"].astype(int) == cur_t_end)
    slice_df = df_lbl[slice_mask].copy()
    slice_indices = list(slice_df.index)
    total_students_slice = len(slice_indices)

    # State for student index within slice
    student_idx_key = "studio_student_idx_in_slice"
    if student_idx_key not in st.session_state or st.session_state[student_idx_key] >= total_students_slice:
        st.session_state[student_idx_key] = 0

    cur_student_pos = st.session_state[student_idx_key]
    if total_students_slice > 0:
        cur_row_idx = slice_indices[cur_student_pos]
        cur_row = df_lbl.loc[cur_row_idx]
        cur_track_id = str(cur_row["track_id"])
    else:
        cur_row_idx = None
        cur_row = None
        cur_track_id = None

    # ---------------- Top Navigation Bar ----------------
    top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([1, 4, 1, 4, 2])

    # Slice prev/next
    if top_c1.button("⏮️ Trước", key="btn_slice_prev", disabled=cur_slice_idx == 0):
        st.session_state[slice_idx_key] = max(0, cur_slice_idx - 1)
        st.session_state[student_idx_key] = 0
        st.rerun()

    slice_labels = []
    for s_i, s_row in time_slices.iterrows():
        t0_s = int(s_row["t_start_ms"]) / 1000.0
        t1_s = int(s_row["t_end_ms"]) / 1000.0
        sub_s = df_lbl[(df_lbl["t_start_ms"].astype(int) == s_row["t_start_ms"]) & (df_lbl["t_end_ms"].astype(int) == s_row["t_end_ms"])]
        done_s = int(((sub_s["annotation_state"] == "VALID") & sub_s["participation_label"].ne("")) | (sub_s["annotation_state"].isin(["OCCLUDED", "INVALID"]))).sum()
        status_tag = "[✓ Xong]" if done_s == len(sub_s) else f"[{done_s}/{len(sub_s)}]"
        slice_labels.append(f"Mốc #{s_i + 1}/{total_slices} ({t0_s:g}s-{t1_s:g}s) • {len(sub_s)} SV {status_tag}")

    chosen_slice_label = top_c2.selectbox(
        "Mốc 4s:",
        slice_labels,
        index=cur_slice_idx,
        label_visibility="collapsed",
        key="select_slice_box",
    )
    new_slice_idx = slice_labels.index(chosen_slice_label)
    if new_slice_idx != cur_slice_idx:
        st.session_state[slice_idx_key] = new_slice_idx
        st.session_state[student_idx_key] = 0
        st.rerun()

    if top_c3.button("Tiếp ⏭️", key="btn_slice_next", disabled=cur_slice_idx == total_slices - 1):
        st.session_state[slice_idx_key] = min(total_slices - 1, cur_slice_idx + 1)
        st.session_state[student_idx_key] = 0
        st.rerun()

    # Student dropdown in slice
    student_labels = []
    for pos_i, r_idx in enumerate(slice_indices):
        r = df_lbl.loc[r_idx]
        t_id = str(r["track_id"])
        p_lbl = str(r["participation_label"] or r["annotation_state"])
        zone = str(r.get("zone_id", "M-C"))
        is_d = "✓" if (r["annotation_state"] == "VALID" and r["participation_label"]) or r["annotation_state"] in ("OCCLUDED", "INVALID") else "○"
        student_labels.append(f"{is_d} SV #{pos_i + 1}: Track #{t_id} ({zone}) -> {p_lbl or 'Chưa duyệt'}")

    if student_labels:
        chosen_student_label = top_c4.selectbox(
            "SV:",
            student_labels,
            index=cur_student_pos,
            label_visibility="collapsed",
            key="select_student_box",
        )
        new_student_pos = student_labels.index(chosen_student_label)
        if new_student_pos != cur_student_pos:
            st.session_state[student_idx_key] = new_student_pos
            st.rerun()

    # View options
    show_kpts = top_c5.checkbox("Khung xương", value=True, key="opt_show_kpts")
    bo_che_mat = top_c5.checkbox(
        "Bỏ che mặt (Xem rõ mặt)",
        value=True,
        key="opt_no_blur_face",
        help="Mặc định bật: KHÔNG che mặt để quan sát rõ mắt, biểu cảm, hướng đầu sinh viên. Bỏ chọn để làm mờ mặt.",
    )
    blur_face = not bo_che_mat
    auto_jump = top_c5.checkbox("Tự nhảy SV", value=True, key="opt_auto_jump")

    # ---------------- 2-Column Main Workspace ----------------
    left_col, right_col = st.columns([6, 5])

    # ---------------- LEFT COLUMN: CLASSROOM VIEW & TIMELINE ----------------
    with left_col:
        st.markdown(
            f"##### 🏫 Khung hình {cur_t_start/1000:g}s ➔ {cur_t_end/1000:g}s ({total_students_slice} sinh viên)"
        )
        st.caption(
            "Chú thích khung màu: 🟢 Xanh lá = CAO (1) • 🟡 Vàng = VỪA (2) • 🔴 Đỏ = THẤP (3) • ⚪ Xám = BỊ CHE (4) • 🔲 Viền trắng dày = Đang chọn | "
            + ("🟢 **BỎ CHE MẶT** (Nhìn rõ khuôn mặt)" if bo_che_mat else "🛡️ **Đang làm mờ mặt**")
        )

        mid_t_ms = (cur_t_start + cur_t_end) / 2.0
        frame_rgb = extract_frame_at_ms(v_path, mid_t_ms)

        if frame_rgb is not None:
            # Detections near middle timestamp
            sub_detections = pd.DataFrame()
            if not df_pose.empty:
                diffs = (df_pose["timestamp_ms"] - mid_t_ms).abs()
                near_mask = diffs <= 350.0
                sub_detections = df_pose[near_mask]

            # Build label map for tracks in current slice
            track_labels_map = {}
            for _, r in slice_df.iterrows():
                track_labels_map[str(r["track_id"])] = (str(r["participation_label"]), str(r["annotation_state"]))

            annotated_scene = render_classroom_scene(
                frame_rgb,
                sub_detections,
                selected_track_id=cur_track_id,
                labels_by_track=track_labels_map,
                show_skeleton=show_kpts,
                blur_face=blur_face,
            )
            st.image(annotated_scene, use_column_width=True)
        else:
            st.warning("Chưa mở được video hoặc frame tại mốc này.")

        # 5-Keyframe Progression Strip
        st.markdown(
            f"##### 🎞️ 5 khung hình diễn biến 4 giây của SV đang chọn: Track #{cur_track_id} ({cur_t_start/1000:g}s ➔ {cur_t_end/1000:g}s)"
        )

        if cur_track_id is not None:
            strip_frames = render_progression_strip(
                v_path,
                df_pose,
                cur_track_id,
                float(cur_t_start),
                float(cur_t_end),
                show_skeleton=show_kpts,
                blur_face=blur_face,
            )
            strip_cols = st.columns(5)
            for idx_kpt, (lbl_kpt, img_kpt) in enumerate(strip_frames):
                strip_cols[idx_kpt].image(img_kpt, caption=lbl_kpt, use_column_width=True)

        # Overview statistics line
        c_high = int((slice_df["participation_label"] == HIGH).sum())
        c_mod = int((slice_df["participation_label"] == MODERATE).sum())
        c_low = int((slice_df["participation_label"] == LOW).sum())
        c_occ = int((slice_df["annotation_state"] == "OCCLUDED").sum())
        st.info(
            f"🏛️ **Tổng quan cả lớp ở mốc này ({total_students_slice} SV):** 🟢 CAO: {c_high} • 🟡 VỪA: {c_mod} • 🔴 THẤP: {c_low} • ⚪ BỊ CHE: {c_occ}\n\n"
            f"👤 **Track #{cur_track_id} đang chọn:** {cur_row.get('notes', '')} • Vùng: {cur_row.get('zone_id', 'M-C')}"
        )

    # ---------------- RIGHT COLUMN: FAST 1-CLICK LABELING ----------------
    with right_col:
        # Progress metrics
        complete_mask = (
            (df_lbl["annotation_state"].eq("VALID") & df_lbl["participation_label"].ne("")) |
            df_lbl["annotation_state"].isin(["OCCLUDED", "INVALID"])
        )
        total_all = len(df_lbl)
        done_all = int(complete_mask.sum())
        slice_done = int((
            (slice_df["annotation_state"].eq("VALID") & slice_df["participation_label"].ne("")) |
            slice_df["annotation_state"].isin(["OCCLUDED", "INVALID"])
        ).sum())

        st.markdown(
            f"📊 **Toàn video:** {done_all}/{total_all} ({done_all/max(1, total_all)*100:.1f}%) | **Mốc này:** {slice_done}/{total_students_slice} SV đã duyệt"
        )
        st.progress(done_all / max(1, total_all))

        # Approve entire classroom at this time slice
        if st.button("✓ DUYỆT CẢ LỚP Ở MỐC NÀY & SANG MỐC GIÂY TIẾP (Space)", type="primary", use_container_width=True):
            _batch_approve_slice(df_lbl, slice_indices, annotator_full_name, l_path, master_csv_path)
            st.session_state[slice_idx_key] = min(total_slices - 1, cur_slice_idx + 1)
            st.session_state[student_idx_key] = 0
            st.success("Đã duyệt tất cả sinh viên ở mốc này!")
            st.rerun()

        # Batch assign buttons
        st.caption("Gán đồng loạt cả lớp ở mốc này:")
        b_c1, b_c2, b_c3 = st.columns(3)
        if b_c1.button("🔴 Tất cả THẤP", use_container_width=True):
            _batch_set_label(df_lbl, slice_indices, LOW, "VALID", annotator_full_name, l_path, master_csv_path)
            st.rerun()
        if b_c2.button("🟡 Tất cả VỪA", use_container_width=True):
            _batch_set_label(df_lbl, slice_indices, MODERATE, "VALID", annotator_full_name, l_path, master_csv_path)
            st.rerun()
        if b_c3.button("🟢 Tất cả CAO", use_container_width=True):
            _batch_set_label(df_lbl, slice_indices, HIGH, "VALID", annotator_full_name, l_path, master_csv_path)
            st.rerun()

        st.divider()

        # Current Student Labeling Controls
        if cur_row is not None:
            cur_p_label = str(cur_row["participation_label"])
            cur_state = str(cur_row["annotation_state"])
            display_stat = cur_p_label if cur_state == "VALID" and cur_p_label else cur_state
            st.markdown(
                f"##### 👤 SV {cur_student_pos + 1}/{total_students_slice}: Track #{cur_track_id} ➔ **{display_stat or 'Chưa duyệt'}**"
            )

            # 4 BIG LABEL BUTTONS
            l_col1, l_col2 = st.columns(2)
            l_col3, l_col4 = st.columns(2)

            btn_cao = l_col1.button("🟢 [Phím 1] CAO (HIGH)", use_container_width=True)
            btn_vua = l_col2.button("🟡 [Phím 2] VỪA (MODERATE)", use_container_width=True)
            btn_thap = l_col3.button("🔴 [Phím 3] THẤP (LOW)", use_container_width=True)
            btn_che = l_col4.button("⚪ [Phím 4] BỊ CHE (OCCLUDED)", use_container_width=True)

            selected_btn_label = None
            if btn_cao:
                selected_btn_label = (HIGH, "VALID")
            elif btn_vua:
                selected_btn_label = (MODERATE, "VALID")
            elif btn_thap:
                selected_btn_label = (LOW, "VALID")
            elif btn_che:
                selected_btn_label = ("", "OCCLUDED")

            # Checkboxes for Behaviors (Khong can go chu)
            st.caption("Hành vi của sinh viên đang chọn (Tích chọn 1 hoặc nhiều):")
            current_behavior_str = str(cur_row.get("atomic_behavior", ""))
            existing_behaviors = [b.strip() for b in current_behavior_str.split(";") if b.strip()]

            bh_col1, bh_col2 = st.columns(2)
            selected_behaviors = []

            for i, (b_name, b_code) in enumerate(BEHAVIOR_CHOICES):
                target_col = bh_col1 if i % 2 == 0 else bh_col2
                checked = target_col.checkbox(
                    b_name,
                    value=(b_code in existing_behaviors or b_name in existing_behaviors),
                    key=f"chk_bh_{cur_row['window_id']}_{b_code}",
                )
                if checked:
                    selected_behaviors.append(b_code)

            # Context metadata dropdowns
            st.caption("Ngữ cảnh & Vị trí:")
            m_col1, m_col2 = st.columns(2)
            state_options = ["VALID", "OCCLUDED", "TRANSITION", "INVALID"]
            cur_state_val = cur_state if cur_state in state_options else "VALID"
            new_state = m_col1.selectbox("Trạng thái:", state_options, index=state_options.index(cur_state_val), key="sb_state")

            phases = ["lecture", "individual_work", "discussion", "UNKNOWN"]
            cur_phase_val = cur_row.get("class_phase", "lecture")
            new_phase = m_col2.selectbox("Pha học:", phases, index=phases.index(cur_phase_val) if cur_phase_val in phases else 0, key="sb_phase")

            m_col3, m_col4 = st.columns(2)
            zones = ["F-L", "F-C", "F-R", "M-L", "M-C", "M-R", "B-L", "B-C", "B-R", "FRONT", "MIDDLE", "BACK"]
            cur_zone_val = cur_row.get("zone_id", "M-C")
            new_zone = m_col3.selectbox("Vùng ghế:", zones, index=zones.index(cur_zone_val) if cur_zone_val in zones else 4, key="sb_zone")

            occlusions = ["none", "partial", "heavy"]
            cur_occ_val = cur_row.get("occlusion_level", "none")
            new_occ = m_col4.selectbox("Che khuất:", occlusions, index=occlusions.index(cur_occ_val) if cur_occ_val in occlusions else 0, key="sb_occ")

            new_notes = st.text_input("Ghi chú:", value=str(cur_row.get("notes", "")), key="txt_notes")

            # Handle 1-click button press or Save button
            save_clicked = st.button("💾 Lưu SV Này & Sang SV Kế (Enter)", type="secondary", use_container_width=True)

            if selected_btn_label is not None or save_clicked:
                target_label, target_state = selected_btn_label if selected_btn_label else (cur_p_label, new_state)
                if target_state == "OCCLUDED":
                    target_label = ""

                _save_student_label(
                    df_lbl,
                    cur_row_idx,
                    target_label=target_label,
                    target_state=target_state,
                    behaviors="; ".join(selected_behaviors),
                    phase=new_phase,
                    zone=new_zone,
                    occlusion=new_occ,
                    notes=new_notes,
                    annotator_id=annotator_full_name,
                    label_file_path=l_path,
                    master_csv_path=master_csv_path,
                )

                # Auto jump to next student if enabled
                if auto_jump or save_clicked:
                    if cur_student_pos + 1 < total_students_slice:
                        st.session_state[student_idx_key] = cur_student_pos + 1
                    else:
                        # Advance to next slice if available
                        if cur_slice_idx + 1 < total_slices:
                            st.session_state[slice_idx_key] = cur_slice_idx + 1
                            st.session_state[student_idx_key] = 0
                st.rerun()

            # Fast forward across whole 60s for this track
            if st.button("⚡ Duyệt SV Này Suốt Cả Video", use_container_width=True):
                _apply_track_to_all_windows(
                    df_lbl,
                    cur_track_id,
                    cur_p_label,
                    cur_state,
                    annotator_full_name,
                    l_path,
                    master_csv_path,
                )
                st.success(f"Đã áp dụng nhãn cho Track #{cur_track_id} trên toàn bộ video!")
                st.rerun()


def _save_student_label(
    df: pd.DataFrame,
    row_idx: int,
    target_label: str,
    target_state: str,
    behaviors: str,
    phase: str,
    zone: str,
    occlusion: str,
    notes: str,
    annotator_id: str,
    label_file_path: Path,
    master_csv_path: Path,
):
    """Save an updated row to local session CSV and sync to master review CSV."""
    df.at[row_idx, "annotation_state"] = target_state
    df.at[row_idx, "participation_label"] = target_label if target_state == "VALID" else ""
    df.at[row_idx, "label_source"] = "HUMAN" if target_state == "VALID" else ""
    df.at[row_idx, "label_rule_version"] = ""
    df.at[row_idx, "annotator_id"] = annotator_id
    df.at[row_idx, "atomic_behavior"] = behaviors
    df.at[row_idx, "class_phase"] = phase
    df.at[row_idx, "zone_id"] = zone
    df.at[row_idx, "occlusion_level"] = occlusion
    df.at[row_idx, "notes"] = notes
    df.at[row_idx, "annotation_guide_version"] = "v0.3-studio"
    df.at[row_idx, "annotation_confidence"] = 0.8
    df.at[row_idx, "phase_verified"] = phase != "UNKNOWN"
    df.at[row_idx, "phase_source"] = "HUMAN_WINDOW_REVIEW" if phase != "UNKNOWN" else "UNKNOWN"

    # Write to local session CSV
    label_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(label_file_path, index=False, encoding="utf-8")

    # Sync to master review CSV if window_id matches
    if master_csv_path.is_file():
        win_id = df.at[row_idx, "window_id"]
        _sync_row_to_master(master_csv_path, win_id, df.loc[row_idx].to_dict())


def _batch_approve_slice(df: pd.DataFrame, slice_indices: List[int], annotator_id: str, label_file_path: Path, master_csv_path: Path):
    """Approve all students in slice (mark as VALID if label exists, or OCCLUDED)."""
    for idx in slice_indices:
        st_val = df.at[idx, "annotation_state"]
        lbl_val = df.at[idx, "participation_label"]
        if not lbl_val and st_val == "VALID":
            # Default to MODERATE if unspecified
            lbl_val = MODERATE
            df.at[idx, "participation_label"] = lbl_val
        df.at[idx, "annotator_id"] = annotator_id
        df.at[idx, "label_source"] = "HUMAN" if st_val == "VALID" else ""
        df.at[idx, "annotation_confidence"] = 0.8
    df.to_csv(label_file_path, index=False, encoding="utf-8")


def _batch_set_label(df: pd.DataFrame, slice_indices: List[int], label: str, state: str, annotator_id: str, label_file_path: Path, master_csv_path: Path):
    """Batch assign label to all students in slice."""
    for idx in slice_indices:
        df.at[idx, "participation_label"] = label if state == "VALID" else ""
        df.at[idx, "annotation_state"] = state
        df.at[idx, "label_source"] = "HUMAN" if state == "VALID" else ""
        df.at[idx, "annotator_id"] = annotator_id
        df.at[idx, "annotation_confidence"] = 0.8
    df.to_csv(label_file_path, index=False, encoding="utf-8")


def _apply_track_to_all_windows(df: pd.DataFrame, track_id: str, label: str, state: str, annotator_id: str, label_file_path: Path, master_csv_path: Path):
    """Fast-forward apply this label to this track across all windows."""
    mask = df["track_id"].astype(str) == str(track_id)
    for idx in df[mask].index:
        df.at[idx, "participation_label"] = label if state == "VALID" else ""
        df.at[idx, "annotation_state"] = state
        df.at[idx, "label_source"] = "HUMAN" if state == "VALID" else ""
        df.at[idx, "annotator_id"] = annotator_id
        df.at[idx, "annotation_confidence"] = 0.8
    df.to_csv(label_file_path, index=False, encoding="utf-8")


def _sync_row_to_master(master_path: Path, window_id: str, row_dict: dict):
    """Update matching window_id row in master review CSV."""
    try:
        master_df = read_annotation_csv(master_path)
        m_idx = master_df.index[master_df["window_id"] == window_id]
        if not m_idx.empty:
            target_idx = m_idx[0]
            for col, val in row_dict.items():
                if col in master_df.columns:
                    master_df.at[target_idx, col] = val
            master_df.to_csv(master_path, index=False, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not sync window %s to master CSV: %s", window_id, exc)


# ----------------------------------------------------------------------
# BƯỚC 3: KIỂM ĐỊNH & XUẤT FILE CSV
# ----------------------------------------------------------------------
def _render_step_3(person_name: str, annotator_full_name: str, master_csv_path: Path):
    st.subheader("BƯỚC 3: KIỂM ĐỊNH & XUẤT FILE CSV")
    st.caption("Kiểm tra tính hợp lệ của nhãn theo chuẩn đồ án và tải file CSV đã hoàn tất.")

    label_path_str = st.session_state.get("studio_label_path")
    if not label_path_str or not Path(label_path_str).is_file():
        st.warning("Chưa có file nhãn để kiểm định.")
        return

    l_path = Path(label_path_str)
    df = read_annotation_csv(l_path)

    # 1. Distribution Summary
    total = len(df)
    n_high = int((df["participation_label"] == HIGH).sum())
    n_mod = int((df["participation_label"] == MODERATE).sum())
    n_low = int((df["participation_label"] == LOW).sum())
    n_occ = int((df["annotation_state"] == "OCCLUDED").sum())
    n_inv = int((df["annotation_state"] == "INVALID").sum())
    n_done = int((
        (df["annotation_state"].eq("VALID") & df["participation_label"].ne("")) |
        df["annotation_state"].isin(["OCCLUDED", "INVALID"])
    ).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng số cửa sổ", total)
    c2.metric("🟢 CAO (HIGH)", f"{n_high} ({n_high/max(1, total)*100:.1f}%)")
    c3.metric("🟡 VỪA (MOD)", f"{n_mod} ({n_mod/max(1, total)*100:.1f}%)")
    c4.metric("🔴 THẤP (LOW)", f"{n_low} ({n_low/max(1, total)*100:.1f}%)")
    c5.metric("⚪ BỊ CHE / KHÁC", f"{n_occ + n_inv} ({(n_occ+n_inv)/max(1, total)*100:.1f}%)")

    st.write(f"**Tiến độ gán nhãn:** {n_done}/{total} ({n_done/max(1, total)*100:.1f}%) hoàn thành.")
    st.progress(n_done / max(1, total))

    # 2. Schema Validation
    valid, errors = validate_annotation_dataframe(df)
    if valid:
        st.success("✅ File nhãn hoàn toàn hợp lệ theo chuẩn dataset_qa! Không có lỗi dữ liệu.")
    else:
        st.error(f"⚠️ Phát hiện {len(errors)} cảnh báo / lỗi schema:")
        for err in errors[:5]:
            st.write(f"- {err}")

    # 3. Download & Sync
    st.divider()
    d_col1, d_col2 = st.columns(2)
    csv_bytes = l_path.read_bytes()
    d_col1.download_button(
        label=f"📥 Tải File CSV Đã Gán ({l_path.name})",
        data=csv_bytes,
        file_name=l_path.name,
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    if master_csv_path.is_file():
        if d_col2.button(f"🔄 Đồng Bộ Toàn Bộ Vào Master ({master_csv_path.name})", use_container_width=True):
            # Sync all rows
            master_df = read_annotation_csv(master_csv_path)
            synced_count = 0
            for _, row in df.iterrows():
                w_id = row["window_id"]
                m_match = master_df.index[master_df["window_id"] == w_id]
                if not m_match.empty:
                    for col in df.columns:
                        if col in master_df.columns:
                            master_df.at[m_match[0], col] = row[col]
                    synced_count += 1
            master_df.to_csv(master_csv_path, index=False, encoding="utf-8")
            st.success(f"Đã đồng bộ {synced_count} cửa sổ vào file tổng {master_csv_path.name} thành công!")
