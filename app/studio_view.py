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
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from scripts.dataset_qa import validate_annotation_dataframe
from src.contracts.annotation_io import read_annotation_csv, write_annotation_csv, save_annotation_row
from src.pose.annotation_preview import render_annotation_preview
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
from src.pose.ai_bundle import (
    build_web_llm_prompt,
    build_window_contact_sheet,
    load_suggestions_map,
    parse_and_validate_suggestions,
    propose_local_suggestion,
    save_suggestions_map,
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


def show_image(image, caption=None, target=None):
    """Display image safely across all Streamlit versions without deprecation warnings."""
    ctx = target if target is not None else st
    try:
        ctx.image(image, caption=caption, width="stretch")
    except (TypeError, ValueError):
        try:
            ctx.image(image, caption=caption, use_container_width=True)
        except Exception:
            ctx.image(image, caption=caption)


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

    # Studio Mode Selector
    mode_options = [
        "🎯 Gán 600 Cửa Sổ Chuẩn (Review V0.3 - Khuyên Dùng)",
        "🗂️ Gán Nhãn Tập Đầy Đủ (Train / Val / Test — Khác Bộ 600)",
        "👥 Duyệt Theo Mốc 4s Toàn Lớp (Theo Session / Cắt Đoạn)",
    ]
    studio_mode = st.radio(
        "Lựa chọn chế độ làm việc Studio:",
        mode_options,
        horizontal=True,
        key="studio_mode_radio",
    )

    if studio_mode == mode_options[0]:
        _render_600_windows_studio(person_name, annotator_full_name, master_csv_path)
    elif studio_mode == mode_options[1]:
        _render_full_datasets_studio(person_name, annotator_full_name, master_csv_path)
    else:
        # 3-step session workflow
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
# CHẾ ĐỘ 1 & 2: GÁN TRỰC TIẾP 600 CỬA SỔ CHUẨN HOẶC TẬP ĐẦY ĐỦ (TRAIN / VAL / TEST)
# ----------------------------------------------------------------------
def _render_600_windows_studio(person_name: str, annotator_full_name: str, master_csv_path: Path):
    """Interactive Studio view for direct review of 600 sampled windows."""
    if person_name not in ("Phong", "Tài"):
        st.info("Bộ 600 dùng phiếu riêng của Phong và Tài. Hãy chọn đúng người gán.")
        return
    p_code = "b" if "phong" in person_name.lower() else "a"
    cand_path = ROOT / f"data/annotations/review_v03_independent_20261124/annotator_{p_code}.csv"
    _render_window_dataset_studio(
        person_name,
        annotator_full_name,
        master_csv_path,
        cand_path=cand_path,
        key_prefix="600",
        section_title=f"🎯 GÁN NHÃN 600 CỬA SỔ CHUẨN — {annotator_full_name}",
    )


def _render_full_datasets_studio(person_name: str, annotator_full_name: str, master_csv_path: Path):
    """Interactive Studio view for full train / val / test splits and other annotation sets."""
    if person_name not in ("Phong", "Tài"):
        st.info("Chế độ gán tập đầy đủ dùng phiếu riêng của Phong và Tài. Hãy chọn đúng người gán.")
        return
    p_code = "b" if "phong" in person_name.lower() else "a"
    catalog = [
        ("train_full_ai_assist", "🚂 Tập TRAIN đầy đủ (9.888 cửa sổ · class_1..5, session_001..002)"),
        ("val_full_ai_assist", "🧪 Tập VAL đầy đủ (8.617 cửa sổ · session_003..004)"),
        ("test_full_ai_assist", "🎯 Tập TEST đầy đủ (4.889 cửa sổ · session_005..006)"),
        ("external_review_20261109", "🆕 Hai video mới mở rộng (130 cửa sổ · external_review)"),
        ("human_review_v0_2", "📦 Bộ gán nhãn chính v0.2 (600 cửa sổ cũ)"),
        ("review_v03", "📋 Bộ gán nhãn v0.3 gốc (600 cửa sổ YOLO11)"),
    ]
    choices: List[Tuple[Path, str]] = []
    seen_paths = set()
    for dir_name, label in catalog:
        p = ROOT / "data/annotations" / dir_name / f"annotator_{p_code}.csv"
        if p.is_file():
            choices.append((p, label))
            seen_paths.add(p.resolve())
    if (
        master_csv_path.is_file()
        and master_csv_path.parent.name != "review_v03_independent_20261124"
        and master_csv_path.resolve() not in seen_paths
    ):
        choices.append((master_csv_path, f"📄 Phiếu đang chọn ({master_csv_path.parent.name}/{master_csv_path.name})"))

    if not choices:
        st.warning("Chưa tìm thấy phiếu tập train/val/test đầy đủ. Hãy chạy `python scripts/prepare_full_split_annotations.py`.")
        return

    default_idx = 0
    if master_csv_path.is_file():
        for idx_c, (c_path, _) in enumerate(choices):
            if c_path.resolve() == master_csv_path.resolve():
                default_idx = idx_c
                break

    selected_idx = st.selectbox(
        "Chọn tập dữ liệu đầy đủ / bộ khác cần gán:",
        list(range(len(choices))),
        index=default_idx,
        format_func=lambda idx_c: choices[int(idx_c)][1] if isinstance(idx_c, (int, np.integer)) else str(idx_c),
        key=f"full_ds_picker:{person_name}",
    )
    if not isinstance(selected_idx, (int, np.integer)):
        selected_idx = int(selected_idx) if str(selected_idx).isdigit() else default_idx
    cand_path, cand_label = choices[int(selected_idx)]
    _render_window_dataset_studio(
        person_name,
        annotator_full_name,
        master_csv_path,
        cand_path=cand_path,
        key_prefix=f"full_{cand_path.parent.name}",
        section_title=f"🗂️ GÁN NHÃN TẬP ĐẦY ĐỦ — {annotator_full_name}",
        subtitle=cand_label,
    )


def _render_window_dataset_studio(
    person_name: str,
    annotator_full_name: str,
    master_csv_path: Path,
    *,
    cand_path: Path,
    key_prefix: str,
    section_title: str,
    subtitle: str = "",
):
    """Shared interactive window-by-window Studio reviewer for 600-window or full train/val/test CSVs."""
    st.subheader(section_title)
    if subtitle:
        st.caption(f"{subtitle} · Xem clip đầy đủ, xác minh đúng track và ghi bằng chứng cho từng cửa sổ.")
    else:
        st.caption("Xem clip đầy đủ, xác minh đúng track và ghi bằng chứng cho từng cửa sổ.")

    if not cand_path.is_file():
        st.error(f"Chưa tìm thấy file cửa sổ tại {cand_path}")
        return

    df_600 = read_annotation_csv(cand_path)
    total_600 = len(df_600)

    # Compute completed mask
    is_done_mask = (
        (df_600["annotation_state"].eq("VALID") & df_600["participation_label"].ne("")) |
        df_600["annotation_state"].isin(["OCCLUDED", "INVALID"])
    )

    # Auto-save progress alert
    done_count = int(is_done_mask.sum())
    if done_count > 0 and done_count < total_600:
        st.success(
            f"💾 **Tự động lưu tiến độ vào ổ cứng:** Đã lưu **{done_count}/{total_600}** cửa sổ trên máy (`{cand_path.name}`). "
            f"Bạn có thể yên tâm tắt máy/nghỉ ngơi bất kỳ lúc nào, hôm sau mở lại tool sẽ **tự động mở tiếp từ cửa sổ chưa gán** kế tiếp!"
        )
    elif done_count == total_600:
        st.success(f"🎉 **Đã hoàn thành xuất sắc toàn bộ {total_600}/{total_600} cửa sổ!**")

    # 1. Filters Row
    f_col1, f_col2, f_col3 = st.columns([3, 4, 3])
    split_counts = df_600["split"].astype(str).str.lower().value_counts().to_dict() if "split" in df_600.columns else {}
    all_label = f"Tất cả ({total_600} cửa sổ)"
    filter_opts = [all_label, "Chưa gán nhãn (còn lại)", "Đã gán nhãn"]
    if len(split_counts) > 1:
        if split_counts.get("test", 0) > 0:
            filter_opts.append(f"Tập TEST ({split_counts['test']} mẫu · session_005, 006)" if total_600 == 600 else f"Tập TEST ({split_counts['test']} mẫu)")
        if split_counts.get("val", 0) > 0:
            filter_opts.append(f"Tập VAL ({split_counts['val']} mẫu · session_003, 004)" if total_600 == 600 else f"Tập VAL ({split_counts['val']} mẫu)")
        if split_counts.get("train", 0) > 0:
            filter_opts.append(f"Tập TRAIN ({split_counts['train']} mẫu · session_001, 002, classes)" if total_600 == 600 else f"Tập TRAIN ({split_counts['train']} mẫu)")
    filter_opts.append("Theo phiên học cụ thể...")

    filter_choice = f_col1.selectbox(
        "Lọc danh sách cửa sổ:",
        filter_opts,
        key=f"filter_{key_prefix}_choice",
    )

    specific_session = None
    if filter_choice == "Theo phiên học cụ thể...":
        all_sessions = sorted(df_600["session_id"].dropna().unique().tolist())
        specific_session = f_col2.selectbox("Chọn phiên học:", all_sessions, key=f"filter_{key_prefix}_sess")

    # Apply filter mask
    if filter_choice == "Chưa gán nhãn (còn lại)":
        filtered_indices = list(df_600.index[~is_done_mask])
    elif filter_choice == "Đã gán nhãn":
        filtered_indices = list(df_600.index[is_done_mask])
    elif filter_choice.startswith("Tập TEST"):
        filtered_indices = list(df_600.index[df_600["split"].astype(str).str.lower() == "test"])
    elif filter_choice.startswith("Tập VAL"):
        filtered_indices = list(df_600.index[df_600["split"].astype(str).str.lower() == "val"])
    elif filter_choice.startswith("Tập TRAIN"):
        filtered_indices = list(df_600.index[df_600["split"].astype(str).str.lower() == "train"])
    elif specific_session is not None:
        filtered_indices = list(df_600.index[df_600["session_id"] == specific_session])
    else:
        filtered_indices = list(df_600.index)

    if not filtered_indices:
        st.info("Không có cửa sổ nào phù hợp với bộ lọc đã chọn.")
        return

    # Filtered pos state (coerce safely if stale browser/widget state stored a string)
    box_key = f"select_{key_prefix}_window_pos"
    pending_key = "studio_pending_window:" + (person_name if key_prefix == "600" else f"{key_prefix}:{person_name}")
    if pending_key in st.session_state:
        pending_id = st.session_state.pop(pending_key)
        positions = [p for p, i in enumerate(filtered_indices) if str(df_600.at[i, "window_id"]) == pending_id]
        st.session_state[box_key] = positions[0] if positions else 0

    def _safe_pos(val) -> Optional[int]:
        if isinstance(val, bool):
            return None
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, str) and val.strip().lstrip("-").isdigit():
            return int(val.strip())
        return None

    parsed_pos = _safe_pos(st.session_state.get(box_key))
    if parsed_pos is None or parsed_pos < 0 or parsed_pos >= len(filtered_indices):
        unfin = [p for p, i in enumerate(filtered_indices) if not is_done_mask.loc[i]]
        st.session_state[box_key] = unfin[0] if unfin else 0
    elif st.session_state.get(box_key) != parsed_pos:
        st.session_state[box_key] = parsed_pos

    def move_600(step: int):
        cur_p = _safe_pos(st.session_state.get(box_key)) or 0
        st.session_state[box_key] = max(0, min(len(filtered_indices) - 1, cur_p + step))

    def next_unfin_600():
        cur_p = _safe_pos(st.session_state.get(box_key)) or 0
        rem = [p for p, i in enumerate(filtered_indices) if p > cur_p and not is_done_mask.loc[i]]
        if not rem:
            rem = [p for p, i in enumerate(filtered_indices) if not is_done_mask.loc[i]]
        if rem:
            st.session_state[box_key] = rem[0]

    cur_pos = st.session_state[box_key]
    cur_df_idx = filtered_indices[cur_pos]
    cur_row = df_600.loc[cur_df_idx]

    # 2. Navigation bar
    n_col1, n_col2, n_col3, n_col4 = st.columns([1, 6, 1, 3])

    n_col1.button("⏮️ Trước", key=f"btn_{key_prefix}_prev", on_click=move_600, args=(-1,), disabled=cur_pos == 0)

    # Precompute dropdown labels in milliseconds even for 9,888-window splits.
    sub_df = df_600.loc[filtered_indices]
    splits_col = sub_df["split"] if "split" in sub_df.columns else [""] * len(filtered_indices)
    opt_labels = [
        f"{'✓' if is_done_mask.at[idx] else '○'} #{idx + 1}/{total_600} [{s}] {sess} | Track #{trk} ({float(t0 or 0) / 1000:g}s-{float(t1 or 0) / 1000:g}s) ➔ {lbl or st_v or 'Chưa gán'}"
        for idx, s, sess, trk, t0, t1, lbl, st_v in zip(
            filtered_indices,
            splits_col,
            sub_df["session_id"],
            sub_df["track_id"],
            sub_df["t_start_ms"],
            sub_df["t_end_ms"],
            sub_df["participation_label"],
            sub_df["annotation_state"],
        )
    ]

    window_options = list(range(len(filtered_indices)))
    n_col2.selectbox(
        "Cửa sổ đang chọn:",
        window_options,
        format_func=lambda pos_i: opt_labels[int(pos_i)] if isinstance(pos_i, (int, np.integer)) or str(pos_i).isdigit() else str(pos_i),
        label_visibility="collapsed",
        key=box_key,
    )

    n_col3.button("Tiếp ⏭️", key=f"btn_{key_prefix}_next", on_click=move_600, args=(1,), disabled=cur_pos == len(filtered_indices) - 1)

    # Next unfinished button
    unfin_any = any(not is_done_mask.loc[i] for i in filtered_indices)
    n_col4.button("⏩ Đến cửa sổ chưa gán", key=f"btn_{key_prefix}_unfin", on_click=next_unfin_600, disabled=not unfin_any)

    # Toggles bar
    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    show_kpts_600 = t_c1.checkbox("Khung xương SV", value=True, key=f"opt_{key_prefix}_kpts", help="Vẽ khung xương vàng cho sinh viên đang chọn")
    show_all_kpts_600 = t_c2.checkbox("Xương cả lớp", value=False, key=f"opt_{key_prefix}_all_kpts", help="Mặc định tắt: chỉ vẽ khung xương cho SV đang chọn để tránh rối mắt và đè khung xương")
    bo_che_mat_600 = t_c3.checkbox(
        "Bỏ che mặt trong phiên gán nhãn riêng tư",
        value=True,
        key=f"opt_{key_prefix}_no_blur_default_on",
        help="Chỉ bật khi được phép xem video gốc riêng tư.",
    )
    blur_face_600 = not bo_che_mat_600
    auto_jump_600 = t_c4.checkbox("Tự nhảy cửa sổ kế", value=True, key=f"opt_{key_prefix}_auto_jump")
    ai_assisted_600 = st.checkbox(
        "🤖 Hiển thị gợi ý AI hỗ trợ (Cách 1 Local + Cách 2 ChatGPT/Gemini — tắt nếu đang chấm mù)",
        value=True,
        key=f"opt_{key_prefix}_ai_assisted",
    )

    # 3. Resolve Video & Pose Path
    session_id = str(cur_row["session_id"])
    camera_id = str(cur_row.get("camera_id", "cam_A"))
    track_id = str(cur_row["track_id"])
    t_start_ms = float(cur_row["t_start_ms"])
    t_end_ms = float(cur_row["t_end_ms"])

    v_str = str(cur_row.get("video_path", ""))
    v_path = Path(v_str) if Path(v_str).is_file() else ROOT / v_str
    if not v_path.is_file():
        v_path = ROOT / "data/raw" / session_id / "cam_A.mp4"
    if not v_path.is_file():
        v_path = ROOT / "data/raw" / f"{session_id}.mp4"

    p_path = ROOT / "data/poses" / f"{session_id}_{camera_id}_poses.csv"

    df_pose = load_pose_dataframe(str(p_path)) if p_path.is_file() else pd.DataFrame()
    preview_ready = False
    suggestions_path = cand_path.parent / "ai_bundles" / "ai_suggestions.json"
    suggestions_map = load_suggestions_map(suggestions_path) if ai_assisted_600 else {}
    if ai_assisted_600:
        suggestions_map.update(load_suggestions_map(cand_path.parent / "ai_bundles" / "codex_suggestions.json"))
    cur_wid = str(cur_row["window_id"])
    cur_suggestion = suggestions_map.get(cur_wid) if ai_assisted_600 else None

    if ai_assisted_600:
        st.caption(f"Đang đọc {len(suggestions_map)} gợi ý từ ai_suggestions.csv + codex_suggestions.csv trong {suggestions_path.parent}. Gợi ý chưa phải nhãn đã duyệt.")
        with st.expander("🧩 Cách 1 + Cách 2: Tạo gói ảnh / gợi ý AI cho cửa sổ này & nhập JSON từ ChatGPT Plus / Gemini Pro"):
            c_ai1, c_ai2 = st.columns(2)
            if c_ai1.button("⚡ Chạy AI Local đề xuất cho cửa sổ này", key=f"btn_ai_local:{key_prefix}:{cur_wid}", use_container_width=True):
                if df_pose.empty or not v_path.is_file():
                    st.error("Thiếu video hoặc pose CSV để chạy AI Local.")
                else:
                    local_sug = propose_local_suggestion(
                        cur_row.to_dict(),
                        df_pose,
                        video_path=v_path,
                        class_phase="lecture",
                        model_id="local-weak-v0.2",
                    )
                    save_suggestions_map(suggestions_path, [local_sug])
                    st.success("Đã tạo gợi ý AI Local (chưa lưu vào phiếu HUMAN).")
                    st.rerun()

            if c_ai2.button("🖼️ Tạo ảnh ghép + Prompt (ChatGPT Plus / Gemini Pro)", key=f"btn_ai_sheet:{key_prefix}:{cur_wid}", use_container_width=True):
                if df_pose.empty or not v_path.is_file():
                    st.error("Thiếu video hoặc pose CSV để xuất ảnh ghép.")
                else:
                    sheet_path = cand_path.parent / "ai_bundles" / "images" / f"{cur_wid}.jpg"
                    _, b_meta = build_window_contact_sheet(
                        v_path,
                        df_pose,
                        cur_row.to_dict(),
                        output_path=sheet_path,
                        class_phase="lecture",
                        phase_verified=True,
                        show_skeleton=show_kpts_600,
                        blur_face=blur_face_600,
                    )
                    st.session_state[f"ai_sheet_meta:{key_prefix}:{cur_wid}"] = b_meta
                    st.success(f"Đã xuất ảnh ghép: {sheet_path.name}")

            sheet_meta = st.session_state.get(f"ai_sheet_meta:{key_prefix}:{cur_wid}")
            sheet_file = cand_path.parent / "ai_bundles" / "images" / f"{cur_wid}.jpg"
            if sheet_meta and sheet_file.is_file():
                st.download_button(
                    "📥 Tải ảnh ghép (.jpg) để kéo vào ChatGPT Plus / Gemini Pro",
                    data=sheet_file.read_bytes(),
                    file_name=sheet_file.name,
                    mime="image/jpeg",
                    key=f"dl_sheet:{key_prefix}:{cur_wid}",
                )
                st.code(build_web_llm_prompt([sheet_meta]), language="text")

            pasted_json = st.text_area(
                "Dán JSON kết quả từ ChatGPT Plus / Gemini Pro (tự tách khỏi ```json):",
                key=f"txt_ai_import:{key_prefix}:{person_name}",
                height=110,
            )
            if st.button("📥 Nhập JSON đề xuất vào hàng chờ duyệt", key=f"btn_ai_import:{key_prefix}:{person_name}"):
                if not pasted_json.strip():
                    st.warning("Hãy dán nội dung JSON trước.")
                else:
                    try:
                        imported_recs = parse_and_validate_suggestions(pasted_json, suggestion_source="VISION_LLM_IMPORT")
                        save_suggestions_map(suggestions_path, imported_recs)
                        st.success(f"Đã nhập {len(imported_recs)} đề xuất (chỉ làm gợi ý, chưa lưu HUMAN).")
                        st.rerun()
                    except (ValueError, json.JSONDecodeError) as exc:
                        st.error(f"JSON không hợp lệ: {exc}")

    # 4. Two-Column Workspace
    left_col, right_col = st.columns([6, 5])

    # ---------------- LEFT COLUMN: CLASSROOM VIEW & 5-KEYFRAME STRIP ----------------
    with left_col:
        if v_path.is_file() and p_path.is_file():
            try:
                cache_dir = Path(tempfile.gettempdir()) / "cvf25prj10_annotation_preview"
                clip, meta = render_annotation_preview(v_path, p_path, track_id, t_start_ms, t_end_ms,
                    cache_dir, anonymize=blur_face_600, skeleton=show_kpts_600)
                clip_bytes = clip.read_bytes()
                st.video(clip_bytes, format="video/mp4", loop=True)
                st.caption(f"Clip toàn cửa sổ: track được phát hiện {meta['target_visible_frames']}/{meta['frames']} khung. Kiểm tra đúng người và xem đủ đoạn.")
                preview_ready = True
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(f"Không tạo được clip đúng người: {exc}")
        else:
            st.error(f"Thiếu video hoặc pose trên máy chạy ứng dụng: {v_path} | {p_path}")
        st.markdown(
            f"##### 🏫 Toàn cảnh lớp học: **{session_id}** • Track **#{track_id}** ({t_start_ms/1000:g}s ➔ {t_end_ms/1000:g}s)"
        )
        st.caption(
            "Chú thích khung màu: 🟢 CAO • 🟡 VỪA • 🔴 THẤP • ⚪ BỊ CHE • 🔲 **Viền trắng dày là SV đang gán** | "
            + ("🟢 **BỎ CHE MẶT** (Nhìn rõ nét khuôn mặt)" if bo_che_mat_600 else "🛡️ **Đang làm mờ mặt**")
        )

        mid_t_ms = (t_start_ms + t_end_ms) / 2.0
        frame_rgb = extract_frame_at_ms(v_path, mid_t_ms)

        if frame_rgb is not None:
            sub_detections = pd.DataFrame()
            if not df_pose.empty:
                diffs = (df_pose["timestamp_ms"] - mid_t_ms).abs()
                sub = df_pose[diffs <= 350.0].copy()
                if not sub.empty:
                    sub["time_diff"] = (sub["timestamp_ms"] - mid_t_ms).abs()
                    sub_detections = sub.sort_values("time_diff").drop_duplicates(subset=["track_id"]).drop(columns=["time_diff"])

            track_map = {track_id: (str(cur_row["participation_label"]), str(cur_row["annotation_state"]))}
            annotated_scene = render_classroom_scene(
                frame_rgb,
                sub_detections,
                selected_track_id=track_id,
                labels_by_track=track_map,
                show_skeleton=show_kpts_600,
                show_all_skeletons=show_all_kpts_600,
                blur_face=blur_face_600,
            )
            show_image(annotated_scene)
        else:
            st.warning(f"Chưa mở được video: {v_path}")

        # 5-Keyframe Progression Strip
        st.markdown(
            f"##### 🎞️ 5 khung hình diễn biến 4 giây của SV đang chọn: Track #{track_id} (+0.0s ➔ +4.0s)"
        )
        strip_frames = render_progression_strip(
            v_path,
            df_pose,
            track_id,
            t_start_ms,
            t_end_ms,
            show_skeleton=show_kpts_600,
            blur_face=blur_face_600,
        ) if not df_pose.empty and {"track_id", "timestamp_ms"} <= set(df_pose.columns) else []
        if not strip_frames:
            st.warning("Không có pose hợp lệ để tạo dải ảnh; hãy kiểm tra file pose.")
        strip_cols = st.columns(5)
        for idx_kpt, (lbl_kpt, img_kpt) in enumerate(strip_frames):
            show_image(img_kpt, caption=lbl_kpt, target=strip_cols[idx_kpt])

        context_summary = st.empty()

    # ---------------- RIGHT COLUMN: FAST 1-CLICK LABELING ----------------
    with right_col:
        # Progress Bar
        done_600 = int(is_done_mask.sum())
        pct_done = done_600 / max(1, total_600) * 100.0
        st.markdown(
            f"📊 **Tiến độ {total_600} cửa sổ:** **{done_600}/{total_600}** ({pct_done:.1f}%) hoàn thành | Còn **{total_600 - done_600}** cửa sổ"
        )
        st.progress(done_600 / max(1, total_600))

        cur_p_lbl = str(cur_row.get("participation_label", ""))
        cur_st_val = str(cur_row.get("annotation_state", "VALID"))
        disp_status = cur_p_lbl if cur_st_val == "VALID" and cur_p_lbl else cur_st_val
        st.markdown(
            f"##### 👤 Cửa sổ #{cur_df_idx + 1}/{total_600}: Track #{track_id} ({session_id}) ➔ **{disp_status or 'Chưa gán'}**"
        )

        # Callback updates state before rendering, so the selected button stays highlighted.
        wk = f"{person_name}:{cur_row['window_id']}" if key_prefix == "600" else f"{key_prefix}:{person_name}:{cur_row['window_id']}"
        draft_key = f"studio_draft_label:{wk}"
        state_key = f"sb_{key_prefix}_state:{wk}"
        draft_label = st.session_state.get(draft_key, cur_p_lbl)
        draft_state = st.session_state.get(state_key, cur_st_val)

        if cur_suggestion is not None:
            sug_lbl_map = {HIGH: "🟢 CAO (HIGH)", MODERATE: "🟡 VỪA (MODERATE)", LOW: "🔴 THẤP (LOW)", "": "—"}
            sug_state = cur_suggestion.get("suggested_state", "VALID")
            sug_lbl = cur_suggestion.get("suggested_label", "")
            sug_behs = cur_suggestion.get("suggested_behaviors", [])
            sug_conf = float(cur_suggestion.get("confidence", 0.0))
            sug_src = cur_suggestion.get("suggestion_source", "AI")
            sug_rat = cur_suggestion.get("rationale", "")
            sug_rej = cur_suggestion.get("rejection_reason", "")

            with st.container(border=True):
                st.markdown(
                    f"**🤖 Đề xuất AI ({sug_src})** · Trạng thái: **`{sug_state}`** · "
                    f"Nhãn: **{sug_lbl_map.get(sug_lbl, sug_lbl)}** · Tin cậy: `{sug_conf:.2f}`"
                )
                st.caption(f"**Hành vi đề xuất:** {', '.join(sug_behs) if sug_behs else 'Không có'}")
                if sug_rat:
                    st.caption(f"**Căn cứ:** {sug_rat}")
                if sug_rej:
                    st.warning(f"**Lý do từ chối / Cảnh báo:** {sug_rej}")

                def _apply_ai_to_draft():
                    st.session_state[draft_key] = sug_lbl
                    st.session_state[state_key] = sug_state
                    for _, b_code in BEHAVIOR_CHOICES:
                        st.session_state[f"chk_{key_prefix}_{cur_row['window_id']}_{b_code}"] = b_code in sug_behs
                    if cur_suggestion.get("suggested_phase") in ("lecture", "individual_work", "discussion"):
                        st.session_state[f"sb_{key_prefix}_phase_lecture_default:{wk}"] = cur_suggestion["suggested_phase"]
                    if cur_suggestion.get("suggested_occlusion") in ("none", "partial", "heavy"):
                        st.session_state[f"sb_{key_prefix}_occ:{wk}"] = cur_suggestion["suggested_occlusion"]
                    if cur_suggestion.get("suggested_phase_evidence"):
                        st.session_state[f"phase_evidence:{person_name}:{wk}"] = cur_suggestion["suggested_phase_evidence"]
                    if sug_rat:
                        st.session_state[f"txt_{key_prefix}_notes:{wk}"] = sug_rat[:240]
                    st.session_state[f"ai_used:{wk}"] = sug_src

                ai_btn_c1, ai_btn_c2 = st.columns(2)
                ai_btn_c1.button(
                    "📝 Điền gợi ý để sửa",
                    key=f"btn_apply_ai:{wk}",
                    on_click=_apply_ai_to_draft,
                    use_container_width=True,
                )
                ai_btn_c2.caption("Sửa các lựa chọn bên dưới nếu cần, rồi bấm Duyệt & lưu ở cuối phiếu.")
                st.session_state[f"ai_seen:{wk}"] = dict(cur_suggestion)
                if not str(cur_row.get("annotator_id", "")).strip() and draft_key not in st.session_state:
                    _apply_ai_to_draft()
                    st.session_state[f"ai_auto_applied:{wk}"] = True
                    draft_label = st.session_state[draft_key]
                    draft_state = st.session_state[state_key]
        elif not ai_assisted_600:
            st.session_state.pop(f"ai_seen:{wk}", None)
            st.session_state.pop(f"ai_used:{wk}", None)
            if st.session_state.pop(f"ai_auto_applied:{wk}", False):
                for key_to_clear in (
                    draft_key,
                    state_key,
                    f"sb_{key_prefix}_phase_lecture_default:{wk}",
                    f"sb_{key_prefix}_occ:{wk}",
                    f"phase_evidence:{person_name}:{wk}",
                    f"txt_{key_prefix}_notes:{wk}",
                ):
                    st.session_state.pop(key_to_clear, None)
                for _, b_code in BEHAVIOR_CHOICES:
                    st.session_state.pop(f"chk_{key_prefix}_{cur_row['window_id']}_{b_code}", None)
                draft_label = cur_p_lbl
                draft_state = cur_st_val

        def choose_label(label, state):
            st.session_state.pop(f"ai_auto_applied:{wk}", None)
            st.session_state[draft_key] = label
            st.session_state[state_key] = state

        label_options = [
            ("🟢 CAO (HIGH)", HIGH, "VALID"),
            ("🟡 VỪA (MODERATE)", MODERATE, "VALID"),
            ("🔴 THẤP (LOW)", LOW, "VALID"),
            ("⚪ BỊ CHE (OCCLUDED)", "", "OCCLUDED"),
        ]
        columns = list(st.columns(2)) + list(st.columns(2))
        selected_name = None
        for number, (column, (name, label, state)) in enumerate(zip(columns, label_options)):
            active = draft_state == state and (state != "VALID" or draft_label == label)
            if active:
                selected_name = name
            column.button(("✓ " if active else "") + name,
                          key=f"studio_label_button:{wk}:{number}",
                          type="primary" if active else "secondary",
                          use_container_width=True, on_click=choose_label, args=(label, state))
        if selected_name:
            st.info(f"**✓ Đang chọn: {selected_name}** — Bấm Lưu bên dưới để ghi nhận.")
        else:
            st.info("Chưa chọn mức tham gia. Chọn một nút phía trên hoặc trạng thái phù hợp bên dưới.")

        # 8 Behavior Checkboxes
        st.caption("Hành vi quan sát được (Tích chọn 1 hoặc nhiều):")
        current_beh_str = str(cur_row.get("atomic_behavior", ""))
        existing_behs = [b.strip() for b in current_beh_str.split(";") if b.strip()]

        bh_col1, bh_col2 = st.columns(2)
        selected_behs = []
        for i, (b_name, b_code) in enumerate(BEHAVIOR_CHOICES):
            target_c = bh_col1 if i % 2 == 0 else bh_col2
            is_chk = target_c.checkbox(
                b_name,
                value=(b_code in existing_behs or b_name in existing_behs),
                key=f"chk_{key_prefix}_{cur_row['window_id']}_{b_code}",
            )
            if is_chk:
                selected_behs.append(b_code)

        # Context Dropdowns
        st.caption("Ngữ cảnh & Vị trí:")
        m_c1, m_c2 = st.columns(2)
        state_opts = ["VALID", "OCCLUDED", "TRANSITION", "INVALID"]
        s_val = cur_st_val if cur_st_val in state_opts else "VALID"
        new_st_600 = m_c1.selectbox("Trạng thái:", state_opts, index=state_opts.index(s_val), key=f"sb_{key_prefix}_state:{wk}")

        phase_opts = ["lecture", "individual_work", "discussion", "UNKNOWN"]
        # Unreviewed rows contain legacy metadata, not an annotator choice.
        # Use the requested lecture default; preserve completed human reviews.
        reviewed = bool(str(cur_row.get("annotator_id", "")).strip()) and cur_st_val in ANNOTATION_STATES
        p_val = cur_row.get("class_phase", "lecture") if reviewed else "lecture"
        new_phase_600 = m_c2.selectbox("Pha học:", phase_opts,
            index=phase_opts.index(p_val) if p_val in phase_opts and p_val != "UNKNOWN" else 0,
            key=f"sb_{key_prefix}_phase_lecture_default:{wk}")

        m_c3, m_c4 = st.columns(2)
        zone_opts = ["F-L", "F-C", "F-R", "M-L", "M-C", "M-R", "B-L", "B-C", "B-R", "FRONT", "MIDDLE", "BACK"]
        zone_opts = [""] + zone_opts

        # Keep memory separate for each person's file, recording and track.
        sess_trk_key = f"{cand_path.resolve()}:{session_id}:{camera_id}:{track_id}"
        memory = st.session_state.setdefault("track_zone_map_v2", {})
        remembered_zone = memory.get(sess_trk_key, "")
        if not remembered_zone:
            same = df_600[df_600.session_id.astype(str).eq(session_id)
                & df_600.camera_id.astype(str).eq(camera_id)
                & df_600.track_id.astype(str).eq(track_id)]
            zones = same.zone_id.fillna("").astype(str).str.strip()
            unique_zones = zones[zones.ne("")].unique()
            if len(unique_zones) == 1:
                remembered_zone = unique_zones[0]
        stored_zone = str(cur_row.get("zone_id", "")).strip()
        z_val = stored_zone or remembered_zone
        zone_key = f"sb_{key_prefix}_zone_v2:{wk}"
        if not stored_zone and remembered_zone and zone_key not in st.session_state:
            st.session_state[zone_key] = remembered_zone
        def remember_zone():
            st.session_state["track_zone_map_v2"][sess_trk_key] = st.session_state[zone_key]
        new_zone_600 = m_c3.selectbox("Vùng ghế:", zone_opts,
            index=zone_opts.index(z_val) if z_val in zone_opts else 0,
            key=zone_key, on_change=remember_zone)

        occ_opts = ["none", "partial", "heavy"]
        o_val = cur_row.get("occlusion_level", "none")
        new_occ_600 = m_c4.selectbox("Che khuất:", occ_opts, index=occ_opts.index(o_val) if o_val in occ_opts else 0, key=f"sb_{key_prefix}_occ:{wk}")

        new_notes_600 = st.text_input("Ghi chú:", value=str(cur_row.get("notes", "")), key=f"txt_{key_prefix}_notes:{wk}")
        phase_changed = new_phase_600 != str(cur_row.get("class_phase", "UNKNOWN"))
        context_summary.info(
            f"👤 **Thông số Track #{track_id}:** Vùng ghế: **{new_zone_600 or 'Chưa xác định'}** "
            f"• Pha đang chọn: **{new_phase_600}** "
            f"{'(chưa lưu)' if phase_changed else ''} • Tập: **{cur_row.get('split', '').upper()}**\n\n"
            f"📝 Ghi chú: {new_notes_600}"
        )
        phase_basis_options = {
            "lecture": ["Quan sát thấy giáo viên đang giảng bài trước lớp", "Quan sát thấy hoạt động trình bày bài học trước lớp"],
            "individual_work": ["Quan sát được yêu cầu làm bài cá nhân", "Đã đối chiếu lịch hoạt động: làm bài cá nhân"],
            "discussion": ["Quan sát được yêu cầu thảo luận nhóm", "Đã đối chiếu lịch hoạt động: thảo luận nhóm"],
        }
        selected_basis = st.selectbox("Chọn căn cứ pha hoạt động",
            [""] + phase_basis_options.get(new_phase_600, []) + ["Căn cứ khác / đã lưu"],
            format_func=lambda value: value or "— Chọn căn cứ đã xác minh —",
            key=f"phase_basis:{person_name}:{wk}:{new_phase_600}",
            help="Chỉ chọn điều đã quan sát hoặc đối chiếu thực tế. Chọn một pha chưa tự xác minh được pha đó.")
        additional_basis = st.text_input("Căn cứ bổ sung (hoặc nhập căn cứ khác)",
            value=str(cur_row.get("phase_evidence", "")), key=f"phase_evidence:{person_name}:{wk}")
        phase_evidence = "; ".join(value for value in (
            selected_basis if selected_basis != "Căn cứ khác / đã lưu" else "", additional_basis.strip()) if value)
        st.caption("Đánh giá toàn cửa sổ theo hướng dẫn v0.2: HIGH có tín hiệu phù hợp rõ; MODERATE có tín hiệu phù hợp yếu hoặc không liên tục; LOW cần bằng chứng không phù hợp kéo dài. Không suy luận tâm lý hoặc tự coi mặt làm mờ là bị che.")
        confirmed = st.checkbox("Tôi đã xem toàn bộ clip, xác định đúng người và bằng chứng", key=f"studio_confirm:{wk}")

        # Save & Navigation
        assisted_review = st.session_state.get(f"ai_seen:{wk}")
        save_btn_600 = st.button(
            "✅ Duyệt lựa chọn hiện tại → Lưu HUMAN & Tiếp" if assisted_review else "💾 Lưu cửa sổ này",
            type="primary" if assisted_review else "secondary",
            width="stretch",
            key=f"btn_save_{key_prefix}:{wk}",
            help="Bấm duyệt xác nhận bạn đã xem clip, kiểm tra và chịu trách nhiệm về lựa chọn hiện tại.",
        )
        if save_btn_600 and assisted_review:
            confirmed = True

        def _validate_submission(t_lbl: str, t_st: str, is_confirmed: bool) -> List[str]:
            if t_st not in ("VALID", "OCCLUDED", "INVALID", "TRANSITION"):
                return ["Chọn trạng thái hợp lệ."]
            if t_st == "VALID":
                probs = []
                if not preview_ready: probs.append("Chưa mở được clip đúng người.")
                if not is_confirmed: probs.append("Chưa xác nhận đã xem toàn bộ clip.")
                if not selected_behs: probs.append("Chưa chọn hành vi quan sát.")
                if new_phase_600 == "UNKNOWN" or not phase_evidence.strip(): probs.append("Chưa xác định pha và căn cứ.")
                if t_lbl not in CANONICAL_TRAINING_LABELS: probs.append("Chưa chọn mức tham gia bằng nút CAO/VỪA/THẤP.")
                return probs
            return []

        if save_btn_600:
            target_lbl, target_st = draft_label, new_st_600
            if target_st == "OCCLUDED":
                target_lbl = ""

            problems = _validate_submission(target_lbl, target_st, confirmed)
            if problems:
                st.error("Chưa lưu: " + " ".join(problems))
                return

            try:
                _save_600_window(
                    df_600=df_600,
                    row_idx=cur_df_idx,
                    target_label=target_lbl,
                    target_state=target_st,
                    behaviors="; ".join(selected_behs),
                    phase=new_phase_600,
                    zone=new_zone_600,
                    occlusion=new_occ_600,
                    notes=new_notes_600,
                    annotator_id=annotator_full_name,
                    primary_file_path=cand_path,
                    person_name=person_name,
                    phase_evidence=phase_evidence,
                    confirmed=confirmed,
                    ai_suggestion_source=(assisted_review or {}).get("suggestion_source", ""),
                    ai_suggestion=assisted_review,
                )
            except (ValueError, OSError) as exc:
                st.error(f"Không lưu được nhãn: {exc}")
                return

            # Auto advance to next unfinished window
            if auto_jump_600 or assisted_review:
                cur = st.session_state[box_key]
                rem_unfin = [p for p, i in enumerate(filtered_indices) if p > cur and not is_done_mask.loc[i]]
                if not rem_unfin:
                    rem_unfin = [p for p, i in enumerate(filtered_indices) if p < cur and not is_done_mask.loc[i]]
                if rem_unfin:
                    st.session_state[pending_key] = str(df_600.at[filtered_indices[rem_unfin[0]], "window_id"])
            st.rerun()

        # Fast 15-second track propagation ONLY on full train split (never 600 blind set or val/test).
        is_train_row = str(cur_row.get("split", "")).lower() == "train"
        future_15s_indices = (
            find_next_track_window_indices(df_600, cur_df_idx, is_done_mask, max_span_ms=15000.0)
            if key_prefix != "600" and is_train_row
            else []
        )
        if future_15s_indices:
            span_end_ms = max(float(df_600.at[i, "t_end_ms"]) for i in future_15s_indices)
            total_span_cnt = 1 + len(future_15s_indices)
            with st.container(border=True):
                st.markdown("**⏱️ Gán nhanh ~15s cùng sinh viên (chỉ áp dụng tập TRAIN)**")
                st.caption(
                    f"Nếu **Track #{track_id}** giữ nguyên hoạt động từ **{t_start_ms/1000:g}s ➔ {span_end_ms/1000:g}s**, "
                    f"có thể áp dụng cùng nhãn này cho cửa sổ hiện tại + **{len(future_15s_indices)}** cửa sổ chưa gán kế tiếp."
                )
                if st.checkbox("🎞️ Xem nhanh 5 ảnh suốt 15s trước khi gán", key=f"chk_15s_prev_{key_prefix}:{wk}"):
                    span_strip = render_progression_strip(
                        v_path, df_pose, track_id, t_start_ms, span_end_ms,
                        show_skeleton=show_kpts_600, blur_face=blur_face_600,
                    ) if not df_pose.empty and {"track_id", "timestamp_ms"} <= set(df_pose.columns) else []
                    if span_strip:
                        sp_cols = st.columns(len(span_strip))
                        for idx_sp, (lbl_sp, img_sp) in enumerate(span_strip):
                            show_image(img_sp, caption=lbl_sp, target=sp_cols[idx_sp])
                    else:
                        st.warning("Chưa tạo được dải ảnh 15s cho track này.")

                save_15s_btn = st.button(
                    f"⚡ Lưu & Gán nhanh {total_span_cnt} cửa sổ (~15s)",
                    key=f"btn_save_15s_{key_prefix}:{wk}",
                    use_container_width=True,
                )
                if save_15s_btn:
                    eff_confirmed = bool(confirmed or assisted_review)
                    target_lbl, target_st = draft_label, new_st_600
                    if target_st == "OCCLUDED":
                        target_lbl = ""
                    problems = _validate_submission(target_lbl, target_st, eff_confirmed)
                    if problems:
                        st.error("Chưa lưu 15s: " + " ".join(problems))
                        return
                    try:
                        saved_ids = _save_train_15s_span(
                            df_600=df_600,
                            cur_row_idx=cur_df_idx,
                            future_indices=future_15s_indices,
                            target_label=target_lbl,
                            target_state=target_st,
                            behaviors="; ".join(selected_behs),
                            phase=new_phase_600,
                            zone=new_zone_600,
                            occlusion=new_occ_600,
                            notes=new_notes_600,
                            annotator_id=annotator_full_name,
                            primary_file_path=cand_path,
                            person_name=person_name,
                            phase_evidence=phase_evidence,
                            confirmed=eff_confirmed,
                            ai_suggestion_source=(assisted_review or {}).get("suggestion_source", ""),
                            ai_suggestion=assisted_review,
                        )
                    except (ValueError, OSError) as exc:
                        st.error(f"Không lưu được nhóm 15s: {exc}")
                        return

                    saved_set = set(saved_ids)
                    cur = st.session_state[box_key]
                    rem_unfin = [
                        p for p, i in enumerate(filtered_indices)
                        if p > cur and not is_done_mask.loc[i] and str(df_600.at[i, "window_id"]) not in saved_set
                    ]
                    if not rem_unfin:
                        rem_unfin = [
                            p for p, i in enumerate(filtered_indices)
                            if not is_done_mask.loc[i] and str(df_600.at[i, "window_id"]) not in saved_set
                        ]
                    if rem_unfin:
                        st.session_state[pending_key] = str(df_600.at[filtered_indices[rem_unfin[0]], "window_id"])
                    st.rerun()

        # Batch apply caption
        if key_prefix == "600":
            st.caption("Mỗi cửa sổ được xem và lưu riêng; không tự áp dụng nhãn cho toàn bộ track.")
        else:
            st.caption("Tập TRAIN hỗ trợ gán nhanh tối đa ~15s cùng Track khi hoạt động giống nhau; tập VAL/TEST vẫn duyệt từng cửa sổ.")

        # Download button
        st.divider()
        st.download_button(
            label=f"📥 Tải File CSV ({total_600} Cửa Sổ · {cand_path.name})",
            data=cand_path.read_bytes(),
            file_name=cand_path.name,
            mime="text/csv",
            key=f"dl_csv_{key_prefix}:{cand_path.name}",
            use_container_width=True,
        )


def find_next_track_window_indices(
    df_rows: pd.DataFrame,
    row_idx: int,
    is_done_mask: pd.Series,
    *,
    max_span_ms: float = 15000.0,
) -> List[int]:
    """Find unfinished TRAIN windows of the same track within [cur_t0, cur_t0 + max_span_ms]."""
    cur = df_rows.loc[row_idx]
    if str(cur.get("split", "")).lower() != "train":
        return []
    cur_sess = str(cur["session_id"])
    cur_cam = str(cur.get("camera_id", "cam_A"))
    cur_trk = str(cur["track_id"])
    cur_t0 = float(cur["t_start_ms"])
    t0_num = pd.to_numeric(df_rows["t_start_ms"], errors="coerce")
    t1_num = pd.to_numeric(df_rows["t_end_ms"], errors="coerce")
    match = (
        (df_rows.index != row_idx)
        & (~is_done_mask)
        & df_rows["split"].astype(str).str.lower().eq("train")
        & df_rows["session_id"].astype(str).eq(cur_sess)
        & df_rows["camera_id"].astype(str).eq(cur_cam)
        & df_rows["track_id"].astype(str).eq(cur_trk)
        & (t0_num >= cur_t0)
        & (t1_num <= cur_t0 + float(max_span_ms) + 1e-6)
    )
    sub = df_rows[match].assign(_t0=t0_num[match]).sort_values("_t0")
    return [int(i) for i in sub.index]


def _save_train_15s_span(
    df_600,
    cur_row_idx,
    future_indices: List[int],
    target_label,
    target_state,
    behaviors,
    phase,
    zone,
    occlusion,
    notes,
    annotator_id,
    primary_file_path,
    person_name,
    phase_evidence,
    confirmed,
    ai_suggestion_source: str = "",
    ai_suggestion=None,
) -> List[str]:
    """Save the current train window plus consecutive ~15s windows of the same track."""
    cur_wid = str(df_600.at[cur_row_idx, "window_id"])
    saved_ids: List[str] = []
    _save_600_window(
        df_600=df_600, row_idx=cur_row_idx, target_label=target_label, target_state=target_state,
        behaviors=behaviors, phase=phase, zone=zone, occlusion=occlusion, notes=notes,
        annotator_id=annotator_id, primary_file_path=primary_file_path, person_name=person_name,
        phase_evidence=phase_evidence, confirmed=confirmed,
        ai_suggestion_source=ai_suggestion_source, ai_suggestion=ai_suggestion,
    )
    saved_ids.append(cur_wid)
    for f_idx in future_indices:
        _save_600_window(
            df_600=df_600, row_idx=f_idx, target_label=target_label, target_state=target_state,
            behaviors=behaviors, phase=phase, zone=zone, occlusion=occlusion, notes=notes,
            annotator_id=annotator_id, primary_file_path=primary_file_path, person_name=person_name,
            phase_evidence=phase_evidence, confirmed=confirmed,
            ai_suggestion_source=ai_suggestion_source, ai_suggestion=ai_suggestion,
            span_source_window_id=cur_wid, span_seconds=15.0,
        )
        saved_ids.append(str(df_600.at[f_idx, "window_id"]))
    return saved_ids


def _save_600_window(
    df_600, row_idx, target_label, target_state, behaviors, phase, zone,
    occlusion, notes, annotator_id, primary_file_path, person_name,
    phase_evidence, confirmed, ai_suggestion_source: str = "", ai_suggestion=None,
    span_source_window_id: str = "", span_seconds: float = 0.0,
):
    """Save one window (or audited 15s train span) in the current annotator's independent CSV."""
    if target_state == "VALID":
        if phase == "UNKNOWN" or not phase_evidence.strip() or not behaviors.strip() or not confirmed or target_label not in CANONICAL_TRAINING_LABELS:
            raise ValueError("Chưa đủ bằng chứng để lưu nhãn VALID.")
    evidence_payload = {
        "guide_version": "v0.2",
        "method": "WHOLE_WINDOW_HUMAN_REVIEW",
        "basis": "DIRECT_HUMAN_OBSERVATION",
        "confirmed": confirmed,
    }
    if ai_suggestion_source:
        evidence_payload["ai_suggestion_source"] = ai_suggestion_source
        evidence_payload["method"] = "AI_ASSISTED_HUMAN_REVIEW"
        evidence_payload["blind_review"] = False
        evidence_payload["ai_suggestion"] = ai_suggestion
        evidence_payload["decision"] = "ACCEPTED" if ai_suggestion and (
            target_state == ai_suggestion.get("suggested_state")
            and (target_label if target_state == "VALID" else "") == ai_suggestion.get("suggested_label", "")
            and set(b.strip() for b in behaviors.split(";") if b.strip()) == set(ai_suggestion.get("suggested_behaviors", []))
            and phase == ai_suggestion.get("suggested_phase")
        ) else "EDITED"
    if span_source_window_id:
        evidence_payload["method"] = "TRACK_15S_PROPAGATED_HUMAN_REVIEW"
        evidence_payload["propagated_from_window_id"] = span_source_window_id
        evidence_payload["span_seconds"] = float(span_seconds)
    updates = {
        "annotation_state": target_state,
        "participation_label": target_label if target_state == "VALID" else "",
        "label_source": "HUMAN" if target_state == "VALID" else "",
        "label_rule_version": "", "annotator_id": annotator_id,
        "atomic_behavior": behaviors, "class_phase": phase, "zone_id": zone,
        "occlusion_level": occlusion, "notes": notes,
        "annotation_guide_version": "v0.2", "annotation_confidence": 0.8,
        "phase_verified": bool(phase != "UNKNOWN" and phase_evidence.strip()),
        "phase_source": "HUMAN_WINDOW_REVIEW" if phase != "UNKNOWN" and phase_evidence.strip() else "UNKNOWN",
        "phase_evidence": phase_evidence.strip(),
        "annotation_evidence": json.dumps(evidence_payload, ensure_ascii=False),
    }
    save_annotation_row(primary_file_path, str(df_600.at[row_idx, "window_id"]), updates,
                        fill_track_zone=True)


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
                show_image(f_s_drawn, caption=f"Khung hình bắt đầu: {start_s:.1f}s", target=p_col1)
            else:
                p_col1.warning("Chưa đọc được khung hình bắt đầu.")

            if f_end is not None:
                f_e_drawn = f_end.copy()
                cv2.putText(f_e_drawn, f"DIEM KET THUC: {end_s:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 200, 30), 3)
                show_image(f_e_drawn, caption=f"Khung hình kết thúc: {end_s:.1f}s", target=p_col2)
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
    slice_box_key = "select_slice_box"
    if slice_box_key not in st.session_state or st.session_state[slice_box_key] >= total_slices:
        # Auto-resume at the first unfinished slice
        first_unfin_s = 0
        for s_i, s_row in time_slices.iterrows():
            sub_s = df_lbl[(df_lbl["t_start_ms"].astype(int) == s_row["t_start_ms"]) & (df_lbl["t_end_ms"].astype(int) == s_row["t_end_ms"])]
            done_s = int(((sub_s["annotation_state"] == "VALID") & sub_s["participation_label"].ne("")) | (sub_s["annotation_state"].isin(["OCCLUDED", "INVALID"]))).sum()
            if done_s < len(sub_s):
                first_unfin_s = s_i
                break
        st.session_state[slice_box_key] = first_unfin_s

    student_box_key = "select_student_box"
    if student_box_key not in st.session_state:
        st.session_state[student_box_key] = 0

    def move_slice(step: int):
        st.session_state[slice_box_key] = max(0, min(total_slices - 1, st.session_state[slice_box_key] + step))
        st.session_state[student_box_key] = 0

    cur_slice_idx = st.session_state[slice_box_key]
    cur_t_start = int(time_slices.loc[cur_slice_idx, "t_start_ms"])
    cur_t_end = int(time_slices.loc[cur_slice_idx, "t_end_ms"])

    # Windows in current time slice
    slice_mask = (df_lbl["t_start_ms"].astype(int) == cur_t_start) & (df_lbl["t_end_ms"].astype(int) == cur_t_end)
    slice_df = df_lbl[slice_mask].copy()
    slice_indices = list(slice_df.index)
    total_students_slice = len(slice_indices)

    if st.session_state[student_box_key] >= total_students_slice:
        # Auto-resume at first unfinished student in this slice
        first_unfin_std = 0
        for pos_i, r_idx in enumerate(slice_indices):
            r = df_lbl.loc[r_idx]
            is_d = (r["annotation_state"] == "VALID" and r["participation_label"]) or r["annotation_state"] in ("OCCLUDED", "INVALID")
            if not is_d:
                first_unfin_std = pos_i
                break
        st.session_state[student_box_key] = first_unfin_std

    cur_student_pos = st.session_state[student_box_key]
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
    top_c1.button("⏮️ Trước", key="btn_slice_prev", on_click=move_slice, args=(-1,), disabled=cur_slice_idx == 0)

    slice_labels = []
    for s_i, s_row in time_slices.iterrows():
        t0_s = int(s_row["t_start_ms"]) / 1000.0
        t1_s = int(s_row["t_end_ms"]) / 1000.0
        sub_s = df_lbl[(df_lbl["t_start_ms"].astype(int) == s_row["t_start_ms"]) & (df_lbl["t_end_ms"].astype(int) == s_row["t_end_ms"])]
        done_s = int(((sub_s["annotation_state"] == "VALID") & sub_s["participation_label"].ne("")) | (sub_s["annotation_state"].isin(["OCCLUDED", "INVALID"]))).sum()
        status_tag = "[✓ Xong]" if done_s == len(sub_s) else f"[{done_s}/{len(sub_s)}]"
        slice_labels.append(f"Mốc #{s_i + 1}/{total_slices} ({t0_s:g}s-{t1_s:g}s) • {len(sub_s)} SV {status_tag}")

    top_c2.selectbox(
        "Mốc 4s:",
        list(range(total_slices)),
        format_func=lambda idx: slice_labels[idx],
        label_visibility="collapsed",
        key=slice_box_key,
    )

    top_c3.button("Tiếp ⏭️", key="btn_slice_next", on_click=move_slice, args=(1,), disabled=cur_slice_idx == total_slices - 1)

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
        top_c4.selectbox(
            "SV:",
            list(range(len(student_labels))),
            format_func=lambda idx: student_labels[idx],
            label_visibility="collapsed",
            key=student_box_key,
        )

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
            show_image(annotated_scene)
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
                show_image(img_kpt, caption=lbl_kpt, target=strip_cols[idx_kpt])

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
            cur_phase_val = cur_row.get("class_phase")
            if not cur_phase_val or str(cur_phase_val).strip() in ("", "nan", "UNKNOWN"):
                cur_phase_val = "lecture"
            new_phase = m_col2.selectbox("Pha học:", phases, index=phases.index(cur_phase_val) if cur_phase_val in phases else 0, key="sb_phase")

            m_col3, m_col4 = st.columns(2)
            zones = ["F-L", "F-C", "F-R", "M-L", "M-C", "M-R", "B-L", "B-C", "B-R", "FRONT", "MIDDLE", "BACK"]

            # Track-zone memory for Step 2
            sess_trk_key = f"{cur_row.get('session_id', '')}:{cur_track_id}"
            remembered_zone = st.session_state.get("track_zone_map", {}).get(sess_trk_key, "")
            if not remembered_zone:
                same_trk_rows = df_lbl[df_lbl["track_id"].astype(str) == str(cur_track_id)]
                valid_zones = same_trk_rows["zone_id"].dropna().loc[lambda z: z.astype(str).str.strip().ne("")]
                if not valid_zones.empty:
                    remembered_zone = str(valid_zones.iloc[-1]).strip()
                    if "track_zone_map" not in st.session_state:
                        st.session_state["track_zone_map"] = {}
                    st.session_state["track_zone_map"][sess_trk_key] = remembered_zone

            cur_zone_val = str(cur_row.get("zone_id", "")).strip()
            if (not cur_zone_val or cur_zone_val in ("nan", "")) and remembered_zone:
                cur_zone_val = remembered_zone
            if not cur_zone_val or cur_zone_val in ("nan", ""):
                cur_zone_val = "M-C"

            new_zone = m_col3.selectbox("Vùng ghế:", zones, index=zones.index(cur_zone_val) if cur_zone_val in zones else 4, key="sb_zone")
            if new_zone:
                if "track_zone_map" not in st.session_state:
                    st.session_state["track_zone_map"] = {}
                st.session_state["track_zone_map"][sess_trk_key] = new_zone

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
                        st.session_state[student_box_key] = cur_student_pos + 1
                    else:
                        # Advance to next slice if available
                        if cur_slice_idx + 1 < total_slices:
                            st.session_state[slice_box_key] = cur_slice_idx + 1
                            st.session_state[student_box_key] = 0
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

    # Update track-zone memory
    if zone:
        try:
            import streamlit as st
            if "track_zone_map" not in st.session_state:
                st.session_state["track_zone_map"] = {}
            sess_id = str(df.at[row_idx, "session_id"]) if "session_id" in df.columns else ""
            trk_id = str(df.at[row_idx, "track_id"]) if "track_id" in df.columns else ""
            if sess_id and trk_id:
                st.session_state["track_zone_map"][f"{sess_id}:{trk_id}"] = zone
                same_trk = (df["session_id"].astype(str) == sess_id) & (df["track_id"].astype(str) == trk_id)
                empty_z = same_trk & (df["zone_id"].isna() | df["zone_id"].astype(str).str.strip().eq(""))
                if empty_z.any():
                    df.loc[empty_z, "zone_id"] = zone
        except Exception:
            pass

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

