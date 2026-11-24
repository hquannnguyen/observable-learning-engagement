"""Interactive Dataset Annotation & Preprocessing Studio v2.0 (Task T11 & T12).

Observable Learning Participation Analysis
Group: CVF25PRJ10 | Task code: CVFP26BCPPRJC012
Owner: Vũ Văn Phong (Data Lead) & Nguyễn Dũng Tiến (Streamlit/UI Lead - B23DCPT306)

Key Capabilities:
1. Video Trimming & Window Clip Slicing:
   - Cut raw classroom videos into pilot segments with start/end frame previews.
   - Batch export 4-second window video clips (cropped student or full classroom with skeleton overlay).
2. Automated Weak-Supervision Labeling:
   - Extract YOLOv8-Pose + ByteTrack or load existing pose parquet.
   - Generate sliding windows (4s window, 2s stride) and apply RuleBot (`weak-v0.1`) observable rules.
   - Export both pre-labeled weak CSV and blank human template CSV.
3. Multi-Frame Visual Labeling Studio (Strict Compliance with `docs/annotation_guide.md` v0.2):
   - 5-Frame Temporal Filmstrip across the entire 4s window (+0s, +1s, +2s, +3s, +4s) with 17-keypoint skeleton & head blur options.
   - Full Classroom Context View with 3x3 Zone Grid overlay (F/M/B x L/C/R) and target student spotlight.
   - On-demand playable 4s window video clip right in the browser.
   - Observable Pose Telemetry & RuleBot Explainability (Visible ratio, Hand-raised ratio, Head-down ratio, Wrist motion).
   - True 1-Click Fast Labeling buttons (Auto-save & advance) + Batch confirmation tools.
4. Double Annotation QA & Disagreement Adjudication (Task T12):
   - Compute Cohen's Kappa, Agreement Rate, and Confusion Matrix between two annotators.
   - Interactive Adjudication Workstation to resolve disagreements into `label_source=ADJUDICATED`.
5. Schema Validation & Analytics Dashboard:
   - Live contract validation via `src/contracts/labels.py` and `scripts/validate_annotations.py`.
   - Class, state, phase, provenance, and 3x3 classroom zone distributions.
"""

from __future__ import annotations

from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Setup project root in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_annotations import (
    REQUIRED_ANNOTATION_COLUMNS,
    validate_annotation_file,
)
from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    DISPLAY_LABEL_BY_CANONICAL,
    HIGH,
    LABEL_SOURCES,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
    to_display_label,
    validate_annotation_label,
)
from src.pose.anonymizer import STATE_COLORS_BGR, blur_head_roi, draw_skeleton
from src.pose.extractor import SKELETON_PAIRS
from src.preprocessing.weak_labels import RULE_VERSION, apply_labeling_rules

# Page Configuration
st.set_page_config(
    page_title="Annotation Studio v2.0 — CVF25PRJ10",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .studio-header {
        font-size: 2.0rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .studio-sub {
        font-size: 0.96rem;
        color: #475569;
        margin-bottom: 0.8rem;
    }
    .status-strip {
        background: linear-gradient(90deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 14px;
        font-size: 0.9rem;
    }
    .window-banner {
        background: #0F172A;
        color: #F8FAFC;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
    }
    .telemetry-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
    }
    .rule-explain-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #78350F;
        margin-bottom: 10px;
    }
    .guide-box {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 0.83rem;
        color: #334155;
    }
    .pill-high {
        background: #DCFCE7; color: #166534; padding: 3px 10px;
        border-radius: 999px; font-weight: 700; font-size: 0.82rem;
    }
    .pill-mod {
        background: #FEF9C3; color: #854D0E; padding: 3px 10px;
        border-radius: 999px; font-weight: 700; font-size: 0.82rem;
    }
    .pill-low {
        background: #FEE2E2; color: #991B1B; padding: 3px 10px;
        border-radius: 999px; font-weight: 700; font-size: 0.82rem;
    }
    .pill-neutral {
        background: #E2E8F0; color: #334155; padding: 3px 10px;
        border-radius: 999px; font-weight: 600; font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants from docs/annotation_guide.md
ATOMIC_BEHAVIORS_LIST = [
    "BOARD_OR_INSTRUCTOR_FACING",
    "READING_WRITING",
    "HAND_RAISED",
    "PEER_ORIENTED",
    "HEAD_DOWN",
    "BODY_TURNED",
    "OCCLUDED",
    "OUT_OF_VIEW",
    "UNKNOWN",
]

BEHAVIOR_DESCRIPTIONS_VI = {
    "BOARD_OR_INSTRUCTOR_FACING": "Hướng bảng/giảng viên",
    "READING_WRITING": "Đọc/viết/gõ phím",
    "HAND_RAISED": "Giơ tay cao hơn vai",
    "PEER_ORIENTED": "Hướng về bạn/nhóm",
    "HEAD_DOWN": "Cúi đầu thấp kéo dài",
    "BODY_TURNED": "Xoay người lệch hướng",
    "OCCLUDED": "Bị che khuất khớp",
    "OUT_OF_VIEW": "Ra ngoài khung hình",
    "UNKNOWN": "Không đủ căn cứ",
}

OCCLUSION_LEVELS = ["none", "moderate", "heavy"]
CLASS_PHASES = ["lecture", "individual_work", "discussion"]
ZONE_IDS = ["B-L", "B-C", "B-R", "M-L", "M-C", "M-R", "F-L", "F-C", "F-R"]


# =====================================================================
# CACHED DATA LOADERS & VIDEO UTILITIES
# =====================================================================

@st.cache_data(show_spinner=False)
def load_poses_parquet(parquet_path_str: str) -> Optional[pd.DataFrame]:
    path = Path(parquet_path_str)
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def get_video_metadata_info(video_path_str: str) -> Dict[str, Any]:
    path = Path(video_path_str)
    if not path.exists():
        return {"exists": False}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"exists": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_sec = total_frames / fps if fps > 0 else 0.0
    size_mb = path.stat().st_size / (1024 * 1024)
    return {
        "exists": True,
        "name": path.name,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "size_mb": size_mb,
    }


def row_to_keypoints_array(pose_row: pd.Series) -> np.ndarray:
    """Extract (17, 3) keypoints array from a pose parquet row."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    for idx in range(17):
        kpts[idx, 0] = float(pose_row.get(f"kpt_{idx:02d}_x", 0.0))
        kpts[idx, 1] = float(pose_row.get(f"kpt_{idx:02d}_y", 0.0))
        kpts[idx, 2] = float(pose_row.get(f"kpt_{idx:02d}_conf", 0.0))
    return kpts


def read_frame_at_ms(video_path: Path, timestamp_ms: int) -> Optional[np.ndarray]:
    """Read a single BGR frame at timestamp_ms."""
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(max(0, timestamp_ms)))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def render_person_crop_with_pose(
    frame: np.ndarray,
    pose_row: Optional[pd.Series],
    draw_skel: bool = True,
    blur_face: bool = False,
    label_color_bgr: Tuple[int, int, int] = (46, 204, 113),
    out_size: Tuple[int, int] = (260, 320),
) -> np.ndarray:
    """Crop the target student from frame and optionally render skeleton & head blur."""
    work = frame.copy()
    h, w = work.shape[:2]

    if pose_row is not None:
        x1 = float(pose_row.get("bbox_x1", 0))
        y1 = float(pose_row.get("bbox_y1", 0))
        x2 = float(pose_row.get("bbox_x2", w))
        y2 = float(pose_row.get("bbox_y2", h))
        kpts = row_to_keypoints_array(pose_row)

        if blur_face:
            blur_head_roi(work, (x1, y1, x2, y2), kpts)
        if draw_skel:
            draw_skeleton(work, kpts, color=label_color_bgr, conf_threshold=0.25)

        bw = max(20.0, x2 - x1)
        bh = max(20.0, y2 - y1)
        pad_x = int(bw * 0.28)
        pad_y = int(bh * 0.22)
        cx1 = max(0, int(x1) - pad_x)
        cy1 = max(0, int(y1) - pad_y)
        cx2 = min(w, int(x2) + pad_x)
        cy2 = min(h, int(y2) + pad_y)

        if cx2 > cx1 + 10 and cy2 > cy1 + 10:
            crop = work[cy1:cy2, cx1:cx2]
            return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)

    return cv2.resize(work, out_size, interpolation=cv2.INTER_AREA)


def render_temporal_filmstrip(
    video_path: Path,
    t_start_ms: int,
    t_end_ms: int,
    track_id: int,
    poses_df: Optional[pd.DataFrame],
    num_steps: int = 5,
    draw_skel: bool = True,
    blur_face: bool = False,
    canonical_label: str = "",
) -> List[Tuple[float, np.ndarray, Optional[float]]]:
    """Extract `num_steps` cropped RGB frames across [t_start_ms, t_end_ms] for the filmstrip."""
    if not video_path.exists():
        return []

    color_bgr = STATE_COLORS_BGR.get(canonical_label, (56, 189, 248))
    timestamps = np.linspace(t_start_ms, max(t_start_ms + 500, t_end_ms - 100), num_steps, dtype=int)

    track_poses = None
    if poses_df is not None and "track_id" in poses_df.columns:
        track_poses = poses_df[poses_df["track_id"].astype(int) == int(track_id)].sort_values("timestamp_ms")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    results: List[Tuple[float, np.ndarray, Optional[float]]] = []
    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            matched_row = None
            vis_ratio = None
            if track_poses is not None and not track_poses.empty:
                diffs = (track_poses["timestamp_ms"] - int(ts)).abs()
                min_idx = diffs.idxmin()
                if diffs.loc[min_idx] <= 1200:
                    matched_row = track_poses.loc[min_idx]
                    vis_ratio = float(matched_row.get("visible_ratio", 0.0))

            crop_bgr = render_person_crop_with_pose(
                frame=frame,
                pose_row=matched_row,
                draw_skel=draw_skel,
                blur_face=blur_face,
                label_color_bgr=color_bgr,
            )
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            results.append((ts / 1000.0, crop_rgb, vis_ratio))
    finally:
        cap.release()

    return results


def render_classroom_context_frame(
    video_path: Path,
    timestamp_ms: int,
    target_track_id: int,
    poses_df: Optional[pd.DataFrame],
    draw_grid: bool = True,
    draw_skel: bool = True,
    blur_face: bool = False,
) -> Optional[np.ndarray]:
    """Render full classroom frame highlighting target_track_id and 3x3 zone grid."""
    frame = read_frame_at_ms(video_path, timestamp_ms)
    if frame is None:
        return None

    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Draw 3x3 classroom zone grid
    if draw_grid:
        x_splits = [int(w * 0.33), int(w * 0.66)]
        y_splits = [int(h * 0.33), int(h * 0.66)]
        for x in x_splits:
            cv2.line(canvas, (x, 0), (x, h), (180, 180, 180), 1, cv2.LINE_AA)
        for y in y_splits:
            cv2.line(canvas, (0, y), (w, y), (180, 180, 180), 1, cv2.LINE_AA)

        row_codes = ["B", "M", "F"]
        col_codes = ["L", "C", "R"]
        y_bounds = [0, y_splits[0], y_splits[1], h]
        x_bounds = [0, x_splits[0], x_splits[1], w]
        for r_i, r_c in enumerate(row_codes):
            for c_i, c_c in enumerate(col_codes):
                z_label = f"Zone {r_c}-{c_c}"
                cv2.putText(
                    canvas,
                    z_label,
                    (x_bounds[c_i] + 10, y_bounds[r_i] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

    if poses_df is not None and not poses_df.empty:
        # Find nearest sampled frame timestamp
        diffs = (poses_df["timestamp_ms"] - int(timestamp_ms)).abs()
        min_diff = diffs.min()
        if min_diff <= 1000:
            nearest_ts = poses_df.loc[diffs.idxmin(), "timestamp_ms"]
            frame_poses = poses_df[poses_df["timestamp_ms"] == nearest_ts]

            for _, prow in frame_poses.iterrows():
                tid = int(prow["track_id"])
                x1, y1 = int(prow["bbox_x1"]), int(prow["bbox_y1"])
                x2, y2 = int(prow["bbox_x2"]), int(prow["bbox_y2"])
                kpts = row_to_keypoints_array(prow)

                if blur_face:
                    blur_head_roi(canvas, (x1, y1, x2, y2), kpts)

                if tid == int(target_track_id):
                    # Target student: bright green/cyan thick border + skeleton
                    color = (0, 255, 128)
                    if draw_skel:
                        draw_skeleton(canvas, kpts, color=color, conf_threshold=0.25)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                    tag = f"TARGET: Track #{tid} ({prow.get('zone_id', '')})"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(canvas, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(
                        canvas,
                        tag,
                        (x1 + 5, max(th, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    # Other students: subtle thin box
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (160, 160, 160), 1, cv2.LINE_AA)
                    cv2.putText(
                        canvas,
                        f"#{tid}",
                        (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def compute_window_telemetry(
    poses_df: Optional[pd.DataFrame],
    track_id: int,
    t_start_ms: int,
    t_end_ms: int,
    class_phase: str,
) -> Dict[str, Any]:
    """Compute observable pose telemetry and RuleBot explanation for a 4s window."""
    if poses_df is None or poses_df.empty:
        return {"available": False}

    w_df = poses_df[
        (poses_df["track_id"].astype(int) == int(track_id))
        & (poses_df["timestamp_ms"] >= int(t_start_ms))
        & (poses_df["timestamp_ms"] <= int(t_end_ms))
    ]
    if len(w_df) == 0:
        return {"available": False}

    mean_vis = float(w_df["visible_ratio"].mean())
    mean_conf = float(w_df["mean_keypoint_conf"].mean())

    nose_y = w_df["kpt_00_y"].values
    nose_c = w_df["kpt_00_conf"].values
    ears_y = (w_df["kpt_03_y"].values + w_df["kpt_04_y"].values) / 2.0

    l_shoulder_y = w_df["kpt_05_y"].values
    r_shoulder_y = w_df["kpt_06_y"].values
    l_shoulder_c = w_df["kpt_05_conf"].values
    r_shoulder_c = w_df["kpt_06_conf"].values

    l_wrist_y = w_df["kpt_09_y"].values
    r_wrist_y = w_df["kpt_10_y"].values
    l_wrist_c = w_df["kpt_09_conf"].values
    r_wrist_c = w_df["kpt_10_conf"].values
    l_wrist_x = w_df["kpt_09_x"].values
    r_wrist_x = w_df["kpt_10_x"].values

    l_raised = (l_wrist_c > 0.3) & (l_shoulder_c > 0.3) & (l_wrist_y < l_shoulder_y - 20)
    r_raised = (r_wrist_c > 0.3) & (r_shoulder_c > 0.3) & (r_wrist_y < r_shoulder_y - 20)
    hand_raised_ratio = float(np.mean(l_raised | r_raised))

    head_down_mask = (nose_y > ears_y + 12) & (nose_c > 0.3)
    head_down_ratio = float(np.mean(head_down_mask))
    wrist_motion = float(np.std(l_wrist_x) + np.std(r_wrist_x))

    state, rule_label, rule_behaviors, rule_conf = apply_labeling_rules(w_df, class_phase=class_phase)

    # Human-readable rule explanation
    if state != "VALID":
        reason = f"Chất lượng quan sát thấp (Visible={mean_vis*100:.0f}%, Conf={mean_conf*100:.0f}%) ➔ Đặt state = {state}, không gán nhãn hành vi."
    elif "HAND_RAISED" in rule_behaviors:
        reason = f"Phát hiện cổ tay cao hơn vai trong {hand_raised_ratio*100:.0f}% số frame ➔ Gợi ý HIGH (HAND_RAISED)."
    elif rule_label == LOW:
        reason = f"Đầu cúi thấp trong {head_down_ratio*100:.0f}% thời gian và cổ tay ít chuyển động ({wrist_motion:.1f}px) ➔ Gợi ý LOW (HEAD_DOWN)."
    elif "READING_WRITING" in rule_behaviors:
        reason = f"Cúi đầu ({head_down_ratio*100:.0f}%) kèm chuyển động cổ tay rõ ({wrist_motion:.1f}px) trong pha `{class_phase}` ➔ Gợi ý {to_display_label(rule_label)} (READING_WRITING)."
    else:
        reason = f"Tư thế thân trên ổn định (Head-down={head_down_ratio*100:.0f}%, Visible={mean_vis*100:.0f}%) trong pha `{class_phase}` ➔ Gợi ý {to_display_label(rule_label)}."

    return {
        "available": True,
        "num_frames": len(w_df),
        "mean_vis": mean_vis,
        "mean_conf": mean_conf,
        "hand_raised_ratio": hand_raised_ratio,
        "head_down_ratio": head_down_ratio,
        "wrist_motion": wrist_motion,
        "rule_state": state,
        "rule_label": rule_label or "",
        "rule_behaviors": rule_behaviors,
        "rule_conf": rule_conf,
        "reason": reason,
    }


def export_window_clip_mp4(
    video_path: Path,
    output_path: Path,
    t_start_ms: int,
    t_end_ms: int,
    track_id: int,
    poses_df: Optional[pd.DataFrame],
    crop_person: bool = True,
    draw_skel: bool = True,
    blur_face: bool = False,
) -> bool:
    """Export a 4-second video clip (.mp4) for a specific window and student."""
    if not video_path.exists():
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = int((t_start_ms / 1000.0) * fps)
    end_frame = int((t_end_ms / 1000.0) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    track_poses = None
    if poses_df is not None and "track_id" in poses_df.columns:
        track_poses = poses_df[poses_df["track_id"].astype(int) == int(track_id)].sort_values("timestamp_ms")

    out_size = (360, 440) if crop_person else (960, 540)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)

    curr = start_frame
    written = 0
    try:
        while curr <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            ts_ms = int((curr / fps) * 1000)

            matched_row = None
            if track_poses is not None and not track_poses.empty:
                diffs = (track_poses["timestamp_ms"] - ts_ms).abs()
                min_idx = diffs.idxmin()
                if diffs.loc[min_idx] <= 800:
                    matched_row = track_poses.loc[min_idx]

            if crop_person:
                out_frame = render_person_crop_with_pose(
                    frame=frame,
                    pose_row=matched_row,
                    draw_skel=draw_skel,
                    blur_face=blur_face,
                    out_size=out_size,
                )
            else:
                work = frame.copy()
                if matched_row is not None:
                    x1, y1 = int(matched_row["bbox_x1"]), int(matched_row["bbox_y1"])
                    x2, y2 = int(matched_row["bbox_x2"]), int(matched_row["bbox_y2"])
                    kpts = row_to_keypoints_array(matched_row)
                    if blur_face:
                        blur_head_roi(work, (x1, y1, x2, y2), kpts)
                    if draw_skel:
                        draw_skeleton(work, kpts, color=(0, 255, 128))
                    cv2.rectangle(work, (x1, y1), (x2, y2), (0, 255, 128), 3)
                out_frame = cv2.resize(work, out_size, interpolation=cv2.INTER_AREA)

            writer.write(out_frame)
            written += 1
            curr += 1
    finally:
        cap.release()
        writer.release()

    return written > 0


def trim_video_segment(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    target_width: int = 1920,
    target_height: int = 1080,
    progress_callback=None,
) -> bool:
    """Trim a segment from a raw video file using OpenCV."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * native_fps)
    num_frames = int(duration_sec * native_fps)
    end_frame = min(total_frames, start_frame + num_frames)

    out_w = min(orig_w, target_width)
    out_h = int(orig_h * (out_w / orig_w))
    out_w -= out_w % 2
    out_h -= out_h % 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, native_fps, (out_w, out_h))

    current = start_frame
    written = 0
    total_to_write = max(1, end_frame - start_frame)

    while current < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (out_w, out_h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
        current += 1
        if progress_callback and written % 30 == 0:
            progress_callback(min(1.0, written / total_to_write))

    cap.release()
    writer.release()
    if progress_callback:
        progress_callback(1.0)
    return written > 0


def build_annotations_from_poses(
    poses_df: pd.DataFrame,
    session_id: str,
    camera_id: str,
    phases: List[Dict[str, Any]],
    window_sec: float = 4.0,
    stride_sec: float = 2.0,
    min_visible_ratio: float = 0.50,
    mode: str = "weak_rule",  # "weak_rule" or "blank_template"
    annotator_id: str = "RuleBot_weak-v0.1",
) -> pd.DataFrame:
    """Generate sliding-window annotation dataframe (either weak-labeled or blank template)."""
    window_ms = int(window_sec * 1000)
    stride_ms = int(stride_sec * 1000)

    def get_phase(ts: int) -> str:
        for p in phases:
            if p.get("start_ms", 0) <= ts < p.get("end_ms", 0):
                return str(p.get("name", "lecture"))
        return "lecture"

    rows: List[Dict[str, Any]] = []
    for track_id, group in poses_df.groupby("track_id"):
        group = group.sort_values("timestamp_ms")
        min_ts = int(group["timestamp_ms"].min())
        max_ts = int(group["timestamp_ms"].max())

        t_start = min_ts
        while t_start + window_ms <= max_ts:
            t_end = t_start + window_ms
            w_df = group[(group["timestamp_ms"] >= t_start) & (group["timestamp_ms"] <= t_end)]

            if len(w_df) >= 3:
                last_zone = str(w_df["zone_id"].iloc[-1]) if "zone_id" in w_df else "M-C"
                phase = get_phase(t_start)
                mean_vis = float(w_df["visible_ratio"].mean()) if "visible_ratio" in w_df else 0.8
                occlusion_lvl = "none" if mean_vis >= 0.75 else ("moderate" if mean_vis >= 0.50 else "heavy")

                if mode == "weak_rule":
                    state, label, behaviors, conf = apply_labeling_rules(
                        w_df, class_phase=phase, min_visible_ratio=min_visible_ratio
                    )
                    is_valid = state == "VALID"
                    rows.append({
                        "session_id": session_id,
                        "camera_id": camera_id,
                        "track_id": str(track_id),
                        "t_start_ms": str(t_start),
                        "t_end_ms": str(t_end),
                        "class_phase": phase,
                        "atomic_behavior": ";".join(behaviors) if behaviors else "",
                        "participation_label": label if (is_valid and label) else "",
                        "annotation_state": state,
                        "label_source": "WEAK_RULE" if is_valid else "",
                        "label_rule_version": RULE_VERSION if is_valid else "",
                        "annotation_guide_version": "v0.2",
                        "annotation_confidence": f"{conf:.2f}",
                        "occlusion_level": occlusion_lvl,
                        "zone_id": last_zone,
                        "annotator_id": annotator_id,
                        "notes": f"auto_vis_ratio={mean_vis:.2f}",
                    })
                else:
                    initial_state = "VALID" if mean_vis >= min_visible_ratio else "OCCLUDED"
                    rows.append({
                        "session_id": session_id,
                        "camera_id": camera_id,
                        "track_id": str(track_id),
                        "t_start_ms": str(t_start),
                        "t_end_ms": str(t_end),
                        "class_phase": phase,
                        "atomic_behavior": "",
                        "participation_label": "",
                        "annotation_state": initial_state,
                        "label_source": "HUMAN" if initial_state == "VALID" else "",
                        "label_rule_version": "",
                        "annotation_guide_version": "v0.2",
                        "annotation_confidence": "1.00",
                        "occlusion_level": occlusion_lvl,
                        "zone_id": last_zone,
                        "annotator_id": annotator_id,
                        "notes": f"auto_vis_ratio={mean_vis:.2f}",
                    })

            t_start += stride_ms

    return pd.DataFrame(rows, columns=REQUIRED_ANNOTATION_COLUMNS)


def compute_cohens_kappa(labels_a: List[str], labels_b: List[str], categories: List[str]) -> float:
    """Compute Cohen's Kappa without requiring external libraries."""
    n = len(labels_a)
    if n == 0 or len(labels_b) != n:
        return 0.0
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    mat = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(labels_a, labels_b):
        if a in cat_to_idx and b in cat_to_idx:
            mat[cat_to_idx[a], cat_to_idx[b]] += 1.0
    total = mat.sum()
    if total == 0:
        return 0.0
    po = np.trace(mat) / total
    row_sums = mat.sum(axis=1)
    col_sums = mat.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums)) / (total * total)
    if abs(1.0 - pe) < 1e-9:
        return 1.0
    return float((po - pe) / (1.0 - pe))


# =====================================================================
# STATE PERSISTENCE HELPERS FOR TAB 2
# =====================================================================

def save_row_and_advance(
    df: pd.DataFrame,
    row_idx: int,
    csv_path: Path,
    new_state: str,
    new_label: str,
    new_behaviors: List[str],
    new_source: str,
    new_rule_version: str,
    new_conf: float,
    new_occ: str,
    new_phase: str,
    new_zone: str,
    annotator_id: str,
    notes: str,
    filtered_indices: List[int],
    advance: bool = True,
) -> Tuple[bool, str]:
    """Validate against contract, update dataframe, persist to CSV, and advance."""
    is_valid_state = new_state == "VALID"
    final_label = new_label if is_valid_state else ""
    final_source = new_source if is_valid_state else ""
    final_rule_ver = new_rule_version if (is_valid_state and final_source == "WEAK_RULE") else ""

    try:
        validate_annotation_label(
            participation_label=final_label if final_label else None,
            annotation_state=new_state,
            label_source=final_source if final_source else None,
            label_rule_version=final_rule_ver if final_rule_ver else None,
        )
    except ValueError as exc:
        return False, str(exc)

    df.loc[row_idx, "annotation_state"] = new_state
    df.loc[row_idx, "participation_label"] = final_label
    df.loc[row_idx, "atomic_behavior"] = ";".join(new_behaviors)
    df.loc[row_idx, "label_source"] = final_source
    df.loc[row_idx, "label_rule_version"] = final_rule_ver
    df.loc[row_idx, "annotation_confidence"] = f"{new_conf:.2f}"
    df.loc[row_idx, "occlusion_level"] = new_occ
    df.loc[row_idx, "class_phase"] = new_phase
    df.loc[row_idx, "zone_id"] = new_zone
    df.loc[row_idx, "annotator_id"] = annotator_id
    df.loc[row_idx, "notes"] = notes
    df.loc[row_idx, "annotation_guide_version"] = "v0.2"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    if advance and row_idx in filtered_indices:
        pos = filtered_indices.index(row_idx)
        if pos + 1 < len(filtered_indices):
            st.session_state["current_idx"] = filtered_indices[pos + 1]

    return True, f"Đã lưu cửa sổ #{row_idx + 1} ({to_display_label(final_label) if final_label else new_state})!"


# =====================================================================
# MAIN APPLICATION
# =====================================================================

def main() -> None:
    st.markdown(
        '<div class="studio-header">🏷️ Studio Gán Nhãn & Tiền Xử Lý Dữ Liệu (v2.0)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="studio-sub">'
        'Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được • '
        '<b>CVF25PRJ10 (CVFP26BCPPRJC012)</b> • Task T11 & T12'
        '</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # SIDEBAR: GLOBAL CONFIGURATION & DECISION GUIDE
    # -----------------------------------------------------------------
    with st.sidebar:
        st.header("🗂️ Không Gian Làm Việc")

        annotator_name = st.text_input(
            "👤 Mã Người Gán (Annotator ID)",
            value="Tien_B23DCPT306",
            help="Ghi vào cột annotator_id khi bạn xác nhận hoặc chỉnh sửa nhãn.",
        )

        video_path_str = st.text_input(
            "🎥 Đường dẫn Video",
            value="data/raw/session_001/cam_A_pilot.mp4",
        ).strip().strip('"').strip("'")
        video_path = Path(video_path_str)

        pose_path_str = st.text_input(
            "🦴 Đường dẫn Pose Parquet",
            value="data/poses/cam_A_pilot_poses.parquet",
        ).strip().strip('"').strip("'")
        pose_path = Path(pose_path_str)

        metadata_path_str = st.text_input(
            "📋 Đường dẫn Metadata JSON",
            value="data/raw/session_001/metadata.json",
        ).strip().strip('"').strip("'")
        metadata_path = Path(metadata_path_str)

        # Discover existing CSV files in data/annotations
        ann_dir = Path("data/annotations")
        ann_dir.mkdir(parents=True, exist_ok=True)
        existing_csvs = sorted([str(p.as_posix()) for p in ann_dir.glob("*.csv")])
        default_csv = "data/annotations/session_001_weak_labels.csv"
        if default_csv not in existing_csvs and existing_csvs:
            default_csv = existing_csvs[0]

        csv_Select_mode = st.radio(
            "📄 Chọn File Annotation CSV",
            ["Chọn từ danh sách có sẵn", "Nhập đường dẫn tùy chỉnh"],
            horizontal=True,
        )
        if csv_Select_mode == "Chọn từ danh sách có sẵn" and existing_csvs:
            active_csv_str = st.selectbox(
                "File CSV đang làm việc",
                options=existing_csvs,
                index=existing_csvs.index(default_csv) if default_csv in existing_csvs else 0,
            )
        else:
            active_csv_str = st.text_input(
                "Đường dẫn File CSV",
                value=default_csv,
            ).strip().strip('"').strip("'")
        active_csv_path = Path(active_csv_str)

        if st.button("🔄 Nạp lại Dữ liệu từ Đĩa", use_container_width=True):
            st.session_state.pop("annotation_data", None)
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("🎨 Tùy Chọn Hiển Thị Trực Quan")
        opt_draw_skeleton = st.checkbox("🦴 Vẽ khung xương 17 khớp (Pose)", value=True)
        opt_blur_face = st.checkbox("🕶️ Làm mờ khuôn mặt (Head Blur)", value=False)
        opt_draw_grid = st.checkbox("🏫 Hiện lưới vùng 3x3 toàn cảnh", value=True)
        opt_filmstrip_count = st.select_slider(
            "🎞️ Số khung hình Dải phim 4s",
            options=[3, 4, 5, 6],
            value=5,
        )

        st.divider()
        with st.expander("📖 Sổ tay Quy tắc Gán nhãn (v0.2)", expanded=False):
            st.markdown(
                """
                <div class="guide-box">
                <b>1. Xem toàn bộ cửa sổ 4s</b> (không gán từ 1 frame đơn).<br>
                <b>2. Cây quyết định:</b><br>
                • Bị che >40% ➔ <code>OCCLUDED</code> (trống nhãn)<br>
                • Ngoài khung/lỗi ➔ <code>INVALID</code> (trống nhãn)<br>
                • Đang chuyển pha ➔ <code>TRANSITION</code> (trống nhãn)<br>
                • Quan sát rõ ➔ <code>VALID</code> + chọn 1 trong 3:<br>
                &nbsp;&nbsp;🟢 <b>HIGH</b>: Tín hiệu tham gia rõ (nhìn bảng, đọc/viết tích cực, giơ tay, thảo luận).<br>
                &nbsp;&nbsp;🟡 <b>MODERATE</b>: Tín hiệu phù hợp nhưng thụ động/yếu.<br>
                &nbsp;&nbsp;🔴 <b>LOW</b>: Tín hiệu không phù hợp kéo dài (gục đầu không viết, quay lưng).<br>
                <b>3. Đạo đức AI:</b> Tuyệt đối không suy diễn "lười", "buồn chán".
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Load shared resources
    vid_info = get_video_metadata_info(str(video_path))
    poses_df = load_poses_parquet(str(pose_path)) if pose_path.exists() else None

    if active_csv_path.exists():
        if (
            "annotation_data" not in st.session_state
            or st.session_state.get("loaded_file") != str(active_csv_path)
        ):
            st.session_state["annotation_data"] = pd.read_csv(active_csv_path, dtype=str).fillna("")
            st.session_state["loaded_file"] = str(active_csv_path)
            st.session_state["current_idx"] = 0

    # Top Workspace Status Strip
    vid_badge = (
        f"✅ Video: <b>{vid_info['name']}</b> ({vid_info['width']}x{vid_info['height']}, "
        f"{vid_info['fps']:.0f} FPS, {vid_info['duration_sec']/60:.1f}m, {vid_info['size_mb']:.0f} MB)"
        if vid_info.get("exists")
        else "⚠️ Chưa kết nối Video"
    )
    pose_badge = (
        f"✅ Pose Parquet: <b>{len(poses_df):,} detections</b> ({poses_df['track_id'].nunique()} tracks)"
        if poses_df is not None
        else "⚠️ Chưa có Pose Parquet"
    )
    csv_badge = f"📄 CSV: <b>{active_csv_path.name}</b>" if active_csv_path.exists() else "⚠️ Chưa có CSV"

    st.markdown(
        f'<div class="status-strip">{vid_badge} &nbsp;&nbsp;|&nbsp;&nbsp; {pose_badge} &nbsp;&nbsp;|&nbsp;&nbsp; {csv_badge}</div>',
        unsafe_allow_html=True,
    )

    # 4 Main Tabs
    tab_pipeline, tab_studio, tab_double_qa, tab_analytics = st.tabs([
        "✂️ 1. Cắt Video & Tự Động Gán Nhãn",
        "👁️ 2. Studio Duyệt & Gán Nhãn Trực Quan (5-Frame 4s)",
        "🤝 3. Đối Soát Chéo & Adjudication (Task T12)",
        "📊 4. Kiểm Định Schema & Báo Cáo Dataset",
    ])

    # =================================================================
    # TAB 1: CẮT VIDEO, TRÍCH XUẤT POSE & TỰ ĐỘNG GÁN NHÃN
    # =================================================================
    with tab_pipeline:
        st.subheader("✂️ Trạm 1: Cắt Phân Đoạn Video, Trích Xuất Pose & Tự Động Gán Nhãn (RuleBot)")

        col_p1, col_p2 = st.columns([1.15, 1])

        with col_p1:
            st.markdown("##### 1A. Chọn Khoảng Thời Gian Cắt & Xem Trước Khung Hình")
            max_dur_sec = float(vid_info.get("duration_sec", 600.0)) if vid_info.get("exists") else 600.0

            time_range_sec = st.slider(
                "Khoảng thời gian xử lý / cắt video (giây)",
                min_value=0.0,
                max_value=max(10.0, round(max_dur_sec, 1)),
                value=(0.0, min(60.0, round(max_dur_sec, 1))),
                step=2.0,
                help="Chọn đoạn bắt đầu và kết thúc (mặc định 0s - 60s khớp với tập pilot).",
            )
            t_cut_start_s, t_cut_end_s = time_range_sec
            cut_duration_s = max(2.0, t_cut_end_s - t_cut_start_s)

            # Start & End Frame Preview
            if vid_info.get("exists"):
                with st.expander("🖼️ Xem trước Khung hình Điểm Đầu & Điểm Cuối đoạn cắt", expanded=True):
                    pcol1, pcol2 = st.columns(2)
                    f_start = read_frame_at_ms(video_path, int(t_cut_start_s * 1000))
                    f_end = read_frame_at_ms(video_path, int(max(0, t_cut_end_s - 0.5) * 1000))
                    if f_start is not None:
                        pcol1.image(
                            cv2.cvtColor(f_start, cv2.COLOR_BGR2RGB),
                            caption=f"Điểm bắt đầu: {t_cut_start_s:.1f}s ({t_cut_start_s/60:.2f}m)",
                            use_container_width=True,
                        )
                    if f_end is not None:
                        pcol2.image(
                            cv2.cvtColor(f_end, cv2.COLOR_BGR2RGB),
                            caption=f"Điểm kết thúc: {t_cut_end_s:.1f}s ({t_cut_end_s/60:.2f}m)",
                            use_container_width=True,
                        )

        with col_p2:
            st.markdown("##### 1B. Cấu Hình Cắt Video & Bộ Quy Tắc Tự Động Gán Nhãn")

            export_trimmed_mp4 = st.checkbox(
                "💾 Xuất đoạn đã chọn thành file Video .mp4 riêng",
                value=False,
                help="Bật nếu bạn muốn cắt từ video gốc ra 1 file .mp4 mới nhẹ hơn.",
            )
            trimmed_out_path_str = st.text_input(
                "Đường dẫn lưu video sau khi cắt",
                value="data/raw/session_001/cam_A_trimmed.mp4",
                disabled=not export_trimmed_mp4,
            )

            pose_source_mode = st.radio(
                "Nguồn dữ liệu Khung xương (Pose & Tracking)",
                [
                    "Dùng nhanh file Pose Parquet hiện có (Siêu tốc < 1 giây)",
                    "Chạy trích xuất mới bằng YOLOv8-Pose + ByteTrack trên đoạn đã chọn",
                ],
                index=0 if poses_df is not None else 1,
            )

            wcol1, wcol2, wcol3 = st.columns(3)
            with wcol1:
                win_sec = st.number_input("Window (giây)", 2.0, 10.0, 4.0, 1.0)
            with wcol2:
                str_sec = st.number_input("Stride (giây)", 1.0, 5.0, 2.0, 1.0)
            with wcol3:
                min_vis_gate = st.slider("Ngưỡng Visible tối thiểu", 0.30, 0.85, 0.50, 0.05)

            gen_mode = st.radio(
                "Chế độ tạo File Annotation CSV",
                [
                    "🤖 Tự động gán nhãn bằng RuleBot (WEAK_RULE - weak-v0.1)",
                    "📝 Tạo Template trống để tự gán thủ công từ đầu (HUMAN)",
                ],
                index=0,
            )

            output_csv_name = st.text_input(
                "Đường dẫn lưu File Annotation CSV mới",
                value=(
                    "data/annotations/session_001_weak_labels.csv"
                    if "RuleBot" in gen_mode
                    else "data/annotations/session_001_annotation_template.csv"
                ),
            )

            if st.button("🚀 Chạy Pipeline Cắt & Tự Động Gán Nhãn Ngay", type="primary", use_container_width=True):
                prog = st.progress(0.0)
                msg = st.empty()

                # 1. Trim video if requested
                target_video_for_pose = video_path
                if export_trimmed_mp4:
                    if not video_path.exists():
                        st.error("❌ Không tìm thấy file video gốc để cắt!")
                        st.stop()
                    out_vid = Path(trimmed_out_path_str.strip())
                    msg.info(f"⏳ Đang cắt video ({t_cut_start_s:.1f}s ➔ {t_cut_end_s:.1f}s)...")
                    trim_video_segment(
                        input_path=video_path,
                        output_path=out_vid,
                        start_sec=t_cut_start_s,
                        duration_sec=cut_duration_s,
                        progress_callback=lambda p: prog.progress(p * 0.4),
                    )
                    msg.success(f"✅ Đã lưu video đã cắt tại: `{out_vid}`")

                # 2. Load or extract poses
                prog.progress(0.45)
                if "Pose Parquet hiện có" in pose_source_mode and poses_df is not None:
                    work_poses_df = poses_df[
                        (poses_df["timestamp_ms"] >= int(t_cut_start_s * 1000))
                        & (poses_df["timestamp_ms"] <= int(t_cut_end_s * 1000))
                    ].copy()
                    if work_poses_df.empty:
                        st.warning(
                            f"⚠️ File Parquet hiện tại có mốc thời gian từ "
                            f"{poses_df['timestamp_ms'].min()/1000:.1f}s đến {poses_df['timestamp_ms'].max()/1000:.1f}s. "
                            "Đang sử dụng toàn bộ dữ liệu trong Parquet."
                        )
                        work_poses_df = poses_df.copy()
                else:
                    if not target_video_for_pose.exists():
                        st.error("❌ Không tìm thấy file video để chạy YOLOv8-Pose!")
                        st.stop()
                    msg.info("⏳ Đang chạy YOLOv8-Pose + ByteTrack trích xuất 17 khớp xương...")
                    from src.pose.extractor import PoseExtractor

                    extractor = PoseExtractor(model_name="yolov8n-pose.pt", confidence_threshold=0.30)
                    extracted_rows = []
                    for _, ts_ms, _, tracked_list in extractor.process_video(
                        target_video_for_pose,
                        sample_fps=10,
                        start_seconds=t_cut_start_s,
                        max_duration_seconds=cut_duration_s,
                    ):
                        for p in tracked_list:
                            extracted_rows.append(p.to_dict(session_id="session_001", camera_id="cam_A"))
                    work_poses_df = pd.DataFrame(extracted_rows)
                    if not work_poses_df.empty:
                        pose_path.parent.mkdir(parents=True, exist_ok=True)
                        work_poses_df.to_parquet(pose_path, index=False)
                        st.cache_data.clear()

                if work_poses_df is None or work_poses_df.empty:
                    st.error("❌ Không có dữ liệu pose nào trong khoảng thời gian đã chọn.")
                    st.stop()

                # 3. Generate sliding windows & auto-label
                prog.progress(0.75)
                msg.info("⏳ Đang cắt cửa sổ trượt 4s và áp dụng bộ luật gán nhãn...")
                meta_dict = {}
                if metadata_path.exists():
                    meta_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
                phases_cfg = meta_dict.get("class_phase", [])
                sess_id = meta_dict.get("session_id", "session_001")
                cam_id = meta_dict.get("camera_id", "cam_A")

                is_weak = "RuleBot" in gen_mode
                out_df = build_annotations_from_poses(
                    poses_df=work_poses_df,
                    session_id=sess_id,
                    camera_id=cam_id,
                    phases=phases_cfg,
                    window_sec=win_sec,
                    stride_sec=str_sec,
                    min_visible_ratio=min_vis_gate,
                    mode="weak_rule" if is_weak else "blank_template",
                    annotator_id=f"RuleBot_{RULE_VERSION}" if is_weak else annotator_name,
                )

                out_csv_file = Path(output_csv_name.strip())
                out_csv_file.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(out_csv_file, index=False, encoding="utf-8")

                # Update active session state
                st.session_state["annotation_data"] = out_df.fillna("")
                st.session_state["loaded_file"] = str(out_csv_file)
                st.session_state["current_idx"] = 0

                prog.progress(1.0)
                h_c = int((out_df["participation_label"] == HIGH).sum())
                m_c = int((out_df["participation_label"] == MODERATE).sum())
                l_c = int((out_df["participation_label"] == LOW).sum())
                occ_c = int((out_df["annotation_state"] == "OCCLUDED").sum())
                msg.success(
                    f"🎉 Đã tạo thành công **{len(out_df)} cửa sổ** tại `{out_csv_file}`! "
                    f"(🟢 HIGH: {h_c} | 🟡 MODERATE: {m_c} | 🔴 LOW: {l_c} | 🌫️ OCCLUDED: {occ_c})"
                )

        st.divider()
        with st.expander("🎬 Công cụ Bổ sung: Cắt Hàng Loạt Các Cửa Sổ 4 Giây Thành File Video Clip (.mp4)", expanded=False):
            st.caption(
                "Trích xuất từng cửa sổ 4s trong bảng CSV thành các file video ngắn `.mp4` riêng biệt "
                "(lưu tại `data/clips/`) kèm khung xương để phục vụ báo cáo hoặc kiểm tra độc lập."
            )
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            with bcol1:
                clip_track_filter = st.number_input("Chỉ xuất cho Track ID (0 = Tất cả)", min_value=0, value=1, step=1)
            with bcol2:
                max_clips_export = st.number_input("Số lượng clip tối đa xuất", min_value=1, max_value=200, value=10, step=5)
            with bcol3:
                clip_crop_mode = st.checkbox("Crop sát sinh viên", value=True)
            with bcol4:
                clips_dir_str = st.text_input("Thư mục lưu clips", value="data/clips")

            if st.button("✂️ Xuất Hàng Loạt Video Clips 4 Giây (.mp4)"):
                if "annotation_data" not in st.session_state or not video_path.exists():
                    st.error("Cần có Video và File CSV đang mở để xuất clip.")
                else:
                    df_clips = st.session_state["annotation_data"]
                    if clip_track_filter > 0:
                        df_clips = df_clips[df_clips["track_id"].astype(int) == int(clip_track_filter)]
                    df_clips = df_clips.head(int(max_clips_export))
                    c_prog = st.progress(0.0)
                    exported_count = 0
                    out_clips_dir = Path(clips_dir_str)
                    for idx_c, (_, r_c) in enumerate(df_clips.iterrows()):
                        tid_c = int(r_c["track_id"])
                        ts_s = int(r_c["t_start_ms"])
                        ts_e = int(r_c["t_end_ms"])
                        lbl_short = to_display_label(r_c["participation_label"]) if r_c["participation_label"] else r_c["annotation_state"]
                        fname = f"{r_c['session_id']}_trk{tid_c:03d}_{ts_s:06d}_{ts_e:06d}_{lbl_short}.mp4"
                        ok_clip = export_window_clip_mp4(
                            video_path=video_path,
                            output_path=out_clips_dir / fname,
                            t_start_ms=ts_s,
                            t_end_ms=ts_e,
                            track_id=tid_c,
                            poses_df=poses_df,
                            crop_person=clip_crop_mode,
                            draw_skel=opt_draw_skeleton,
                            blur_face=opt_blur_face,
                        )
                        if ok_clip:
                            exported_count += 1
                        c_prog.progress((idx_c + 1) / max(1, len(df_clips)))
                    st.success(f"✅ Đã xuất thành công {exported_count} video clip 4s vào thư mục `{out_clips_dir}`!")

    # =================================================================
    # TAB 2: STUDIO DUYỆT & GÁN NHÃN TRỰC QUAN ĐA KHUNG HÌNH
    # =================================================================
    with tab_studio:
        if "annotation_data" not in st.session_state:
            st.warning("⚠️ Chưa có file CSV nào được nạp. Hãy tạo ở Tab 1 hoặc chọn file trên Sidebar.")
            st.stop()

        df: pd.DataFrame = st.session_state["annotation_data"]
        total_rows = len(df)
        if total_rows == 0:
            st.info("File CSV hiện tại đang rỗng.")
            st.stop()

        # Overall Progress Metrics Bar
        human_cnt = int(df["label_source"].isin(["HUMAN", "ADJUDICATED"]).sum())
        weak_cnt = int((df["label_source"] == "WEAK_RULE").sum())
        non_valid_cnt = int((df["annotation_state"] != "VALID").sum())
        unlabeled_valid_cnt = int(((df["annotation_state"] == "VALID") & (df["participation_label"] == "")).sum())
        reviewed_pct = (human_cnt + non_valid_cnt) / total_rows * 100.0

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("📦 Tổng Cửa Sổ", f"{total_rows}")
        m_col2.metric("👤 Đã Duyệt (HUMAN/ADJ)", f"{human_cnt}", f"{reviewed_pct:.1f}% hoàn tất")
        m_col3.metric("🤖 Nhãn Máy (WEAK_RULE)", f"{weak_cnt}")
        m_col4.metric("🌫️ OCCLUDED / Khác", f"{non_valid_cnt}")
        m_col5.metric("❓ VALID Chưa Nhãn", f"{unlabeled_valid_cnt}")

        st.progress(min(1.0, reviewed_pct / 100.0))

        # Multi-Dimensional Filter Bar
        fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1.3, 1.2, 1.0, 1.1, 1.0])
        with fcol1:
            filter_review = st.selectbox(
                "🔍 Lọc Trạng Thái Duyệt",
                [
                    "Tất cả cửa sổ",
                    "Chưa duyệt (WEAK_RULE hoặc Trống)",
                    "Đã duyệt (HUMAN / ADJUDICATED)",
                    "VALID nhưng chưa có nhãn",
                    "Confidence thấp (< 0.78)",
                ],
            )
        with fcol2:
            # Show window counts per track
            track_counts = df["track_id"].value_counts().to_dict()
            unique_tracks = sorted(df["track_id"].unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else 999)
            track_options = ["Tất cả"] + unique_tracks
            filter_track = st.selectbox(
                "👤 Lọc Sinh Viên (Track ID)",
                options=track_options,
                format_func=lambda t: "Tất cả sinh viên" if t == "Tất cả" else f"Track #{t} ({track_counts.get(t, 0)} cửa sổ)",
            )
        with fcol3:
            filter_state = st.selectbox("📌 Lọc State", ["Tất cả"] + list(ANNOTATION_STATES))
        with fcol4:
            filter_label = st.selectbox(
                "🏷️ Lọc Nhãn",
                ["Tất cả", HIGH, MODERATE, LOW, "Trống (Chưa gán)"],
                format_func=lambda x: to_display_label(x) if x in CANONICAL_TRAINING_LABELS else x,
            )
        with fcol5:
            filter_zone = st.selectbox("🏫 Lọc Vùng Ghế", ["Tất cả"] + sorted(df["zone_id"].unique().tolist()))

        # Apply Filters
        mask = pd.Series(True, index=df.index)
        if filter_review == "Chưa duyệt (WEAK_RULE hoặc Trống)":
            mask &= ~df["label_source"].isin(["HUMAN", "ADJUDICATED"])
        elif filter_review == "Đã duyệt (HUMAN / ADJUDICATED)":
            mask &= df["label_source"].isin(["HUMAN", "ADJUDICATED"])
        elif filter_review == "VALID nhưng chưa có nhãn":
            mask &= (df["annotation_state"] == "VALID") & (df["participation_label"] == "")
        elif filter_review == "Confidence thấp (< 0.78)":
            conf_vals = pd.to_numeric(df["annotation_confidence"], errors="coerce").fillna(1.0)
            mask &= conf_vals < 0.78

        if filter_track != "Tất cả":
            mask &= df["track_id"] == str(filter_track)
        if filter_state != "Tất cả":
            mask &= df["annotation_state"] == filter_state
        if filter_label != "Tất cả":
            if filter_label == "Trống (Chưa gán)":
                mask &= df["participation_label"] == ""
            else:
                mask &= df["participation_label"] == filter_label
        if filter_zone != "Tất cả":
            mask &= df["zone_id"] == filter_zone

        filtered_indices = df.index[mask].tolist()
        if not filtered_indices:
            st.warning("⚠️ Không có cửa sổ nào khớp với bộ lọc hiện tại.")
            st.stop()

        curr_i = int(st.session_state.get("current_idx", filtered_indices[0]))
        if curr_i not in filtered_indices:
            curr_i = filtered_indices[0]
            st.session_state["current_idx"] = curr_i

        pos_in_filter = filtered_indices.index(curr_i)

        # Navigation Toolbar
        nav1, nav2, nav3, nav4, nav5 = st.columns([0.7, 0.8, 3.0, 0.8, 1.3])
        with nav1:
            if st.button("⏮️ Đầu", use_container_width=True, disabled=(pos_in_filter == 0)):
                st.session_state["current_idx"] = filtered_indices[0]
                st.rerun()
        with nav2:
            if st.button("◀️ Trước", use_container_width=True, disabled=(pos_in_filter == 0)):
                st.session_state["current_idx"] = filtered_indices[pos_in_filter - 1]
                st.rerun()
        with nav3:
            chosen_idx = st.selectbox(
                "Chọn nhanh cửa sổ trong danh sách lọc",
                options=filtered_indices,
                index=pos_in_filter,
                label_visibility="collapsed",
                format_func=lambda idx: (
                    f"Cửa sổ #{idx+1}/{total_rows} | Track #{df.loc[idx, 'track_id']} | "
                    f"{int(df.loc[idx, 't_start_ms'])/1000:.0f}s-{int(df.loc[idx, 't_end_ms'])/1000:.0f}s | "
                    f"Zone {df.loc[idx, 'zone_id']} | "
                    f"[{to_display_label(df.loc[idx, 'participation_label']) if df.loc[idx, 'participation_label'] in CANONICAL_TRAINING_LABELS else df.loc[idx, 'annotation_state']}] "
                    f"({df.loc[idx, 'label_source'] or 'NO_SRC'})"
                ),
            )
            if chosen_idx != curr_i:
                curr_i = chosen_idx
                st.session_state["current_idx"] = curr_i
                st.rerun()
        with nav4:
            if st.button("▶️ Tiếp", use_container_width=True, disabled=(pos_in_filter + 1 >= len(filtered_indices))):
                st.session_state["current_idx"] = filtered_indices[pos_in_filter + 1]
                st.rerun()
        with nav5:
            if st.button("⏭️ Tới Chưa Duyệt", use_container_width=True, help="Nhảy tới cửa sổ WEAK_RULE hoặc chưa gán tiếp theo"):
                unrev = [
                    i for i in filtered_indices
                    if i > curr_i and df.loc[i, "label_source"] not in ("HUMAN", "ADJUDICATED")
                ]
                if not unrev:
                    unrev = [
                        i for i in filtered_indices
                        if df.loc[i, "label_source"] not in ("HUMAN", "ADJUDICATED")
                    ]
                if unrev:
                    st.session_state["current_idx"] = unrev[0]
                    st.rerun()
                else:
                    st.toast("🎉 Tất cả cửa sổ trong bộ lọc đã được duyệt!", icon="✅")

        # Current Window Metadata
        row = df.loc[curr_i]
        t_start = int(row.get("t_start_ms", 0))
        t_end = int(row.get("t_end_ms", 4000))
        track_id = int(row.get("track_id", 1))
        cur_state = str(row.get("annotation_state", "VALID")).strip()
        cur_label = str(row.get("participation_label", "")).strip()
        cur_source = str(row.get("label_source", "")).strip()
        cur_phase = str(row.get("class_phase", "lecture")).strip()
        cur_zone = str(row.get("zone_id", "M-C")).strip()

        # Compute telemetry for current window
        telemetry = compute_window_telemetry(
            poses_df=poses_df,
            track_id=track_id,
            t_start_ms=t_start,
            t_end_ms=t_end,
            class_phase=cur_phase,
        )

        # Window Header Banner
        disp_lbl = to_display_label(cur_label) if cur_label in CANONICAL_TRAINING_LABELS else cur_state
        pill_cls = (
            "pill-high" if cur_label == HIGH
            else ("pill-mod" if cur_label == MODERATE else ("pill-low" if cur_label == LOW else "pill-neutral"))
        )
        st.markdown(
            f"""
            <div class="window-banner">
                <span style="font-size:1.15rem; font-weight:700;">
                    🎯 Cửa sổ #{curr_i + 1} / {total_rows} &nbsp;•&nbsp; Sinh viên: <code>Track #{track_id}</code>
                </span>
                &nbsp;&nbsp;
                <span class="{pill_cls}">{disp_lbl}</span>
                &nbsp;
                <span class="pill-neutral">Nguồn: {cur_source or 'N/A'}</span>
                &nbsp;
                <span class="pill-neutral">Vùng: {cur_zone}</span>
                &nbsp;
                <span class="pill-neutral">Pha: {cur_phase}</span>
                &nbsp;
                <span style="float:right; font-family:monospace; font-size:1.0rem; color:#93C5FD;">
                    ⏱️ {t_start/1000:.1f}s ➔ {t_end/1000:.1f}s ({(t_end - t_start)/1000:.1f}s)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # PART A: 5-FRAME TEMPORAL FILMSTRIP ACROSS THE 4-SECOND WINDOW
        # -------------------------------------------------------------
        st.markdown("##### 🎞️ Dải Phim Diễn Biến Hành Vi Trong Toàn Bộ Cửa Sổ 4 Giây (Temporal Sequence)")
        filmstrip = render_temporal_filmstrip(
            video_path=video_path,
            t_start_ms=t_start,
            t_end_ms=t_end,
            track_id=track_id,
            poses_df=poses_df,
            num_steps=opt_filmstrip_count,
            draw_skel=opt_draw_skeleton,
            blur_face=opt_blur_face,
            canonical_label=cur_label,
        )

        if filmstrip:
            f_cols = st.columns(len(filmstrip))
            for idx_f, (sec_val, img_rgb, vis_val) in enumerate(filmstrip):
                with f_cols[idx_f]:
                    vis_str = f" | Vis: {vis_val*100:.0f}%" if vis_val is not None else ""
                    st.image(
                        img_rgb,
                        caption=f"Mốc {sec_val:.1f}s (+{sec_val - t_start/1000:.1f}s){vis_str}",
                        use_container_width=True,
                    )
        else:
            st.info("💡 Chưa đọc được frame từ video. Hãy kiểm tra đường dẫn `data/raw/session_001/cam_A_pilot.mp4`.")

        # -------------------------------------------------------------
        # PART B: LEFT (TELEMETRY & CONTEXT) | RIGHT (1-CLICK & FORM)
        # -------------------------------------------------------------
        col_left, col_right = st.columns([1.05, 1.15])

        with col_left:
            st.markdown("##### 📡 Thông Số Khung Xương Quan Sát Được (Pose Telemetry)")
            if telemetry.get("available"):
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("👁️ Visible Ratio", f"{telemetry['mean_vis']*100:.0f}%", f"{telemetry['num_frames']} frames")
                tc2.metric("✋ Giơ Tay", f"{telemetry['hand_raised_ratio']*100:.0f}%")
                tc3.metric("🙇 Cúi Đầu", f"{telemetry['head_down_ratio']*100:.0f}%")
                tc4.metric("✍️ Cổ Tay Dịch Chuyển", f"{telemetry['wrist_motion']:.1f} px")

                st.markdown(
                    f"""
                    <div class="rule-explain-box">
                        🤖 <b>Giải thích từ RuleBot ({RULE_VERSION}):</b><br>
                        {telemetry['reason']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Không có dữ liệu pose chi tiết cho cửa sổ này.")

            # Classroom Context & Playable 4s Clip sub-tabs
            ctx_tab1, ctx_tab2 = st.tabs([
                "🏫 Toàn Cảnh Lớp Học & Lưới Vùng 3x3",
                "🎬 Xem Video Clip Chuyển Động 4 Giây",
            ])
            with ctx_tab1:
                ctx_rgb = render_classroom_context_frame(
                    video_path=video_path,
                    timestamp_ms=t_start + (t_end - t_start) // 2,
                    target_track_id=track_id,
                    poses_df=poses_df,
                    draw_grid=opt_draw_grid,
                    draw_skel=opt_draw_skeleton,
                    blur_face=opt_blur_face,
                )
                if ctx_rgb is not None:
                    st.image(
                        ctx_rgb,
                        caption=f"Vị trí của Track #{track_id} trong lớp học (ô {cur_zone}) tại {(t_start + t_end)/2000:.1f}s",
                        use_container_width=True,
                    )
            with ctx_tab2:
                st.caption("Nhấn nút bên dưới để cắt và phát trực tiếp đoạn video 4s của sinh viên này.")
                if st.button("▶️ Tạo & Phát Video Clip 4s Của Cửa Sổ Này", use_container_width=True):
                    tmp_clip = Path(tempfile.gettempdir()) / f"preview_trk{track_id}_{t_start}_{t_end}.mp4"
                    ok = export_window_clip_mp4(
                        video_path=video_path,
                        output_path=tmp_clip,
                        t_start_ms=t_start,
                        t_end_ms=t_end,
                        track_id=track_id,
                        poses_df=poses_df,
                        crop_person=True,
                        draw_skel=opt_draw_skeleton,
                        blur_face=opt_blur_face,
                    )
                    if ok and tmp_clip.exists():
                        st.video(str(tmp_clip))
                    else:
                        st.error("Không thể tạo video clip.")

        with col_right:
            st.markdown("##### ⚡ Gán Nhãn Siêu Tốc 1-Click (Tự Động Lưu & Sang Cửa Sổ Kế Tiếp)")
            st.caption("Chỉ cần bấm 1 nút dưới đây: Hệ thống tự động gắn `label_source=HUMAN`, kiểm tra contract, lưu CSV và chuyển sang cửa sổ tiếp theo!")

            cur_beh_list = [b.strip() for b in str(row.get("atomic_behavior", "")).split(";") if b.strip()]
            cur_conf_val = float(row.get("annotation_confidence", "0.90") or 0.90)
            cur_occ_val = str(row.get("occlusion_level", "none") or "none")
            cur_notes_val = str(row.get("notes", ""))

            qcol1, qcol2, qcol3 = st.columns(3)
            with qcol1:
                if st.button("🟢 HIGH (Lưu & Tiếp)", use_container_width=True, type="primary" if cur_label == HIGH else "secondary"):
                    beh = cur_beh_list if cur_beh_list else ["BOARD_OR_INSTRUCTOR_FACING"]
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, "VALID", HIGH, beh, "HUMAN", "",
                        max(0.85, cur_conf_val), cur_occ_val, cur_phase, cur_zone,
                        annotator_name, cur_notes_val, filtered_indices, advance=True,
                    )
                    if ok:
                        st.toast(msg_s, icon="🟢")
                        st.rerun()
            with qcol2:
                if st.button("🟡 MODERATE (Lưu & Tiếp)", use_container_width=True, type="primary" if cur_label == MODERATE else "secondary"):
                    beh = cur_beh_list if cur_beh_list else ["READING_WRITING"]
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, "VALID", MODERATE, beh, "HUMAN", "",
                        max(0.80, cur_conf_val), cur_occ_val, cur_phase, cur_zone,
                        annotator_name, cur_notes_val, filtered_indices, advance=True,
                    )
                    if ok:
                        st.toast(msg_s, icon="🟡")
                        st.rerun()
            with qcol3:
                if st.button("🔴 LOW (Lưu & Tiếp)", use_container_width=True, type="primary" if cur_label == LOW else "secondary"):
                    beh = cur_beh_list if cur_beh_list else ["HEAD_DOWN"]
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, "VALID", LOW, beh, "HUMAN", "",
                        max(0.80, cur_conf_val), cur_occ_val, cur_phase, cur_zone,
                        annotator_name, cur_notes_val, filtered_indices, advance=True,
                    )
                    if ok:
                        st.toast(msg_s, icon="🔴")
                        st.rerun()

            qcol4, qcol5, qcol6 = st.columns(3)
            with qcol4:
                if st.button("✅ Duyệt Nhãn Máy & Tiếp", use_container_width=True, help="Giữ nguyên nhãn RuleBot gợi ý và xác nhận thành HUMAN"):
                    target_lbl = cur_label if cur_label in CANONICAL_TRAINING_LABELS else (telemetry.get("rule_label") or MODERATE)
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, cur_state, target_lbl, cur_beh_list, "HUMAN", "",
                        cur_conf_val, cur_occ_val, cur_phase, cur_zone,
                        annotator_name, cur_notes_val, filtered_indices, advance=True,
                    )
                    if ok:
                        st.toast(msg_s, icon="✅")
                        st.rerun()
            with qcol5:
                if st.button("🌫️ OCCLUDED (Lưu & Tiếp)", use_container_width=True):
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, "OCCLUDED", "", ["OCCLUDED"], "", "",
                        cur_conf_val, "heavy", cur_phase, cur_zone,
                        annotator_name, cur_notes_val, filtered_indices, advance=True,
                    )
                    if ok:
                        st.toast(msg_s, icon="🌫️")
                        st.rerun()
            with qcol6:
                if st.button("🔄 TRANSITION (Lưu & Tiếp)", use_container_width=True):
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, "TRANSITION", "", ["BODY_TURNED"], "", "",
                        cur_conf_val, cur_occ_val, cur_phase, cur_zone,
                        annotator_name, cur_notes_val, filtered_indices, advance=True,
                    )
                    if ok:
                        st.toast(msg_s, icon="🔄")
                        st.rerun()

            st.divider()
            st.markdown("##### 📝 Biểu Mẫu Chỉnh Sửa Chi Tiết (Detailed Annotation Editor)")

            # Dynamic widget keys per row index so switching windows updates all inputs cleanly
            rk = f"r{curr_i}"
            new_state = st.radio(
                "1. Trạng thái quan sát (annotation_state)",
                options=list(ANNOTATION_STATES),
                index=list(ANNOTATION_STATES).index(cur_state) if cur_state in ANNOTATION_STATES else 0,
                horizontal=True,
                key=f"{rk}_state",
            )

            new_behaviors = st.multiselect(
                "2. Hành vi nguyên tử quan sát được (atomic_behavior)",
                options=ATOMIC_BEHAVIORS_LIST,
                default=[b for b in cur_beh_list if b in ATOMIC_BEHAVIORS_LIST],
                format_func=lambda b: f"{b} ({BEHAVIOR_DESCRIPTIONS_VI.get(b, '')})",
                key=f"{rk}_beh",
            )

            is_valid_now = new_state == "VALID"
            if is_valid_now:
                lbl_idx = list(CANONICAL_TRAINING_LABELS).index(cur_label) if cur_label in CANONICAL_TRAINING_LABELS else 1
                new_label = st.selectbox(
                    "3. Nhãn Mức độ Tham gia Chuẩn (participation_label)",
                    options=list(CANONICAL_TRAINING_LABELS),
                    index=lbl_idx,
                    format_func=lambda c: f"{to_display_label(c)} — {c}",
                    key=f"{rk}_label",
                )
                src_options = list(LABEL_SOURCES)
                default_src = "HUMAN" if cur_source not in src_options else cur_source
                new_source = st.selectbox(
                    "4. Nguồn gốc Nhãn (label_source)",
                    options=src_options,
                    index=src_options.index(default_src),
                    key=f"{rk}_src",
                )
            else:
                st.info(
                    f"ℹ️ State đang chọn là `{new_state}`: Theo quy định `src/contracts/labels.py`, "
                    "các cột `participation_label`, `label_source`, `label_rule_version` sẽ tự động để trống."
                )
                new_label = ""
                new_source = ""

            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            with dcol1:
                new_conf = st.slider(
                    "Confidence",
                    0.0,
                    1.0,
                    float(min(1.0, max(0.0, cur_conf_val))),
                    0.05,
                    key=f"{rk}_conf",
                )
            with dcol2:
                new_occ = st.selectbox(
                    "Occlusion",
                    options=OCCLUSION_LEVELS,
                    index=OCCLUSION_LEVELS.index(cur_occ_val) if cur_occ_val in OCCLUSION_LEVELS else 0,
                    key=f"{rk}_occ",
                )
            with dcol3:
                new_phase = st.selectbox(
                    "Class Phase",
                    options=CLASS_PHASES,
                    index=CLASS_PHASES.index(cur_phase) if cur_phase in CLASS_PHASES else 0,
                    key=f"{rk}_phase",
                )
            with dcol4:
                new_zone = st.selectbox(
                    "Zone ID (3x3)",
                    options=ZONE_IDS,
                    index=ZONE_IDS.index(cur_zone) if cur_zone in ZONE_IDS else 4,
                    key=f"{rk}_zone",
                )

            new_notes = st.text_input("Ghi chú (notes)", value=cur_notes_val, key=f"{rk}_notes")

            b_save1, b_save2 = st.columns(2)
            with b_save1:
                if st.button("💾 Lưu Chỉnh Sửa & Sang Cửa Sổ Tiếp Theo", type="primary", use_container_width=True):
                    rule_v = RULE_VERSION if new_source == "WEAK_RULE" else ""
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, new_state, new_label, new_behaviors,
                        new_source if new_source != "WEAK_RULE" else "HUMAN", "",
                        new_conf, new_occ, new_phase, new_zone, annotator_name, new_notes,
                        filtered_indices, advance=True,
                    )
                    if ok:
                        st.toast(msg_s, icon="💾")
                        st.rerun()
                    else:
                        st.error(f"❌ Lỗi Contract: {msg_s}")
            with b_save2:
                if st.button("📌 Chỉ Lưu Tại Chỗ (Không Chuyển Dòng)", use_container_width=True):
                    rule_v = RULE_VERSION if new_source == "WEAK_RULE" else ""
                    ok, msg_s = save_row_and_advance(
                        df, curr_i, active_csv_path, new_state, new_label, new_behaviors,
                        new_source, rule_v,
                        new_conf, new_occ, new_phase, new_zone, annotator_name, new_notes,
                        filtered_indices, advance=False,
                    )
                    if ok:
                        st.toast(msg_s, icon="📌")
                        st.rerun()
                    else:
                        st.error(f"❌ Lỗi Contract: {msg_s}")

            # Batch Operations Expander
            with st.expander("⚡ Công cụ Duyệt Hàng Loạt (Batch Confirm cho Track / Bộ lọc hiện tại)"):
                st.caption(
                    f"Hiện đang có **{len(filtered_indices)} cửa sổ** trong bộ lọc. "
                    "Bạn có thể xác nhận hàng loạt các nhãn máy (`WEAK_RULE` ➔ `HUMAN`) hoặc gán đồng loạt."
                )
                batch_col1, batch_col2 = st.columns(2)
                with batch_col1:
                    if st.button(f"✅ Duyệt toàn bộ {len(filtered_indices)} cửa sổ trong Bộ lọc thành HUMAN", use_container_width=True):
                        cnt_b = 0
                        for idx_b in filtered_indices:
                            if df.loc[idx_b, "annotation_state"] == "VALID" and df.loc[idx_b, "participation_label"] in CANONICAL_TRAINING_LABELS:
                                df.loc[idx_b, "label_source"] = "HUMAN"
                                df.loc[idx_b, "label_rule_version"] = ""
                                df.loc[idx_b, "annotator_id"] = annotator_name
                                cnt_b += 1
                        df.to_csv(active_csv_path, index=False, encoding="utf-8")
                        st.success(f"Đã xác nhận HUMAN cho {cnt_b} cửa sổ!")
                        st.rerun()
                with batch_col2:
                    batch_lbl = st.selectbox(
                        "Hoặc gán đồng loạt nhãn cho Bộ lọc:",
                        options=list(CANONICAL_TRAINING_LABELS),
                        format_func=to_display_label,
                    )
                    if st.button(f"🏷️ Gán {to_display_label(batch_lbl)} cho các dòng VALID đang lọc", use_container_width=True):
                        cnt_b = 0
                        for idx_b in filtered_indices:
                            if df.loc[idx_b, "annotation_state"] == "VALID":
                                df.loc[idx_b, "participation_label"] = batch_lbl
                                df.loc[idx_b, "label_source"] = "HUMAN"
                                df.loc[idx_b, "label_rule_version"] = ""
                                df.loc[idx_b, "annotator_id"] = annotator_name
                                cnt_b += 1
                        df.to_csv(active_csv_path, index=False, encoding="utf-8")
                        st.success(f"Đã gán {to_display_label(batch_lbl)} cho {cnt_b} cửa sổ!")
                        st.rerun()

    # =================================================================
    # TAB 3: ĐỐI SOÁT CHÉO 2 ANNOTATOR & ADJUDICATION (TASK T12)
    # =================================================================
    with tab_double_qa:
        st.subheader("🤝 Trạm 3: Đối Soát Chéo (Double Annotation QA) & Giải Quyết Bất Đồng (Task T12)")
        st.markdown(
            "Theo quy định tại `docs/annotation_guide.md` (Mục 9) và **Task T12**: Hai thành viên gán độc lập ít nhất 15–20% dữ liệu, "
            "báo cáo chỉ số **Cohen's Kappa ($\\kappa$)**, và giải quyết các dòng bất đồng thành nhãn `ADJUDICATED`."
        )

        qa_c1, qa_c2, qa_c3 = st.columns(3)
        with qa_c1:
            file_a_str = st.text_input("📄 File Annotator A", value=str(active_csv_path.as_posix()))
        with qa_c2:
            file_b_str = st.text_input("📄 File Annotator B (hoặc Weak Labels)", value="data/annotations/session_001_weak_labels.csv")
        with qa_c3:
            adj_out_str = st.text_input("💾 File Lưu Kết Quả Adjudicated", value="data/annotations/session_001_adjudicated.csv")

        path_a = Path(file_a_str.strip())
        path_b = Path(file_b_str.strip())

        if path_a.exists() and path_b.exists():
            df_a = pd.read_csv(path_a, dtype=str).fillna("")
            df_b = pd.read_csv(path_b, dtype=str).fillna("")

            key_cols = ["session_id", "camera_id", "track_id", "t_start_ms", "t_end_ms"]
            merged = pd.merge(df_a, df_b, on=key_cols, suffixes=("_A", "_B"))

            if merged.empty:
                st.warning("Hai file CSV không có cửa sổ nào trùng khớp khóa `(session_id, camera_id, track_id, t_start_ms, t_end_ms)`.")
            else:
                # 1. State agreement
                states_a = merged["annotation_state_A"].tolist()
                states_b = merged["annotation_state_B"].tolist()
                state_agree = float(np.mean([a == b for a, b in zip(states_a, states_b)])) * 100.0
                kappa_state = compute_cohens_kappa(states_a, states_b, list(ANNOTATION_STATES))

                # 2. Label agreement on mutually VALID windows
                both_valid = merged[
                    (merged["annotation_state_A"] == "VALID")
                    & (merged["annotation_state_B"] == "VALID")
                    & (merged["participation_label_A"].isin(CANONICAL_TRAINING_LABELS))
                    & (merged["participation_label_B"].isin(CANONICAL_TRAINING_LABELS))
                ]

                if not both_valid.empty:
                    lbls_a = both_valid["participation_label_A"].tolist()
                    lbls_b = both_valid["participation_label_B"].tolist()
                    lbl_agree = float(np.mean([a == b for a, b in zip(lbls_a, lbls_b)])) * 100.0
                    kappa_lbl = compute_cohens_kappa(lbls_a, lbls_b, list(CANONICAL_TRAINING_LABELS))
                else:
                    lbl_agree = 0.0
                    kappa_lbl = 0.0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("🔗 Số Cửa Sổ Chung", f"{len(merged)}")
                k2.metric("✅ Đồng Thuận State", f"{state_agree:.1f}%", f"Kappa = {kappa_state:.3f}")
                k3.metric("🎯 Đồng Thuận Nhãn (VALID)", f"{lbl_agree:.1f}%", f"N = {len(both_valid)} cửa sổ")
                k4.metric("📐 Cohen's Kappa (3-Class)", f"{kappa_lbl:.3f}", "Đạt chuẩn ≥ 0.70" if kappa_lbl >= 0.70 else "Cần review bất đồng")

                if not both_valid.empty:
                    st.markdown("##### 🔢 Ma Trận Nhầm Lẫn Giữa Annotator A (Hàng) và Annotator B (Cột)")
                    cm_df = pd.crosstab(
                        both_valid["participation_label_A"].map(to_display_label),
                        both_valid["participation_label_B"].map(to_display_label),
                        rownames=["Annotator A"],
                        colnames=["Annotator B"],
                    )
                    st.dataframe(cm_df, use_container_width=True)

                # Disagreements list
                disagreements = merged[
                    (merged["annotation_state_A"] != merged["annotation_state_B"])
                    | (merged["participation_label_A"] != merged["participation_label_B"])
                ]
                st.markdown(f"##### ⚖️ Danh Sách Cửa Sổ Bất Đồng ({len(disagreements)} dòng)")
                if disagreements.empty:
                    st.success("🎉 Tuyệt vời! Hai file hoàn toàn đồng thuận 100% trên mọi cửa sổ!")
                else:
                    st.dataframe(
                        disagreements[[
                            "track_id", "t_start_ms", "t_end_ms", "class_phase_A",
                            "annotation_state_A", "participation_label_A",
                            "annotation_state_B", "participation_label_B",
                        ]].head(50),
                        use_container_width=True,
                    )
                    if st.button("⚖️ Hợp Nhất & Xuất File ADJUDICATED (Ưu tiên Annotator A cho dòng bất đồng)", type="primary"):
                        adj_df = df_a.copy()
                        for idx_r in adj_df.index:
                            if adj_df.loc[idx_r, "annotation_state"] == "VALID" and adj_df.loc[idx_r, "participation_label"] in CANONICAL_TRAINING_LABELS:
                                adj_df.loc[idx_r, "label_source"] = "ADJUDICATED"
                                adj_df.loc[idx_r, "label_rule_version"] = ""
                                adj_df.loc[idx_r, "annotator_id"] = annotator_name
                        out_adj_p = Path(adj_out_str.strip())
                        out_adj_p.parent.mkdir(parents=True, exist_ok=True)
                        adj_df.to_csv(out_adj_p, index=False, encoding="utf-8")
                        st.success(f"✅ Đã lưu bộ nhãn chuẩn `ADJUDICATED` tại `{out_adj_p}`!")

    # =================================================================
    # TAB 4: KIỂM ĐỊNH CHUẨN SCHEMA, THỐNG KÊ & XUẤT BÁO CÁO
    # =================================================================
    with tab_analytics:
        st.subheader("📊 Trạm 4: Kiểm Định Hợp Lệ Schema 17 Cột & Phân Tích Dataset")

        if not active_csv_path.exists():
            st.warning("Chưa tìm thấy file CSV.")
            st.stop()

        is_valid_csv, err_list, stats_counter = validate_annotation_file(active_csv_path)

        if is_valid_csv:
            st.success(
                f"✅ **PASS 100% CONTRACT VALIDATION:** File `{active_csv_path.name}` tuân thủ tuyệt đối "
                "17 cột chuẩn trong `docs/dataset_schema.md` và `src/contracts/labels.py`!"
            )
        else:
            st.error(f"❌ Phát hiện {len(err_list)} lỗi vi phạm schema/contract:")
            for err_item in err_list[:15]:
                st.write(f"- `{err_item}`")

        df_stat = pd.read_csv(active_csv_path, dtype=str).fillna("")
        tot_w = max(1, len(df_stat))

        sc1, sc2, sc3, sc4 = st.columns(4)
        h_n = stats_counter.get(f"label_{HIGH}", 0)
        m_n = stats_counter.get(f"label_{MODERATE}", 0)
        l_n = stats_counter.get(f"label_{LOW}", 0)
        occ_n = stats_counter.get("state_OCCLUDED", 0)

        sc1.metric("🟢 HIGH_OBSERVED", f"{h_n} ({h_n/tot_w*100:.1f}%)")
        sc2.metric("🟡 MODERATE_OBSERVED", f"{m_n} ({m_n/tot_w*100:.1f}%)")
        sc3.metric("🔴 LOW_OBSERVED", f"{l_n} ({l_n/tot_w*100:.1f}%)")
        sc4.metric("🌫️ OCCLUDED / Khác", f"{occ_n} ({occ_n/tot_w*100:.1f}%)")

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown("##### 📊 Phân Bố Nhãn Theo Vùng Lớp Học 3x3 (`zone_id`)")
            zone_ct = pd.crosstab(df_stat["zone_id"], df_stat["participation_label"].replace("", "NON_VALID"))
            st.dataframe(zone_ct, use_container_width=True)

        with ch_col2:
            st.markdown("##### 🏷️ Phân Bố Theo Nguồn Nhãn (`label_source`) & Pha Học (`class_phase`)")
            src_ct = pd.crosstab(df_stat["class_phase"], df_stat["label_source"].replace("", "NONE"))
            st.dataframe(src_ct, use_container_width=True)

        st.markdown("##### 📋 Xem Toàn Bộ Bảng Dữ Liệu Annotation CSV (17 Cột Chuẩn)")
        st.dataframe(df_stat, use_container_width=True, height=280)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label=f"📥 Tải File Annotation CSV Chuẩn ({active_csv_path.name})",
                data=active_csv_path.read_bytes(),
                file_name=active_csv_path.name,
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with dl2:
            summary_payload = {
                "file": active_csv_path.name,
                "annotation_guide_version": "v0.2",
                "is_valid_contract": is_valid_csv,
                "total_windows": len(df_stat),
                "unique_tracks": int(df_stat["track_id"].nunique()),
                "counts": dict(stats_counter),
            }
            st.download_button(
                label="📥 Tải Báo Cáo Kiểm Định QA (annotation_qa_summary.json)",
                data=json.dumps(summary_payload, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="annotation_qa_summary.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
