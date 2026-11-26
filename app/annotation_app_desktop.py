#!/usr/bin/env python3
"""Native Windows desktop conversion of `app/annotation_app.py` (.exe-ready).

Replicates the full Streamlit independent-annotation tool as a fast Tkinter app:

1. Person picker (Phong / Tài / Quân) writing to the same `annotator_a/b.csv`.
2. Tab "Studio (Chọn nhanh)": whole-window quick labels with AI suggestions and
   ~15s train span propagation — same CSV effects as `app/desktop_studio.py`.
3. Tab "Rubric (4 khoảng)": the strict 4-time-bin evidence rubric from
   `app/annotation_app.py`, validated with `validate_annotation_dataframe` and
   saved via `save_annotation_row` (identical schema, `label_source=HUMAN`).

This module intentionally avoids importing `app.studio_view` /
`app.annotation_app` so the packaged executable does not bundle Streamlit.
The save/rubric helpers below are faithful copies of the referenced logic.

Run:  python app/annotation_app_desktop.py
Build: python -m PyInstaller --onefile --name AnnotationAppDesktop app/annotation_app_desktop.py
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import threading
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

def _resolve_root() -> Path:
    """Resolve the project root when running either as a script or a frozen exe.

    Priority: ``app_root.txt`` next to the exe → folder containing the exe →
    script's parent directory. A valid root must contain ``data/annotations``.
    """
    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        marker = exe_dir / "app_root.txt"
        if marker.is_file():
            try:
                candidates.append(Path(marker.read_text(encoding="utf-8").strip().strip('"')))
            except OSError:
                pass
        candidates.append(exe_dir)
        candidates.append(exe_dir.parent)
    candidates.append(Path(__file__).resolve().parents[1])
    for cand in candidates:
        if (cand / "data" / "annotations").is_dir():
            return cand
    return Path(__file__).resolve().parents[1]


ROOT = _resolve_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_qa import validate_annotation_dataframe
from src.contracts.annotation_io import read_annotation_csv, save_annotation_row
from src.contracts.annotation_rubric import EVIDENCE_OPTIONS, RUBRIC_VERSION, evidence_label
from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    HIGH,
    LOW,
    MODERATE,
)
from src.pose.annotation_preview import compose_target_frame, nearest_frame_boxes

logger = logging.getLogger("annotation_app_desktop")

# ---------------------------------------------------------------------------
# Constants (mirrors of app/annotation_app.py + app/desktop_studio.py)
# ---------------------------------------------------------------------------

PEOPLE = {
    "Phong": ("Vũ Văn Phong", "annotator_b.csv"),
    "Tài": ("Vũ Đức Tài", "annotator_a.csv"),
    "Quân": ("Nguyễn Hữu Quân", "annotator_a.csv"),
}
DATASET_NAMES = {
    "review_v03_independent_20261124": "600 cửa sổ · phiếu độc lập Phong và Tài",
    "train_full_ai_assist": "Tập TRAIN đầy đủ (9.888 cửa sổ · class_1..5, session_001..002)",
    "val_full_ai_assist": "Tập VAL đầy đủ (8.617 cửa sổ · session_003..004)",
    "test_full_ai_assist": "Tập TEST đầy đủ (4.889 cửa sổ · session_005..006)",
    "review_v03": "Bộ gán nhãn v0.3 (600 cửa sổ YOLO11)",
    "human_review_v0_2": "Dataset cũ · bộ gán nhãn chính",
    "external_review_20261109": "Hai video mới · đánh giá bổ sung",
}
DATASET_PRIORITY = {
    "review_v03_independent_20261124": 0,
    "train_full_ai_assist": 1,
    "val_full_ai_assist": 2,
    "test_full_ai_assist": 3,
    "human_review_v0_2": 4,
    "external_review_20261109": 5,
}

PHASES = {
    "UNKNOWN": "Chưa xác định",
    "lecture": "Giảng bài",
    "individual_work": "Làm bài cá nhân",
    "discussion": "Thảo luận",
}
STATES = {
    "VALID": "Đủ bằng chứng — gán mức tham gia",
    "OCCLUDED": "Bị che khuất",
    "TRANSITION": "Đang chuyển hoạt động/trạng thái",
    "INVALID": "Không đủ bằng chứng / chưa rõ nhiệm vụ / lỗi ID hoặc video",
}
LABELS = dict(zip(CANONICAL_TRAINING_LABELS, ["Cao", "Trung bình", "Thấp"]))

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

PHASE_BASIS_DEFAULT = {
    "lecture": [
        "Quan sát thấy hoạt động trình bày bài học trước lớp",
        "Quan sát thấy giáo viên đang giảng bài trước lớp",
    ],
    "individual_work": [
        "Quan sát được yêu cầu làm bài cá nhân",
        "Đã đối chiếu lịch hoạt động: làm bài cá nhân",
    ],
    "discussion": [
        "Quan sát được yêu cầu thảo luận nhóm",
        "Đã đối chiếu lịch hoạt động: thảo luận nhóm",
    ],
}

ZONE_OPTIONS = ["", "F-L", "F-C", "F-R", "M-L", "M-C", "M-R", "B-L", "B-C", "B-R", "FRONT", "MIDDLE", "BACK"]
OCCLUSION_OPTIONS = ["none", "partial", "heavy"]
STATE_OPTIONS = ["VALID", "OCCLUDED", "TRANSITION", "INVALID"]
PHASE_OPTIONS = ["lecture", "individual_work", "discussion", "UNKNOWN"]

EV_DISPLAY = [f"{code} — {name}" for code, name in EVIDENCE_OPTIONS.items()]
EV_FROM_DISPLAY = {f"{code} — {name}": code for code, name in EVIDENCE_OPTIONS.items()}
STATE_DISPLAY = [f"{code} — {name}" for code, name in STATES.items()]
STATE_FROM_DISPLAY = {f"{code} — {name}": code for code, name in STATES.items()}
PHASE_DISPLAY = [f"{code} — {name}" for code, name in PHASES.items()]
PHASE_FROM_DISPLAY = {f"{code} — {name}": code for code, name in PHASES.items()}
LABEL_DISPLAY = [f"{code} — {name}" for code, name in LABELS.items()]
LABEL_FROM_DISPLAY = {f"{code} — {name}": code for code, name in LABELS.items()}
OCC_DISPLAY = {"none": "Không", "partial": "Một phần", "heavy": "Nhiều"}

GUIDE_TEXT = """HƯỚNG DẪN CHỌN NHÃN
1. Xem đúng track ID trong toàn bộ khoảng thời gian, xác định pha hoạt động thực tế.
2. Chọn "Đủ bằng chứng (VALID)" khi có thể đánh giá; chọn che khuất/chuyển hoạt động/không đủ dữ liệu khi cần.
3. Với cửa sổ đủ bằng chứng: Cao = nhiều tín hiệu rõ phù hợp nhiệm vụ; Trung bình = tín hiệu yếu hoặc không liên tục;
   Thấp = nhiều tín hiệu không phù hợp nhiệm vụ trong phần lớn cửa sổ.

Cúi đầu khi đọc/viết có thể phù hợp; quay sang bạn khi thảo luận có thể phù hợp.
KHÔNG suy luận cảm xúc, ý định hoặc mức hiểu bài. Nếu chưa biết pha, xem thêm ngữ cảnh bằng video gốc; không đoán pha để ép nhãn.

CƠ SỞ GÁN NHÃN: HÀNH VI NÀO PHÙ HỢP VỚI TỪNG HOẠT ĐỘNG?
| Hoạt động        | Bằng chứng có thể phù hợp                                   | Chưa đủ để kết luận                    |
|------------------|--------------------------------------------------------------|----------------------------------------|
| Giảng bài        | Thân/đầu hướng vùng giảng dạy; ghi chép phù hợp bài đang diễn ra | Chỉ ngồi yên, chỉ nhìn thẳng, chỉ cúi đầu |
| Làm bài cá nhân  | Đọc tài liệu bài tập; tay viết/thao tác nhiệm vụ đã xác định  | Cúi đầu nhưng không thấy đang làm gì    |
| Thảo luận        | Hướng về nhóm và có cử chỉ/trao đổi liên quan nhiệm vụ        | Chỉ quay sang bạn; không biết nội dung   |

Không phù hợp cần hành vi cụ thể mâu thuẫn với nhiệm vụ đã biết. Không tự coi dùng điện thoại, quay đầu hoặc cúi đầu là
không phù hợp. Nếu chưa rõ, chọn "Nhìn thấy nhưng chưa rõ"; nếu bị che hoặc không tìm đúng người, chọn "Bị che / không xác định được".

BẢNG QUY ĐỔI THÍ NGHIỆM: ≥3/4 khoảng phù hợp → Cao; ≥3/4 khoảng không phù hợp → Thấp;
≥3/4 khoảng có bằng chứng rõ nhưng xen kẽ → Trung bình. Nếu ít hơn 3/4 khoảng rõ, không gán mức tham gia.
Đây là lựa chọn vận hành của nhóm để thử nghiệm, không phải ngưỡng khoa học đã được xác nhận.
"""


# ---------------------------------------------------------------------------
# Save helpers (faithful copies of app/studio_view.py logic, streamlit-free)
# ---------------------------------------------------------------------------

def _save_independent_window(
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
    primary_file_path: Path,
    phase_evidence: str,
    confirmed: bool,
    ai_suggestion_source: str = "",
    ai_suggestion: Optional[Dict[str, Any]] = None,
    span_source_window_id: str = "",
    span_seconds: float = 0.0,
    annotation_confidence: float = 0.8,
    annotation_evidence_override: Optional[str] = None,
) -> None:
    """Save one window in the current annotator's independent CSV."""
    if target_state == "VALID":
        if (
            phase == "UNKNOWN"
            or not phase_evidence.strip()
            or not behaviors.strip()
            or not confirmed
            or target_label not in CANONICAL_TRAINING_LABELS
        ):
            raise ValueError("Chưa đủ bằng chứng để lưu nhãn VALID.")

    if annotation_evidence_override is not None:
        evidence_payload: Dict[str, Any] = json.loads(annotation_evidence_override)
    else:
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
            evidence_payload["decision"] = (
                "ACCEPTED"
                if ai_suggestion
                and (
                    target_state == ai_suggestion.get("suggested_state")
                    and (target_label if target_state == "VALID" else "") == ai_suggestion.get("suggested_label", "")
                    and {b.strip() for b in behaviors.split(";") if b.strip()} == set(ai_suggestion.get("suggested_behaviors", []))
                    and phase == ai_suggestion.get("suggested_phase")
                )
                else "EDITED"
            )
        if span_source_window_id:
            evidence_payload["method"] = "TRACK_15S_PROPAGATED_HUMAN_REVIEW"
            evidence_payload["propagated_from_window_id"] = span_source_window_id
            evidence_payload["span_seconds"] = float(span_seconds)

    updates = {
        "annotation_state": target_state,
        "participation_label": target_label if target_state == "VALID" else "",
        "label_source": "HUMAN" if target_state == "VALID" else "",
        "label_rule_version": "",
        "annotator_id": annotator_id,
        "atomic_behavior": behaviors,
        "class_phase": phase,
        "zone_id": zone,
        "occlusion_level": occlusion,
        "notes": notes,
        "annotation_guide_version": "v0.2",
        "annotation_confidence": annotation_confidence,
        "phase_verified": bool(phase != "UNKNOWN" and phase_evidence.strip()),
        "phase_source": "HUMAN_WINDOW_REVIEW" if phase != "UNKNOWN" and phase_evidence.strip() else "UNKNOWN",
        "phase_evidence": phase_evidence.strip(),
        "annotation_evidence": json.dumps(evidence_payload, ensure_ascii=False),
    }
    save_annotation_row(primary_file_path, str(df.at[row_idx, "window_id"]), updates, fill_track_zone=True)


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
    t0_num = pd.to_numeric(df_rows["t_start_ms"], errors="coerce")
    t1_num = pd.to_numeric(df_rows["t_end_ms"], errors="coerce")
    match = (
        (df_rows.index != row_idx)
        & (~is_done_mask)
        & df_rows["split"].astype(str).str.lower().eq("train")
        & df_rows["session_id"].astype(str).eq(str(cur["session_id"]))
        & df_rows["camera_id"].astype(str).eq(str(cur.get("camera_id", "cam_A")))
        & df_rows["track_id"].astype(str).eq(str(cur["track_id"]))
        & (t0_num >= float(cur["t_start_ms"]))
        & (t1_num <= float(cur["t_start_ms"]) + float(max_span_ms) + 1e-6)
    )
    sub = df_rows[match].assign(_t0=t0_num[match]).sort_values("_t0")
    return [int(i) for i in sub.index]


# ---------------------------------------------------------------------------
# Media decoding (copy of app/desktop_studio.py, streamlit-free dependencies)
# ---------------------------------------------------------------------------

def extract_window_media_in_memory(
    video_path: Path,
    pose_df: pd.DataFrame,
    session_id: str,
    camera_id: str,
    track_id: str,
    t_start_ms: float,
    t_end_ms: float,
    *,
    anonymize: bool = False,
    skeleton: bool = True,
    n_loop_frames: int = 16,
) -> Tuple[List[np.ndarray], List[Tuple[str, np.ndarray]], str]:
    """Decode sampled window frames in a single OpenCV pass (no disk/ffmpeg overhead)."""
    if not video_path.is_file():
        return [], [], f"Thiếu video: {video_path}"
    if pose_df.empty:
        return [], [], "Thiếu dữ liệu pose CSV."

    scoped = pose_df
    if {"session_id", "camera_id"} <= set(pose_df.columns):
        scoped = pose_df[
            pose_df["session_id"].astype(str).eq(str(session_id))
            & pose_df["camera_id"].astype(str).eq(str(camera_id))
        ]
    scoped = scoped[scoped["timestamp_ms"].between(t_start_ms - 75.0, t_end_ms + 75.0)].copy()
    target = scoped[scoped["track_id"].astype(str).eq(str(track_id))]
    if target.empty:
        return [], [], f"Không thấy track #{track_id} trong khoảng {t_start_ms / 1000:.1f}–{t_end_ms / 1000:.1f}s."

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], f"Không mở được video: {video_path}"

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        left = float(target["bbox_x1"].min())
        top = float(target["bbox_y1"].min())
        right = float(target["bbox_x2"].max())
        bottom = float(target["bbox_y2"].max())
        pad_x, pad_y = (right - left) * 0.14, (bottom - top) * 0.10
        crop_box = (
            max(0, int(left - pad_x)),
            max(0, int(top - pad_y)),
            min(width, int(right + pad_x)),
            min(height, int(bottom + pad_y)),
        )

        timestamps = np.linspace(float(t_start_ms), float(t_end_ms), max(5, int(n_loop_frames)))
        strip_ts_indices = set(np.linspace(0, len(timestamps) - 1, 5).round().astype(int).tolist())

        loop_rgb: List[np.ndarray] = []
        strip_items: List[Tuple[str, np.ndarray]] = []
        matched = 0

        start_frame = max(0, round(float(timestamps[0]) * native_fps / 1000.0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cur_frame = start_frame

        for idx, ts in enumerate(timestamps):
            desired = max(0, round(float(ts) * native_fps / 1000.0))
            if desired - cur_frame > 30:
                cap.set(cv2.CAP_PROP_POS_FRAMES, desired)
                cur_frame = desired
            while cur_frame < desired:
                if not cap.grab():
                    break
                cur_frame += 1
            ok, bgr = cap.read()
            cur_frame += 1
            if not ok or bgr is None:
                continue

            boxes = nearest_frame_boxes(scoped, float(ts))
            canvas_bgr, found = compose_target_frame(
                bgr,
                boxes,
                track_id,
                crop_box,
                float(ts),
                anonymize=anonymize,
                skeleton=skeleton,
            )
            matched += int(found)
            canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
            loop_rgb.append(canvas_rgb)

            if idx in strip_ts_indices and len(strip_items) < 5:
                crop_rgb = canvas_rgb[48:552, 896:].copy()
                rel_s = (float(ts) - float(t_start_ms)) / 1000.0
                lbl = f"+{rel_s:.1f}s ({ts / 1000:.1f}s)"
                strip_items.append((lbl, crop_rgb))

        status = (
            f"Đã dựng {len(loop_rgb)} khung hình (thấy track #{track_id} ở {matched}/{len(loop_rgb)} khung)."
            if loop_rgb
            else f"Không đọc được khung hình cho track #{track_id}."
        )
        return loop_rgb, strip_items, status
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# AI suggestion I/O (faithful copies of src/pose/ai_bundle.py loaders)
#
# src.pose.ai_bundle imports torch at module level; the desktop exe must stay
# light, so only the suggestion-loading helpers are duplicated here. These are
# pure json/csv/numpy code — no torch. Source of truth: src/pose/ai_bundle.py.
# ---------------------------------------------------------------------------

VALID_BEHAVIOR_CODES = (
    "BOARD_OR_INSTRUCTOR_FACING",
    "READING_WRITING",
    "HAND_RAISED",
    "PEER_ORIENTED",
    "HEAD_DOWN_EXTENDED",
    "TURNING_AWAY",
    "OCCLUDED_KEYPOINTS",
    "UNCLEAR",
)


def _normalize_label(raw_label: Any) -> str:
    from src.contracts.labels import CANONICAL_BY_DISPLAY_LABEL

    val = str(raw_label or "").strip().upper()
    if not val:
        return ""
    if val in CANONICAL_TRAINING_LABELS:
        return val
    if val in CANONICAL_BY_DISPLAY_LABEL:
        return CANONICAL_BY_DISPLAY_LABEL[val]
    raise ValueError(f"Nhãn không hợp lệ trong gợi ý AI: {raw_label!r}")


def parse_and_validate_suggestions(
    raw_text: str,
    *,
    suggestion_source: str = "VISION_LLM_IMPORT",
) -> List[Dict[str, Any]]:
    """Parse JSON (even fenced inside Markdown from ChatGPT/Gemini) and validate fields."""
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    elif not text.startswith("[") and not text.startswith("{"):
        bracket = re.search(r"(\[[\s\S]*\])", text)
        if bracket:
            text = bracket.group(1)

    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("suggestions", [parsed])
    if not isinstance(parsed, list):
        raise ValueError("JSON gợi ý phải là một danh sách (array) các object theo window_id.")

    normalized: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Mỗi phần tử gợi ý phải là một JSON object.")
        wid = str(item.get("window_id", "")).strip()
        if not wid:
            raise ValueError("Thiếu window_id trong phần tử JSON.")
        state = str(item.get("suggested_state") or item.get("state") or "VALID").strip().upper()
        if state not in ANNOTATION_STATES:
            raise ValueError(f"Trạng thái không hợp lệ cho {wid}: {state!r}")

        raw_lbl = item.get("suggested_label") if "suggested_label" in item else item.get("label", "")
        label = _normalize_label(raw_lbl) if state == "VALID" else ""
        if state == "VALID" and label not in CANONICAL_TRAINING_LABELS:
            raise ValueError(f"Cửa sổ VALID {wid} thiếu nhãn hợp lệ (HIGH/MODERATE/LOW).")

        raw_behs = item.get("suggested_behaviors") if "suggested_behaviors" in item else item.get("behaviors", [])
        if isinstance(raw_behs, str):
            raw_behs = [b.strip() for b in raw_behs.split(";") if b.strip()]
        behaviors = [str(b).strip().upper() for b in (raw_behs or []) if str(b).strip()]

        if any(b not in VALID_BEHAVIOR_CODES for b in behaviors):
            raise ValueError(f"Hành vi không hợp lệ: {wid}")
        if state == "VALID" and not behaviors:
            raise ValueError(f"Thiếu hành vi cho gợi ý VALID: {wid}")
        confidence = float(item.get("confidence", 0.75))
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Confidence không hợp lệ: {wid}")
        phase = str(item.get("suggested_phase") or item.get("class_phase") or "UNKNOWN").strip()
        if phase not in ("lecture", "individual_work", "discussion", "UNKNOWN"):
            raise ValueError(f"Pha không hợp lệ: {wid}")
        normalized.append({
            "window_id": wid,
            "suggested_state": state,
            "suggested_label": label,
            "suggested_behaviors": behaviors,
            "suggested_phase": phase,
            "suggested_phase_evidence": str(
                item.get("suggested_phase_evidence")
                or item.get("phase_evidence")
                or ""
            ).strip(),
            "suggested_occlusion": str(item.get("suggested_occlusion") or item.get("occlusion_level") or ("heavy" if state == "OCCLUDED" else "none")).strip(),
            "rationale": str(item.get("rationale", "")).strip(),
            "rejection_reason": str(item.get("rejection_reason", "")).strip(),
            "confidence": confidence,
            "gate_reason": str(item.get("gate_reason", "VISION_LLM" if suggestion_source == "VISION_LLM_IMPORT" else "ACCEPTED")),
            "needs_vision_review": bool(item.get("needs_vision_review", state != "VALID")),
            "suggestion_source": str(item.get("suggestion_source") or suggestion_source),
            "model_id": str(item.get("model_id", "")),
            "bundle_image_path": str(item.get("bundle_image_path", "")),
        })
    return normalized


def _read_suggestions_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not csv_path.read_text(encoding="utf-8").strip():
        return {}
    df = pd.read_csv(csv_path, dtype=object, keep_default_na=False)
    if df.empty:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        wid = str(row.get("window_id", "")).strip()
        if not wid:
            continue
        if wid in out:
            raise ValueError(f"window_id trùng trong CSV gợi ý: {wid}")
        raw_behs = str(row.get("suggested_behaviors", ""))
        behs = [b.strip() for b in raw_behs.split(";") if b.strip()]
        raw_conf = str(row.get("confidence", "")).strip()
        conf = float(raw_conf) if raw_conf else 0.0
        raw_nvr = str(row.get("needs_vision_review", "")).strip().lower()
        out[wid] = {
            "window_id": wid,
            "suggested_state": str(row.get("suggested_state", "VALID")).strip(),
            "suggested_label": str(row.get("suggested_label", "")).strip(),
            "suggested_behaviors": behs,
            "suggested_phase": str(row.get("suggested_phase", "lecture")).strip(),
            "suggested_phase_evidence": str(row.get("suggested_phase_evidence", "")).strip(),
            "suggested_occlusion": str(row.get("suggested_occlusion", "none")).strip(),
            "confidence": conf,
            "gate_reason": str(row.get("gate_reason", "")).strip(),
            "needs_vision_review": raw_nvr in ("true", "1", "yes"),
            "suggestion_source": str(row.get("suggestion_source", "")).strip(),
            "rationale": str(row.get("rationale", "")).strip(),
            "rejection_reason": str(row.get("rejection_reason", "")).strip(),
            "model_id": str(row.get("model_id", "")).strip(),
            "bundle_image_path": str(row.get("bundle_image_path", "")).strip(),
        }
    validated = parse_and_validate_suggestions(json.dumps(list(out.values())))
    return {r["window_id"]: r for r in validated}


def load_suggestions_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load suggestions keyed by window_id from CSV or JSON (prefers newer sibling CSV)."""
    json_path = path.with_suffix(".json") if path.suffix == ".csv" else path
    csv_path = json_path.with_suffix(".csv")
    if csv_path.is_file() and (not json_path.is_file() or csv_path.stat().st_mtime_ns >= json_path.stat().st_mtime_ns):
        return _read_suggestions_csv(csv_path)
    if not json_path.is_file():
        return {}
    text = json_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    raw = json.loads(text)
    if isinstance(raw, list):
        return {str(item["window_id"]): item for item in raw if "window_id" in item}
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    return {}


# ---------------------------------------------------------------------------
# Controller (headless-testable state + persistence)
# ---------------------------------------------------------------------------

class AnnotationAppController:
    """State + persistence manager mirroring `app/annotation_app.py` behaviour."""

    ZONE_MEMORY_PATH = ROOT / "data" / "annotations" / "track_zone_memory.json"

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.pose_cache: Dict[Path, pd.DataFrame] = {}
        self.suggestions_cache: Dict[Path, Dict[str, Dict[str, Any]]] = {}
        self.media_cache: "OrderedDict[Tuple[str, bool, bool], Tuple[List[np.ndarray], List[Tuple[str, np.ndarray]], str]]" = OrderedDict()
        self.track_zone_memory: Dict[str, str] = {}
        self._zone_file_cache: Optional[Dict[str, str]] = None

        self.person_short: str = ""
        self.person_full: str = ""
        self.annotator_filename: str = ""

        self.dataset_path: Optional[Path] = None
        self.dataset_dir_name: str = ""
        self.df: pd.DataFrame = pd.DataFrame()
        self.filtered_indices: List[int] = []
        self.cur_pos: int = 0

    # -- person / dataset ---------------------------------------------------

    def set_person(self, short_name: str) -> List[Tuple[str, str, Path]]:
        """Return sorted (dir_name, label, csv_path) dataset candidates for the person."""
        full_name, filename = PEOPLE[short_name]
        self.person_short, self.person_full, self.annotator_filename = short_name, full_name, filename
        ann_root = self.root / "data" / "annotations"
        forms = sorted(
            ann_root.glob(f"**/{filename}"),
            key=lambda p: (DATASET_PRIORITY.get(p.parent.name, 6), str(p)),
        )
        return [(p.parent.name, DATASET_NAMES.get(p.parent.name, str(p.parent.relative_to(ann_root))), p) for p in forms]

    def load_dataset(self, csv_path: Path) -> pd.DataFrame:
        self.dataset_path = Path(csv_path)
        self.dataset_dir_name = self.dataset_path.parent.name
        self.df = read_annotation_csv(self.dataset_path)
        self.load_suggestions(force_reload=True)
        return self.df

    # -- masks / filters ----------------------------------------------------

    def done_mask(self) -> pd.Series:
        """Same completion rule as annotation_app.py's rubric tab."""
        if self.df.empty:
            return pd.Series(dtype=bool)
        return self.df["annotation_state"].isin(ANNOTATION_STATES) & self.df["annotator_id"].ne("")

    def load_suggestions(self, *, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        if self.dataset_path is None:
            return {}
        bundles_dir = self.dataset_path.parent / "ai_bundles"
        if force_reload or bundles_dir not in self.suggestions_cache:
            merged: Dict[str, Dict[str, Any]] = {}
            merged.update(load_suggestions_map(bundles_dir / "ai_suggestions.json"))
            merged.update(load_suggestions_map(bundles_dir / "codex_suggestions.json"))
            self.suggestions_cache[bundles_dir] = merged
        return self.suggestions_cache[bundles_dir]

    def get_pose_df(self, pose_path: Path) -> pd.DataFrame:
        if pose_path not in self.pose_cache:
            self.pose_cache[pose_path] = pd.read_csv(pose_path) if pose_path.is_file() else pd.DataFrame()
        return self.pose_cache[pose_path]

    def apply_filter(self, filter_mode: str, specific_session: str = "") -> List[int]:
        if self.df.empty:
            self.filtered_indices = []
            self.cur_pos = 0
            return []
        done = self.done_mask()
        if filter_mode == "Chưa gán (còn lại)":
            idxs = list(self.df.index[~done])
        elif filter_mode == "Đã gán":
            idxs = list(self.df.index[done])
        elif filter_mode == "Theo phiên học..." and specific_session:
            idxs = list(self.df.index[self.df["session_id"].astype(str).eq(specific_session)])
        else:
            idxs = list(self.df.index)
        self.filtered_indices = idxs
        unfin = [p for p, i in enumerate(idxs) if not done.loc[i]]
        self.cur_pos = unfin[0] if unfin else 0
        return idxs

    def current_row(self) -> Optional[pd.Series]:
        if not self.filtered_indices or not (0 <= self.cur_pos < len(self.filtered_indices)):
            return None
        return self.df.loc[self.filtered_indices[self.cur_pos]]

    def current_index(self) -> Optional[int]:
        if not self.filtered_indices or not (0 <= self.cur_pos < len(self.filtered_indices)):
            return None
        return int(self.filtered_indices[self.cur_pos])

    # -- persistent zone memory (survives restarts, shared across datasets) --

    @staticmethod
    def zone_key(session_id: str, camera_id: str, track_id: Any) -> str:
        return f"{session_id}:{camera_id}:{track_id}"

    def _load_zone_file(self) -> Dict[str, str]:
        if self._zone_file_cache is None:
            try:
                text = self.ZONE_MEMORY_PATH.read_text(encoding="utf-8")
                data = json.loads(text)
                self._zone_file_cache = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self._zone_file_cache = {}
        return self._zone_file_cache

    def _save_zone_file(self, key: str, zone: str) -> None:
        store = self._load_zone_file()
        store[key] = zone
        try:
            tmp = self.ZONE_MEMORY_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.ZONE_MEMORY_PATH)
        except OSError:
            logger.warning("Khong luu duoc bo nho vung ghe: %s", self.ZONE_MEMORY_PATH)

    def remembered_zone_for_row(self, row: pd.Series) -> str:
        stored = str(row.get("zone_id", "")).strip()
        if stored:
            return stored
        if self.dataset_path is None:
            return ""
        session_id = str(row["session_id"])
        camera_id = str(row.get("camera_id", "cam_A") or "cam_A")
        track_id = str(row["track_id"])
        key = self.zone_key(session_id, camera_id, track_id)
        # 1) remembered within this app session
        if key in self.track_zone_memory:
            return self.track_zone_memory[key]
        # 2) persistent memory file — survives restarts, works across datasets
        persistent = self._load_zone_file().get(key, "")
        if persistent:
            return persistent
        # 3) same track already labelled inside the current phiếu
        same = self.df[
            self.df["session_id"].astype(str).eq(session_id)
            & self.df["camera_id"].astype(str).eq(camera_id)
            & self.df["track_id"].astype(str).eq(track_id)
        ]
        zones = same["zone_id"].fillna("").astype(str).str.strip()
        uniq = zones[zones.ne("")].unique()
        if len(uniq) == 1:
            self.track_zone_memory[key] = uniq[0]
            return uniq[0]
        return ""

    # -- persistence ---------------------------------------------------------

    def resolve_media_paths(self, row: pd.Series) -> Tuple[Path, Path]:
        session_id = str(row["session_id"])
        camera_id = str(row.get("camera_id", "cam_A") or "cam_A")
        v_str = str(row.get("video_path", "") or "").strip()
        v_path = Path(v_str) if v_str and Path(v_str).is_file() else (self.root / v_str if v_str else Path())
        if not v_str or not v_path.is_file():
            for cand in (
                self.root / "data/raw" / session_id / f"{camera_id}.mp4",
                self.root / "data/raw" / session_id / "cam_A.mp4",
                self.root / "data/raw" / f"{session_id}.mp4",
            ):
                if cand.is_file():
                    v_path = cand
                    break
        p_path = self.root / "data/poses" / f"{session_id}_{camera_id}_poses.csv"
        if not p_path.is_file() and row.get("pose_overlay_path"):
            alt = Path(row["pose_overlay_path"]).parent / "poses.csv"
            p_path = alt if alt.is_absolute() else self.root / alt
        return v_path, p_path

    def save_studio_current(
        self,
        *,
        target_label: str,
        target_state: str,
        behaviors: List[str],
        phase: str,
        zone: str,
        occlusion: str,
        notes: str,
        phase_evidence: str,
        confirmed: bool,
        ai_suggestion: Optional[Dict[str, Any]] = None,
        span_15s: bool = False,
    ) -> List[str]:
        """Studio-tab save: whole-window review with optional ~15s train span."""
        if self.dataset_path is None:
            raise ValueError("Chưa chọn file CSV.")
        row = self.current_row()
        if row is None:
            raise ValueError("Không có cửa sổ đang chọn.")
        cur_idx = self.current_index()
        clean_lbl = "" if target_state == "OCCLUDED" else target_label
        beh_str = "; ".join(behaviors)
        ai_src = (ai_suggestion or {}).get("suggestion_source", "")
        saved_ids: List[str] = []

        use_span = span_15s and str(row.get("split", "")).lower() == "train"
        if use_span:
            future_idxs = find_next_track_window_indices(self.df, cur_idx, self.done_mask(), max_span_ms=15000.0)
        else:
            future_idxs = []

        _save_independent_window(
            df=self.df, row_idx=cur_idx, target_label=clean_lbl, target_state=target_state,
            behaviors=beh_str, phase=phase, zone=zone, occlusion=occlusion, notes=notes,
            annotator_id=self.person_full, primary_file_path=self.dataset_path,
            phase_evidence=phase_evidence, confirmed=confirmed,
            ai_suggestion_source=ai_src, ai_suggestion=ai_suggestion,
        )
        saved_ids.append(str(row["window_id"]))
        for f_idx in future_idxs:
            _save_independent_window(
                df=self.df, row_idx=f_idx, target_label=clean_lbl, target_state=target_state,
                behaviors=beh_str, phase=phase, zone=zone, occlusion=occlusion, notes=notes,
                annotator_id=self.person_full, primary_file_path=self.dataset_path,
                phase_evidence=phase_evidence, confirmed=confirmed,
                ai_suggestion_source=ai_src, ai_suggestion=ai_suggestion,
                span_source_window_id=str(row["window_id"]), span_seconds=15.0,
            )
            saved_ids.append(str(self.df.at[f_idx, "window_id"]))

        if zone:
            key = self.zone_key(str(row["session_id"]), str(row.get("camera_id", "cam_A") or "cam_A"), row["track_id"])
            self.track_zone_memory[key] = zone
            self._save_zone_file(key, zone)

        self.df = read_annotation_csv(self.dataset_path)
        return saved_ids

    def save_rubric_current(
        self,
        *,
        observations: List[Optional[str]],
        phase: str,
        phase_evidence: str,
        state: str,
        label: Optional[str],
        behavior: str,
        confidence: float,
        occlusion: str,
        notes: str,
        confirmed: bool,
    ) -> None:
        """Rubric-tab save: exact validation + payload of app/annotation_app.py."""
        if self.dataset_path is None:
            raise ValueError("Chưa chọn file CSV.")
        row = self.current_row()
        if row is None:
            raise ValueError("Không có cửa sổ đang chọn.")
        if not state:
            raise ValueError("Chọn khả năng quan sát. Với VALID, cần xác định pha thực tế và mức tham gia.")
        if state == "VALID":
            if not label or phase == "UNKNOWN":
                raise ValueError("Chọn khả năng quan sát. Với VALID, cần xác định pha thực tế và mức tham gia.")
            if not confirmed or not behavior.strip() or not phase_evidence.strip():
                raise ValueError("Cần ghi hành vi cụ thể, bằng chứng xác định nhiệm vụ và xác nhận đã xem đúng người trong toàn bộ đoạn.")
            rubric_label, rubric_reason = evidence_label(observations)
            if rubric_label is None or label != rubric_label:
                shown = LABELS.get(rubric_label, "Chưa đủ bằng chứng") if rubric_label else "Chưa đủ bằng chứng"
                raise ValueError(f"Bằng chứng chưa phù hợp với nhãn đã chọn. Theo bảng tiêu chí: {shown}. {rubric_reason}")

        updates = {
            "class_phase": phase,
            "annotation_state": state,
            "participation_label": label if state == "VALID" else "",
            "label_source": "HUMAN" if state == "VALID" else "",
            "label_rule_version": "",
            "annotator_id": self.person_full,
            "atomic_behavior": behavior,
            "annotation_confidence": confidence,
            "occlusion_level": occlusion,
            "notes": notes,
            "annotation_guide_version": RUBRIC_VERSION,
            "phase_verified": phase != "UNKNOWN",
            "phase_evidence": phase_evidence.strip(),
            "annotation_evidence": json.dumps(
                {
                    "rubric_version": RUBRIC_VERSION,
                    "observations": observations,
                    "basis": "DIRECT_HUMAN_OBSERVATION",
                    "confirmed": confirmed,
                },
                ensure_ascii=False,
            ),
            "phase_source": "HUMAN_WINDOW_REVIEW" if phase != "UNKNOWN" else "UNKNOWN",
        }
        idx = self.current_index()
        candidate = self.df.loc[[idx]].copy()
        for key, value in updates.items():
            candidate.at[idx, key] = value
        valid, errors = validate_annotation_dataframe(candidate)
        if not valid:
            raise ValueError("; ".join(errors))
        save_annotation_row(self.dataset_path, str(row["window_id"]), updates)
        self.df = read_annotation_csv(self.dataset_path)


# ---------------------------------------------------------------------------
# Tkinter application
# ---------------------------------------------------------------------------

class AnnotationAppDesktop(tk.Tk):
    """Desktop conversion of app/annotation_app.py (Studio tab + Rubric tab)."""

    def __init__(self, root_dir: Path = ROOT):
        super().__init__()
        self.ctrl = AnnotationAppController(root_dir)
        self.title("CVF25PRJ10 · Gán nhãn độc lập — Studio + Rubric (Desktop)")
        self.geometry("1460x900")
        self.minsize(1200, 740)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self._datasets: List[Tuple[str, str, Path]] = []
        self._opt_labels: List[str] = []
        self._loop_tk_frames: List[ImageTk.PhotoImage] = []
        self._strip_tk_imgs: List[ImageTk.PhotoImage] = []
        self._play_idx = 0
        self._playing = True
        self._play_job: Optional[str] = None
        self._render_seq = 0
        self._ai_seen_cur: Optional[Dict[str, Any]] = None
        self._future_15s_count = 0

        # --- variables -------------------------------------------------------
        self.var_person = tk.StringVar()
        self.var_dataset = tk.StringVar()
        self.var_filter = tk.StringVar(value="Chưa gán (còn lại)")
        self.var_session_filter = tk.StringVar()
        self.var_window_jump = tk.StringVar()

        self.var_skeleton = tk.BooleanVar(value=True)
        self.var_no_blur = tk.BooleanVar(value=True)
        self.var_auto_jump = tk.BooleanVar(value=True)
        self.var_ai_on = tk.BooleanVar(value=True)

        # studio tab
        self.var_s_label = tk.StringVar(value="")
        self.var_s_state = tk.StringVar(value="VALID")
        self.var_s_phase = tk.StringVar(value="lecture")
        self.var_s_zone = tk.StringVar(value="")
        self.var_s_occlusion = tk.StringVar(value="none")
        self.var_s_notes = tk.StringVar(value="")
        self.var_s_basis = tk.StringVar(value="")
        self.var_s_evidence = tk.StringVar(value="")
        self.var_s_confirmed = tk.BooleanVar(value=False)
        self.var_beh = {code: tk.BooleanVar(value=False) for _, code in BEHAVIOR_CHOICES}

        # rubric tab
        self.var_r_bins: List[tk.StringVar] = [tk.StringVar(value="") for _ in range(4)]
        self.var_r_phase = tk.StringVar(value="UNKNOWN")
        self.var_r_evidence = tk.StringVar(value="")
        self.var_r_state = tk.StringVar(value="")
        self.var_r_label = tk.StringVar(value="")
        self.var_r_behavior = tk.StringVar(value="")
        self.var_r_occlusion = tk.StringVar(value="none")
        self.var_r_notes = tk.StringVar(value="")
        self.var_r_confidence = tk.DoubleVar(value=0.8)
        self.var_r_confirmed = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_shortcuts()
        self._tick_video_loop()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(10, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text="Bạn là:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        cb_person = ttk.Combobox(top, textvariable=self.var_person, values=list(PEOPLE), state="readonly", width=8)
        cb_person.pack(side=tk.LEFT, padx=(4, 10))
        cb_person.bind("<<ComboboxSelected>>", lambda _e: self._on_person_changed())

        ttk.Label(top, text="Bộ video cần gán:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.cb_dataset = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=46)
        self.cb_dataset.pack(side=tk.LEFT, padx=(4, 10))
        self.cb_dataset.bind("<<ComboboxSelected>>", lambda _e: self._on_dataset_changed())

        ttk.Label(top, text="Lọc:").pack(side=tk.LEFT)
        cb_filter = ttk.Combobox(
            top, textvariable=self.var_filter,
            values=["Tất cả", "Chưa gán (còn lại)", "Đã gán", "Theo phiên học..."],
            state="readonly", width=17,
        )
        cb_filter.pack(side=tk.LEFT, padx=(4, 6))
        cb_filter.bind("<<ComboboxSelected>>", lambda _e: self._on_filter_changed())
        self.cb_sess = ttk.Combobox(top, textvariable=self.var_session_filter, state="readonly", width=12)
        self.cb_sess.pack(side=tk.LEFT, padx=(0, 8))
        self.cb_sess.bind("<<ComboboxSelected>>", lambda _e: self._on_filter_changed())

        ttk.Button(top, text="📖 Hướng dẫn", command=self._show_guide, width=11).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(top, text="💾 Sao lưu phiếu", command=self._export_csv, width=13).pack(side=tk.RIGHT)

        bar2 = ttk.Frame(self, padding=(10, 2, 10, 4))
        bar2.pack(fill=tk.X)
        ttk.Button(bar2, text="⏮ Trước [←]", command=lambda: self._step_pos(-1), width=11).pack(side=tk.LEFT)
        self.cb_window = ttk.Combobox(bar2, textvariable=self.var_window_jump, state="readonly", width=52)
        self.cb_window.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.cb_window.bind("<<ComboboxSelected>>", lambda _e: self._on_window_selected())
        ttk.Button(bar2, text="Tiếp [→] ⏭", command=lambda: self._step_pos(1), width=10).pack(side=tk.LEFT)
        ttk.Button(bar2, text="⏩ Chưa gán kế", command=self._jump_next_unfinished, width=13).pack(side=tk.LEFT, padx=(4, 8))

        ttk.Checkbutton(bar2, text="Khung xương", variable=self.var_skeleton, command=self._reload_current_media).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Checkbutton(bar2, text="Bỏ che mặt", variable=self.var_no_blur, command=self._reload_current_media).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(bar2, text="🤖 Gợi ý AI (bỏ tích để chấm mù)", variable=self.var_ai_on, command=self._sync_ai_panel).pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_progress = ttk.Label(bar2, text="Tiến độ: —", font=("Segoe UI", 9, "bold"), foreground="#1a5fb4")
        self.lbl_progress.pack(side=tk.RIGHT)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: shared media preview (video loop + 5-frame strip)
        left = ttk.Frame(paned)
        paned.add(left, weight=6)
        vid_hdr = ttk.Frame(left)
        vid_hdr.pack(fill=tk.X, pady=(0, 4))
        self.lbl_media_status = ttk.Label(vid_hdr, text="Chọn người gán và bộ dữ liệu để bắt đầu.", font=("Segoe UI", 9, "bold"))
        self.lbl_media_status.pack(side=tk.LEFT)
        self.btn_play = ttk.Button(vid_hdr, text="⏸ Tạm dừng [Space]", command=self._toggle_play, width=17)
        self.btn_play.pack(side=tk.RIGHT)
        self.canvas_video = tk.Label(left, bg="#181818", anchor=tk.CENTER)
        self.canvas_video.pack(fill=tk.BOTH, expand=True)
        ttk.Label(left, text="🎞️ 5 khung hình diễn biến cửa sổ:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(6, 2))
        strip_frame = ttk.Frame(left)
        strip_frame.pack(fill=tk.X)
        self.strip_panels: List[Tuple[ttk.Label, ttk.Label]] = []
        for idx in range(5):
            cell = ttk.Frame(strip_frame, borderwidth=1, relief=tk.GROOVE)
            cell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            img_lbl = ttk.Label(cell, anchor=tk.CENTER)
            img_lbl.pack(fill=tk.BOTH, expand=True)
            txt_lbl = ttk.Label(cell, text="", font=("Segoe UI", 8))
            txt_lbl.pack()
            self.strip_panels.append((img_lbl, txt_lbl))

        # Right: notebook with two tabs (Studio / Rubric)
        right_outer = ttk.Frame(paned)
        paned.add(right_outer, weight=5)
        canvas_scroll = tk.Canvas(right_outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self.right_form = ttk.Frame(canvas_scroll, padding=(6, 4))
        self.right_form.bind("<Configure>", lambda _e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=self.right_form, anchor="nw")
        canvas_scroll.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse-wheel scrolling for the annotation form (hover-follows-pointer).
        def _on_form_wheel(event) -> None:
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _enter_form(_event=None) -> None:
            canvas_scroll.bind_all("<MouseWheel>", _on_form_wheel)

        def _leave_form(event) -> None:
            # detail == "NotifyInferior" means we moved onto a child widget — keep binding.
            if getattr(event, "detail", "") != "NotifyInferior":
                canvas_scroll.unbind_all("<MouseWheel>")

        canvas_scroll.bind("<Enter>", _enter_form)
        canvas_scroll.bind("<Leave>", _leave_form)
        _enter_form()

        self.lbl_win_header = ttk.Label(self.right_form, text="👤 Cửa sổ", font=("Segoe UI", 11, "bold"), wraplength=520)
        self.lbl_win_header.pack(anchor=tk.W, pady=(0, 6))

        self.notebook = ttk.Notebook(self.right_form)
        self.notebook.pack(fill=tk.X, pady=(0, 6))
        self.tab_studio = ttk.Frame(self.notebook, padding=(6, 6))
        self.tab_rubric = ttk.Frame(self.notebook, padding=(6, 6))
        self.notebook.add(self.tab_studio, text="🚀 Studio (Chọn nhanh)")
        self.notebook.add(self.tab_rubric, text="📋 Rubric (4 khoảng)")
        self._build_studio_tab()
        self._build_rubric_tab()

        self.lbl_status_footer = ttk.Label(
            self,
            text="Phím tắt: [1/2/3/4] CAO·VỪA·THẤP·BỊ CHE (Studio) · [Enter] Lưu tab đang mở & Tiếp · [F] Gán nhanh ~15s Train · [←/→] Đổi cửa sổ · [Space] Phát/dừng",
            padding=(10, 4),
            foreground="#555555",
        )
        self.lbl_status_footer.pack(fill=tk.X, side=tk.BOTTOM)

    # -- Studio tab -----------------------------------------------------------

    def _build_studio_tab(self) -> None:
        f = self.tab_studio
        self.ai_frame = ttk.LabelFrame(f, text="🤖 Đề xuất AI", padding=8)
        self.ai_frame.pack(fill=tk.X, pady=(0, 8))
        self.lbl_ai_summary = ttk.Label(self.ai_frame, text="", font=("Segoe UI", 9, "bold"), wraplength=470)
        self.lbl_ai_summary.pack(anchor=tk.W)
        self.lbl_ai_detail = ttk.Label(self.ai_frame, text="", font=("Segoe UI", 8), wraplength=470, foreground="#444444")
        self.lbl_ai_detail.pack(anchor=tk.W, pady=(2, 4))
        ttk.Button(self.ai_frame, text="📝 Điền lại gợi ý vào phiếu", command=self._apply_current_ai_to_draft).pack(anchor=tk.W)

        lbl_grid = ttk.Frame(f)
        lbl_grid.pack(fill=tk.X, pady=(0, 6))
        self.label_buttons: Dict[Tuple[str, str], tk.Button] = {}
        btn_specs = [
            (0, 0, "[1] 🟢 CAO (HIGH)", HIGH, "VALID", "#1e7e34"),
            (0, 1, "[2] 🟡 VỪA (MODERATE)", MODERATE, "VALID", "#c69500"),
            (1, 0, "[3] 🔴 THẤP (LOW)", LOW, "VALID", "#bd2130"),
            (1, 1, "[4] ⚪ BỊ CHE (OCCLUDED)", "", "OCCLUDED", "#545b62"),
        ]
        for r_i, c_i, text_b, l_val, s_val, bg_hex in btn_specs:
            lbl_grid.columnconfigure(c_i, weight=1)
            btn = tk.Button(
                lbl_grid, text=text_b, font=("Segoe UI", 10, "bold"), pady=7, relief=tk.RAISED,
                command=lambda lv=l_val, sv=s_val: self._choose_label(lv, sv),
            )
            btn.grid(row=r_i, column=c_i, sticky="ew", padx=3, pady=3)
            self.label_buttons[(l_val, s_val)] = btn
        self._btn_colors = {(l_val, s_val): bg_hex for _, _, _, l_val, s_val, bg_hex in btn_specs}
        self.lbl_selected_badge = ttk.Label(f, text="Chưa chọn mức tham gia.", font=("Segoe UI", 9, "bold"), foreground="#0b5394")
        self.lbl_selected_badge.pack(anchor=tk.W, pady=(2, 6))

        beh_box = ttk.LabelFrame(f, text="Hành vi quan sát được (tích chọn ≥ 1)", padding=6)
        beh_box.pack(fill=tk.X, pady=(0, 8))
        beh_box.columnconfigure(0, weight=1)
        beh_box.columnconfigure(1, weight=1)
        for i, (b_name, b_code) in enumerate(BEHAVIOR_CHOICES):
            ttk.Checkbutton(beh_box, text=b_name, variable=self.var_beh[b_code]).grid(row=i // 2, column=i % 2, sticky="w", padx=4, pady=2)

        ctx = ttk.LabelFrame(f, text="Ngữ cảnh & Vị trí", padding=6)
        ctx.pack(fill=tk.X, pady=(0, 8))
        for c in (1, 3):
            ctx.columnconfigure(c, weight=1)
        ttk.Label(ctx, text="Trạng thái:").grid(row=0, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(ctx, textvariable=self.var_s_state, values=STATE_OPTIONS, state="readonly", width=13).grid(row=0, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(ctx, text="Pha học:").grid(row=0, column=2, sticky="w", padx=3, pady=2)
        cb_phase = ttk.Combobox(ctx, textvariable=self.var_s_phase, values=PHASE_OPTIONS, state="readonly", width=15)
        cb_phase.grid(row=0, column=3, sticky="ew", padx=3, pady=2)
        cb_phase.bind("<<ComboboxSelected>>", lambda _e: self._refresh_phase_basis_options())

        ttk.Label(ctx, text="Vùng ghế:").grid(row=1, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(ctx, textvariable=self.var_s_zone, values=ZONE_OPTIONS, state="readonly", width=13).grid(row=1, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(ctx, text="Che khuất:").grid(row=1, column=2, sticky="w", padx=3, pady=2)
        ttk.Combobox(ctx, textvariable=self.var_s_occlusion, values=OCCLUSION_OPTIONS, state="readonly", width=15).grid(row=1, column=3, sticky="ew", padx=3, pady=2)

        ttk.Label(ctx, text="Căn cứ pha:").grid(row=2, column=0, sticky="w", padx=3, pady=2)
        self.cb_phase_basis = ttk.Combobox(ctx, textvariable=self.var_s_basis, state="readonly")
        self.cb_phase_basis.grid(row=2, column=1, columnspan=3, sticky="ew", padx=3, pady=2)
        self.cb_phase_basis.bind("<<ComboboxSelected>>", lambda _e: self._on_basis_picked())

        ttk.Label(ctx, text="Căn cứ khác:").grid(row=3, column=0, sticky="w", padx=3, pady=2)
        ttk.Entry(ctx, textvariable=self.var_s_evidence).grid(row=3, column=1, columnspan=3, sticky="ew", padx=3, pady=2)
        ttk.Label(ctx, text="Ghi chú:").grid(row=4, column=0, sticky="w", padx=3, pady=2)
        ttk.Entry(ctx, textvariable=self.var_s_notes).grid(row=4, column=1, columnspan=3, sticky="ew", padx=3, pady=2)

        ttk.Checkbutton(f, text="Tôi đã xem toàn bộ clip, xác định đúng người và bằng chứng", variable=self.var_s_confirmed).pack(anchor=tk.W, pady=(2, 6))

        self.btn_save = tk.Button(
            f, text="✅ Duyệt & Lưu HUMAN & Tiếp [Enter]", font=("Segoe UI", 11, "bold"),
            bg="#198754", fg="white", pady=9, command=self._save_studio,
        )
        self.btn_save.pack(fill=tk.X, pady=(0, 6))

        self.span15_frame = ttk.LabelFrame(f, text="⏱️ Gán nhanh ~15s cùng Track (chỉ tập TRAIN)", padding=8)
        self.lbl_15s_info = ttk.Label(self.span15_frame, text="", wraplength=470, font=("Segoe UI", 8))
        self.lbl_15s_info.pack(anchor=tk.W, pady=(0, 4))
        self.btn_save_15s = tk.Button(
            self.span15_frame, text="⚡ Lưu & Gán nhanh ~15s [F]", font=("Segoe UI", 10, "bold"),
            bg="#0d6efd", fg="white", pady=6, command=lambda: self._save_studio(span_15s=True),
        )
        self.btn_save_15s.pack(fill=tk.X)

    # -- Rubric tab -----------------------------------------------------------

    def _build_rubric_tab(self) -> None:
        f = self.tab_rubric
        ttk.Label(
            f,
            text="Chia đoạn thành 4 khoảng gần bằng nhau. Đánh dấu hành vi của đúng người trong từng khoảng; các lựa chọn này do bạn quan sát, không lấy từ mô hình.",
            wraplength=470, font=("Segoe UI", 8), foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 6))

        bins_box = ttk.LabelFrame(f, text="Bằng chứng theo 4 khoảng thời gian", padding=6)
        bins_box.pack(fill=tk.X, pady=(0, 4))
        for part in range(4):
            row = ttk.Frame(bins_box)
            row.pack(fill=tk.X, pady=2)
            self.lbl_bins_time = getattr(self, "lbl_bins_time", None)
            ttk.Label(row, text=f"Khoảng {part + 1}:", width=9).pack(side=tk.LEFT)
            cb = ttk.Combobox(row, textvariable=self.var_r_bins[part], values=[""] + EV_DISPLAY, width=42)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_rubric_result())
        self.lbl_rubric_result = ttk.Label(f, text="Theo bảng tiêu chí thí điểm: —", font=("Segoe UI", 9, "bold"), foreground="#0b5394", wraplength=470)
        self.lbl_rubric_result.pack(anchor=tk.W, pady=(2, 6))

        form = ttk.LabelFrame(f, text="Phiếu gán nhãn", padding=6)
        form.pack(fill=tk.X, pady=(0, 8))
        for c in (1,):
            form.columnconfigure(c, weight=1)
        ttk.Label(form, text="Pha hoạt động thực tế:").grid(row=0, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(form, textvariable=self.var_r_phase, values=PHASE_DISPLAY, state="readonly").grid(row=0, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(form, text="Bằng chứng pha / nhiệm vụ:").grid(row=1, column=0, sticky="w", padx=3, pady=2)
        ttk.Entry(form, textvariable=self.var_r_evidence).grid(row=1, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(form, text="Khả năng quan sát:").grid(row=2, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(form, textvariable=self.var_r_state, values=[""] + STATE_DISPLAY, state="readonly").grid(row=2, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(form, text="Mức tham gia (nếu VALID):").grid(row=3, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(form, textvariable=self.var_r_label, values=[""] + LABEL_DISPLAY, state="readonly").grid(row=3, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(form, text="Hành vi quan sát được:").grid(row=4, column=0, sticky="w", padx=3, pady=2)
        ttk.Entry(form, textvariable=self.var_r_behavior).grid(row=4, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(form, text="(VD: READING_WRITING; HAND_RAISED — dấu ; nếu nhiều)", foreground="#777777").grid(row=5, column=1, sticky="w", padx=3)
        ttk.Label(form, text="Mức che khuất:").grid(row=6, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(form, textvariable=self.var_r_occlusion, values=OCCLUSION_OPTIONS, state="readonly").grid(row=6, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(form, text="Độ tin cậy của người gán:").grid(row=7, column=0, sticky="w", padx=3, pady=2)
        tk.Scale(form, from_=0.0, to=1.0, resolution=0.05, variable=self.var_r_confidence, orient=tk.HORIZONTAL, showvalue=False).grid(row=7, column=1, sticky="ew", padx=3, pady=2)
        self.lbl_conf_value = ttk.Label(form, text="0.80")
        self.lbl_conf_value.grid(row=7, column=1, sticky="e", padx=(0, 6))
        self.var_r_confidence.trace_add("write", lambda *_a: self.lbl_conf_value.config(text=f"{self.var_r_confidence.get():.2f}"))
        ttk.Label(form, text="Ghi chú:").grid(row=8, column=0, sticky="nw", padx=3, pady=2)
        self.txt_r_notes = tk.Text(form, height=3, font=("Segoe UI", 9), wrap=tk.WORD)
        self.txt_r_notes.grid(row=8, column=1, sticky="ew", padx=3, pady=2)

        ttk.Checkbutton(f, text="Tôi đã xem hết đoạn, xác định đúng người và xác nhận bằng chứng theo bảng tiêu chí", variable=self.var_r_confirmed).pack(anchor=tk.W, pady=(2, 4))

        btn_row = ttk.Frame(f)
        btn_row.pack(fill=tk.X, pady=(2, 6))
        tk.Button(
            btn_row, text="✅ Lưu & đến cửa sổ chưa gán tiếp [Enter]", font=("Segoe UI", 10, "bold"),
            bg="#198754", fg="white", pady=8, command=lambda: self._save_rubric(advance=True),
        ).pack(fill=tk.X, side=tk.TOP)
        tk.Button(btn_row, text="💾 Lưu tại cửa sổ này", pady=5, command=lambda: self._save_rubric(advance=False)).pack(fill=tk.X, side=tk.TOP, pady=(4, 0))
        tk.Button(btn_row, text="🎬 Xem video gốc toàn cảnh (ngữ cảnh, chỉ dùng local)", pady=4, command=self._play_context_video).pack(fill=tk.X, side=tk.TOP, pady=(4, 0))

    # ------------------------------------------------------------ handlers --

    def _bind_shortcuts(self) -> None:
        def _is_typing(event) -> bool:
            return isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox))

        self.bind("1", lambda e: None if _is_typing(e) else self._choose_label(HIGH, "VALID"))
        self.bind("2", lambda e: None if _is_typing(e) else self._choose_label(MODERATE, "VALID"))
        self.bind("3", lambda e: None if _is_typing(e) else self._choose_label(LOW, "VALID"))
        self.bind("4", lambda e: None if _is_typing(e) else self._choose_label("", "OCCLUDED"))
        self.bind("<Return>", lambda e: (self._save_active_tab(), "break")[1] if not _is_typing(e) else None)
        self.bind("<Control-s>", lambda e: (self._save_active_tab(), "break")[1])
        self.bind("f", lambda e: None if _is_typing(e) else self._save_studio(span_15s=True))
        self.bind("<Left>", lambda e: None if _is_typing(e) else self._step_pos(-1))
        self.bind("<Right>", lambda e: None if _is_typing(e) else self._step_pos(1))
        self.bind("<space>", lambda e: None if _is_typing(e) else self._toggle_play())

    def _save_active_tab(self) -> None:
        if str(self.notebook.index(self.notebook.select())) == "0":
            self._save_studio()
        else:
            self._save_rubric(advance=True)

    # -- person / dataset / filter -------------------------------------------

    def _on_person_changed(self) -> None:
        if not self.var_person.get():
            return
        self._datasets = self.ctrl.set_person(self.var_person.get())
        labels = [lbl for _, lbl, _ in self._datasets]
        self.cb_dataset["values"] = labels
        if labels:
            self.var_dataset.set(labels[0])
            self._on_dataset_changed()

    def _on_dataset_changed(self) -> None:
        sel = self.var_dataset.get()
        match = next((p for _, lbl, p in self._datasets if lbl == sel), None)
        if match is None:
            return
        self.ctrl.load_dataset(match)
        sessions = sorted(self.ctrl.df["session_id"].dropna().astype(str).unique().tolist()) if not self.ctrl.df.empty else []
        self.cb_sess["values"] = sessions
        if sessions and self.var_session_filter.get() not in sessions:
            self.var_session_filter.set(sessions[0])
        self._on_filter_changed()

    def _on_filter_changed(self) -> None:
        self.ctrl.apply_filter(self.var_filter.get(), self.var_session_filter.get())
        self._refresh_window_dropdown()
        self._load_current_window()

    def _refresh_window_dropdown(self) -> None:
        df = self.ctrl.df
        done = self.ctrl.done_mask()
        idxs = self.ctrl.filtered_indices
        total = len(df)
        if not idxs:
            self._opt_labels = []
            self.cb_window["values"] = []
            self.var_window_jump.set("")
            return
        sub = df.loc[idxs]
        splits = sub["split"] if "split" in sub.columns else [""] * len(idxs)
        self._opt_labels = [
            f"{'✓' if done.at[i] else '○'} #{i + 1}/{total} [{sp}] {sess} | Trk #{trk} ({float(t0 or 0) / 1000:g}-{float(t1 or 0) / 1000:g}s) → {lbl or st_v or 'Chưa gán'}"
            for i, sp, sess, trk, t0, t1, lbl, st_v in zip(
                idxs, splits, sub["session_id"], sub["track_id"], sub["t_start_ms"], sub["t_end_ms"],
                sub["participation_label"], sub["annotation_state"],
            )
        ]
        self.cb_window["values"] = self._opt_labels
        if 0 <= self.ctrl.cur_pos < len(self._opt_labels):
            self.var_window_jump.set(self._opt_labels[self.ctrl.cur_pos])

    def _on_window_selected(self) -> None:
        pos = self.cb_window.current()
        if pos >= 0:
            self.ctrl.cur_pos = pos
            self._load_current_window()

    def _step_pos(self, delta: int) -> None:
        if not self.ctrl.filtered_indices:
            return
        nxt = max(0, min(len(self.ctrl.filtered_indices) - 1, self.ctrl.cur_pos + delta))
        if nxt != self.ctrl.cur_pos:
            self.ctrl.cur_pos = nxt
            if self._opt_labels:
                self.var_window_jump.set(self._opt_labels[nxt])
            self._load_current_window()

    def _jump_next_unfinished(self) -> None:
        done = self.ctrl.done_mask()
        idxs = self.ctrl.filtered_indices
        rem = [p for p, i in enumerate(idxs) if p > self.ctrl.cur_pos and not done.loc[i]]
        if not rem:
            rem = [p for p, i in enumerate(idxs) if not done.loc[i]]
        if rem:
            self.ctrl.cur_pos = rem[0]
            if self._opt_labels:
                self.var_window_jump.set(self._opt_labels[rem[0]])
            self._load_current_window()

    # -- studio form ----------------------------------------------------------

    def _refresh_phase_basis_options(self) -> None:
        opts = [""] + PHASE_BASIS_DEFAULT.get(self.var_s_phase.get(), [])
        self.cb_phase_basis["values"] = opts

    def _on_basis_picked(self) -> None:
        picked = self.var_s_basis.get().strip()
        if picked and not self.var_s_evidence.get().strip():
            self.var_s_evidence.set(picked)

    def _choose_label(self, label: str, state: str) -> None:
        self.var_s_label.set(label)
        self.var_s_state.set(state)
        self._update_label_button_highlights()

    def _update_label_button_highlights(self) -> None:
        cur_lbl = self.var_s_label.get()
        cur_st = self.var_s_state.get()
        names = {
            (HIGH, "VALID"): "🟢 CAO (HIGH)",
            (MODERATE, "VALID"): "🟡 VỪA (MODERATE)",
            (LOW, "VALID"): "🔴 THẤP (LOW)",
            ("", "OCCLUDED"): "⚪ BỊ CHE (OCCLUDED)",
        }
        selected_txt = None
        for key, btn in self.label_buttons.items():
            lv, sv = key
            active = cur_st == sv and (sv != "VALID" or cur_lbl == lv)
            if active:
                selected_txt = names[key]
                btn.config(bg=self._btn_colors[key], fg="white", relief=tk.SUNKEN)
            else:
                btn.config(bg="#f0f0f0", fg="#222222", relief=tk.RAISED)
        self.lbl_selected_badge.config(text=f"✓ Đang chọn: {selected_txt}" if selected_txt else "Chưa chọn mức tham gia.")

    def _sync_ai_panel(self) -> None:
        row = self.ctrl.current_row()
        if row is None:
            return
        wid = str(row["window_id"])
        suggestions = self.ctrl.load_suggestions()
        sug = suggestions.get(wid) if self.var_ai_on.get() else None
        self._ai_seen_cur = dict(sug) if sug else None

        if sug is not None:
            self.ai_frame.pack(fill=tk.X, pady=(0, 8), before=self.label_buttons[(HIGH, "VALID")].master)
            lbl_map = {HIGH: "🟢 CAO", MODERATE: "🟡 VỪA", LOW: "🔴 THẤP", "": "—"}
            self.lbl_ai_summary.config(
                text=f"{sug.get('suggestion_source', 'AI')} · {sug.get('suggested_state')} · {lbl_map.get(sug.get('suggested_label', ''), '')} · p={float(sug.get('confidence', 0)):.2f}"
            )
            behs = ", ".join(sug.get("suggested_behaviors", [])) or "Không có"
            det = f"Hành vi: {behs}"
            if sug.get("rationale"):
                det += f"\nCăn cứ: {sug['rationale']}"
            if sug.get("rejection_reason"):
                det += f"\n⚠ Cảnh báo: {sug['rejection_reason']}"
            self.lbl_ai_detail.config(text=det)
            self.btn_save.config(text="✅ Duyệt & Lưu HUMAN & Tiếp [Enter]", bg="#198754")
            if not str(row.get("annotator_id", "")).strip():
                self._apply_current_ai_to_draft()
        else:
            self.ai_frame.pack_forget()
            self.btn_save.config(text="💾 Lưu cửa sổ này & Tiếp [Enter]", bg="#0d6efd")
            if not str(row.get("annotator_id", "")).strip():
                self._reset_studio_form_from_row(row)

    def _apply_current_ai_to_draft(self) -> None:
        sug = self._ai_seen_cur
        if not sug:
            return
        self.var_s_label.set(sug.get("suggested_label", ""))
        self.var_s_state.set(sug.get("suggested_state", "VALID"))
        s_behs = set(sug.get("suggested_behaviors", []))
        for _, code in BEHAVIOR_CHOICES:
            self.var_beh[code].set(code in s_behs)
        if sug.get("suggested_phase") in ("lecture", "individual_work", "discussion"):
            self.var_s_phase.set(sug["suggested_phase"])
        self._refresh_phase_basis_options()
        if sug.get("suggested_occlusion") in OCCLUSION_OPTIONS:
            self.var_s_occlusion.set(sug["suggested_occlusion"])
        if sug.get("suggested_phase_evidence"):
            self.var_s_evidence.set(sug["suggested_phase_evidence"])
        if sug.get("rationale"):
            self.var_s_notes.set(str(sug["rationale"])[:240])
        self._update_label_button_highlights()

    def _reset_studio_form_from_row(self, row: pd.Series) -> None:
        p_lbl = str(row.get("participation_label", ""))
        st_v = str(row.get("annotation_state", "")) or "VALID"
        self.var_s_label.set(p_lbl)
        self.var_s_state.set(st_v if st_v in STATE_OPTIONS else "VALID")
        raw_beh = str(row.get("atomic_behavior", ""))
        cur_behs = {b.strip() for b in raw_beh.split(";") if b.strip()}
        for _, code in BEHAVIOR_CHOICES:
            self.var_beh[code].set(code in cur_behs)
        reviewed = bool(str(row.get("annotator_id", "")).strip()) and st_v in ANNOTATION_STATES
        p_val = str(row.get("class_phase", "lecture")) if reviewed else "lecture"
        self.var_s_phase.set(p_val if p_val in PHASE_OPTIONS and p_val != "UNKNOWN" else "lecture")
        self._refresh_phase_basis_options()
        self.var_s_basis.set("")
        self.var_s_evidence.set(str(row.get("phase_evidence", "")))
        self.var_s_zone.set(self.ctrl.remembered_zone_for_row(row))
        o_val = str(row.get("occlusion_level", "none") or "none")
        self.var_s_occlusion.set(o_val if o_val in OCCLUSION_OPTIONS else "none")
        self.var_s_notes.set(str(row.get("notes", "")))
        self.var_s_confirmed.set(False)
        self._update_label_button_highlights()

    def _reset_rubric_form_from_row(self, row: pd.Series) -> None:
        try:
            saved_evidence = json.loads(row.get("annotation_evidence", "") or "{}")
        except (TypeError, ValueError):
            saved_evidence = {}
        stored_bins = saved_evidence.get("observations", [None] * 4)
        for part in range(4):
            previous = stored_bins[part] if isinstance(stored_bins, list) and len(stored_bins) == 4 else None
            if previous in EVIDENCE_OPTIONS:
                code, name = previous, EVIDENCE_OPTIONS[previous]
                self.var_r_bins[part].set(f"{code} — {name}")
            else:
                self.var_r_bins[part].set("")
        phase = str(row.get("class_phase", "UNKNOWN"))
        self.var_r_phase.set(f"{phase} — {PHASES[phase]}" if phase in PHASES else PHASE_DISPLAY[0])
        self.var_r_evidence.set(str(row.get("phase_evidence", "")))
        state = str(row.get("annotation_state", "")) or ""
        self.var_r_state.set(f"{state} — {STATES[state]}" if state in STATES else "")
        label = str(row.get("participation_label", "")) or ""
        self.var_r_label.set(f"{label} — {LABELS[label]}" if label in LABELS else "")
        self.var_r_behavior.set(str(row.get("atomic_behavior", "")))
        occ = str(row.get("occlusion_level", "none") or "none")
        self.var_r_occlusion.set(occ if occ in OCCLUSION_OPTIONS else "none")
        self.var_r_notes.set(str(row.get("notes", "")))
        try:
            conf = float(row.get("annotation_confidence"))
            if pd.isna(conf):
                raise ValueError
        except (TypeError, ValueError):
            conf = 0.8
        self.var_r_confidence.set(round(min(1.0, max(0.0, conf)), 2))
        self.var_r_confirmed.set(False)
        self.txt_r_notes.delete("1.0", tk.END)
        self.txt_r_notes.insert("1.0", str(row.get("notes", "")))
        self._refresh_rubric_result()

    def _refresh_rubric_result(self) -> None:
        observations = [EV_FROM_DISPLAY.get(v) if v else None for v in (b.get() for b in self.var_r_bins)]
        rubric_label, reason = evidence_label(observations)
        shown = LABELS.get(rubric_label, "Chưa đủ bằng chứng") if rubric_label else "Chưa đủ bằng chứng"
        self.lbl_rubric_result.config(text=f"Theo bảng tiêu chí thí điểm: {shown}. {reason}")

    # -- window loading --------------------------------------------------------

    def _load_current_window(self) -> None:
        total = len(self.ctrl.df)
        done_cnt = int(self.ctrl.done_mask().sum()) if total else 0
        self.lbl_progress.config(text=f"Tiến độ: {done_cnt}/{total} ({done_cnt / max(1, total) * 100:.1f}%)")

        row = self.ctrl.current_row()
        if row is None:
            self.lbl_win_header.config(text="Không có cửa sổ phù hợp bộ lọc.")
            return

        cur_idx = self.ctrl.current_index()
        disp = row.get("participation_label") or row.get("annotation_state") or "Chưa gán"
        self.lbl_win_header.config(
            text=f"👤 #{cur_idx + 1}/{total} · Track #{row['track_id']} ({row['session_id']} · {float(row['t_start_ms']) / 1000:g}–{float(row['t_end_ms']) / 1000:g}s · Vùng {row.get('zone_id', 'UNKNOWN')}) → {disp}"
        )
        self._reset_studio_form_from_row(row)
        self._reset_rubric_form_from_row(row)
        self._sync_ai_panel()

        # 15s train quick-label panel availability
        is_train = str(row.get("split", "")).lower() == "train"
        fut = find_next_track_window_indices(self.ctrl.df, cur_idx, self.ctrl.done_mask(), max_span_ms=15000.0) if is_train else []
        self._future_15s_count = len(fut)
        if fut:
            span_end = max(float(self.ctrl.df.at[i, "t_end_ms"]) for i in fut)
            self.lbl_15s_info.config(
                text=f"Track #{row['track_id']} còn {len(fut)} cửa sổ chưa gán trong khoảng {float(row['t_start_ms']) / 1000:g}s → {span_end / 1000:g}s."
            )
            self.btn_save_15s.config(text=f"⚡ Lưu & Gán nhanh {1 + len(fut)} cửa sổ (~15s) [F]")
            self.span15_frame.pack(fill=tk.X, pady=(0, 8))
        else:
            self.span15_frame.pack_forget()

        self._reload_current_media()

    def _reload_current_media(self) -> None:
        row = self.ctrl.current_row()
        if row is None:
            return
        self._render_seq += 1
        seq_id = self._render_seq
        wid = str(row["window_id"])
        anonymize = not self.var_no_blur.get()
        skeleton = bool(self.var_skeleton.get())
        cache_key = (wid, anonymize, skeleton)

        if cache_key in self.ctrl.media_cache:
            loop_rgb, strip_items, status = self.ctrl.media_cache[cache_key]
            self.ctrl.media_cache.move_to_end(cache_key)
            self._apply_media_to_widgets(loop_rgb, strip_items, status)
            return

        self.lbl_media_status.config(text="⏳ Đang giải mã nhanh cửa sổ từ video gốc...")
        row_dict = row.to_dict()

        def _worker() -> None:
            v_path, p_path = self.ctrl.resolve_media_paths(row)
            pose_df = self.ctrl.get_pose_df(p_path)
            loop_rgb, strip_items, status = extract_window_media_in_memory(
                v_path, pose_df,
                str(row_dict["session_id"]), str(row_dict.get("camera_id", "cam_A") or "cam_A"),
                str(row_dict["track_id"]), float(row_dict["t_start_ms"]), float(row_dict["t_end_ms"]),
                anonymize=anonymize, skeleton=skeleton,
            )

            def _apply_on_ui() -> None:
                if seq_id != self._render_seq:
                    return
                if loop_rgb:
                    self.ctrl.media_cache[cache_key] = (loop_rgb, strip_items, status)
                    if len(self.ctrl.media_cache) > 16:
                        self.ctrl.media_cache.popitem(last=False)
                self._apply_media_to_widgets(loop_rgb, strip_items, status)

            self.after(0, _apply_on_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_media_to_widgets(
        self,
        loop_rgb: List[np.ndarray],
        strip_items: List[Tuple[str, np.ndarray]],
        status: str,
    ) -> None:
        self.lbl_media_status.config(text=status)
        self._play_idx = 0
        w_target = max(640, self.canvas_video.winfo_width() or 720)
        h_target = max(300, min(430, int(w_target * 592 / 1280)))
        self._loop_tk_frames = []
        for arr in loop_rgb:
            pil_img = Image.fromarray(arr).resize((w_target, h_target), Image.Resampling.BILINEAR)
            self._loop_tk_frames.append(ImageTk.PhotoImage(pil_img))
        if self._loop_tk_frames:
            self.canvas_video.config(image=self._loop_tk_frames[0], text="")
        else:
            self.canvas_video.config(image="", text=status, fg="#ffaaaa")

        self._strip_tk_imgs = []
        for idx, (img_lbl, txt_lbl) in enumerate(self.strip_panels):
            if idx < len(strip_items):
                cap_txt, crop_arr = strip_items[idx]
                tk_crop = ImageTk.PhotoImage(Image.fromarray(crop_arr).resize((132, 172), Image.Resampling.BILINEAR))
                self._strip_tk_imgs.append(tk_crop)
                img_lbl.config(image=tk_crop, text="")
                txt_lbl.config(text=cap_txt)
            else:
                img_lbl.config(image="", text="—")
                txt_lbl.config(text="")

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.btn_play.config(text="⏸ Tạm dừng [Space]" if self._playing else "▶ Phát tiếp [Space]")

    def _tick_video_loop(self) -> None:
        if self._playing and self._loop_tk_frames:
            self._play_idx = (self._play_idx + 1) % len(self._loop_tk_frames)
            self.canvas_video.config(image=self._loop_tk_frames[self._play_idx])
        self._play_job = self.after(220, self._tick_video_loop)

    def _play_context_video(self) -> None:
        """Play the raw session video around the window (context only, local)."""
        row = self.ctrl.current_row()
        if row is None:
            return
        v_path, _ = self.ctrl.resolve_media_paths(row)
        if not v_path.is_file():
            messagebox.showwarning("Không có video", f"Không tìm thấy video: {v_path}")
            return
        t0 = max(0.0, float(row["t_start_ms"]) / 1000.0 - 5.0)
        t1 = float(row["t_end_ms"]) / 1000.0 + 2.0
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            messagebox.showwarning("Không mở được video", str(v_path))
            return
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
        win = f"Video goc (ngu canh) — Esc de thoat | {v_path.name}"
        while True:
            ok, frame = cap.read()
            if not ok or cap.get(cv2.CAP_PROP_POS_MSEC) > t1 * 1000.0:
                break
            disp = cv2.resize(frame, (960, int(960 * frame.shape[0] / max(1, frame.shape[1]))))
            cv2.imshow(win, disp)
            if cv2.waitKey(max(1, int(1000 / max(1.0, cap.get(cv2.CAP_PROP_FPS))))) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    # -- saves ------------------------------------------------------------------

    def _save_studio(self, *, span_15s: bool = False) -> None:
        if span_15s and self._future_15s_count <= 0:
            return
        selected_behs = [code for _, code in BEHAVIOR_CHOICES if self.var_beh[code].get()]
        pe = self.var_s_evidence.get().strip() or self.var_s_basis.get().strip()
        confirmed = bool(self.var_s_confirmed.get() or self._ai_seen_cur or self.var_s_label.get())
        try:
            saved_ids = self.ctrl.save_studio_current(
                target_label=self.var_s_label.get(),
                target_state=self.var_s_state.get(),
                behaviors=selected_behs,
                phase=self.var_s_phase.get(),
                zone=self.var_s_zone.get(),
                occlusion=self.var_s_occlusion.get(),
                notes=self.var_s_notes.get(),
                phase_evidence=pe,
                confirmed=confirmed,
                ai_suggestion=self._ai_seen_cur,
                span_15s=span_15s,
            )
        except (ValueError, OSError) as exc:
            messagebox.showwarning("Chưa thể lưu", str(exc))
            return

        saved_set = set(saved_ids)
        self.lbl_status_footer.config(text=f"✅ Đã lưu {len(saved_ids)} cửa sổ ({saved_ids[0] if saved_ids else ''}) xuống ổ cứng.")
        if self.var_auto_jump.get() or self._ai_seen_cur or span_15s:
            self._advance_after_save(saved_set)
        self._refresh_window_dropdown()
        self._load_current_window()

    def _save_rubric(self, *, advance: bool) -> None:
        observations = [EV_FROM_DISPLAY.get(v) if v else None for v in (b.get() for b in self.var_r_bins)]
        try:
            self.ctrl.save_rubric_current(
                observations=observations,
                phase=PHASE_FROM_DISPLAY.get(self.var_r_phase.get(), "UNKNOWN"),
                phase_evidence=self.var_r_evidence.get(),
                state=STATE_FROM_DISPLAY.get(self.var_r_state.get(), ""),
                label=LABEL_FROM_DISPLAY.get(self.var_r_label.get()) or None,
                behavior=self.var_r_behavior.get(),
                confidence=float(self.var_r_confidence.get()),
                occlusion=self.var_r_occlusion.get(),
                notes=self.txt_r_notes.get("1.0", tk.END).strip(),
                confirmed=bool(self.var_r_confirmed.get()),
            )
        except (ValueError, OSError) as exc:
            messagebox.showwarning("Chưa thể lưu", str(exc))
            return
        wid = str(self.ctrl.current_row()["window_id"]) if self.ctrl.current_row() is not None else ""
        self.lbl_status_footer.config(text=f"✅ Đã lưu nhãn rubric của {self.ctrl.person_short} ({wid}).")
        if advance:
            self._advance_after_save(set())
        self._refresh_window_dropdown()
        self._load_current_window()

    def _advance_after_save(self, saved_set: set) -> None:
        prev_pos = self.ctrl.cur_pos
        if self.var_filter.get() == "Chưa gán (còn lại)":
            self.ctrl.apply_filter(self.var_filter.get(), self.var_session_filter.get())
            self.ctrl.cur_pos = min(prev_pos, max(0, len(self.ctrl.filtered_indices) - 1))
        else:
            done = self.ctrl.done_mask()
            idxs = self.ctrl.filtered_indices
            rem = [
                p for p, i in enumerate(idxs)
                if p > prev_pos and not done.loc[i] and str(self.ctrl.df.at[i, "window_id"]) not in saved_set
            ]
            if not rem:
                rem = [
                    p for p, i in enumerate(idxs)
                    if not done.loc[i] and str(self.ctrl.df.at[i, "window_id"]) not in saved_set
                ]
            if rem:
                self.ctrl.cur_pos = rem[0]

    # -- utilities ----------------------------------------------------------------

    def _show_guide(self) -> None:
        win = tk.Toplevel(self)
        win.title("Hướng dẫn gán nhãn")
        win.geometry("760x640")
        txt = tk.Text(win, wrap=tk.WORD, font=("Segoe UI", 10), padx=12, pady=10)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", GUIDE_TEXT)
        txt.config(state=tk.DISABLED)
        btns = ttk.Frame(win, padding=6)
        btns.pack(fill=tk.X)
        guide_path = ROOT / "docs" / "annotation_guide.md"
        rubric_path = ROOT / "docs" / "ANNOTATION_RUBRIC_PILOT.md"
        if guide_path.is_file():
            ttk.Button(btns, text="Mở annotation_guide.md", command=lambda: self._open_file(guide_path)).pack(side=tk.LEFT, padx=4)
        if rubric_path.is_file():
            ttk.Button(btns, text="Mở ANNOTATION_RUBRIC_PILOT.md", command=lambda: self._open_file(rubric_path)).pack(side=tk.LEFT, padx=4)

    @staticmethod
    def _open_file(path: Path) -> None:
        try:
            import os

            os.startfile(str(path))  # type: ignore[attr-defined]  # Windows-only by design
        except OSError as exc:
            messagebox.showwarning("Không mở được file", f"{path}\n{exc}")

    def _export_csv(self) -> None:
        if self.ctrl.dataset_path is None or not self.ctrl.dataset_path.is_file():
            messagebox.showwarning("Chưa có phiếu", "Hãy chọn người gán và bộ dữ liệu trước.")
            return
        person = self.ctrl.person_short or "annotator"
        default_name = f"{self.ctrl.dataset_path.parent.name}_{person}.csv"
        target = filedialog.asksaveasfilename(
            title="Sao lưu phiếu của bạn",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv"), ("Tất cả", "*.*")],
        )
        if not target:
            return
        try:
            shutil.copy2(self.ctrl.dataset_path, target)
        except OSError as exc:
            messagebox.showwarning("Chưa lưu được bản sao", str(exc))
            return
        messagebox.showinfo("Đã sao lưu", f"Đã tạo bản sao:\n{target}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root_dir = ROOT
    if not (root_dir / "data" / "annotations").is_dir():
        # Frozen exe launched outside the repo: let the user point at the project folder.
        import tkinter as _tk

        probe = _tk.Tk()
        probe.withdraw()
        chosen = filedialog.askdirectory(title="Chọn thư mục dự án (chứa data/annotations)")
        probe.destroy()
        if chosen:
            root_dir = Path(chosen)
            # Persist the choice so later launches skip the dialog.
            try:
                marker_path = Path(sys.executable).resolve().parent / "app_root.txt" if getattr(sys, "frozen", False) else ROOT / "app_root.txt"
                marker_path.write_text(str(root_dir), encoding="utf-8")
            except OSError:
                pass
    try:
        app = AnnotationAppDesktop(root_dir)
    except Exception:
        logger.exception("Lỗi khởi động tool")
        crash_log = Path(sys.executable).resolve().parent / "annotation_app_desktop_crash.log" if getattr(sys, "frozen", False) else ROOT / "annotation_app_desktop_crash.log"
        import traceback

        try:
            crash_log.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        raise
    app.mainloop()


if __name__ == "__main__":
    main()
