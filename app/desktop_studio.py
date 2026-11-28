#!/usr/bin/env python3
"""Native Windows Desktop Studio (.exe-ready) for fast, zero-reload behavioral annotation.

Eliminates Streamlit's full-script rerun latency by decoding each 4s window once into
RAM via OpenCV and storing all drafts/edits in native Tkinter state (0ms button clicks),
while reading/writing the exact same CSVs & AI suggestions as `app/annotation_app.py`.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.studio_view import (
    BEHAVIOR_CHOICES,
    _save_600_window,
    _save_train_15s_span,
    find_next_track_window_indices,
)
from src.contracts.annotation_io import read_annotation_csv
from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    HIGH,
    LOW,
    MODERATE,
)
from src.pose.ai_bundle import load_suggestions_map
from src.pose.annotation_preview import (
    compose_target_frame,
    fit_image,
    nearest_frame_boxes,
)

logger = logging.getLogger("desktop_studio")

PEOPLE_OPTIONS = [
    ("Tài", "Vũ Đức Tài", "a"),
    ("Phong", "Vũ Văn Phong", "b"),
    ("Quân", "Nguyễn Hữu Quân", "a"),
]

DATASET_CATALOG = [
    ("review_v03_independent_20261124", "🎯 600 cửa sổ chuẩn (Review V0.3)"),
    ("train_full_ai_assist", "🚂 Tập TRAIN đầy đủ (9.888 cửa sổ)"),
    ("val_full_ai_assist", "🧪 Tập VAL đầy đủ (8.617 cửa sổ)"),
    ("test_full_ai_assist", "🎯 Tập TEST đầy đủ (4.889 cửa sổ)"),
    ("external_review_20261109", "🆕 Hai video mới mở rộng (130 cửa sổ)"),
    ("human_review_v0_2", "📦 Bộ gán nhãn chính v0.2 (600 cửa sổ cũ)"),
    ("review_v03", "📋 Bộ gán nhãn v0.3 gốc (600 cửa sổ YOLO11)"),
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


def discover_datasets_for_person(root: Path, annotator_code: str) -> List[Tuple[str, str, Path]]:
    """Return existing (dir_name, label, csv_path) entries for the selected annotator."""
    ann_root = root / "data" / "annotations"
    target_name = f"annotator_{annotator_code}.csv"
    found: List[Tuple[str, str, Path]] = []
    seen = set()
    for dir_name, label in DATASET_CATALOG:
        cand = ann_root / dir_name / target_name
        if cand.is_file():
            found.append((dir_name, label, cand))
            seen.add(cand.resolve())
    for cand in sorted(ann_root.glob(f"**/{target_name}")):
        if cand.resolve() not in seen:
            found.append((cand.parent.name, f"📄 {cand.parent.name}", cand))
    return found


def resolve_video_and_pose_paths(root: Path, row: Dict[str, Any]) -> Tuple[Path, Path]:
    """Resolve raw session video and pose CSV paths matching studio_view logic."""
    session_id = str(row["session_id"])
    camera_id = str(row.get("camera_id", "cam_A") or "cam_A")
    v_str = str(row.get("video_path", "") or "").strip()
    v_path = Path(v_str) if v_str and Path(v_str).is_file() else (root / v_str if v_str else Path())
    if not v_str or not v_path.is_file():
        for cand in (
            root / "data/raw" / session_id / f"{camera_id}.mp4",
            root / "data/raw" / session_id / "cam_A.mp4",
            root / "data/raw" / f"{session_id}.mp4",
        ):
            if cand.is_file():
                v_path = cand
                break
    p_path = root / "data/poses" / f"{session_id}_{camera_id}_poses.csv"
    return v_path, p_path


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
        return [], [], f"Không thấy track #{track_id} trong khoảng {t_start_ms/1000:.1f}–{t_end_ms/1000:.1f}s."

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
                # Extract crop directly from the annotated right panel of canvas_rgb
                crop_rgb = canvas_rgb[48:552, 896:].copy()
                rel_s = (float(ts) - float(t_start_ms)) / 1000.0
                lbl = f"+{rel_s:.1f}s ({ts/1000:.1f}s)"
                strip_items.append((lbl, crop_rgb))

        status = (
            f"Đã dựng {len(loop_rgb)} khung hình (thấy track #{track_id} ở {matched}/{len(loop_rgb)} khung)."
            if loop_rgb
            else f"Không đọc được khung hình cho track #{track_id}."
        )
        return loop_rgb, strip_items, status
    finally:
        cap.release()


class DesktopStudioController:
    """Headless testable state + persistence manager for Desktop Studio."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.pose_cache: Dict[Path, pd.DataFrame] = {}
        self.suggestions_cache: Dict[Path, Dict[str, Dict[str, Any]]] = {}
        self.media_cache: OrderedDict[Tuple[str, bool, bool], Tuple[List[np.ndarray], List[Tuple[str, np.ndarray]], str]] = OrderedDict()
        self.track_zone_memory: Dict[str, str] = {}

        self.person_short: str = PEOPLE_OPTIONS[0][0]
        self.person_full: str = PEOPLE_OPTIONS[0][1]
        self.person_code: str = PEOPLE_OPTIONS[0][2]

        self.dataset_dir_name: str = ""
        self.dataset_path: Optional[Path] = None
        self.df: pd.DataFrame = pd.DataFrame()
        self.filtered_indices: List[int] = []
        self.cur_pos: int = 0

    def set_person(self, short_name: str) -> List[Tuple[str, str, Path]]:
        for s_n, full_n, code in PEOPLE_OPTIONS:
            if s_n == short_name:
                self.person_short, self.person_full, self.person_code = s_n, full_n, code
                break
        return discover_datasets_for_person(self.root, self.person_code)

    def load_dataset(self, csv_path: Path) -> pd.DataFrame:
        self.dataset_path = Path(csv_path)
        self.dataset_dir_name = self.dataset_path.parent.name
        self.df = read_annotation_csv(self.dataset_path)
        self.load_suggestions(force_reload=True)
        return self.df

    def done_mask(self) -> pd.Series:
        if self.df.empty:
            return pd.Series(dtype=bool)
        return (
            (self.df["annotation_state"].eq("VALID") & self.df["participation_label"].ne(""))
            | self.df["annotation_state"].isin(["OCCLUDED", "INVALID"])
        )

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
            if pose_path.is_file():
                self.pose_cache[pose_path] = pd.read_csv(pose_path)
            else:
                self.pose_cache[pose_path] = pd.DataFrame()
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

    def remembered_zone_for_row(self, row: pd.Series) -> str:
        stored = str(row.get("zone_id", "")).strip()
        if stored:
            return stored
        if self.dataset_path is None:
            return ""
        sess = str(row["session_id"])
        cam = str(row.get("camera_id", "cam_A"))
        trk = str(row["track_id"])
        key = f"{self.dataset_path.resolve()}:{sess}:{cam}:{trk}"
        if key in self.track_zone_memory:
            return self.track_zone_memory[key]
        same = self.df[
            self.df["session_id"].astype(str).eq(sess)
            & self.df["camera_id"].astype(str).eq(cam)
            & self.df["track_id"].astype(str).eq(trk)
        ]
        zones = same["zone_id"].fillna("").astype(str).str.strip()
        uniq = zones[zones.ne("")].unique()
        if len(uniq) == 1:
            self.track_zone_memory[key] = uniq[0]
            return uniq[0]
        return ""

    def save_current(
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
        if self.dataset_path is None:
            raise ValueError("Chưa chọn file CSV.")
        row = self.current_row()
        if row is None:
            raise ValueError("Không có cửa sổ đang chọn.")
        cur_idx = int(self.filtered_indices[self.cur_pos])
        clean_lbl = "" if target_state == "OCCLUDED" else target_label
        if target_state == "VALID":
            if phase == "UNKNOWN" or not phase_evidence.strip() or not behaviors or not confirmed or clean_lbl not in CANONICAL_TRAINING_LABELS:
                raise ValueError("Cần đủ trạng thái VALID, mức tham gia, hành vi, pha, căn cứ và xác nhận.")

        beh_str = "; ".join(behaviors)
        ai_src = (ai_suggestion or {}).get("suggestion_source", "")
        saved_ids: List[str] = []

        if span_15s and self.dataset_dir_name != "review_v03_independent_20261124" and str(row.get("split", "")).lower() == "train":
            future_idxs = find_next_track_window_indices(self.df, cur_idx, self.done_mask(), max_span_ms=15000.0)
            saved_ids = _save_train_15s_span(
                df_600=self.df,
                cur_row_idx=cur_idx,
                future_indices=future_idxs,
                target_label=clean_lbl,
                target_state=target_state,
                behaviors=beh_str,
                phase=phase,
                zone=zone,
                occlusion=occlusion,
                notes=notes,
                annotator_id=self.person_full,
                primary_file_path=self.dataset_path,
                person_name=self.person_short,
                phase_evidence=phase_evidence,
                confirmed=confirmed,
                ai_suggestion_source=ai_src,
                ai_suggestion=ai_suggestion,
            )
        else:
            _save_600_window(
                df_600=self.df,
                row_idx=cur_idx,
                target_label=clean_lbl,
                target_state=target_state,
                behaviors=beh_str,
                phase=phase,
                zone=zone,
                occlusion=occlusion,
                notes=notes,
                annotator_id=self.person_full,
                primary_file_path=self.dataset_path,
                person_name=self.person_short,
                phase_evidence=phase_evidence,
                confirmed=confirmed,
                ai_suggestion_source=ai_src,
                ai_suggestion=ai_suggestion,
            )
            saved_ids = [str(row["window_id"])]

        if zone:
            key = f"{self.dataset_path.resolve()}:{row['session_id']}:{row.get('camera_id', 'cam_A')}:{row['track_id']}"
            self.track_zone_memory[key] = zone

        # Refresh in-memory DataFrame cleanly from disk after atomic save.
        self.df = read_annotation_csv(self.dataset_path)
        return saved_ids


class DesktopStudioApp(tk.Tk):
    """Fast native Tkinter desktop window with async frame decode and 0ms interaction."""

    def __init__(self, root_dir: Path = ROOT):
        super().__init__()
        self.ctrl = DesktopStudioController(root_dir)
        self.title("CVF25PRJ10 · Desktop Studio Gán Nhãn Nhanh (Zero-Reload)")
        self.geometry("1420x860")
        self.minsize(1160, 720)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self._datasets: List[Tuple[str, str, Path]] = []
        self._opt_labels: List[str] = []
        self._loop_tk_frames: List[ImageTk.PhotoImage] = []
        self._strip_tk_imgs: List[ImageTk.PhotoImage] = []
        self._play_idx: int = 0
        self._playing: bool = True
        self._play_job: Optional[str] = None
        self._render_seq: int = 0
        self._ai_seen_cur: Optional[Dict[str, Any]] = None
        self._future_15s_count: int = 0

        self.var_person = tk.StringVar(value=PEOPLE_OPTIONS[0][0])
        self.var_dataset = tk.StringVar()
        self.var_filter = tk.StringVar(value="Chưa gán (còn lại)")
        self.var_session_filter = tk.StringVar()
        self.var_window_jump = tk.StringVar()

        self.var_skeleton = tk.BooleanVar(value=True)
        self.var_no_blur = tk.BooleanVar(value=True)
        self.var_auto_jump = tk.BooleanVar(value=True)
        self.var_ai_on = tk.BooleanVar(value=True)

        self.var_draft_label = tk.StringVar(value="")
        self.var_state = tk.StringVar(value="VALID")
        self.var_phase = tk.StringVar(value="lecture")
        self.var_zone = tk.StringVar(value="")
        self.var_occlusion = tk.StringVar(value="none")
        self.var_notes = tk.StringVar(value="")
        self.var_phase_basis = tk.StringVar(value="")
        self.var_phase_evidence = tk.StringVar(value="")
        self.var_confirmed = tk.BooleanVar(value=False)
        self.var_beh: Dict[str, tk.BooleanVar] = {code: tk.BooleanVar(value=False) for _, code in BEHAVIOR_CHOICES}

        self._build_ui()
        self._bind_shortcuts()
        self._on_person_changed()
        self._tick_video_loop()

    def _build_ui(self) -> None:
        # Top control bar
        top = ttk.Frame(self, padding=(10, 6))
        top.pack(fill=tk.X)

        ttk.Label(top, text="Người gán:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        cb_person = ttk.Combobox(
            top,
            textvariable=self.var_person,
            values=[p[0] for p in PEOPLE_OPTIONS],
            state="readonly",
            width=8,
        )
        cb_person.pack(side=tk.LEFT, padx=(4, 10))
        cb_person.bind("<<ComboboxSelected>>", lambda _e: self._on_person_changed())

        ttk.Label(top, text="Bộ dữ liệu:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.cb_dataset = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=34)
        self.cb_dataset.pack(side=tk.LEFT, padx=(4, 10))
        self.cb_dataset.bind("<<ComboboxSelected>>", lambda _e: self._on_dataset_changed())

        ttk.Label(top, text="Lọc:").pack(side=tk.LEFT)
        cb_filter = ttk.Combobox(
            top,
            textvariable=self.var_filter,
            values=["Tất cả", "Chưa gán (còn lại)", "Đã gán", "Theo phiên học..."],
            state="readonly",
            width=18,
        )
        cb_filter.pack(side=tk.LEFT, padx=(4, 6))
        cb_filter.bind("<<ComboboxSelected>>", lambda _e: self._on_filter_changed())

        self.cb_sess = ttk.Combobox(top, textvariable=self.var_session_filter, state="readonly", width=13)
        self.cb_sess.pack(side=tk.LEFT, padx=(0, 8))
        self.cb_sess.bind("<<ComboboxSelected>>", lambda _e: self._on_filter_changed())

        ttk.Button(top, text="⏮ Trước [←]", command=lambda: self._step_pos(-1), width=11).pack(side=tk.LEFT, padx=2)
        self.cb_window = ttk.Combobox(top, textvariable=self.var_window_jump, state="readonly", width=44)
        self.cb_window.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.cb_window.bind("<<ComboboxSelected>>", lambda _e: self._on_window_selected())
        ttk.Button(top, text="Tiếp [→] ⏭", command=lambda: self._step_pos(1), width=11).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="⏩ Chưa gán kế", command=self._jump_next_unfinished, width=14).pack(side=tk.LEFT, padx=(4, 0))

        # Second bar: toggles & progress
        bar2 = ttk.Frame(self, padding=(10, 2, 10, 6))
        bar2.pack(fill=tk.X)
        ttk.Checkbutton(bar2, text="Khung xương SV", variable=self.var_skeleton, command=self._reload_current_media).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(bar2, text="Bỏ che mặt (riêng tư)", variable=self.var_no_blur, command=self._reload_current_media).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(bar2, text="Tự nhảy cửa sổ kế", variable=self.var_auto_jump).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(bar2, text="🤖 Gợi ý AI (mặc định bật — bỏ tích để chấm mù)", variable=self.var_ai_on, command=self._on_ai_toggle).pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_progress = ttk.Label(bar2, text="Tiến độ: 0/0", font=("Segoe UI", 9, "bold"), foreground="#1a5fb4")
        self.lbl_progress.pack(side=tk.RIGHT)
        self.prog_bar = ttk.Progressbar(bar2, mode="determinate", length=180)
        self.prog_bar.pack(side=tk.RIGHT, padx=(0, 8))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Main workspace split
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Left column: video canvas + 5 strip
        left = ttk.Frame(paned)
        paned.add(left, weight=6)

        vid_hdr = ttk.Frame(left)
        vid_hdr.pack(fill=tk.X, pady=(0, 4))
        self.lbl_media_status = ttk.Label(vid_hdr, text="Đang tải khung hình...", font=("Segoe UI", 9, "bold"))
        self.lbl_media_status.pack(side=tk.LEFT)
        self.btn_play = ttk.Button(vid_hdr, text="⏸ Tạm dừng [Space]", command=self._toggle_play, width=18)
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
            txt_lbl = ttk.Label(cell, text=f"#{idx + 1}", font=("Segoe UI", 8))
            txt_lbl.pack()
            self.strip_panels.append((img_lbl, txt_lbl))

        # Right column: fast scrollable form
        right_outer = ttk.Frame(paned)
        paned.add(right_outer, weight=4)

        canvas_scroll = tk.Canvas(right_outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self.right_form = ttk.Frame(canvas_scroll, padding=(6, 4))
        self.right_form.bind("<Configure>", lambda _e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=self.right_form, anchor="nw")
        canvas_scroll.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_win_header = ttk.Label(self.right_form, text="👤 Cửa sổ", font=("Segoe UI", 11, "bold"), wraplength=480)
        self.lbl_win_header.pack(anchor=tk.W, pady=(0, 6))

        # AI suggestion card
        self.ai_frame = ttk.LabelFrame(self.right_form, text="🤖 Đề xuất AI", padding=8)
        self.ai_frame.pack(fill=tk.X, pady=(0, 8))
        self.lbl_ai_summary = ttk.Label(self.ai_frame, text="", font=("Segoe UI", 9, "bold"), wraplength=460)
        self.lbl_ai_summary.pack(anchor=tk.W)
        self.lbl_ai_detail = ttk.Label(self.ai_frame, text="", font=("Segoe UI", 8), wraplength=460, foreground="#444444")
        self.lbl_ai_detail.pack(anchor=tk.W, pady=(2, 4))
        ttk.Button(self.ai_frame, text="📝 Điền lại gợi ý vào phiếu", command=self._apply_current_ai_to_draft).pack(anchor=tk.W)

        # 4 label buttons
        lbl_grid = ttk.Frame(self.right_form)
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
                lbl_grid,
                text=text_b,
                font=("Segoe UI", 10, "bold"),
                pady=7,
                relief=tk.RAISED,
                command=lambda lv=l_val, sv=s_val: self._choose_label(lv, sv),
            )
            btn.grid(row=r_i, column=c_i, sticky="ew", padx=3, pady=3)
            self.label_buttons[(l_val, s_val)] = btn
        self._btn_colors = {(l_val, s_val): bg_hex for _, _, _, l_val, s_val, bg_hex in btn_specs}

        self.lbl_selected_badge = ttk.Label(self.right_form, text="Chưa chọn mức tham gia.", font=("Segoe UI", 9, "bold"), foreground="#0b5394")
        self.lbl_selected_badge.pack(anchor=tk.W, pady=(2, 6))

        # 8 behavior checkboxes
        beh_box = ttk.LabelFrame(self.right_form, text="Hành vi quan sát được (tích chọn ≥ 1)", padding=6)
        beh_box.pack(fill=tk.X, pady=(0, 8))
        beh_box.columnconfigure(0, weight=1)
        beh_box.columnconfigure(1, weight=1)
        for i, (b_name, b_code) in enumerate(BEHAVIOR_CHOICES):
            ttk.Checkbutton(beh_box, text=b_name, variable=self.var_beh[b_code]).grid(
                row=i // 2, column=i % 2, sticky="w", padx=4, pady=2
            )

        # Context fields
        ctx = ttk.LabelFrame(self.right_form, text="Ngữ cảnh & Vị trí", padding=6)
        ctx.pack(fill=tk.X, pady=(0, 8))
        for c in (1, 3):
            ctx.columnconfigure(c, weight=1)
        ttk.Label(ctx, text="Trạng thái:").grid(row=0, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(ctx, textvariable=self.var_state, values=STATE_OPTIONS, state="readonly", width=13).grid(row=0, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(ctx, text="Pha học:").grid(row=0, column=2, sticky="w", padx=3, pady=2)
        cb_phase = ttk.Combobox(ctx, textvariable=self.var_phase, values=PHASE_OPTIONS, state="readonly", width=15)
        cb_phase.grid(row=0, column=3, sticky="ew", padx=3, pady=2)
        cb_phase.bind("<<ComboboxSelected>>", lambda _e: self._refresh_phase_basis_options())

        ttk.Label(ctx, text="Vùng ghế:").grid(row=1, column=0, sticky="w", padx=3, pady=2)
        ttk.Combobox(ctx, textvariable=self.var_zone, values=ZONE_OPTIONS, state="readonly", width=13).grid(row=1, column=1, sticky="ew", padx=3, pady=2)
        ttk.Label(ctx, text="Che khuất:").grid(row=1, column=2, sticky="w", padx=3, pady=2)
        ttk.Combobox(ctx, textvariable=self.var_occlusion, values=OCCLUSION_OPTIONS, state="readonly", width=15).grid(row=1, column=3, sticky="ew", padx=3, pady=2)

        ttk.Label(ctx, text="Căn cứ pha:").grid(row=2, column=0, sticky="w", padx=3, pady=2)
        self.cb_phase_basis = ttk.Combobox(ctx, textvariable=self.var_phase_basis, state="readonly")
        self.cb_phase_basis.grid(row=2, column=1, columnspan=3, sticky="ew", padx=3, pady=2)
        self.cb_phase_basis.bind("<<ComboboxSelected>>", lambda _e: self._on_basis_picked())

        ttk.Label(ctx, text="Căn cứ khác:").grid(row=3, column=0, sticky="w", padx=3, pady=2)
        ttk.Entry(ctx, textvariable=self.var_phase_evidence).grid(row=3, column=1, columnspan=3, sticky="ew", padx=3, pady=2)

        ttk.Label(ctx, text="Ghi chú:").grid(row=4, column=0, sticky="w", padx=3, pady=2)
        ttk.Entry(ctx, textvariable=self.var_notes).grid(row=4, column=1, columnspan=3, sticky="ew", padx=3, pady=2)

        ttk.Checkbutton(
            self.right_form,
            text="Tôi đã xem toàn bộ clip, xác định đúng người và bằng chứng",
            variable=self.var_confirmed,
        ).pack(anchor=tk.W, pady=(2, 6))

        self.btn_save = tk.Button(
            self.right_form,
            text="✅ Duyệt & Lưu HUMAN & Tiếp [Enter]",
            font=("Segoe UI", 11, "bold"),
            bg="#198754",
            fg="white",
            pady=9,
            command=lambda: self._save_and_advance(span_15s=False),
        )
        self.btn_save.pack(fill=tk.X, pady=(0, 6))

        # 15s train span panel
        self.span15_frame = ttk.LabelFrame(self.right_form, text="⏱️ Gán nhanh ~15s cùng Track (chỉ tập TRAIN)", padding=8)
        self.lbl_15s_info = ttk.Label(self.span15_frame, text="", wraplength=460, font=("Segoe UI", 8))
        self.lbl_15s_info.pack(anchor=tk.W, pady=(0, 4))
        self.btn_save_15s = tk.Button(
            self.span15_frame,
            text="⚡ Lưu & Gán nhanh ~15s [F]",
            font=("Segoe UI", 10, "bold"),
            bg="#0d6efd",
            fg="white",
            pady=6,
            command=lambda: self._save_and_advance(span_15s=True),
        )
        self.btn_save_15s.pack(fill=tk.X)

        self.lbl_status_footer = ttk.Label(
            self,
            text="Phím tắt: [1] CAO · [2] VỪA · [3] THẤP · [4] BỊ CHE · [Enter / Ctrl+S] Lưu & Tiếp · [F] Gán 15s Train · [←/→] Đổi cửa sổ",
            padding=(10, 4),
            foreground="#555555",
        )
        self.lbl_status_footer.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_shortcuts(self) -> None:
        def _is_typing(event) -> bool:
            return isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text))

        self.bind("1", lambda e: None if _is_typing(e) else self._choose_label(HIGH, "VALID"))
        self.bind("2", lambda e: None if _is_typing(e) else self._choose_label(MODERATE, "VALID"))
        self.bind("3", lambda e: None if _is_typing(e) else self._choose_label(LOW, "VALID"))
        self.bind("4", lambda e: None if _is_typing(e) else self._choose_label("", "OCCLUDED"))
        self.bind("<Return>", lambda _e: self._save_and_advance(span_15s=False))
        self.bind("<Control-s>", lambda _e: self._save_and_advance(span_15s=False))
        self.bind("f", lambda e: None if _is_typing(e) else self._save_and_advance(span_15s=True))
        self.bind("<Left>", lambda e: None if _is_typing(e) else self._step_pos(-1))
        self.bind("<Right>", lambda e: None if _is_typing(e) else self._step_pos(1))
        self.bind("<space>", lambda e: None if _is_typing(e) else self._toggle_play())

    def _on_person_changed(self) -> None:
        self._datasets = self.ctrl.set_person(self.var_person.get())
        labels = [lbl for _, lbl, _ in self._datasets]
        self.cb_dataset["values"] = labels
        if labels:
            self.var_dataset.set(labels[0])
            self._on_dataset_changed()

    def _on_dataset_changed(self) -> None:
        sel_lbl = self.var_dataset.get()
        match = next((p for _, lbl, p in self._datasets if lbl == sel_lbl), None)
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
            f"{'✓' if done.at[i] else '○'} #{i + 1}/{total} [{sp}] {sess} | Trk #{trk} ({float(t0 or 0)/1000:g}-{float(t1 or 0)/1000:g}s) → {lbl or st_v or 'Chưa gán'}"
            for i, sp, sess, trk, t0, t1, lbl, st_v in zip(
                idxs, splits, sub["session_id"], sub["track_id"],
                sub["t_start_ms"], sub["t_end_ms"],
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
            self.var_window_jump.set(self._opt_labels[rem[0]])
            self._load_current_window()

    def _refresh_phase_basis_options(self) -> None:
        opts = [""] + PHASE_BASIS_DEFAULT.get(self.var_phase.get(), [])
        self.cb_phase_basis["values"] = opts

    def _on_basis_picked(self) -> None:
        picked = self.var_phase_basis.get().strip()
        if picked and not self.var_phase_evidence.get().strip():
            self.var_phase_evidence.set(picked)

    def _choose_label(self, label: str, state: str) -> None:
        self.var_draft_label.set(label)
        self.var_state.set(state)
        self._update_label_button_highlights()

    def _update_label_button_highlights(self) -> None:
        cur_lbl = self.var_draft_label.get()
        cur_st = self.var_state.get()
        names = {(HIGH, "VALID"): "🟢 CAO (HIGH)", (MODERATE, "VALID"): "🟡 VỪA (MODERATE)", (LOW, "VALID"): "🔴 THẤP (LOW)", ("", "OCCLUDED"): "⚪ BỊ CHE (OCCLUDED)"}
        selected_txt = None
        for key, btn in self.label_buttons.items():
            lv, sv = key
            active = cur_st == sv and (sv != "VALID" or cur_lbl == lv)
            base_bg = self._btn_colors[key]
            if active:
                selected_txt = names[key]
                btn.config(bg=base_bg, fg="white", relief=tk.SUNKEN)
            else:
                btn.config(bg="#f0f0f0", fg="#222222", relief=tk.RAISED)
        self.lbl_selected_badge.config(
            text=f"✓ Đang chọn: {selected_txt}" if selected_txt else "Chưa chọn mức tham gia."
        )

    def _on_ai_toggle(self) -> None:
        self._sync_ai_panel()

    def _sync_ai_panel(self) -> None:
        row = self.ctrl.current_row()
        if row is None:
            return
        wid = str(row["window_id"])
        suggestions = self.ctrl.load_suggestions()
        sug = suggestions.get(wid) if self.var_ai_on.get() else None
        self._ai_seen_cur = dict(sug) if sug else None

        if sug is not None:
            self.ai_frame.pack(fill=tk.X, pady=(0, 8), after=self.lbl_win_header)
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
                self._reset_form_from_row(row)

    def _apply_current_ai_to_draft(self) -> None:
        sug = self._ai_seen_cur
        if not sug:
            return
        self.var_draft_label.set(sug.get("suggested_label", ""))
        self.var_state.set(sug.get("suggested_state", "VALID"))
        s_behs = set(sug.get("suggested_behaviors", []))
        for _, code in BEHAVIOR_CHOICES:
            self.var_beh[code].set(code in s_behs)
        if sug.get("suggested_phase") in ("lecture", "individual_work", "discussion"):
            self.var_phase.set(sug["suggested_phase"])
        self._refresh_phase_basis_options()
        if sug.get("suggested_occlusion") in OCCLUSION_OPTIONS:
            self.var_occlusion.set(sug["suggested_occlusion"])
        if sug.get("suggested_phase_evidence"):
            self.var_phase_evidence.set(sug["suggested_phase_evidence"])
        if sug.get("rationale"):
            self.var_notes.set(str(sug["rationale"])[:240])
        self._update_label_button_highlights()

    def _reset_form_from_row(self, row: pd.Series) -> None:
        p_lbl = str(row.get("participation_label", ""))
        st_v = str(row.get("annotation_state", "")) or "VALID"
        self.var_draft_label.set(p_lbl)
        self.var_state.set(st_v if st_v in STATE_OPTIONS else "VALID")
        raw_beh = str(row.get("atomic_behavior", ""))
        cur_behs = {b.strip() for b in raw_beh.split(";") if b.strip()}
        for _, code in BEHAVIOR_CHOICES:
            self.var_beh[code].set(code in cur_behs)

        reviewed = bool(str(row.get("annotator_id", "")).strip()) and st_v in ANNOTATION_STATES
        p_val = str(row.get("class_phase", "lecture")) if reviewed else "lecture"
        self.var_phase.set(p_val if p_val in PHASE_OPTIONS and p_val != "UNKNOWN" else "lecture")
        self._refresh_phase_basis_options()
        self.var_phase_basis.set("")
        self.var_phase_evidence.set(str(row.get("phase_evidence", "")))
        self.var_zone.set(self.ctrl.remembered_zone_for_row(row))
        o_val = str(row.get("occlusion_level", "none") or "none")
        self.var_occlusion.set(o_val if o_val in OCCLUSION_OPTIONS else "none")
        self.var_notes.set(str(row.get("notes", "")))
        self.var_confirmed.set(False)
        self._update_label_button_highlights()

    def _load_current_window(self) -> None:
        total = len(self.ctrl.df)
        done_cnt = int(self.ctrl.done_mask().sum()) if total else 0
        self.lbl_progress.config(text=f"Tiến độ: {done_cnt}/{total} ({done_cnt/max(1, total)*100:.1f}%)")
        self.prog_bar["maximum"] = max(1, total)
        self.prog_bar["value"] = done_cnt

        row = self.ctrl.current_row()
        if row is None:
            self.lbl_win_header.config(text="Không có cửa sổ phù hợp bộ lọc.")
            return

        cur_idx = int(self.ctrl.filtered_indices[self.ctrl.cur_pos])
        disp = row.get("participation_label") or row.get("annotation_state") or "Chưa gán"
        self.lbl_win_header.config(
            text=f"👤 #{cur_idx + 1}/{total} · Track #{row['track_id']} ({row['session_id']} · {float(row['t_start_ms'])/1000:g}–{float(row['t_end_ms'])/1000:g}s) → {disp}"
        )
        self._reset_form_from_row(row)
        self._sync_ai_panel()

        # 15s train quick-label panel
        is_train_full = self.ctrl.dataset_dir_name != "review_v03_independent_20261124" and str(row.get("split", "")).lower() == "train"
        fut = (
            find_next_track_window_indices(self.ctrl.df, cur_idx, self.ctrl.done_mask(), max_span_ms=15000.0)
            if is_train_full
            else []
        )
        self._future_15s_count = len(fut)
        if fut:
            span_end = max(float(self.ctrl.df.at[i, "t_end_ms"]) for i in fut)
            self.lbl_15s_info.config(
                text=f"Track #{row['track_id']} còn {len(fut)} cửa sổ chưa gán trong khoảng {float(row['t_start_ms'])/1000:g}s → {span_end/1000:g}s."
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
            v_path, p_path = resolve_video_and_pose_paths(self.ctrl.root, row_dict)
            pose_df = self.ctrl.get_pose_df(p_path)
            loop_rgb, strip_items, status = extract_window_media_in_memory(
                v_path,
                pose_df,
                str(row_dict["session_id"]),
                str(row_dict.get("camera_id", "cam_A") or "cam_A"),
                str(row_dict["track_id"]),
                float(row_dict["t_start_ms"]),
                float(row_dict["t_end_ms"]),
                anonymize=anonymize,
                skeleton=skeleton,
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
        # Fit canvas to current panel size cleanly
        w_target = max(720, self.canvas_video.winfo_width() or 780)
        h_target = max(340, min(460, int(w_target * 592 / 1280)))
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
                tk_crop = ImageTk.PhotoImage(Image.fromarray(crop_arr).resize((136, 178), Image.Resampling.BILINEAR))
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

    def _save_and_advance(self, *, span_15s: bool = False) -> None:
        if span_15s and self._future_15s_count <= 0:
            return
        selected_behs = [code for _, code in BEHAVIOR_CHOICES if self.var_beh[code].get()]
        pe = self.var_phase_evidence.get().strip() or self.var_phase_basis.get().strip()
        confirmed = bool(self.var_confirmed.get() or self._ai_seen_cur or self.var_draft_label.get())
        try:
            saved_ids = self.ctrl.save_current(
                target_label=self.var_draft_label.get(),
                target_state=self.var_state.get(),
                behaviors=selected_behs,
                phase=self.var_phase.get(),
                zone=self.var_zone.get(),
                occlusion=self.var_occlusion.get(),
                notes=self.var_notes.get(),
                phase_evidence=pe,
                confirmed=confirmed,
                ai_suggestion=self._ai_seen_cur,
                span_15s=span_15s,
            )
        except (ValueError, OSError) as exc:
            messagebox.showwarning("Chưa thể lưu", str(exc))
            return

        saved_set = set(saved_ids)
        self.lbl_status_footer.config(text=f"✅ Đã lưu {len(saved_ids)} cửa sổ ({saved_ids[0]}) xuống ổ cứng.")
        if self.var_auto_jump.get() or self._ai_seen_cur or span_15s:
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
        self._refresh_window_dropdown()
        self._load_current_window()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app = DesktopStudioApp(ROOT)
    app.mainloop()


if __name__ == "__main__":
    main()
