"""Standalone Native Desktop GUI v3.0 for Dataset Annotation & Video Preprocessing (Task T11 & T12).

Observable Learning Participation Analysis (CVF25PRJ10 • CVFP26BCPPRJC012)
Owner: Vũ Văn Phong (Data Lead) & Nguyễn Dũng Tiến (UI/Reporting Lead - B23DCPT306)

v3.0 Redesign Highlights:
- Clean, intuitive 3-Step Workflow:
    Bước 1: ✂️ Cắt Video & Tự Động Gán Nhãn
    Bước 2: 👁️ Duyệt & Sửa Nhãn Từng Cửa Sổ 4s
    Bước 3: 📊 Thống Kê, Kiểm Định & Xuất File CSV
- Harmonious Color Hierarchy:
    Neutral slate/white surfaces, single primary CTA color (#2563EB),
    and semantic label buttons that highlight ONLY the currently active label.
- Zero Text Clipping:
    Concise Vietnamese labels, spacious two-column layout, and responsive image scaling.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


def find_project_root() -> Path:
    """Locate project root containing `data/` whether running from source or PyInstaller `.exe`."""
    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([exe_dir, exe_dir.parent, exe_dir.parent.parent, Path.cwd()])
    else:
        file_dir = Path(__file__).resolve().parent
        candidates.extend([file_dir.parent, Path.cwd(), file_dir])

    for cand in candidates:
        if (cand / "data").exists() or (cand / "configs").exists():
            return cand
    return candidates[0]


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_TRAINING_LABELS,
    HIGH,
    LABEL_SOURCES,
    LOW,
    MODERATE,
    to_display_label,
    validate_annotation_label,
)
from src.pose.anonymizer import STATE_COLORS_BGR, blur_head_roi, draw_skeleton
from src.preprocessing.weak_labels import RULE_VERSION, apply_labeling_rules

REQUIRED_ANNOTATION_COLUMNS = [
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

# Concise Vietnamese display names so checkboxes never overflow
BEHAVIOR_ITEMS = [
    ("BOARD_OR_INSTRUCTOR_FACING", "Nhìn bảng / Giảng viên"),
    ("READING_WRITING", "Đọc sách / Viết / Gõ phím"),
    ("HAND_RAISED", "Giơ tay phát biểu"),
    ("PEER_ORIENTED", "Quay sang thảo luận nhóm"),
    ("HEAD_DOWN", "Cúi gục đầu kéo dài"),
    ("BODY_TURNED", "Xoay người hướng khác"),
    ("OCCLUDED", "Bị che khuất khớp xương"),
    ("UNKNOWN", "Không rõ hành vi"),
]

OCCLUSION_LEVELS = ["none", "moderate", "heavy"]
CLASS_PHASES = ["lecture", "individual_work", "discussion"]
ZONE_IDS = ["B-L", "B-C", "B-R", "M-L", "M-C", "M-R", "F-L", "F-C", "F-R"]

# Harmonious Color Palette
PALETTE = {
    "bg_app": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "bg_header": "#0F172A",
    "bg_subbar": "#F1F5F9",
    "border": "#E2E8F0",
    "text_main": "#0F172A",
    "text_muted": "#475569",
    "primary": "#2563EB",
    "primary_light": "#DBEAFE",
    "primary_text": "#1E40AF",
    # Active vs Inactive semantic colors for label buttons
    "high_active_bg": "#16A34A",
    "high_active_fg": "#FFFFFF",
    "high_idle_bg": "#DCFCE7",
    "high_idle_fg": "#166534",
    "mod_active_bg": "#D97706",
    "mod_active_fg": "#FFFFFF",
    "mod_idle_bg": "#FEF3C7",
    "mod_idle_fg": "#92400E",
    "low_active_bg": "#DC2626",
    "low_active_fg": "#FFFFFF",
    "low_idle_bg": "#FEE2E2",
    "low_idle_fg": "#991B1B",
    "occ_active_bg": "#475569",
    "occ_active_fg": "#FFFFFF",
    "occ_idle_bg": "#E2E8F0",
    "occ_idle_fg": "#334155",
}


# =====================================================================
# DATA & POSE HELPERS
# =====================================================================

def load_poses_table(pose_path: Path):
    import pandas as pd

    if not pose_path.exists():
        csv_fb = pose_path.with_suffix(".csv")
        if csv_fb.exists():
            return pd.read_csv(csv_fb)
        return None
    if pose_path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(pose_path)
        except Exception:
            csv_fb = pose_path.with_suffix(".csv")
            if csv_fb.exists():
                return pd.read_csv(csv_fb)
            raise
    return pd.read_csv(pose_path)


def row_to_keypoints_array(pose_row) -> np.ndarray:
    kpts = np.zeros((17, 3), dtype=np.float32)
    for idx in range(17):
        kpts[idx, 0] = float(pose_row.get(f"kpt_{idx:02d}_x", 0.0))
        kpts[idx, 1] = float(pose_row.get(f"kpt_{idx:02d}_y", 0.0))
        kpts[idx, 2] = float(pose_row.get(f"kpt_{idx:02d}_conf", 0.0))
    return kpts


def render_crop_with_pose(
    frame: np.ndarray,
    pose_row,
    draw_skel: bool = True,
    blur_face: bool = False,
    color_bgr: Tuple[int, int, int] = (46, 204, 113),
    out_size: Tuple[int, int] = (156, 185),
) -> np.ndarray:
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
            draw_skeleton(work, kpts, color=color_bgr, conf_threshold=0.25)

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


def compute_cohens_kappa(labels_a: List[str], labels_b: List[str], categories: List[str]) -> float:
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
    pe = float(np.sum(mat.sum(axis=1) * mat.sum(axis=0))) / (total * total)
    if abs(1.0 - pe) < 1e-9:
        return 1.0
    return float((po - pe) / (1.0 - pe))


# =====================================================================
# MAIN APPLICATION
# =====================================================================

class AnnotationStudioDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CVF25PRJ10 — Công Cụ Cắt Video & Gán Nhãn Dữ Liệu (v3.0)")
        self.root.geometry("1440x880")
        self.root.minsize(1260, 780)
        self.root.configure(bg=PALETTE["bg_app"])

        # Configure ttk Theme
        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self.style.configure("TFrame", background=PALETTE["bg_app"])
        self.style.configure("TNotebook", background=PALETTE["bg_app"], borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 10, "bold"),
            padding=[18, 8],
            background="#E2E8F0",
            foreground="#334155",
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", "#2563EB")],
            foreground=[("selected", "#FFFFFF")],
        )

        # File paths & User state
        self.video_path_var = tk.StringVar(value=str((ROOT / "data/raw/session_001/cam_A_pilot.mp4").as_posix()))
        self.pose_path_var = tk.StringVar(value=str((ROOT / "data/poses/cam_A_pilot_poses.parquet").as_posix()))
        self.meta_path_var = tk.StringVar(value=str((ROOT / "data/raw/session_001/metadata.json").as_posix()))
        self.csv_path_var = tk.StringVar(value=str((ROOT / "data/annotations/session_001_weak_labels.csv").as_posix()))
        self.annotator_var = tk.StringVar(value="Tien_B23DCPT306")

        self.opt_draw_skel = tk.BooleanVar(value=True)
        self.opt_blur_face = tk.BooleanVar(value=False)
        self.opt_draw_grid = tk.BooleanVar(value=True)

        # Runtime Data
        self.poses_df = None
        self.rows: List[Dict[str, str]] = []
        self.filtered_indices: List[int] = []
        self.current_idx: int = 0
        self.is_playing_clip: bool = False

        # Image references
        self.filmstrip_imgs: List[ImageTk.PhotoImage] = []
        self.context_img: Optional[ImageTk.PhotoImage] = None
        self.trim_start_img: Optional[ImageTk.PhotoImage] = None
        self.trim_end_img: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._bind_shortcuts()
        self.reload_all_resources()

    # -----------------------------------------------------------------
    # TOP-LEVEL UI LAYOUT
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        # 1. Clean Header Bar
        header = tk.Frame(self.root, bg=PALETTE["bg_header"], padx=18, pady=10)
        header.pack(fill="x")

        tk.Label(
            header,
            text="🎓 CVF25PRJ10 — CÔNG CỤ CẮT VIDEO & GÁN NHÃN DỮ LIỆU",
            bg=PALETTE["bg_header"],
            fg="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        tk.Button(
            header,
            text="⚙️ Đổi File / Cài Đặt",
            command=self.open_settings_dialog,
            bg="#1E293B",
            fg="#E2E8F0",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
        ).pack(side="right", padx=(10, 0))

        self.header_status_lbl = tk.Label(
            header,
            text="Đang tải dữ liệu...",
            bg=PALETTE["bg_header"],
            fg="#93C5FD",
            font=("Segoe UI", 9),
        )
        self.header_status_lbl.pack(side="right")

        # 2. Quick Workflow Guide Strip
        guide_strip = tk.Frame(self.root, bg="#EFF6FF", highlightbackground="#BFDBFE", highlightthickness=1, padx=14, pady=5)
        guide_strip.pack(fill="x", padx=12, pady=(6, 2))

        tk.Label(
            guide_strip,
            text=(
                "💡 QUY TRÌNH 3 BƯỚC:   "
                "① [Bước 1] Cắt video & AI gán nhãn tự động   ➔   "
                "② [Bước 2] Bấm phím 1, 2, 3, 4 hoặc Enter để duyệt từng cửa sổ 4s   ➔   "
                "③ [Bước 3] Kiểm tra & Xuất CSV"
            ),
            bg="#EFF6FF",
            fg="#1E3A8A",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        # 3. Notebook Ordered by Workflow (Step 1 -> Step 2 -> Step 3)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=6)

        self.tab_step1_trim = ttk.Frame(self.notebook)
        self.tab_step2_review = ttk.Frame(self.notebook)
        self.tab_step3_export = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_step1_trim, text="  BƯỚC 1: ✂️ Cắt Video & Tự Động Gán Nhãn  ")
        self.notebook.add(self.tab_step2_review, text="  BƯỚC 2: 👁️ Duyệt & Sửa Nhãn Từng Sinh Viên (Chính)  ")
        self.notebook.add(self.tab_step3_export, text="  BƯỚC 3: 📊 Thống Kê & Xuất File CSV Chuẩn  ")

        self._build_step1_trim()
        self._build_step2_review()
        self._build_step3_export()

        # Default to Step 2 if CSV already has 427 windows loaded, or let user click Step 1 easily
        self.notebook.select(self.tab_step2_review)

    # -----------------------------------------------------------------
    # SETTINGS DIALOG (Keeps main UI free of long truncated file paths!)
    # -----------------------------------------------------------------
    def open_settings_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Cài đặt Đường dẫn File & Người gán nhãn")
        win.geometry("680x280")
        win.configure(bg="#FFFFFF")
        win.transient(self.root)
        win.grab_set()

        pad_frame = tk.Frame(win, bg="#FFFFFF", padx=20, pady=16)
        pad_frame.pack(fill="both", expand=True)

        rows_cfg = [
            ("👤 Mã Người Gán (Annotator ID):", self.annotator_var, None),
            ("🎥 File Video Lớp Học (.mp4):", self.video_path_var, self._browse_video),
            ("🦴 File Khung Xương Pose (.parquet/.csv):", self.pose_path_var, self._browse_pose),
            ("📄 File Lưu Kết Quả Gán Nhãn (.csv):", self.csv_path_var, self._browse_csv),
        ]

        for idx, (label_text, var, browse_cmd) in enumerate(rows_cfg):
            tk.Label(pad_frame, text=label_text, bg="#FFFFFF", font=("Segoe UI", 9, "bold")).grid(
                row=idx, column=0, sticky="w", pady=8
            )
            tk.Entry(pad_frame, textvariable=var, width=48, font=("Segoe UI", 9), bg="#F8FAFC").grid(
                row=idx, column=1, sticky="ew", padx=8, pady=8
            )
            if browse_cmd:
                tk.Button(
                    pad_frame,
                    text="Chọn file...",
                    command=browse_cmd,
                    bg="#E2E8F0",
                    fg="#0F172A",
                    relief="flat",
                    padx=10,
                ).grid(row=idx, column=2, pady=8)

        pad_frame.columnconfigure(1, weight=1)

        tk.Button(
            pad_frame,
            text="✓ Lưu Cài Đặt & Nạp Lại Dữ Liệu",
            command=lambda: [self.reload_all_resources(), win.destroy()],
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=20,
            pady=8,
        ).grid(row=4, column=0, columnspan=3, pady=(16, 0))

    # -----------------------------------------------------------------
    # BƯỚC 1: CẮT VIDEO & TỰ ĐỘNG GÁN NHÃN
    # -----------------------------------------------------------------
    def _build_step1_trim(self) -> None:
        card = tk.Frame(
            self.tab_step1_trim,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=20,
            pady=16,
        )
        card.pack(fill="both", expand=True, pady=4)

        tk.Label(
            card,
            text="BƯỚC 1: CẮT ĐOẠN VIDEO & CHẠY AI TỰ ĐỘNG GÁN NHÃN SƠ BỘ",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            text="Chọn khoảng thời gian video bạn muốn lấy (mặc định từ giây 0 đến giây 60), sau đó bấm nút màu xanh để máy tự động cắt thành các cửa sổ 4 giây và gán nhãn.",
            bg="#FFFFFF",
            fg="#475569",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 12))

        # Parameter bar
        p_bar = tk.Frame(card, bg="#F8FAFC", highlightbackground=PALETTE["border"], highlightthickness=1, padx=16, pady=12)
        p_bar.pack(fill="x")

        tk.Label(p_bar, text="⏱️ Cắt từ giây:", bg="#F8FAFC", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.trim_start_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(
            p_bar, from_=0.0, to=3600.0, increment=5.0, textvariable=self.trim_start_var,
            width=8, font=("Segoe UI", 10), command=self.update_trim_previews
        ).pack(side="left", padx=(6, 18))

        tk.Label(p_bar, text="⏱️ Đến giây:", bg="#F8FAFC", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.trim_end_var = tk.DoubleVar(value=60.0)
        tk.Spinbox(
            p_bar, from_=5.0, to=3600.0, increment=5.0, textvariable=self.trim_end_var,
            width=8, font=("Segoe UI", 10), command=self.update_trim_previews
        ).pack(side="left", padx=(6, 18))

        tk.Label(p_bar, text="Độ dài cửa sổ (s):", bg="#F8FAFC", font=("Segoe UI", 9)).pack(side="left")
        self.win_sec_var = tk.DoubleVar(value=4.0)
        tk.Spinbox(p_bar, from_=2.0, to=10.0, increment=1.0, textvariable=self.win_sec_var, width=5, font=("Segoe UI", 9)).pack(side="left", padx=(4, 14))

        tk.Label(p_bar, text="Bước trượt (s):", bg="#F8FAFC", font=("Segoe UI", 9)).pack(side="left")
        self.stride_sec_var = tk.DoubleVar(value=2.0)
        tk.Spinbox(p_bar, from_=1.0, to=5.0, increment=1.0, textvariable=self.stride_sec_var, width=5, font=("Segoe UI", 9)).pack(side="left", padx=4)

        tk.Button(
            p_bar,
            text="🔄 Xem trước Điểm Đầu & Điểm Cuối",
            command=self.update_trim_previews,
            bg="#E2E8F0",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=4,
        ).pack(side="right")

        # Main Action Buttons Row
        btn_bar = tk.Frame(card, bg="#FFFFFF", pady=14)
        btn_bar.pack(fill="x")

        tk.Button(
            btn_bar,
            text="🚀 BẮT ĐẦU: Tự Động Cắt Cửa Sổ 4s & Gán Nhãn Ngay ➔ Chuyển Sang Bước 2",
            command=self.run_auto_label_pipeline,
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
        ).pack(side="left", padx=(0, 12))

        tk.Button(
            btn_bar,
            text="💾 Xuất Đoạn Video Này Ra File .MP4 Riêng",
            command=self.run_trim_video_file,
            bg="#F1F5F9",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=14,
            pady=10,
            cursor="hand2",
        ).pack(side="left", padx=6)

        tk.Button(
            btn_bar,
            text="🎬 Xuất Các Cửa Sổ 4s Thành Video Clips (.mp4)",
            command=self.run_export_window_clips,
            bg="#F1F5F9",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=14,
            pady=10,
            cursor="hand2",
        ).pack(side="left", padx=6)

        self.trim_prog_lbl = tk.Label(
            card,
            text="✅ Đã có sẵn dữ liệu 427 cửa sổ (0s - 60s). Bạn có thể bấm nút xanh ở trên để chạy lại hoặc sang thẳng BƯỚC 2.",
            bg="#FFFFFF",
            fg="#166534",
            font=("Segoe UI", 9, "bold"),
        )
        self.trim_prog_lbl.pack(anchor="w", pady=(0, 8))

        # Preview Frames Side-by-Side
        prev_grid = tk.Frame(card, bg="#0F172A", padx=12, pady=12)
        prev_grid.pack(fill="both", expand=True)

        self.trim_start_canvas = tk.Label(prev_grid, text="Khung hình Bắt đầu", bg="#0F172A", fg="#94A3B8", font=("Segoe UI", 10))
        self.trim_start_canvas.pack(side="left", fill="both", expand=True, padx=6)

        self.trim_end_canvas = tk.Label(prev_grid, text="Khung hình Kết thúc", bg="#0F172A", fg="#94A3B8", font=("Segoe UI", 10))
        self.trim_end_canvas.pack(side="right", fill="both", expand=True, padx=6)

    # -----------------------------------------------------------------
    # BƯỚC 2: DUYỆT & SỬA NHÃN TỪNG SINH VIÊN (MAIN WORKSPACE)
    # -----------------------------------------------------------------
    def _build_step2_review(self) -> None:
        # 1. Clean Navigation & Filter Bar (All fits comfortably!)
        nav_bar = tk.Frame(
            self.tab_step2_review,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        nav_bar.pack(fill="x", pady=(2, 6))

        tk.Label(nav_bar, text="Bộ lọc:", bg="#FFFFFF", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.filter_review_cb = ttk.Combobox(
            nav_bar,
            values=["Tất cả cửa sổ", "Chưa duyệt (Nhãn máy)", "Đã duyệt (HUMAN)"],
            state="readonly",
            width=20,
        )
        self.filter_review_cb.current(0)
        self.filter_review_cb.pack(side="left", padx=(4, 10))
        self.filter_review_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        tk.Label(nav_bar, text="Sinh viên:", bg="#FFFFFF", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.filter_track_cb = ttk.Combobox(nav_bar, values=["Tất cả"], state="readonly", width=12)
        self.filter_track_cb.current(0)
        self.filter_track_cb.pack(side="left", padx=(4, 12))
        self.filter_track_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        # Previous / Window Selector / Next
        tk.Button(
            nav_bar, text="◀ Trước (←)", command=self.nav_prev,
            bg="#F1F5F9", fg="#0F172A", font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=3
        ).pack(side="left", padx=2)

        self.window_selector_cb = ttk.Combobox(nav_bar, state="readonly", width=34, font=("Segoe UI", 9))
        self.window_selector_cb.pack(side="left", padx=4)
        self.window_selector_cb.bind("<<ComboboxSelected>>", self._on_window_combo_select)

        tk.Button(
            nav_bar, text="Tiếp (→) ▶", command=self.nav_next,
            bg="#F1F5F9", fg="#0F172A", font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=3
        ).pack(side="left", padx=2)

        tk.Button(
            nav_bar,
            text="⏭ Nhảy Tới Chưa Duyệt",
            command=self.nav_next_unreviewed,
            bg=PALETTE["primary_light"],
            fg=PALETTE["primary_text"],
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=3,
        ).pack(side="left", padx=(8, 4))

        # Right side overlays toggle (short text, never overflows)
        tk.Checkbutton(
            nav_bar, text="Mờ mặt", variable=self.opt_blur_face, bg="#FFFFFF", command=self.refresh_current_window
        ).pack(side="right", padx=4)
        tk.Checkbutton(
            nav_bar, text="Khung xương", variable=self.opt_draw_skel, bg="#FFFFFF", command=self.refresh_current_window
        ).pack(side="right", padx=4)

        # 2. Main Two-Column Workspace
        work_area = tk.Frame(self.tab_step2_review, bg=PALETTE["bg_app"])
        work_area.pack(fill="both", expand=True)

        # Left Column: Visual Inspection (Flexible width)
        left_panel = tk.Frame(
            work_area,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=12,
            pady=10,
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Right Column: Clean Step-by-Step Labeling Form (Spacious 520px width, ZERO clipping!)
        right_panel = tk.Frame(
            work_area,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=16,
            pady=10,
            width=520,
        )
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # =============================================================
        # LEFT PANEL CONTENT: VISUAL OBSERVATION
        # =============================================================
        top_left_bar = tk.Frame(left_panel, bg="#FFFFFF")
        top_left_bar.pack(side="top", fill="x", pady=(0, 4))

        self.window_title_lbl = tk.Label(
            top_left_bar,
            text="Cửa sổ #1 / 427 — Sinh viên Track #1 (0.0s ➔ 4.0s)",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 10, "bold"),
        )
        self.window_title_lbl.pack(side="left")

        tk.Button(
            top_left_bar,
            text="▶ Phát Video 4 Giây (Phím Space)",
            command=self.play_4s_clip_live,
            bg="#EFF6FF",
            fg="#1D4ED8",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=2,
            cursor="hand2",
        ).pack(side="right")

        # Pack bottom elements FIRST so they are guaranteed full height and never clipped!
        self.ai_suggest_lbl = tk.Label(
            left_panel,
            text="🤖 Gợi ý của AI: Đang phân tích...",
            bg="#F8FAFC",
            fg="#334155",
            font=("Segoe UI", 9),
            justify="left",
            anchor="w",
            padx=12,
            pady=6,
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
        )
        self.ai_suggest_lbl.pack(side="bottom", fill="x", pady=(4, 0))

        self.filmstrip_frame = tk.Frame(left_panel, bg="#F8FAFC", pady=4)
        self.filmstrip_frame.pack(side="bottom", fill="x", pady=2)
        self.film_labels: List[tk.Label] = []
        self.film_captions: List[tk.Label] = []
        for i in range(5):
            cell = tk.Frame(self.filmstrip_frame, bg="#F8FAFC", padx=3)
            cell.pack(side="left", expand=True)
            img_l = tk.Label(cell, bg="#0F172A")
            img_l.pack()
            cap_l = tk.Label(cell, text=f"+{i}.0s", bg="#F8FAFC", fg="#475569", font=("Segoe UI", 8, "bold"))
            cap_l.pack(pady=(2, 0))
            self.film_labels.append(img_l)
            self.film_captions.append(cap_l)

        tk.Label(
            left_panel,
            text="🎞️ Diễn biến tư thế của sinh viên này qua 5 mốc thời gian trong cửa sổ 4 giây:",
            bg="#FFFFFF",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="bottom", anchor="w", pady=(2, 0))

        # Main Classroom Context Image fills remaining center space
        self.context_canvas = tk.Label(left_panel, bg="#0F172A")
        self.context_canvas.pack(side="top", fill="both", expand=True, pady=(0, 4))

        # =============================================================
        # RIGHT PANEL CONTENT: CLEAN 1-2-3 DECISION PANEL
        # =============================================================
        # Progress Header inside Right Panel
        prog_box = tk.Frame(right_panel, bg="#F8FAFC", highlightbackground=PALETTE["border"], highlightthickness=1, padx=12, pady=8)
        prog_box.pack(fill="x", pady=(0, 12))

        self.progress_summary_lbl = tk.Label(
            prog_box,
            text="Tiến độ duyệt: 0 / 427 cửa sổ (0%)",
            bg="#F8FAFC",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
        )
        self.progress_summary_lbl.pack(anchor="w")

        self.current_status_badge = tk.Label(
            prog_box,
            text="Trạng thái cửa sổ này: Chưa duyệt (Nhãn máy tự động)",
            bg="#F8FAFC",
            fg="#64748B",
            font=("Segoe UI", 8),
        )
        self.current_status_badge.pack(anchor="w", pady=(2, 0))

        # --- MỤC 1: CHỌN MỨC ĐỘ THAM GIA (4 Clear Buttons — Only Active One is Highlighted!) ---
        tk.Label(
            right_panel,
            text="1. Chọn Mức Độ Tham Gia (Bấm chuột hoặc gõ phím 1, 2, 3, 4):",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        btn_grid = tk.Frame(right_panel, bg="#FFFFFF")
        btn_grid.pack(fill="x", pady=(0, 12))

        self.btn_choice_high = tk.Button(
            btn_grid,
            text="[Phím 1]  CAO (HIGH)",
            command=lambda: self.select_or_save_label("VALID", HIGH, auto_advance=True),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=7,
            cursor="hand2",
        )
        self.btn_choice_high.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)

        self.btn_choice_mod = tk.Button(
            btn_grid,
            text="[Phím 2]  VỪA (MODERATE)",
            command=lambda: self.select_or_save_label("VALID", MODERATE, auto_advance=True),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=7,
            cursor="hand2",
        )
        self.btn_choice_mod.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)

        self.btn_choice_low = tk.Button(
            btn_grid,
            text="[Phím 3]  THẤP (LOW)",
            command=lambda: self.select_or_save_label("VALID", LOW, auto_advance=True),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=7,
            cursor="hand2",
        )
        self.btn_choice_low.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)

        self.btn_choice_occ = tk.Button(
            btn_grid,
            text="[Phím 4]  BỊ CHE (OCCLUDED)",
            command=lambda: self.select_or_save_label("OCCLUDED", "", auto_advance=True),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=7,
            cursor="hand2",
        )
        self.btn_choice_occ.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)

        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        # Hidden/Synced state variables for detailed editing
        self.edit_state_var = tk.StringVar(value="VALID")
        self.edit_label_var = tk.StringVar(value=LOW)

        # --- MỤC 2: HÀNH VI QUAN SÁT ĐƯỢC (Concise Vietnamese labels, zero overflow!) ---
        tk.Label(
            right_panel,
            text="2. Hành vi cụ thể quan sát được (Tích chọn 1 hoặc nhiều):",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(2, 4))

        beh_frame = tk.Frame(
            right_panel,
            bg="#F8FAFC",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=10,
            pady=8,
        )
        beh_frame.pack(fill="x", pady=(0, 12))

        self.beh_vars: Dict[str, tk.BooleanVar] = {}
        for idx_b, (b_code, b_vi) in enumerate(BEHAVIOR_ITEMS):
            var = tk.BooleanVar(value=False)
            self.beh_vars[b_code] = var
            cb = tk.Checkbutton(
                beh_frame,
                text=b_vi,
                variable=var,
                bg="#F8FAFC",
                fg="#1E293B",
                activebackground="#F8FAFC",
                font=("Segoe UI", 9),
                anchor="w",
            )
            cb.grid(row=idx_b // 2, column=idx_b % 2, sticky="ew", padx=4, pady=2)

        beh_frame.columnconfigure(0, weight=1)
        beh_frame.columnconfigure(1, weight=1)

        # --- MỤC 3: THÔNG TIN BỐI CẢNH LỚP HỌC ---
        tk.Label(
            right_panel,
            text="3. Bối cảnh & Chi tiết cửa sổ:",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        ctx_grid = tk.Frame(right_panel, bg="#FFFFFF")
        ctx_grid.pack(fill="x", pady=(0, 10))

        tk.Label(ctx_grid, text="Trạng thái:", bg="#FFFFFF", fg="#475569").grid(row=0, column=0, sticky="w", pady=4)
        self.state_combo = ttk.Combobox(ctx_grid, values=list(ANNOTATION_STATES), state="readonly", width=13)
        self.state_combo.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.state_combo.bind("<<ComboboxSelected>>", self._on_state_combo_change)

        tk.Label(ctx_grid, text="Pha học:", bg="#FFFFFF", fg="#475569").grid(row=0, column=2, sticky="w", padx=(12, 0), pady=4)
        self.edit_phase_cb = ttk.Combobox(ctx_grid, values=CLASS_PHASES, state="readonly", width=14)
        self.edit_phase_cb.grid(row=0, column=3, sticky="w", padx=6, pady=4)

        tk.Label(ctx_grid, text="Vùng ghế:", bg="#FFFFFF", fg="#475569").grid(row=1, column=0, sticky="w", pady=4)
        self.edit_zone_cb = ttk.Combobox(ctx_grid, values=ZONE_IDS, state="readonly", width=13)
        self.edit_zone_cb.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        tk.Label(ctx_grid, text="Che khuất:", bg="#FFFFFF", fg="#475569").grid(row=1, column=2, sticky="w", padx=(12, 0), pady=4)
        self.edit_occ_cb = ttk.Combobox(ctx_grid, values=OCCLUSION_LEVELS, state="readonly", width=14)
        self.edit_occ_cb.grid(row=1, column=3, sticky="w", padx=6, pady=4)

        tk.Label(ctx_grid, text="Ghi chú:", bg="#FFFFFF", fg="#475569").grid(row=2, column=0, sticky="w", pady=4)
        self.edit_notes_var = tk.StringVar(value="")
        tk.Entry(ctx_grid, textvariable=self.edit_notes_var, bg="#F8FAFC", font=("Segoe UI", 9)).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=6, pady=4
        )
        ctx_grid.columnconfigure(3, weight=1)

        # --- PRIMARY CTA BUTTONS ---
        tk.Button(
            right_panel,
            text="✓ XÁC NHẬN LƯU & SANG CỬA SỔ TIẾP (Phím Enter)",
            command=lambda: self.save_current_window(advance=True),
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            pady=8,
            cursor="hand2",
        ).pack(fill="x", pady=(4, 4))

        sub_btns = tk.Frame(right_panel, bg="#FFFFFF")
        sub_btns.pack(fill="x")

        tk.Button(
            sub_btns,
            text="📌 Lưu Tại Chỗ",
            command=lambda: self.save_current_window(advance=False),
            bg="#F1F5F9",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=4,
            cursor="hand2",
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        tk.Button(
            sub_btns,
            text="⚡ Duyệt Cả Danh Sách Lọc",
            command=self.batch_confirm_filtered,
            bg="#F1F5F9",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=4,
            cursor="hand2",
        ).pack(side="right", fill="x", expand=True, padx=(4, 0))

    # -----------------------------------------------------------------
    # BƯỚC 3: THỐNG KÊ, KIỂM ĐỊNH & XUẤT FILE CSV
    # -----------------------------------------------------------------
    def _build_step3_export(self) -> None:
        card = tk.Frame(
            self.tab_step3_export,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=20,
            pady=16,
        )
        card.pack(fill="both", expand=True, pady=4)

        top_bar = tk.Frame(card, bg="#FFFFFF")
        top_bar.pack(fill="x", pady=(0, 10))

        tk.Label(
            top_bar,
            text="BƯỚC 3: KIỂM TRA HỢP LỆ & XUẤT CSV",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        tk.Button(
            top_bar,
            text="💾 Xuất Bản Sao CSV (Save As...)",
            command=self.export_csv_copy,
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="right", padx=4)

        tk.Button(
            top_bar,
            text="🤝 Đối Soát Cohen's Kappa (Task T12)",
            command=self.run_double_annotation_kappa,
            bg="#F1F5F9",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="right", padx=4)

        tk.Button(
            top_bar,
            text="🔄 Làm Mới",
            command=self.run_qa_validation_report,
            bg="#F1F5F9",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=6,
        ).pack(side="right", padx=4)

        self.qa_text = tk.Text(
            card,
            font=("Consolas", 10),
            bg="#0F172A",
            fg="#F8FAFC",
            padx=16,
            pady=14,
            relief="flat",
        )
        self.qa_text.pack(fill="both", expand=True)

    # -----------------------------------------------------------------
    # KEYBOARD SHORTCUTS
    # -----------------------------------------------------------------
    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", lambda e: self._shortcut_guard(self.nav_prev))
        self.root.bind("<Right>", lambda e: self._shortcut_guard(self.nav_next))
        self.root.bind("1", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("VALID", HIGH, True)))
        self.root.bind("2", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("VALID", MODERATE, True)))
        self.root.bind("3", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("VALID", LOW, True)))
        self.root.bind("4", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("OCCLUDED", "", True)))
        self.root.bind("<Return>", lambda e: self._shortcut_guard(lambda: self.save_current_window(advance=True)))
        self.root.bind("<space>", lambda e: self._shortcut_guard(self.play_4s_clip_live))

    def _shortcut_guard(self, func) -> None:
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Spinbox, tk.Text)):
            return
        if self.notebook.index(self.notebook.select()) == 1:
            func()

    # -----------------------------------------------------------------
    # DATA LOADING & FILTERING
    # -----------------------------------------------------------------
    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
        if p:
            self.video_path_var.set(p)

    def _browse_pose(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Pose files", "*.parquet *.csv"), ("All files", "*.*")])
        if p:
            self.pose_path_var.set(p)

    def _browse_csv(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.csv_path_var.set(p)

    def reload_all_resources(self) -> None:
        vid_p = Path(self.video_path_var.get().strip())
        pose_p = Path(self.pose_path_var.get().strip())
        csv_p = Path(self.csv_path_var.get().strip())

        try:
            self.poses_df = load_poses_table(pose_p)
        except Exception:
            self.poses_df = None

        self.rows = []
        if csv_p.exists():
            with csv_p.open("r", encoding="utf-8", newline="") as f:
                self.rows = [dict(r) for r in csv.DictReader(f)]

        tracks = sorted(
            {str(r.get("track_id", "")) for r in self.rows if r.get("track_id")},
            key=lambda x: int(x) if x.isdigit() else 999,
        )
        self.filter_track_cb["values"] = ["Tất cả"] + [f"Track #{t}" for t in tracks]
        self.filter_track_cb.current(0)

        self.header_status_lbl.config(
            text=(
                f"🎥 {vid_p.name} ({'OK' if vid_p.exists() else 'Chưa thấy'})  |  "
                f"📄 {len(self.rows)} cửa sổ  |  "
                f"👤 {self.annotator_var.get()}"
            )
        )

        self.apply_filters()
        self.update_trim_previews()
        self.run_qa_validation_report()

    def apply_filters(self) -> None:
        rev_f = self.filter_review_cb.get()
        trk_f = self.filter_track_cb.get()

        target_trk = trk_f.replace("Track #", "").strip() if trk_f.startswith("Track #") else "Tất cả"

        self.filtered_indices = []
        for idx, r in enumerate(self.rows):
            src = r.get("label_source", "").strip()
            if rev_f == "Chưa duyệt (Nhãn máy)" and src in ("HUMAN", "ADJUDICATED"):
                continue
            if rev_f == "Đã duyệt (HUMAN)" and src not in ("HUMAN", "ADJUDICATED"):
                continue
            if target_trk != "Tất cả" and str(r.get("track_id", "")) != target_trk:
                continue
            self.filtered_indices.append(idx)

        combo_items = []
        for i in self.filtered_indices:
            r = self.rows[i]
            lbl = r.get("participation_label", "")
            disp = to_display_label(lbl) if lbl in CANONICAL_TRAINING_LABELS else r.get("annotation_state", "")
            src_tag = "✓ Đã duyệt" if r.get("label_source") in ("HUMAN", "ADJUDICATED") else "Máy gợi ý"
            combo_items.append(
                f"Cửa sổ #{i+1} | Track #{r.get('track_id')} ({int(r.get('t_start_ms', 0))//1000}s-{int(r.get('t_end_ms', 0))//1000}s) ➔ {disp} [{src_tag}]"
            )
        self.window_selector_cb["values"] = combo_items

        if self.filtered_indices:
            if self.current_idx not in self.filtered_indices:
                self.current_idx = self.filtered_indices[0]
            self.refresh_current_window()
        else:
            self.window_title_lbl.config(text="⚠️ Không có cửa sổ nào khớp với bộ lọc hiện tại.")

    def _on_window_combo_select(self, _event=None) -> None:
        sel = self.window_selector_cb.current()
        if 0 <= sel < len(self.filtered_indices):
            self.current_idx = self.filtered_indices[sel]
            self.refresh_current_window()

    def nav_prev(self) -> None:
        if self.current_idx in self.filtered_indices:
            pos = self.filtered_indices.index(self.current_idx)
            if pos > 0:
                self.current_idx = self.filtered_indices[pos - 1]
                self.refresh_current_window()

    def nav_next(self) -> None:
        if self.current_idx in self.filtered_indices:
            pos = self.filtered_indices.index(self.current_idx)
            if pos + 1 < len(self.filtered_indices):
                self.current_idx = self.filtered_indices[pos + 1]
                self.refresh_current_window()

    def nav_next_unreviewed(self) -> None:
        after = [
            i for i in self.filtered_indices
            if i > self.current_idx and self.rows[i].get("label_source", "") not in ("HUMAN", "ADJUDICATED")
        ]
        if not after:
            after = [
                i for i in self.filtered_indices
                if self.rows[i].get("label_source", "") not in ("HUMAN", "ADJUDICATED")
            ]
        if after:
            self.current_idx = after[0]
            self.refresh_current_window()
        else:
            messagebox.showinfo("Hoàn tất", "Tất cả cửa sổ trong danh sách lọc hiện tại đã được duyệt!")

    # -----------------------------------------------------------------
    # RENDERING CURRENT WINDOW & BUTTON COLOR HIERARCHY
    # -----------------------------------------------------------------
    def _update_label_button_styles(self, state: str, label: str) -> None:
        """Highlight ONLY the currently selected label button; keep others soft pastel."""
        is_high = state == "VALID" and label == HIGH
        is_mod = state == "VALID" and label == MODERATE
        is_low = state == "VALID" and label == LOW
        is_occ = state != "VALID"

        self.btn_choice_high.config(
            text="✓ [Phím 1]  CAO (HIGH)" if is_high else "[Phím 1]  CAO (HIGH)",
            bg=PALETTE["high_active_bg"] if is_high else PALETTE["high_idle_bg"],
            fg=PALETTE["high_active_fg"] if is_high else PALETTE["high_idle_fg"],
        )
        self.btn_choice_mod.config(
            text="✓ [Phím 2]  VỪA (MODERATE)" if is_mod else "[Phím 2]  VỪA (MODERATE)",
            bg=PALETTE["mod_active_bg"] if is_mod else PALETTE["mod_idle_bg"],
            fg=PALETTE["mod_active_fg"] if is_mod else PALETTE["mod_idle_fg"],
        )
        self.btn_choice_low.config(
            text="✓ [Phím 3]  THẤP (LOW)" if is_low else "[Phím 3]  THẤP (LOW)",
            bg=PALETTE["low_active_bg"] if is_low else PALETTE["low_idle_bg"],
            fg=PALETTE["low_active_fg"] if is_low else PALETTE["low_idle_fg"],
        )
        occ_txt = f"✓ [Phím 4]  {state}" if is_occ else "[Phím 4]  BỊ CHE (OCCLUDED)"
        self.btn_choice_occ.config(
            text=occ_txt,
            bg=PALETTE["occ_active_bg"] if is_occ else PALETTE["occ_idle_bg"],
            fg=PALETTE["occ_active_fg"] if is_occ else PALETTE["occ_idle_fg"],
        )

    def refresh_current_window(self) -> None:
        if not self.rows or self.current_idx >= len(self.rows):
            return

        if self.current_idx in self.filtered_indices:
            self.window_selector_cb.current(self.filtered_indices.index(self.current_idx))

        r = self.rows[self.current_idx]
        t_start = int(r.get("t_start_ms", 0))
        t_end = int(r.get("t_end_ms", 4000))
        track_id = int(r.get("track_id", 1))
        state = r.get("annotation_state", "VALID").strip()
        label = r.get("participation_label", "").strip()
        source = r.get("label_source", "").strip()
        phase = r.get("class_phase", "lecture").strip()
        zone = r.get("zone_id", "M-C").strip()

        # Update Titles & Progress
        human_c = sum(1 for x in self.rows if x.get("label_source") in ("HUMAN", "ADJUDICATED"))
        pct = human_c / max(1, len(self.rows)) * 100.0
        self.progress_summary_lbl.config(
            text=f"📊 Tiến độ đã duyệt: {human_c} / {len(self.rows)} cửa sổ ({pct:.1f}%)"
        )

        disp_lbl = to_display_label(label) if label in CANONICAL_TRAINING_LABELS else state
        if source in ("HUMAN", "ADJUDICATED"):
            self.current_status_badge.config(
                text=f"✓ Đã được duyệt bởi {r.get('annotator_id', 'HUMAN')} — Nhãn hiện tại: {disp_lbl}",
                fg="#166534",
            )
        else:
            self.current_status_badge.config(
                text=f"🤖 Đang hiển thị gợi ý tự động của máy — Nhãn gợi ý: {disp_lbl}",
                fg="#D97706",
            )

        self.window_title_lbl.config(
            text=f"🎯 Cửa sổ #{self.current_idx + 1} / {len(self.rows)}   •   Sinh viên: Track #{track_id}   •   Thời gian: {t_start/1000:.1f}s ➔ {t_end/1000:.1f}s   •   Ghế: {zone}"
        )

        # Sync Form State
        self.edit_state_var.set(state)
        self.edit_label_var.set(label)
        self.state_combo.set(state if state in ANNOTATION_STATES else "VALID")
        self._update_label_button_styles(state, label)

        active_behs = {b.strip() for b in r.get("atomic_behavior", "").split(";") if b.strip()}
        for b_code, var in self.beh_vars.items():
            var.set(b_code in active_behs)

        self.edit_occ_cb.set(r.get("occlusion_level", "none") or "none")
        self.edit_phase_cb.set(phase if phase in CLASS_PHASES else "lecture")
        self.edit_zone_cb.set(zone if zone in ZONE_IDS else "M-C")
        self.edit_notes_var.set(r.get("notes", ""))

        # Render visuals
        vid_p = Path(self.video_path_var.get().strip())
        self._render_visuals(vid_p, t_start, t_end, track_id, phase, label)

    def _on_state_combo_change(self, _event=None) -> None:
        st_val = self.state_combo.get()
        self.edit_state_var.set(st_val)
        if st_val == "VALID" and not self.edit_label_var.get():
            self.edit_label_var.set(MODERATE)
        elif st_val != "VALID":
            self.edit_label_var.set("")
        self._update_label_button_styles(st_val, self.edit_label_var.get())

    def _render_visuals(
        self,
        video_path: Path,
        t_start_ms: int,
        t_end_ms: int,
        track_id: int,
        class_phase: str,
        canonical_label: str,
    ) -> None:
        draw_skel = self.opt_draw_skel.get()
        blur_face = self.opt_blur_face.get()

        # 1. AI Suggestion & Telemetry text
        if self.poses_df is not None and not self.poses_df.empty:
            w_df = self.poses_df[
                (self.poses_df["track_id"].astype(int) == int(track_id))
                & (self.poses_df["timestamp_ms"] >= int(t_start_ms))
                & (self.poses_df["timestamp_ms"] <= int(t_end_ms))
            ]
            if not w_df.empty:
                mean_vis = float(w_df["visible_ratio"].mean())
                nose_y = w_df["kpt_00_y"].values
                nose_c = w_df["kpt_00_conf"].values
                ears_y = (w_df["kpt_03_y"].values + w_df["kpt_04_y"].values) / 2.0
                l_sh_y, r_sh_y = w_df["kpt_05_y"].values, w_df["kpt_06_y"].values
                l_wr_y, r_wr_y = w_df["kpt_09_y"].values, w_df["kpt_10_y"].values
                l_wr_x, r_wr_x = w_df["kpt_09_x"].values, w_df["kpt_10_x"].values

                hand_up = float(np.mean((l_wr_y < l_sh_y - 20) | (r_wr_y < r_sh_y - 20)))
                head_dn = float(np.mean((nose_y > ears_y + 12) & (nose_c > 0.3)))
                wr_mot = float(np.std(l_wr_x) + np.std(r_wr_x))
                r_state, r_lbl, r_behs, r_conf = apply_labeling_rules(w_df, class_phase=class_phase)
                r_disp = to_display_label(r_lbl) if r_lbl in CANONICAL_TRAINING_LABELS else r_state

                self.ai_suggest_lbl.config(
                    text=(
                        f"🤖 Phân tích tự động từ khung xương:  Máy gợi ý [{r_disp}] (Độ tự tin: {r_conf*100:.0f}%)\n"
                        f"📡 Chỉ số đo được:  Nhìn rõ {mean_vis*100:.0f}% khớp  •  Cúi đầu {head_dn*100:.0f}% thời gian  •  "
                        f"Giơ tay {hand_up*100:.0f}%  •  Cổ tay dịch chuyển {wr_mot:.1f} px"
                    )
                )

        if not video_path.exists():
            return

        color_bgr = STATE_COLORS_BGR.get(canonical_label, (46, 204, 113))
        timestamps = np.linspace(t_start_ms, max(t_start_ms + 500, t_end_ms - 100), 5, dtype=int)

        track_poses = None
        if self.poses_df is not None and not self.poses_df.empty:
            track_poses = self.poses_df[self.poses_df["track_id"].astype(int) == int(track_id)].sort_values("timestamp_ms")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        self.filmstrip_imgs.clear()
        mid_full_frame = None
        try:
            for idx_t, ts in enumerate(timestamps):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                if idx_t == 2:
                    mid_full_frame = frame.copy()

                matched_row = None
                if track_poses is not None and not track_poses.empty:
                    diffs = (track_poses["timestamp_ms"] - int(ts)).abs()
                    min_i = diffs.idxmin()
                    if diffs.loc[min_i] <= 1200:
                        matched_row = track_poses.loc[min_i]

                crop_bgr = render_crop_with_pose(
                    frame,
                    matched_row,
                    draw_skel=draw_skel,
                    blur_face=blur_face,
                    color_bgr=color_bgr,
                    out_size=(145, 150),
                )
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tk_img = ImageTk.PhotoImage(Image.fromarray(crop_rgb))
                self.filmstrip_imgs.append(tk_img)
                self.film_labels[idx_t].config(image=tk_img)
                self.film_captions[idx_t].config(text=f"Giây {ts/1000:.1f}s")

            if mid_full_frame is not None:
                ctx_bgr = self._draw_classroom_context(
                    mid_full_frame,
                    t_start_ms + (t_end_ms - t_start_ms) // 2,
                    track_id,
                    draw_skel,
                    blur_face,
                )
                ctx_resized = cv2.resize(ctx_bgr, (680, 290), interpolation=cv2.INTER_AREA)
                ctx_rgb = cv2.cvtColor(ctx_resized, cv2.COLOR_BGR2RGB)
                self.context_img = ImageTk.PhotoImage(Image.fromarray(ctx_rgb))
                self.context_canvas.config(image=self.context_img)
        finally:
            cap.release()

    def _draw_classroom_context(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        target_track_id: int,
        draw_skel: bool,
        blur_face: bool,
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Subtle 3x3 grid lines
        for frac in (0.33, 0.66):
            cv2.line(canvas, (int(w * frac), 0), (int(w * frac), h), (190, 190, 190), 1)
            cv2.line(canvas, (0, int(h * frac)), (w, int(h * frac)), (190, 190, 190), 1)

        if self.poses_df is not None and not self.poses_df.empty:
            diffs = (self.poses_df["timestamp_ms"] - int(timestamp_ms)).abs()
            if diffs.min() <= 1000:
                nearest_ts = self.poses_df.loc[diffs.idxmin(), "timestamp_ms"]
                f_poses = self.poses_df[self.poses_df["timestamp_ms"] == nearest_ts]
                for _, prow in f_poses.iterrows():
                    tid = int(prow["track_id"])
                    x1, y1 = int(prow["bbox_x1"]), int(prow["bbox_y1"])
                    x2, y2 = int(prow["bbox_x2"]), int(prow["bbox_y2"])
                    kpts = row_to_keypoints_array(prow)
                    if blur_face:
                        blur_head_roi(canvas, (x1, y1, x2, y2), kpts)
                    if tid == int(target_track_id):
                        if draw_skel:
                            draw_skeleton(canvas, kpts, color=(0, 255, 128), conf_threshold=0.25)
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 128), 4)
                        tag = f"SINH VIEN DANG XET: Track #{tid}"
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                        cv2.rectangle(canvas, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, y1), (0, 255, 128), -1)
                        cv2.putText(
                            canvas,
                            tag,
                            (x1 + 6, max(th, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 0),
                            2,
                        )
                    else:
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), (160, 160, 160), 1)
        return canvas

    def play_4s_clip_live(self) -> None:
        if self.is_playing_clip or not self.rows:
            return
        vid_p = Path(self.video_path_var.get().strip())
        if not vid_p.exists():
            return

        r = self.rows[self.current_idx]
        t_start = int(r.get("t_start_ms", 0))
        t_end = int(r.get("t_end_ms", 4000))
        track_id = int(r.get("track_id", 1))
        draw_skel = self.opt_draw_skel.get()
        blur_face = self.opt_blur_face.get()

        self.is_playing_clip = True

        def _worker():
            cap = cv2.VideoCapture(str(vid_p))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t_start))
            step = max(1, int(round(fps / 15.0)))
            f_idx = 0
            try:
                while cap.isOpened() and self.is_playing_clip:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if cur_ms > t_end:
                        break
                    if f_idx % step == 0:
                        ctx_bgr = self._draw_classroom_context(frame, int(cur_ms), track_id, draw_skel, blur_face)
                        ctx_res = cv2.resize(ctx_bgr, (740, 340), interpolation=cv2.INTER_AREA)
                        cv2.putText(
                            ctx_res,
                            f"DANG PHAT VIDEO 4S: {cur_ms/1000:.1f}s / {t_end/1000:.1f}s",
                            (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        rgb = cv2.cvtColor(ctx_res, cv2.COLOR_BGR2RGB)
                        img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
                        self.context_img = img_tk
                        self.context_canvas.config(image=img_tk)
                        time.sleep(0.05)
                    f_idx += 1
            finally:
                cap.release()
                self.is_playing_clip = False

        threading.Thread(target=_worker, daemon=True).start()

    # -----------------------------------------------------------------
    # SAVING & LABEL SELECTION
    # -----------------------------------------------------------------
    def select_or_save_label(self, state: str, canonical_label: str, auto_advance: bool = True) -> None:
        if not self.rows:
            return
        self.edit_state_var.set(state)
        self.state_combo.set(state)
        self.edit_label_var.set(canonical_label)
        self._update_label_button_styles(state, canonical_label)

        if state == "VALID":
            if not any(v.get() for v in self.beh_vars.values()):
                if canonical_label == HIGH:
                    self.beh_vars["BOARD_OR_INSTRUCTOR_FACING"].set(True)
                elif canonical_label == MODERATE:
                    self.beh_vars["READING_WRITING"].set(True)
                else:
                    self.beh_vars["HEAD_DOWN"].set(True)
        else:
            for v in self.beh_vars.values():
                v.set(False)
            if state == "OCCLUDED":
                self.beh_vars["OCCLUDED"].set(True)

        self.save_current_window(advance=auto_advance)

    def save_current_window(self, advance: bool = True) -> None:
        if not self.rows:
            return
        st_val = self.state_combo.get().strip() or "VALID"
        is_valid = st_val == "VALID"
        lbl_val = self.edit_label_var.get().strip() if is_valid else ""
        if is_valid and lbl_val not in CANONICAL_TRAINING_LABELS:
            lbl_val = MODERATE

        src_val = "HUMAN" if is_valid else ""
        rule_ver = ""

        try:
            validate_annotation_label(
                participation_label=lbl_val if lbl_val else None,
                annotation_state=st_val,
                label_source=src_val if src_val else None,
                label_rule_version=rule_ver if rule_ver else None,
            )
        except ValueError as exc:
            messagebox.showerror("Lỗi dữ liệu", str(exc))
            return

        selected_behs = [code for code, var in self.beh_vars.items() if var.get()]
        r = self.rows[self.current_idx]
        r["annotation_state"] = st_val
        r["participation_label"] = lbl_val
        r["atomic_behavior"] = ";".join(selected_behs)
        r["label_source"] = src_val
        r["label_rule_version"] = rule_ver
        r["occlusion_level"] = self.edit_occ_cb.get()
        r["class_phase"] = self.edit_phase_cb.get()
        r["zone_id"] = self.edit_zone_cb.get()
        r["annotator_id"] = self.annotator_var.get().strip() or "Tien_B23DCPT306"
        r["notes"] = self.edit_notes_var.get().strip()
        r["annotation_guide_version"] = "v0.2"

        self._write_rows_to_csv()

        if advance and self.current_idx in self.filtered_indices:
            pos = self.filtered_indices.index(self.current_idx)
            if pos + 1 < len(self.filtered_indices):
                self.current_idx = self.filtered_indices[pos + 1]
        self.apply_filters()

    def batch_confirm_filtered(self) -> None:
        if not self.filtered_indices:
            return
        if not messagebox.askyesno(
            "Duyệt Nhanh Danh Sách Đang Lọc",
            f"Bạn có muốn xác nhận toàn bộ {len(self.filtered_indices)} cửa sổ đang lọc thành đã duyệt (HUMAN) không?",
        ):
            return
        ann_id = self.annotator_var.get().strip() or "Tien_B23DCPT306"
        cnt = 0
        for idx in self.filtered_indices:
            r = self.rows[idx]
            if r.get("annotation_state") == "VALID" and r.get("participation_label") in CANONICAL_TRAINING_LABELS:
                r["label_source"] = "HUMAN"
                r["label_rule_version"] = ""
                r["annotator_id"] = ann_id
                cnt += 1
        self._write_rows_to_csv()
        self.apply_filters()
        messagebox.showinfo("Hoàn tất", f"Đã duyệt thành công {cnt} cửa sổ!")

    def _write_rows_to_csv(self) -> None:
        csv_p = Path(self.csv_path_var.get().strip())
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        with csv_p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_ANNOTATION_COLUMNS)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in REQUIRED_ANNOTATION_COLUMNS})

    # -----------------------------------------------------------------
    # STEP 1 ACTIONS: TRIM & AUTO-LABEL
    # -----------------------------------------------------------------
    def update_trim_previews(self) -> None:
        vid_p = Path(self.video_path_var.get().strip())
        if not vid_p.exists():
            return
        s_ms = int(self.trim_start_var.get() * 1000)
        e_ms = int(max(0.0, self.trim_end_var.get() - 0.5) * 1000)

        cap = cv2.VideoCapture(str(vid_p))
        if not cap.isOpened():
            return
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(s_ms))
            ret1, f1 = cap.read()
            if ret1 and f1 is not None:
                f1_res = cv2.resize(f1, (580, 325), interpolation=cv2.INTER_AREA)
                cv2.putText(f1_res, f"DIEM BAT DAU: {s_ms/1000:.1f}s", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2)
                self.trim_start_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f1_res, cv2.COLOR_BGR2RGB)))
                self.trim_start_canvas.config(image=self.trim_start_img, text="")

            cap.set(cv2.CAP_PROP_POS_MSEC, float(e_ms))
            ret2, f2 = cap.read()
            if ret2 and f2 is not None:
                f2_res = cv2.resize(f2, (580, 325), interpolation=cv2.INTER_AREA)
                cv2.putText(f2_res, f"DIEM KET THUC: {e_ms/1000:.1f}s", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)
                self.trim_end_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f2_res, cv2.COLOR_BGR2RGB)))
                self.trim_end_canvas.config(image=self.trim_end_img, text="")
        finally:
            cap.release()

    def run_auto_label_pipeline(self) -> None:
        if self.poses_df is None or self.poses_df.empty:
            messagebox.showerror("Thiếu dữ liệu", "Chưa tìm thấy file tọa độ khung xương Pose!")
            return

        s_ms = int(self.trim_start_var.get() * 1000)
        e_ms = int(self.trim_end_var.get() * 1000)
        win_ms = int(self.win_sec_var.get() * 1000)
        str_ms = int(self.stride_sec_var.get() * 1000)

        sub_df = self.poses_df[
            (self.poses_df["timestamp_ms"] >= s_ms) & (self.poses_df["timestamp_ms"] <= e_ms)
        ]
        if sub_df.empty:
            sub_df = self.poses_df

        meta_p = Path(self.meta_path_var.get().strip())
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        phases = meta.get("class_phase", [])
        sess_id = meta.get("session_id", "session_001")
        cam_id = meta.get("camera_id", "cam_A")

        def get_phase(ts: int) -> str:
            for p in phases:
                if p.get("start_ms", 0) <= ts < p.get("end_ms", 0):
                    return str(p.get("name", "lecture"))
            return "lecture"

        new_rows: List[Dict[str, str]] = []
        for track_id, grp in sub_df.groupby("track_id"):
            grp = grp.sort_values("timestamp_ms")
            min_t, max_t = int(grp["timestamp_ms"].min()), int(grp["timestamp_ms"].max())
            t_cur = min_t
            while t_cur + win_ms <= max_t:
                t_stop = t_cur + win_ms
                w_df = grp[(grp["timestamp_ms"] >= t_cur) & (grp["timestamp_ms"] <= t_stop)]
                if len(w_df) >= 3:
                    zone = str(w_df["zone_id"].iloc[-1])
                    ph = get_phase(t_cur)
                    m_vis = float(w_df["visible_ratio"].mean())
                    st_val, lbl_val, behs, conf = apply_labeling_rules(w_df, class_phase=ph)
                    is_v = st_val == "VALID"
                    occ = "none" if m_vis >= 0.75 else ("moderate" if m_vis >= 0.50 else "heavy")
                    new_rows.append({
                        "session_id": sess_id,
                        "camera_id": cam_id,
                        "track_id": str(track_id),
                        "t_start_ms": str(t_cur),
                        "t_end_ms": str(t_stop),
                        "class_phase": ph,
                        "atomic_behavior": ";".join(behs),
                        "participation_label": lbl_val if (is_v and lbl_val) else "",
                        "annotation_state": st_val,
                        "label_source": "WEAK_RULE" if is_v else "",
                        "label_rule_version": RULE_VERSION if is_v else "",
                        "annotation_guide_version": "v0.2",
                        "annotation_confidence": f"{conf:.2f}",
                        "occlusion_level": occ,
                        "zone_id": zone,
                        "annotator_id": f"RuleBot_{RULE_VERSION}",
                        "notes": f"auto_vis_ratio={m_vis:.2f}",
                    })
                t_cur += str_ms

        self.rows = new_rows
        self._write_rows_to_csv()
        self.reload_all_resources()
        self.trim_prog_lbl.config(text=f"✅ Đã tự động cắt & gán nhãn {len(new_rows)} cửa sổ 4s thành công!")
        # Automatically jump to Step 2 so user can start reviewing immediately
        self.notebook.select(self.tab_step2_review)

    def run_trim_video_file(self) -> None:
        vid_p = Path(self.video_path_var.get().strip())
        if not vid_p.exists():
            messagebox.showerror("Lỗi", "Không tìm thấy file video gốc!")
            return
        out_p = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile="cam_A_trimmed.mp4",
            filetypes=[("MP4 Video", "*.mp4")],
        )
        if not out_p:
            return
        s_sec = self.trim_start_var.get()
        e_sec = self.trim_end_var.get()
        if e_sec <= s_sec:
            messagebox.showerror("Lỗi", "Giây kết thúc phải lớn hơn giây bắt đầu!")
            return

        self.trim_prog_lbl.config(text=f"⏳ Đang cắt video ({s_sec:.1f}s ➔ {e_sec:.1f}s)...")
        self.root.update_idletasks()

        cap = cv2.VideoCapture(str(vid_p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        s_f = int(s_sec * fps)
        e_f = int(e_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_f)

        writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cur = s_f
        while cur < e_f:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            cur += 1
        cap.release()
        writer.release()

        self.trim_prog_lbl.config(text=f"✅ Đã xuất video đã cắt tại: {out_p}")
        messagebox.showinfo("Thành công", f"Đã lưu video đã cắt tại:\n{out_p}")

    def run_export_window_clips(self) -> None:
        vid_p = Path(self.video_path_var.get().strip())
        if not vid_p.exists() or not self.rows:
            messagebox.showerror("Lỗi", "Cần có Video và danh sách cửa sổ CSV!")
            return
        out_dir = ROOT / "data/clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        subset = self.rows[:10]
        cap = cv2.VideoCapture(str(vid_p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = 0
        for r in subset:
            ts_s = int(r["t_start_ms"])
            ts_e = int(r["t_end_ms"])
            tid = int(r["track_id"])
            lbl = to_display_label(r["participation_label"]) if r["participation_label"] in CANONICAL_TRAINING_LABELS else r["annotation_state"]
            fname = out_dir / f"{r['session_id']}_trk{tid:03d}_{ts_s:06d}_{ts_e:06d}_{lbl}.mp4"
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_s))
            writer = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
            while cap.isOpened():
                ret, frm = cap.read()
                if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > ts_e:
                    break
                writer.write(cv2.resize(frm, (640, 360)))
            writer.release()
            count += 1
        cap.release()
        self.trim_prog_lbl.config(text=f"✅ Đã xuất {count} video clip 4s vào {out_dir}")
        messagebox.showinfo("Hoàn tất", f"Đã xuất {count} video clip 4 giây vào thư mục:\n{out_dir}")

    # -----------------------------------------------------------------
    # STEP 3 ACTIONS: QA & EXPORT
    # -----------------------------------------------------------------
    def export_csv_copy(self) -> None:
        out_p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="session_001_annotated_final.csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not out_p:
            return
        with Path(out_p).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_ANNOTATION_COLUMNS)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in REQUIRED_ANNOTATION_COLUMNS})
        messagebox.showinfo("Xuất File Thành Công", f"Đã lưu file CSV chuẩn 17 cột tại:\n{out_p}")

    def run_qa_validation_report(self) -> None:
        self.qa_text.delete("1.0", "end")
        if not self.rows:
            self.qa_text.insert("end", "Chưa có dữ liệu CSV.\n")
            return

        errors: List[str] = []
        stats: Counter = Counter()
        for idx, r in enumerate(self.rows, start=2):
            stats["total"] += 1
            st_val = r.get("annotation_state", "").strip()
            lbl_val = r.get("participation_label", "").strip()
            src_val = r.get("label_source", "").strip()
            r_ver = r.get("label_rule_version", "").strip()
            stats[f"state_{st_val}"] += 1
            if lbl_val:
                stats[f"label_{lbl_val}"] += 1
            if src_val:
                stats[f"src_{src_val}"] += 1
            try:
                validate_annotation_label(
                    participation_label=lbl_val if lbl_val else None,
                    annotation_state=st_val,
                    label_source=src_val if src_val else None,
                    label_rule_version=r_ver if r_ver else None,
                )
            except ValueError as exc:
                errors.append(f"Dòng {idx}: {exc}")

        tot = max(1, stats["total"])
        lines = [
            "=========================================================================",
            "   BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG & SCHEMA DATASET (CVF25PRJ10 - T11/T12)",
            "=========================================================================",
            f"File CSV hiện tại      : {self.csv_path_var.get()}",
            f"Tổng số cửa sổ 4s      : {stats['total']}",
            f"Kết quả kiểm tra Schema: {'✅ HỢP LỆ 100% (PASS CONTRACT)' if not errors else f'❌ CÓ LỖI ({len(errors)} dòng)'}",
            "",
            "1. THỐNG KÊ THEO MỨC ĐỘ THAM GIA (3 NHÃN HUẤN LUYỆN):",
            f"   • Tham gia Cao (HIGH)        : {stats[f'label_{HIGH}']:>4} cửa sổ ({stats[f'label_{HIGH}']/tot*100:5.1f}%)",
            f"   • Trung bình (MODERATE)      : {stats[f'label_{MODERATE}']:>4} cửa sổ ({stats[f'label_{MODERATE}']/tot*100:5.1f}%)",
            f"   • Tham gia Thấp (LOW)        : {stats[f'label_{LOW}']:>4} cửa sổ ({stats[f'label_{LOW}']/tot*100:5.1f}%)",
            "",
            "2. THỐNG KÊ THEO TRẠNG THÁI QUAN SÁT (STATE):",
        ]
        for s in ANNOTATION_STATES:
            c = stats[f"state_{s}"]
            lines.append(f"   • {s:<12}               : {c:>4} cửa sổ ({c/tot*100:5.1f}%)")

        lines.append("\n3. THỐNG KÊ THEO TIẾN ĐỘ DUYỆT (LABEL SOURCE):")
        lines.append(f"   • Đã do người duyệt (HUMAN)  : {stats['src_HUMAN']:>4} cửa sổ ({stats['src_HUMAN']/tot*100:5.1f}%)")
        lines.append(f"   • Nhãn máy gợi ý (WEAK_RULE) : {stats['src_WEAK_RULE']:>4} cửa sổ ({stats['src_WEAK_RULE']/tot*100:5.1f}%)")
        lines.append(f"   • Đã đối soát (ADJUDICATED)  : {stats['src_ADJUDICATED']:>4} cửa sổ ({stats['src_ADJUDICATED']/tot*100:5.1f}%)")

        if errors:
            lines.append("\n--- DANH SÁCH LỖI CẦN SỬA ---")
            lines.extend(errors[:20])

        self.qa_text.insert("end", "\n".join(lines))

    def run_double_annotation_kappa(self) -> None:
        second_file = filedialog.askopenfilename(
            title="Chọn File CSV thứ 2 để đối soát Cohen's Kappa",
            filetypes=[("CSV files", "*.csv")],
        )
        if not second_file:
            return
        with Path(second_file).open("r", encoding="utf-8", newline="") as f:
            rows_b = list(csv.DictReader(f))

        key_fn = lambda r: (r.get("session_id"), r.get("camera_id"), str(r.get("track_id")), str(r.get("t_start_ms")), str(r.get("t_end_ms")))
        map_a = {key_fn(r): r for r in self.rows}
        map_b = {key_fn(r): r for r in rows_b}
        common = sorted(set(map_a.keys()) & set(map_b.keys()))

        if not common:
            messagebox.showwarning("Không khớp", "Hai file CSV không có cửa sổ nào trùng khớp khóa!")
            return

        valid_keys = [
            k for k in common
            if map_a[k].get("annotation_state") == "VALID"
            and map_b[k].get("annotation_state") == "VALID"
            and map_a[k].get("participation_label") in CANONICAL_TRAINING_LABELS
            and map_b[k].get("participation_label") in CANONICAL_TRAINING_LABELS
        ]
        lbl_a = [map_a[k].get("participation_label", "") for k in valid_keys]
        lbl_b = [map_b[k].get("participation_label", "") for k in valid_keys]
        agree_pct = float(np.mean([a == b for a, b in zip(lbl_a, lbl_b)]) * 100.0) if valid_keys else 0.0
        kappa_lbl = compute_cohens_kappa(lbl_a, lbl_b, list(CANONICAL_TRAINING_LABELS)) if valid_keys else 0.0

        self.qa_text.insert(
            "end",
            f"\n\n=========================================================================\n"
            f"   KẾT QUẢ ĐỐI SOÁT CHÉO 2 ANNOTATOR (TASK T12 — DOUBLE ANNOTATION QA)\n"
            f"=========================================================================\n"
            f"Số cửa sổ cùng VALID            : {len(valid_keys)}\n"
            f"Tỷ lệ đồng thuận nhãn 3-Class   : {agree_pct:.1f}%\n"
            f"Cohen's Kappa (3-Class Label)   : {kappa_lbl:.4f} ({'ĐẠT CHUẨN >= 0.70' if kappa_lbl >= 0.70 else 'CẦN ADJUDICATE'})\n",
        )


def main() -> None:
    root = tk.Tk()
    AnnotationStudioDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
