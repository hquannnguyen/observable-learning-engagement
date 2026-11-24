"""Standalone Native Desktop GUI — VERSION 2 (Multi-Student Classroom Labeling Studio).

Observable Learning Participation Analysis (CVF25PRJ10 • CVFP26BCPPRJC012)
Owner: Vũ Văn Phong (Data Lead) & Nguyễn Dũng Tiến (UI/Reporting Lead - B23DCPT306)

Version 2 Highlights (Multi-Student Scene Annotation):
1. All students present in the same 4-second time bucket (11–18 students simultaneously)
   are displayed with color-coded bounding boxes and label badges on the classroom image:
     - Green  = HIGH_OBSERVED_PARTICIPATION
     - Amber  = MODERATE_OBSERVED_PARTICIPATION
     - Red    = LOW_OBSERVED_PARTICIPATION
     - Gray   = OCCLUDED / TRANSITION / INVALID
2. Direct Interactive Mouse Click on the Classroom Image:
     - Left-Click on any student box: Select that student to inspect their 5-frame filmstrip
       and change their label via keys 1, 2, 3, 4.
     - Right-Click on any student box: Cycle their label (HIGH -> MOD -> LOW -> OCC) on the spot!
3. 1-Click Whole-Classroom Scene Approval:
     - Approve all 15–18 students in the current 4s time bucket as `HUMAN` in one click
       and automatically jump to the next 4-second time bucket (only 28 scenes for the whole 60s video!).
4. 100% Compliant with `docs/dataset_schema.md` (17 columns) and `src/contracts/labels.py`.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
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
from src.pose.extractor import assign_zone_id
try:
    from src.preprocessing.weak_labels import RULE_VERSION, apply_labeling_rules
except ImportError:
    RULE_VERSION = "weak-v0.1"

    def apply_labeling_rules(
        window_poses_df,
        class_phase: str = "lecture",
        min_visible_ratio: float = 0.50,
    ) -> Tuple[str, Optional[str], List[str], float]:
        """Fallback NaN-safe rule-based suggestion helper when src.preprocessing.weak_labels uses tensor API."""
        if len(window_poses_df) < 3:
            return "INVALID", None, ["UNKNOWN"], 0.0

        mean_vis = float(np.nanmean(window_poses_df["visible_ratio"].to_numpy(dtype=float)))
        mean_conf = float(np.nanmean(window_poses_df["mean_keypoint_conf"].to_numpy(dtype=float)))
        if not np.isfinite(mean_vis):
            mean_vis = 0.0
        if not np.isfinite(mean_conf):
            mean_conf = 0.0

        if mean_vis < min_visible_ratio or mean_conf < 0.25:
            return "OCCLUDED", None, ["OCCLUDED"], round(1.0 - mean_vis, 2)

        def _arr(col: str) -> np.ndarray:
            return window_poses_df[col].to_numpy(dtype=float)

        nose_y, nose_c = _arr("kpt_00_y"), _arr("kpt_00_conf")
        ears_y = (_arr("kpt_03_y") + _arr("kpt_04_y")) / 2.0
        l_sh_y, r_sh_y = _arr("kpt_05_y"), _arr("kpt_06_y")
        l_sh_c, r_sh_c = _arr("kpt_05_conf"), _arr("kpt_06_conf")
        l_wr_y, r_wr_y = _arr("kpt_09_y"), _arr("kpt_10_y")
        l_wr_c, r_wr_c = _arr("kpt_09_conf"), _arr("kpt_10_conf")
        l_wr_x, r_wr_x = _arr("kpt_09_x"), _arr("kpt_10_x")

        l_raised = (l_wr_c > 0.3) & (l_sh_c > 0.3) & np.isfinite(l_wr_y) & np.isfinite(l_sh_y) & (l_wrist_lt := (l_wr_y < l_sh_y - 20))
        r_raised = (r_wr_c > 0.3) & (r_sh_c > 0.3) & np.isfinite(r_wr_y) & np.isfinite(r_sh_y) & (r_wr_y < r_sh_y - 20)
        hand_raised_ratio = float(np.mean(l_raised | r_raised))

        valid_hd = np.isfinite(nose_y) & np.isfinite(ears_y) & (nose_c > 0.3)
        head_down_ratio = float(np.mean(valid_hd & (nose_y > ears_y + 12)))

        lw_std = float(np.nanstd(l_wr_x)) if np.any(np.isfinite(l_wr_x)) else 0.0
        rw_std = float(np.nanstd(r_wr_x)) if np.any(np.isfinite(r_wr_x)) else 0.0
        wrist_motion = lw_std + rw_std

        if hand_raised_ratio >= 0.15:
            return "VALID", HIGH, ["HAND_RAISED"], 0.90
        if (head_down_ratio >= 0.35) and (wrist_motion > 2.0):
            if class_phase in {"individual_work", "discussion"}:
                return "VALID", HIGH, ["READING_WRITING"], 0.85
            return "VALID", MODERATE, ["READING_WRITING"], 0.75
        if class_phase == "lecture" and head_down_ratio < 0.40 and mean_vis >= 0.65:
            return "VALID", HIGH, ["BOARD_OR_INSTRUCTOR_FACING"], 0.80
        if head_down_ratio > 0.75 and wrist_motion < 1.0:
            return "VALID", LOW, ["HEAD_DOWN"], 0.75
        return "VALID", MODERATE, ["BOARD_OR_INSTRUCTOR_FACING" if head_down_ratio < 0.5 else "HEAD_DOWN"], 0.70

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

PALETTE = {
    "bg_app": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "bg_header": "#0F172A",
    "border": "#E2E8F0",
    "text_main": "#0F172A",
    "text_muted": "#475569",
    "primary": "#2563EB",
    "primary_light": "#DBEAFE",
    "primary_text": "#1E40AF",
    "scene_green": "#15803D",
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

# Distinct BGR colors for OpenCV multi-student classroom rendering
BOX_COLORS_BGR = {
    HIGH: (46, 204, 113),       # Bright Green
    MODERATE: (15, 196, 241),   # Bright Amber/Gold (BGR)
    LOW: (60, 60, 235),         # Bright Red (BGR)
    "OCCLUDED": (140, 140, 140),
    "TRANSITION": (180, 140, 90),
    "INVALID": (110, 110, 110),
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


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _normalize_track_key(val: Any) -> str:
    """Normalize track_id across int (584), float (584.0), and string ('track_004' / '584')."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    if s.lower().startswith("track_"):
        tail = s[6:].strip()
        if tail.isdigit():
            return str(int(tail))
        return s.lower()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-6:
            return str(int(round(f)))
    except ValueError:
        pass
    return s


def _track_sort_key(val: Any) -> Tuple[int, str]:
    key = _normalize_track_key(val)
    if key.isdigit():
        return (0, f"{int(key):09d}")
    return (1, key)


def row_to_keypoints_array(pose_row) -> np.ndarray:
    kpts = np.zeros((17, 3), dtype=np.float32)
    for idx in range(17):
        x = _safe_float(pose_row.get(f"kpt_{idx:02d}_x", np.nan), np.nan)
        y = _safe_float(pose_row.get(f"kpt_{idx:02d}_y", np.nan), np.nan)
        c = _safe_float(pose_row.get(f"kpt_{idx:02d}_conf", 0.0), 0.0)
        if np.isfinite(x) and np.isfinite(y) and not (x == 0.0 and y == 0.0):
            kpts[idx, 0] = x
            kpts[idx, 1] = y
            kpts[idx, 2] = c
        else:
            kpts[idx, 0] = 0.0
            kpts[idx, 1] = 0.0
            kpts[idx, 2] = 0.0
    return kpts


def render_crop_with_pose(
    frame: np.ndarray,
    pose_row,
    draw_skel: bool = True,
    blur_face: bool = False,
    color_bgr: Tuple[int, int, int] = (46, 204, 113),
    out_size: Tuple[int, int] = (145, 150),
) -> np.ndarray:
    work = frame.copy()
    h, w = work.shape[:2]

    if pose_row is not None:
        x1 = _safe_float(pose_row.get("bbox_x1", np.nan), np.nan)
        y1 = _safe_float(pose_row.get("bbox_y1", np.nan), np.nan)
        x2 = _safe_float(pose_row.get("bbox_x2", np.nan), np.nan)
        y2 = _safe_float(pose_row.get("bbox_y2", np.nan), np.nan)

        if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2) and x2 > x1 and y2 > y1:
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


def get_bucket_ms(row: Dict[str, str]) -> int:
    """Group windows into 2-second time buckets (0, 2000, 4000, ...)."""
    try:
        ts = int(row.get("t_start_ms", 0))
    except ValueError:
        ts = 0
    return (ts // 2000) * 2000


def is_row_reviewed(row: Dict[str, str]) -> bool:
    """Return True if this window has been reviewed/confirmed by a human annotator.

    - VALID rows: label_source is HUMAN or ADJUDICATED.
    - Non-VALID rows (OCCLUDED/TRANSITION/INVALID): label_source must be empty per contract,
      so reviewed status is tracked by annotator_id not being RuleBot.
    """
    st_val = str(row.get("annotation_state", "")).strip()
    src_val = str(row.get("label_source", "")).strip()
    ann_id = str(row.get("annotator_id", "")).strip()
    if st_val == "VALID":
        return src_val in ("HUMAN", "ADJUDICATED")
    return bool(ann_id) and not ann_id.startswith("RuleBot")


# =====================================================================
# VERSION 2 DESKTOP APPLICATION (MULTI-STUDENT CLASSROOM MODE)
# =====================================================================

class AnnotationStudioV2App:
    CONTEXT_DISP_W = 680
    CONTEXT_DISP_H = 278

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CVF25PRJ10 — Studio Gán Nhãn Đa Sinh Viên Cùng Lúc (VERSION 2.1)")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        win_w = min(1400, max(1120, sw - 40))
        win_h = min(840, max(680, sh - 80))
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(1100, 680)
        self.root.configure(bg=PALETTE["bg_app"])

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

        # File paths (always start empty — user chooses their own video/data in Step 1)
        self.video_path_var = tk.StringVar(value="")
        self.pose_path_var = tk.StringVar(value="")
        self.meta_path_var = tk.StringVar(value="")
        self.csv_path_var = tk.StringVar(value="")
        self.annotator_var = tk.StringVar(value="HUMAN_ANNOTATOR")

        self.opt_draw_skel = tk.BooleanVar(value=True)
        self.opt_blur_face = tk.BooleanVar(value=False)
        self.opt_auto_next_student = tk.BooleanVar(value=False)

        # Runtime Data
        self.poses_df = None
        self.rows: List[Dict[str, str]] = []
        self.time_buckets: List[int] = []
        self.bucket_to_rows: Dict[int, List[int]] = {}
        self.current_bucket_ms: int = 0
        self.current_idx: int = 0
        self.is_playing_clip: bool = False

        # Undo stack: list of (changed_rows_dict, prev_bucket_ms, prev_idx, desc)
        self.undo_stack: List[Tuple[Dict[int, Dict[str, str]], int, int, str]] = []

        # Frame cache for instant switching between students within the same 4s bucket:
        # key = (video_path_str, bucket_ms) -> list of (ts_ms, bgr_frame)
        self._frame_cache_key: Optional[Tuple[str, int]] = None
        self._frame_cache_frames: List[Tuple[int, np.ndarray]] = []

        # Interactive clickable bounding boxes on the classroom canvas:
        # List of (x1_disp, y1_disp, x2_disp, y2_disp, row_idx, track_id)
        self.clickable_boxes: List[Tuple[int, int, int, int, int, int]] = []

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
        header = tk.Frame(self.root, bg=PALETTE["bg_header"], padx=18, pady=10)
        header.pack(fill="x")

        tk.Label(
            header,
            text="🎓 STUDIO CẮT VIDEO & GÁN NHÃN DỮ LIỆU (V2)",
            bg=PALETTE["bg_header"],
            fg="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        tk.Button(
            header,
            text="❓ Hướng Dẫn",
            command=self.open_help_dialog,
            bg="#0EA5E9",
            fg="#FFFFFF",
            activebackground="#0284C7",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=3,
            cursor="hand2",
        ).pack(side="right")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=8)

        self.tab_step1_trim = ttk.Frame(self.notebook)
        self.tab_step2_review = ttk.Frame(self.notebook)
        self.tab_step3_export = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_step1_trim, text="  BƯỚC 1: 🎥 Chọn Video & Cắt Đoạn  ")
        self.notebook.add(self.tab_step2_review, text="  BƯỚC 2: 🏫 Gán & Duyệt Nhiều Sinh Viên  ")
        self.notebook.add(self.tab_step3_export, text="  BƯỚC 3: 📊 Kiểm Định & Xuất File CSV  ")

        self._build_step1_trim()
        self._build_step2_review()
        self._build_step3_export()

        # Always start at STEP 1 so the user selects data, trim range, and triggers auto-labeling first!
        self.notebook.select(self.tab_step1_trim)

    # -----------------------------------------------------------------
    # SETTINGS DIALOG
    # -----------------------------------------------------------------
    def open_help_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Hướng dẫn nhanh cách sử dụng Studio Gán Nhãn (Version 2.1)")
        win.geometry("720x520")
        win.configure(bg="#FFFFFF")
        win.transient(self.root)
        win.grab_set()

        box = tk.Frame(win, bg="#FFFFFF", padx=22, pady=16)
        box.pack(fill="both", expand=True)

        tk.Label(
            box,
            text="📖 HƯỚNG DẪN SỬ DỤNG NHANH TRONG 30 GIÂY",
            bg="#FFFFFF",
            fg="#1E3A8A",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        guide_text = (
            "1️⃣ BƯỚC 1 (Chọn Video & Cắt Đoạn):\n"
            "   • Chỉ cần bấm [📂 Chọn Video...] để chọn file Video (.mp4 / .mov / .avi) từ máy của bạn.\n"
            "   • Chọn khoảng giây muốn cắt (Từ giây ➔ Đến giây), rồi bấm nút xanh dương:\n"
            "     [🚀 Cắt Cửa Sổ 4s & Tự Động Gán Nhãn ➔ Sang BƯỚC 2]\n"
            "   • Tool sẽ tự tìm dữ liệu khung xương (Pose) tương ứng hoặc tự chạy AI nhận diện khung xương\n"
            "     nếu là video mới — bạn không phải chọn nhiều file phức tạp ở Bước 1.\n\n"
            "2️⃣ BƯỚC 2 (Duyệt & Sửa Nhãn — Có thể bấm [📂 Mở CSV] để tiếp tục file gán dở):\n"
            "   ★ CÁCH A — Duyệt cả lớp 15–18 người trong 1 lần nhìn (KHUYÊN DÙNG):\n"
            "     • Trên ảnh lớn bên trái, mỗi sinh viên có khung màu theo nhãn máy đoán:\n"
            "       🟢 Xanh lá = CAO (HIGH) | 🟡 Vàng = VỪA (MOD) | 🔴 Đỏ = THẤP (LOW) | ⚪ Xám = BỊ CHE\n"
            "     • Nhìn cả lớp: Ai máy đoán sai màu thì CLICK CHUỘT PHẢI thẳng vào người đó trên ảnh để đổi màu\n"
            "       (hoặc Click chuột trái để chọn rồi gõ phím 1, 2, 3, 4).\n"
            "     • Khi thấy màu của mọi người đã đúng, bấm nút xanh lá [✅ DUYỆT CẢ LỚP] (hoặc phím Space).\n"
            "       Máy tự duyệt hết 15 người trong ảnh và nhảy sang mốc 4 giây tiếp theo!\n\n"
            "   ★ CÁCH B — Duyệt lần lượt từng sinh viên:\n"
            "     • Bấm [✓ Lưu SV Này & Sang SV Kế] (hoặc phím Enter). Khi hết sinh viên cuối cùng của\n"
            "       mốc hiện tại, phần mềm sẽ TỰ ĐỘNG chuyển sang sinh viên đầu tiên của mốc tiếp theo!\n\n"
            "   ★ PHÍM TẮT HỮU ÍCH:\n"
            "     • Phím 1 / 2 / 3 / 4 : Gán CAO / VỪA / THẤP / BỊ CHE cho sinh viên đang chọn.\n"
            "     • Phím Space         : Duyệt cả lớp ở mốc này & sang mốc 4 giây tiếp theo.\n"
            "     • Phím Enter         : Lưu sinh viên hiện tại & sang sinh viên kế tiếp.\n"
            "     • Phím Ctrl + Z      : Hoàn tác (Undo) nếu lỡ tay bấm nhầm.\n"
            "     • Phím ← / →         : Lùi / Tiến từng sinh viên.   PgUp / PgDn : Lùi / Tiến từng mốc 4s.\n\n"
            "3️⃣ BƯỚC 3 (Kiểm Tra Hợp Lệ & Xuất File CSV):\n"
            "   • Vào Tab Bước 3 xem báo cáo [✅ HỢP LỆ 100%] và bấm [💾 Xuất Bản Sao CSV]."
        )

        txt = tk.Text(box, font=("Segoe UI", 9), bg="#F8FAFC", fg="#0F172A", padx=12, pady=10, relief="flat", wrap="word")
        txt.insert("1.0", guide_text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, pady=(0, 10))

        tk.Button(
            box,
            text="✓ Đã Hiểu, Đóng Hướng Dẫn",
            command=win.destroy,
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=20,
            pady=6,
        ).pack()

    # -----------------------------------------------------------------
    # BƯỚC 1: CẮT VIDEO & TỰ ĐỘNG GÁN NHÃN
    # -----------------------------------------------------------------
    def _build_step1_trim(self) -> None:
        card = tk.Frame(
            self.tab_step1_trim,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=18,
            pady=12,
        )
        card.pack(fill="both", expand=True, pady=4)

        tk.Label(
            card,
            text="BƯỚC 1: CHỌN VIDEO & CẮT ĐOẠN CẦN PHÂN TÍCH",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        # --- PHẦN 1A: CHỈ CHỌN 1 FILE VIDEO DUY NHẤT ---
        data_box = tk.Frame(
            card,
            bg="#F8FAFC",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=14,
            pady=10,
        )
        data_box.pack(fill="x", pady=(6, 8))

        tk.Label(data_box, text="🎥 Video đầu vào (.mp4):", bg="#F8FAFC", fg="#1E3A8A", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=2)
        tk.Entry(data_box, textvariable=self.video_path_var, font=("Segoe UI", 9), bg="#FFFFFF").grid(row=0, column=1, sticky="ew", padx=10, pady=2)
        tk.Button(
            data_box, text="📂 Chọn Video...", command=self._browse_video_and_refresh,
            bg=PALETTE["primary"], fg="#FFFFFF", font=("Segoe UI", 9, "bold"), relief="flat", padx=14, pady=4, cursor="hand2"
        ).grid(row=0, column=2, pady=2)

        data_box.columnconfigure(1, weight=1)

        # --- PHẦN 1B: CHỌN KHOẢNG THỜI GIAN CẮT VIDEO & CỬA SỔ TRƯỢT 4 GIÂY ---
        p_bar = tk.Frame(card, bg="#F8FAFC", highlightbackground=PALETTE["border"], highlightthickness=1, padx=14, pady=8)
        p_bar.pack(fill="x", pady=(0, 8))

        tk.Label(p_bar, text="⏱️ Từ giây:", bg="#F8FAFC", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.trim_start_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(
            p_bar, from_=0.0, to=3600.0, increment=5.0, textvariable=self.trim_start_var,
            width=7, font=("Segoe UI", 9), command=self.update_trim_previews
        ).pack(side="left", padx=(4, 14))

        tk.Label(p_bar, text="⏱️ Đến giây:", bg="#F8FAFC", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.trim_end_var = tk.DoubleVar(value=60.0)
        tk.Spinbox(
            p_bar, from_=5.0, to=3600.0, increment=5.0, textvariable=self.trim_end_var,
            width=7, font=("Segoe UI", 9), command=self.update_trim_previews
        ).pack(side="left", padx=(4, 14))

        tk.Label(p_bar, text="Cửa sổ (s):", bg="#F8FAFC", font=("Segoe UI", 9)).pack(side="left")
        self.win_sec_var = tk.DoubleVar(value=4.0)
        tk.Spinbox(p_bar, from_=2.0, to=10.0, increment=1.0, textvariable=self.win_sec_var, width=5, font=("Segoe UI", 9)).pack(side="left", padx=(4, 12))

        tk.Label(p_bar, text="Bước trượt (s):", bg="#F8FAFC", font=("Segoe UI", 9)).pack(side="left")
        self.stride_sec_var = tk.DoubleVar(value=2.0)
        tk.Spinbox(p_bar, from_=1.0, to=5.0, increment=1.0, textvariable=self.stride_sec_var, width=5, font=("Segoe UI", 9)).pack(side="left", padx=4)

        self.opt_export_trimmed_mp4 = tk.BooleanVar(value=False)
        tk.Checkbutton(
            p_bar,
            text="Lưu kèm file .mp4 đã cắt",
            variable=self.opt_export_trimmed_mp4,
            bg="#F8FAFC",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(12, 4))

        tk.Button(
            p_bar,
            text="🔄 Xem Trước Khung Hình Đầu & Cuối",
            command=self.update_trim_previews,
            bg="#E2E8F0",
            fg="#0F172A",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=10,
            pady=3,
        ).pack(side="right")

        # --- PHẦN 1C: NÚT HÀNH ĐỘNG CHÍNH CỦA BƯỚC 1 ---
        btn_bar = tk.Frame(card, bg="#FFFFFF", pady=4)
        btn_bar.pack(fill="x")

        tk.Button(
            btn_bar,
            text="🚀 Cắt Cửa Sổ 4s & Tự Động Gán Nhãn ➔ Sang BƯỚC 2",
            command=self.run_auto_label_pipeline,
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_bar,
            text="💾 Chỉ Cắt & Lưu Video (.mp4)",
            command=self.run_trim_video_file,
            bg="#F1F5F9",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
        ).pack(side="left", padx=4)

        self.trim_prog_lbl = tk.Label(
            card,
            text="ℹ️ Bấm [📂 Chọn Video...] ở trên để chọn video cần cắt và gán nhãn.",
            bg="#FFFFFF",
            fg="#475569",
            font=("Segoe UI", 9, "bold"),
        )
        self.trim_prog_lbl.pack(anchor="w", pady=(2, 4))

        prev_grid = tk.Frame(card, bg="#0F172A", padx=10, pady=8)
        prev_grid.pack(fill="both", expand=True)

        self.trim_start_canvas = tk.Label(prev_grid, text="Chưa chọn video — bấm [📂 Chọn Video...] ở trên", bg="#0F172A", fg="#94A3B8", font=("Segoe UI", 10))
        self.trim_start_canvas.pack(side="left", fill="both", expand=True, padx=6)

        self.trim_end_canvas = tk.Label(prev_grid, text="Khung hình Điểm Kết Thúc sẽ hiện sau khi chọn video", bg="#0F172A", fg="#94A3B8", font=("Segoe UI", 10))
        self.trim_end_canvas.pack(side="right", fill="both", expand=True, padx=6)

    # -----------------------------------------------------------------
    # BƯỚC 2: MULTI-STUDENT CLASSROOM SCENE STUDIO (VERSION 2 CORE)
    # -----------------------------------------------------------------
    def _build_step2_review(self) -> None:
        # 1. Top Navigation Bar by Classroom Time Bucket (28 scenes instead of 427 isolated rows!)
        nav_bar = tk.Frame(
            self.tab_step2_review,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=12,
            pady=7,
        )
        nav_bar.pack(fill="x", pady=(2, 6))

        tk.Label(nav_bar, text="⏱️ Mốc 4s:", bg="#FFFFFF", font=("Segoe UI", 9, "bold")).pack(side="left")

        tk.Button(
            nav_bar, text="⏪ Trước", command=self.nav_prev_bucket,
            bg="#F1F5F9", fg="#0F172A", font=("Segoe UI", 9, "bold"), relief="flat", padx=8, pady=3
        ).pack(side="left", padx=(4, 2))

        self.bucket_selector_cb = ttk.Combobox(nav_bar, state="readonly", width=27, font=("Segoe UI", 9, "bold"))
        self.bucket_selector_cb.pack(side="left", padx=3)
        self.bucket_selector_cb.bind("<<ComboboxSelected>>", self._on_bucket_combo_select)

        tk.Button(
            nav_bar, text="Tiếp ⏩", command=self.nav_next_bucket,
            bg="#F1F5F9", fg="#0F172A", font=("Segoe UI", 9, "bold"), relief="flat", padx=7, pady=3
        ).pack(side="left", padx=(2, 4))

        tk.Button(
            nav_bar, text="⏭ Chưa Duyệt", command=self.nav_next_unreviewed_bucket,
            bg=PALETTE["primary_light"], fg=PALETTE["primary_text"], font=("Segoe UI", 8, "bold"), relief="flat", padx=7, pady=3
        ).pack(side="left", padx=(0, 8))

        tk.Label(nav_bar, text="👤 SV:", bg="#FFFFFF", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.scene_student_cb = ttk.Combobox(nav_bar, state="readonly", width=18, font=("Segoe UI", 9))
        self.scene_student_cb.pack(side="left", padx=3)
        self.scene_student_cb.bind("<<ComboboxSelected>>", self._on_scene_student_select)

        tk.Button(
            nav_bar, text="◀ SV", command=self.nav_prev_student_in_bucket,
            bg="#F1F5F9", fg="#0F172A", font=("Segoe UI", 8, "bold"), relief="flat", padx=5, pady=3
        ).pack(side="left", padx=1)
        tk.Button(
            nav_bar, text="SV ▶", command=self.nav_next_student_in_bucket,
            bg="#F1F5F9", fg="#0F172A", font=("Segoe UI", 8, "bold"), relief="flat", padx=5, pady=3
        ).pack(side="left", padx=1)

        tk.Checkbutton(
            nav_bar, text="Mờ mặt", variable=self.opt_blur_face, bg="#FFFFFF", command=self.refresh_current_window
        ).pack(side="right", padx=3)
        tk.Checkbutton(
            nav_bar, text="Khung xương", variable=self.opt_draw_skel, bg="#FFFFFF", command=self.refresh_current_window
        ).pack(side="right", padx=3)
        tk.Button(
            nav_bar, text="🦴 Đổi Pose", command=self._browse_pose_and_refresh,
            bg="#F1F5F9", fg="#334155", font=("Segoe UI", 8, "bold"), relief="flat", padx=6, pady=3, cursor="hand2"
        ).pack(side="right", padx=2)
        tk.Button(
            nav_bar, text="📂 Mở CSV", command=self.open_existing_csv_to_step2,
            bg="#DCFCE7", fg="#166534", font=("Segoe UI", 8, "bold"), relief="flat", padx=6, pady=3, cursor="hand2"
        ).pack(side="right", padx=2)

        # 2. Main Two-Column Workspace
        work_area = tk.Frame(self.tab_step2_review, bg=PALETTE["bg_app"])
        work_area.pack(fill="both", expand=True)

        left_panel = tk.Frame(
            work_area,
            bg="#FFFFFF",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

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
        # LEFT PANEL: INTERACTIVE MULTI-STUDENT CLASSROOM VIEW
        # =============================================================
        top_left_bar = tk.Frame(left_panel, bg="#FFFFFF")
        top_left_bar.pack(side="top", fill="x", pady=(0, 4))

        self.window_title_lbl = tk.Label(
            top_left_bar,
            text="Chưa có dữ liệu — hãy chọn Video & dữ liệu ở BƯỚC 1 trước",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 10, "bold"),
        )
        self.window_title_lbl.pack(side="left")

        tk.Button(
            top_left_bar,
            text="▶ Phát Video 4s Cả Lớp",
            command=self.play_4s_clip_live,
            bg="#EFF6FF",
            fg="#1D4ED8",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=2,
            cursor="hand2",
        ).pack(side="right")

        # Color Legend Bar right above the classroom image so colors are immediately clear
        legend_bar = tk.Label(
            left_panel,
            text="Chú thích khung màu:   🟢 Xanh lá = CAO (1)   •   🟡 Vàng = VỪA (2)   •   🔴 Đỏ = THẤP (3)   •   ⚪ Xám = BỊ CHE (4)   •   ⬜ Viền trắng = Đang chọn",
            bg="#F8FAFC",
            fg="#334155",
            font=("Segoe UI", 8, "bold"),
            pady=2,
        )
        legend_bar.pack(side="top", fill="x", pady=(0, 2))

        # Pack bottom elements FIRST so they are never clipped
        self.ai_suggest_lbl = tk.Label(
            left_panel,
            text="🤖 Hãy chọn Video & dữ liệu ở BƯỚC 1 để bắt đầu gán nhãn.",
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

        self.filmstrip_title_lbl = tk.Label(
            left_panel,
            text="🎞️ 5 khung hình chi tiết của sinh viên đang chọn (Click vào người khác trên ảnh lớn để đổi):",
            bg="#FFFFFF",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
        )
        self.filmstrip_title_lbl.pack(side="bottom", anchor="w", pady=(2, 0))

        # Main Interactive Classroom Canvas (Supports Left-Click to select & Right-Click to cycle label!)
        self.context_canvas = tk.Label(left_panel, bg="#0F172A", cursor="hand2")
        self.context_canvas.pack(side="top", fill="both", expand=True, pady=(0, 4))
        self.context_canvas.bind("<Button-1>", self._on_classroom_left_click)
        self.context_canvas.bind("<Button-3>", self._on_classroom_right_click)

        # =============================================================
        # RIGHT PANEL: WHOLE-CLASS BATCH & INDIVIDUAL CONTROLS
        # =============================================================
        # 1. WHOLE-CLASSROOM SCENE APPROVAL CARD (The core Version 2 feature!)
        scene_card = tk.Frame(
            right_panel,
            bg="#ECFDF5",
            highlightbackground="#6EE7B7",
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        scene_card.pack(fill="x", pady=(0, 10))

        self.progress_summary_lbl = tk.Label(
            scene_card,
            text="📊 Chưa tải dữ liệu (hãy chọn ở BƯỚC 1)",
            bg="#ECFDF5",
            fg="#065F46",
            font=("Segoe UI", 9, "bold"),
        )
        self.progress_summary_lbl.pack(anchor="w")

        tk.Button(
            scene_card,
            text="✅ DUYỆT CẢ LỚP Ở MỐC NÀY & SANG MỐC GIÂY TIẾP (Space)",
            command=lambda: self.approve_entire_scene(advance_bucket=True),
            bg=PALETTE["scene_green"],
            fg="#FFFFFF",
            activebackground="#166534",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=7,
            cursor="hand2",
        ).pack(fill="x", pady=(6, 4))

        batch_row = tk.Frame(scene_card, bg="#ECFDF5")
        batch_row.pack(fill="x")
        tk.Label(batch_row, text="Gán đồng loạt cả lớp ở mốc này:", bg="#ECFDF5", fg="#047857", font=("Segoe UI", 8, "bold")).pack(side="left")

        tk.Button(
            batch_row, text="Tất cả CAO", command=lambda: self.set_entire_scene_label(HIGH),
            bg="#DCFCE7", fg="#166534", font=("Segoe UI", 8, "bold"), relief="flat", padx=6, pady=1
        ).pack(side="right", padx=2)
        tk.Button(
            batch_row, text="Tất cả VỪA", command=lambda: self.set_entire_scene_label(MODERATE),
            bg="#FEF3C7", fg="#92400E", font=("Segoe UI", 8, "bold"), relief="flat", padx=6, pady=1
        ).pack(side="right", padx=2)
        tk.Button(
            batch_row, text="Tất cả THẤP", command=lambda: self.set_entire_scene_label(LOW),
            bg="#FEE2E2", fg="#991B1B", font=("Segoe UI", 8, "bold"), relief="flat", padx=6, pady=1
        ).pack(side="right", padx=2)

        # 2. SELECTED STUDENT INDIVIDUAL CONTROLS
        stu_header = tk.Frame(right_panel, bg="#FFFFFF")
        stu_header.pack(fill="x", pady=(0, 4))

        self.current_status_badge = tk.Label(
            stu_header,
            text="👤 Chưa chọn sinh viên",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
        )
        self.current_status_badge.pack(side="left")

        tk.Checkbutton(
            stu_header,
            text="Tự nhảy SV",
            variable=self.opt_auto_next_student,
            bg="#FFFFFF",
            fg="#475569",
            font=("Segoe UI", 8),
        ).pack(side="right")

        btn_grid = tk.Frame(right_panel, bg="#FFFFFF")
        btn_grid.pack(fill="x", pady=(0, 10))

        self.btn_choice_high = tk.Button(
            btn_grid,
            text="[Phím 1]  CAO (HIGH)",
            command=lambda: self.select_or_save_label("VALID", HIGH),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=6,
            cursor="hand2",
        )
        self.btn_choice_high.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)

        self.btn_choice_mod = tk.Button(
            btn_grid,
            text="[Phím 2]  VỪA (MODERATE)",
            command=lambda: self.select_or_save_label("VALID", MODERATE),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=6,
            cursor="hand2",
        )
        self.btn_choice_mod.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)

        self.btn_choice_low = tk.Button(
            btn_grid,
            text="[Phím 3]  THẤP (LOW)",
            command=lambda: self.select_or_save_label("VALID", LOW),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=6,
            cursor="hand2",
        )
        self.btn_choice_low.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)

        self.btn_choice_occ = tk.Button(
            btn_grid,
            text="[Phím 4]  BỊ CHE (OCCLUDED)",
            command=lambda: self.select_or_save_label("OCCLUDED", ""),
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=6,
            cursor="hand2",
        )
        self.btn_choice_occ.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)

        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        self.edit_state_var = tk.StringVar(value="VALID")
        self.edit_label_var = tk.StringVar(value=LOW)

        # 3. ATOMIC BEHAVIORS FOR SELECTED STUDENT
        tk.Label(
            right_panel,
            text="Hành vi của sinh viên đang chọn (Tích chọn 1 hoặc nhiều):",
            bg="#FFFFFF",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        beh_frame = tk.Frame(
            right_panel,
            bg="#F8FAFC",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=10,
            pady=6,
        )
        beh_frame.pack(fill="x", pady=(0, 10))

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
            cb.grid(row=idx_b // 2, column=idx_b % 2, sticky="ew", padx=4, pady=1)

        beh_frame.columnconfigure(0, weight=1)
        beh_frame.columnconfigure(1, weight=1)

        # 4. CONTEXT FIELDS
        ctx_grid = tk.Frame(right_panel, bg="#FFFFFF")
        ctx_grid.pack(fill="x", pady=(0, 8))

        tk.Label(ctx_grid, text="Trạng thái:", bg="#FFFFFF", fg="#475569").grid(row=0, column=0, sticky="w", pady=3)
        self.state_combo = ttk.Combobox(ctx_grid, values=list(ANNOTATION_STATES), state="readonly", width=13)
        self.state_combo.grid(row=0, column=1, sticky="w", padx=6, pady=3)
        self.state_combo.bind("<<ComboboxSelected>>", self._on_state_combo_change)

        tk.Label(ctx_grid, text="Pha học:", bg="#FFFFFF", fg="#475569").grid(row=0, column=2, sticky="w", padx=(12, 0), pady=3)
        self.edit_phase_cb = ttk.Combobox(ctx_grid, values=CLASS_PHASES, state="readonly", width=14)
        self.edit_phase_cb.grid(row=0, column=3, sticky="w", padx=6, pady=3)

        tk.Label(ctx_grid, text="Vùng ghế:", bg="#FFFFFF", fg="#475569").grid(row=1, column=0, sticky="w", pady=3)
        self.edit_zone_cb = ttk.Combobox(ctx_grid, values=ZONE_IDS, state="readonly", width=13)
        self.edit_zone_cb.grid(row=1, column=1, sticky="w", padx=6, pady=3)

        tk.Label(ctx_grid, text="Che khuất:", bg="#FFFFFF", fg="#475569").grid(row=1, column=2, sticky="w", padx=(12, 0), pady=3)
        self.edit_occ_cb = ttk.Combobox(ctx_grid, values=OCCLUSION_LEVELS, state="readonly", width=14)
        self.edit_occ_cb.grid(row=1, column=3, sticky="w", padx=6, pady=3)

        tk.Label(ctx_grid, text="Ghi chú:", bg="#FFFFFF", fg="#475569").grid(row=2, column=0, sticky="w", pady=3)
        self.edit_notes_var = tk.StringVar(value="")
        tk.Entry(ctx_grid, textvariable=self.edit_notes_var, bg="#F8FAFC", font=("Segoe UI", 9)).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=6, pady=3
        )
        ctx_grid.columnconfigure(3, weight=1)

        # 5. SAVE BUTTONS FOR SELECTED STUDENT / TRACK + UNDO
        sub_btns = tk.Frame(right_panel, bg="#FFFFFF")
        sub_btns.pack(fill="x", pady=(2, 3))

        tk.Button(
            sub_btns,
            text="✓ Lưu SV Này & Sang SV Kế (Enter)",
            command=lambda: self.save_current_window(advance_student=True),
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            pady=5,
            cursor="hand2",
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))

        tk.Button(
            sub_btns,
            text="↩ Hoàn Tác",
            command=self.undo_last_action,
            bg="#FEF2F2",
            fg="#991B1B",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            pady=5,
            padx=8,
            cursor="hand2",
        ).pack(side="right", padx=(3, 0))

        tk.Button(
            sub_btns,
            text="⚡ Duyệt SV Suốt 60s",
            command=self.approve_selected_track_all_time,
            bg="#F1F5F9",
            fg="#334155",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            pady=5,
            padx=6,
            cursor="hand2",
        ).pack(side="right", padx=(3, 0))

        # Live Action Feedback Toast Bar
        self.toast_lbl = tk.Label(
            right_panel,
            text="💡 Sẵn sàng. Bấm phím 1/2/3/4, Enter (từng SV) hoặc Space (duyệt cả lớp).",
            bg="#F1F5F9",
            fg="#1E40AF",
            font=("Segoe UI", 8, "bold"),
            anchor="w",
            padx=8,
            pady=3,
        )
        self.toast_lbl.pack(fill="x", pady=(2, 0))

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
            text="💾 Xuất File CSV (Save As...)",
            command=self.export_csv_copy,
            bg=PALETTE["primary"],
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
        ).pack(side="right", padx=4)

        tk.Button(
            top_bar,
            text="🎬 Xuất Clip 4s (.mp4)",
            command=self.run_export_window_clips,
            bg="#F1F5F9",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
        ).pack(side="right", padx=4)

        tk.Button(
            top_bar,
            text="🤝 Đối Soát Cohen's Kappa",
            command=self.run_double_annotation_kappa,
            bg="#F1F5F9",
            fg="#0F172A",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
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
        self.root.bind("<Left>", lambda e: self._shortcut_guard(self.nav_prev_student_in_bucket))
        self.root.bind("<Right>", lambda e: self._shortcut_guard(self.nav_next_student_in_bucket))
        self.root.bind("<Prior>", lambda e: self._shortcut_guard(self.nav_prev_bucket))  # PageUp
        self.root.bind("<Next>", lambda e: self._shortcut_guard(self.nav_next_bucket))   # PageDown
        self.root.bind("1", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("VALID", HIGH)))
        self.root.bind("2", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("VALID", MODERATE)))
        self.root.bind("3", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("VALID", LOW)))
        self.root.bind("4", lambda e: self._shortcut_guard(lambda: self.select_or_save_label("OCCLUDED", "")))
        self.root.bind("<Return>", lambda e: self._shortcut_guard(lambda: self.save_current_window(advance_student=True)))
        self.root.bind("<space>", lambda e: self._shortcut_guard(lambda: self.approve_entire_scene(advance_bucket=True)))
        self.root.bind("<Control-z>", lambda e: self._shortcut_guard(self.undo_last_action))
        self.root.bind("<Control-Z>", lambda e: self._shortcut_guard(self.undo_last_action))

    def set_toast(self, message: str, color: str = "#1E40AF") -> None:
        if hasattr(self, "toast_lbl"):
            self.toast_lbl.config(text=message, fg=color)

    def _push_undo(self, row_indices: List[int], desc: str) -> None:
        snapshot = {idx: dict(self.rows[idx]) for idx in row_indices if 0 <= idx < len(self.rows)}
        self.undo_stack.append((snapshot, self.current_bucket_ms, self.current_idx, desc))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo_last_action(self) -> None:
        if not self.undo_stack:
            self.set_toast("ℹ️ Không có thao tác nào để hoàn tác.", "#64748B")
            return
        snapshot, prev_bucket, prev_idx, desc = self.undo_stack.pop()
        for idx, old_row in snapshot.items():
            if 0 <= idx < len(self.rows):
                self.rows[idx] = old_row
        self._write_rows_to_csv()
        self.current_bucket_ms = prev_bucket
        self.current_idx = prev_idx
        self.refresh_current_window()
        self.set_toast(f"↩️ Đã hoàn tác: {desc}", "#B91C1C")

    def _shortcut_guard(self, func) -> None:
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Spinbox, tk.Text)):
            return
        if self.notebook.index(self.notebook.select()) == 1:
            func()

    # -----------------------------------------------------------------
    # DATA LOADING & TIME BUCKET INDEXING
    # -----------------------------------------------------------------
    @staticmethod
    def _get_path(var: tk.StringVar) -> Optional[Path]:
        raw = var.get().strip()
        return Path(raw) if raw else None

    def _find_matching_pose_file(self, vid_p: Path) -> Optional[Path]:
        pose_candidates = [
            vid_p.parent / f"{vid_p.stem}_poses.parquet",
            vid_p.parent / f"{vid_p.stem}_poses.csv",
            ROOT / "data/poses" / f"{vid_p.stem}_poses.parquet",
            ROOT / "data/poses" / f"{vid_p.stem}_poses.csv",
        ]
        for cand in pose_candidates:
            if cand.is_file():
                return cand
        return None

    def _browse_video_and_refresh(self) -> None:
        init_dir = str(ROOT / "data/raw") if (ROOT / "data/raw").exists() else str(ROOT)
        p = filedialog.askopenfilename(
            title="Chọn Video lớp học (.mp4 / .mov / .avi / .mkv)",
            initialdir=init_dir,
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not p:
            return
        vid_p = Path(p)
        self.video_path_var.set(vid_p.as_posix())
        self._frame_cache_key = None
        self._frame_cache_frames.clear()

        # Detect video duration and auto-set sensible trim range
        cap = cv2.VideoCapture(str(vid_p))
        dur_str = ""
        if cap.isOpened():
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                dur_sec = total_frames / fps if fps > 0 and total_frames > 0 else 60.0
                self.trim_start_var.set(0.0)
                self.trim_end_var.set(round(min(60.0, dur_sec), 1))
                dur_str = f" ({dur_sec:.1f}s)"
            finally:
                cap.release()

        # Auto-detect sibling metadata.json if present
        sib_meta = vid_p.parent / "metadata.json"
        self.meta_path_var.set(sib_meta.as_posix() if sib_meta.is_file() else "")

        # Auto-match Pose file for this specific video (hidden from Step 1 to keep Step 1 clean)
        matched_pose = self._find_matching_pose_file(vid_p)
        if matched_pose is not None:
            self.pose_path_var.set(matched_pose.as_posix())
            extra_note = f"  •  Đã có sẵn dữ liệu khung xương ({matched_pose.name})"
        else:
            self.pose_path_var.set("")
            extra_note = "  •  Video mới (sẽ tự trích xuất khung xương khi bấm Bắt đầu)"

        self.reload_all_resources()
        self.trim_prog_lbl.config(
            text=f"✅ Đã chọn video: {vid_p.name}{dur_str}{extra_note}",
            fg="#166534",
        )

    def _browse_pose_and_refresh(self) -> None:
        init_dir = str(ROOT / "data/poses") if (ROOT / "data/poses").exists() else str(ROOT)
        p = filedialog.askopenfilename(
            title="Chọn File Pose (.parquet / .csv)",
            initialdir=init_dir,
            filetypes=[("Pose files", "*.parquet *.csv"), ("All files", "*.*")],
        )
        if p:
            self.pose_path_var.set(Path(p).as_posix())
            self.reload_all_resources()
            self.set_toast(f"🦴 Đã nạp file Pose: {Path(p).name}", "#166534")
            self.trim_prog_lbl.config(text=f"✅ Đã nạp file Pose: {Path(p).name}", fg="#166534")

    def _browse_csv_and_refresh(self) -> None:
        init_dir = str(ROOT / "data/annotations") if (ROOT / "data/annotations").exists() else str(ROOT)
        p = filedialog.askopenfilename(
            title="Mở File CSV nhãn (.csv)",
            initialdir=init_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if p:
            self.csv_path_var.set(Path(p).as_posix())
            self.reload_all_resources()
            self.set_toast(f"📂 Đã mở file CSV: {Path(p).name} ({len(self.rows)} mẫu)", "#166534")

    def open_existing_csv_to_step2(self) -> None:
        self._browse_csv_and_refresh()
        csv_p = self._get_path(self.csv_path_var)
        if csv_p is None or not csv_p.is_file():
            return
        if not self.rows:
            messagebox.showwarning("File CSV trống", "File CSV đã chọn không có dòng dữ liệu nào.")
            return
        self.notebook.select(self.tab_step2_review)
        if self._get_path(self.video_path_var) is None:
            self.set_toast("ℹ️ Đã mở CSV. Hãy chọn Video tương ứng ở BƯỚC 1 nếu cần xem ảnh lớp học.", "#B45309")

    def reload_all_resources(self) -> None:
        pose_p = self._get_path(self.pose_path_var)
        csv_p = self._get_path(self.csv_path_var)

        self.poses_df = None
        if pose_p is not None and pose_p.is_file():
            try:
                df_loaded = load_poses_table(pose_p)
                if df_loaded is not None and not df_loaded.empty and "track_id" in df_loaded.columns:
                    df_loaded = df_loaded.copy()
                    df_loaded["_track_key"] = df_loaded["track_id"].map(_normalize_track_key)
                self.poses_df = df_loaded
            except Exception:
                self.poses_df = None

        self.rows = []
        if csv_p is not None and csv_p.is_file():
            with csv_p.open("r", encoding="utf-8", newline="") as f:
                self.rows = [dict(r) for r in csv.DictReader(f)]

        self._rebuild_time_buckets()

        if self.time_buckets:
            if self.current_bucket_ms not in self.bucket_to_rows:
                self.current_bucket_ms = self.time_buckets[0]
            scene_rows = self.bucket_to_rows[self.current_bucket_ms]
            if self.current_idx not in scene_rows:
                self.current_idx = scene_rows[0]
            self.refresh_current_window()
        else:
            self.bucket_selector_cb.set("")
            self.scene_student_cb.set("")
            self.scene_student_cb["values"] = []

        self.update_trim_previews()
        self.run_qa_validation_report()

    def _rebuild_time_buckets(self) -> None:
        self.bucket_to_rows.clear()
        for idx, r in enumerate(self.rows):
            b_ms = get_bucket_ms(r)
            self.bucket_to_rows.setdefault(b_ms, []).append(idx)

        self.time_buckets = sorted(self.bucket_to_rows.keys())
        # Sort students within each bucket by normalized track_id
        for b_ms in self.time_buckets:
            self.bucket_to_rows[b_ms].sort(
                key=lambda i: _track_sort_key(self.rows[i].get("track_id", ""))
            )

        self._refresh_bucket_combobox_labels()

    def _refresh_bucket_combobox_labels(self) -> None:
        items = []
        for idx_b, b_ms in enumerate(self.time_buckets):
            r_indices = self.bucket_to_rows[b_ms]
            done_c = sum(1 for i in r_indices if is_row_reviewed(self.rows[i]))
            status_tag = "✓ Xong" if done_c == len(r_indices) else f"{done_c}/{len(r_indices)} SV"
            items.append(
                f"Mốc #{idx_b+1}/{len(self.time_buckets)} ({b_ms/1000:.0f}s-{(b_ms+4000)/1000:.0f}s) • {len(r_indices)} SV [{status_tag}]"
            )
        self.bucket_selector_cb["values"] = items
        if self.current_bucket_ms in self.time_buckets:
            self.bucket_selector_cb.current(self.time_buckets.index(self.current_bucket_ms))

    def _refresh_scene_student_combobox(self) -> None:
        r_indices = self.bucket_to_rows.get(self.current_bucket_ms, [])
        stu_items = []
        for i in r_indices:
            r = self.rows[i]
            lbl = r.get("participation_label", "")
            disp = to_display_label(lbl) if lbl in CANONICAL_TRAINING_LABELS else r.get("annotation_state", "")
            chk = "✓" if is_row_reviewed(r) else "•"
            z_raw = str(r.get("zone_id", "")).strip()
            z_disp = z_raw if z_raw and z_raw.lower() not in ("nan", "none") else "?"
            tid_disp = _normalize_track_key(r.get("track_id", "")) or str(r.get("track_id", ""))
            stu_items.append(f"{chk} Track #{tid_disp} ({z_disp}) ➔ {disp}")
        self.scene_student_cb["values"] = stu_items
        if self.current_idx in r_indices:
            self.scene_student_cb.current(r_indices.index(self.current_idx))

    # -----------------------------------------------------------------
    # SCENE & STUDENT NAVIGATION (WITH SEAMLESS CROSS-BUCKET ROLLOVER!)
    # -----------------------------------------------------------------
    def _on_bucket_combo_select(self, _event=None) -> None:
        sel = self.bucket_selector_cb.current()
        if 0 <= sel < len(self.time_buckets):
            self.current_bucket_ms = self.time_buckets[sel]
            r_indices = self.bucket_to_rows[self.current_bucket_ms]
            if r_indices:
                self.current_idx = r_indices[0]
            self.refresh_current_window()

    def _on_scene_student_select(self, _event=None) -> None:
        sel = self.scene_student_cb.current()
        r_indices = self.bucket_to_rows.get(self.current_bucket_ms, [])
        if 0 <= sel < len(r_indices):
            self.current_idx = r_indices[sel]
            self.refresh_current_window()

    def nav_prev_bucket(self) -> None:
        if self.current_bucket_ms in self.time_buckets:
            pos = self.time_buckets.index(self.current_bucket_ms)
            if pos > 0:
                self.current_bucket_ms = self.time_buckets[pos - 1]
                self.current_idx = self.bucket_to_rows[self.current_bucket_ms][0]
                self.refresh_current_window()
                self.set_toast(f"⏪ Đã lùi về Mốc #{pos}/{len(self.time_buckets)}")

    def nav_next_bucket(self) -> None:
        if self.current_bucket_ms in self.time_buckets:
            pos = self.time_buckets.index(self.current_bucket_ms)
            if pos + 1 < len(self.time_buckets):
                self.current_bucket_ms = self.time_buckets[pos + 1]
                self.current_idx = self.bucket_to_rows[self.current_bucket_ms][0]
                self.refresh_current_window()
                self.set_toast(f"⏩ Đã sang Mốc #{pos+2}/{len(self.time_buckets)}")

    def nav_next_unreviewed_bucket(self) -> None:
        if not self.time_buckets:
            return
        cur_pos = self.time_buckets.index(self.current_bucket_ms) if self.current_bucket_ms in self.time_buckets else 0
        search_order = self.time_buckets[cur_pos + 1:] + self.time_buckets[:cur_pos + 1]
        for b_ms in search_order:
            r_indices = self.bucket_to_rows[b_ms]
            unrev = [i for i in r_indices if not is_row_reviewed(self.rows[i])]
            if unrev:
                self.current_bucket_ms = b_ms
                self.current_idx = unrev[0]
                self.refresh_current_window()
                b_num = self.time_buckets.index(b_ms) + 1
                self.set_toast(f"⏭ Nhảy tới Mốc #{b_num} (còn {len(unrev)} SV chưa duyệt)")
                return
        self.set_toast("🎉 Tất cả 100% cửa sổ trong toàn bộ video đã được duyệt!", "#15803D")
        messagebox.showinfo("Hoàn thành 100%", "Tất cả sinh viên ở mọi mốc thời gian đều đã được duyệt! Bạn có thể sang Bước 3 để kiểm tra & xuất CSV.")

    def nav_prev_student_in_bucket(self) -> None:
        r_indices = self.bucket_to_rows.get(self.current_bucket_ms, [])
        if self.current_idx in r_indices:
            pos = r_indices.index(self.current_idx)
            if pos > 0:
                self.current_idx = r_indices[pos - 1]
                self.refresh_current_window()
            elif self.current_bucket_ms in self.time_buckets:
                b_pos = self.time_buckets.index(self.current_bucket_ms)
                if b_pos > 0:
                    prev_b = self.time_buckets[b_pos - 1]
                    self.current_bucket_ms = prev_b
                    self.current_idx = self.bucket_to_rows[prev_b][-1]
                    self.refresh_current_window()
                    self.set_toast(f"◀ Lùi sang SV cuối của Mốc #{b_pos}/{len(self.time_buckets)}")

    def nav_next_student_in_bucket(self) -> None:
        r_indices = self.bucket_to_rows.get(self.current_bucket_ms, [])
        if self.current_idx in r_indices:
            pos = r_indices.index(self.current_idx)
            if pos + 1 < len(r_indices):
                self.current_idx = r_indices[pos + 1]
                self.refresh_current_window()
            elif self.current_bucket_ms in self.time_buckets:
                b_pos = self.time_buckets.index(self.current_bucket_ms)
                if b_pos + 1 < len(self.time_buckets):
                    next_b = self.time_buckets[b_pos + 1]
                    self.current_bucket_ms = next_b
                    self.current_idx = self.bucket_to_rows[next_b][0]
                    self.refresh_current_window()
                    self.set_toast(f"⏩ Hết SV của Mốc #{b_pos+1} ➔ Tự động sang Mốc #{b_pos+2}/{len(self.time_buckets)}!")
                else:
                    self.set_toast("🏁 Đã đến sinh viên cuối cùng của mốc thời gian cuối cùng!", "#15803D")

    # -----------------------------------------------------------------
    # INTERACTIVE MOUSE CLICK ON CLASSROOM IMAGE
    # -----------------------------------------------------------------
    def _find_clicked_student_row(self, event_x: int, event_y: int) -> Optional[int]:
        if not self.clickable_boxes:
            return None
        # Compute offset if label widget is wider/taller than the 680x290 image
        w_lbl = max(self.CONTEXT_DISP_W, self.context_canvas.winfo_width())
        h_lbl = max(self.CONTEXT_DISP_H, self.context_canvas.winfo_height())
        off_x = max(0, (w_lbl - self.CONTEXT_DISP_W) // 2)
        off_y = max(0, (h_lbl - self.CONTEXT_DISP_H) // 2)
        ix = event_x - off_x
        iy = event_y - off_y

        # Check all bounding boxes containing (ix, iy); prefer smallest area box if overlapping
        matches: List[Tuple[int, int]] = []
        for x1, y1, x2, y2, r_idx, _tid in self.clickable_boxes:
            if x1 - 6 <= ix <= x2 + 6 and y1 - 6 <= iy <= y2 + 6:
                area = max(1, (x2 - x1) * (y2 - y1))
                matches.append((area, r_idx))
        if matches:
            matches.sort(key=lambda item: item[0])
            return matches[0][1]

        # Fallback: nearest center within 45px
        best_dist = 45.0
        best_r_idx = None
        for x1, y1, x2, y2, r_idx, _tid in self.clickable_boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = float(np.hypot(ix - cx, iy - cy))
            if dist < best_dist:
                best_dist = dist
                best_r_idx = r_idx
        return best_r_idx

    def _on_classroom_left_click(self, event) -> None:
        """Left-click on any student in the classroom photo selects that student immediately."""
        clicked_idx = self._find_clicked_student_row(event.x, event.y)
        if clicked_idx is not None and clicked_idx != self.current_idx:
            self.current_idx = clicked_idx
            self.refresh_current_window()

    def _on_classroom_right_click(self, event) -> None:
        """Right-click on any student cycles their label (HIGH -> MOD -> LOW -> OCCLUDED) on the spot!"""
        clicked_idx = self._find_clicked_student_row(event.x, event.y)
        if clicked_idx is None:
            return
        self.current_idx = clicked_idx
        r = self.rows[clicked_idx]
        cur_st = r.get("annotation_state", "VALID")
        cur_lbl = r.get("participation_label", "")

        if cur_st != "VALID":
            next_st, next_lbl = "VALID", HIGH
        elif cur_lbl == HIGH:
            next_st, next_lbl = "VALID", MODERATE
        elif cur_lbl == MODERATE:
            next_st, next_lbl = "VALID", LOW
        elif cur_lbl == LOW:
            next_st, next_lbl = "OCCLUDED", ""
        else:
            next_st, next_lbl = "VALID", HIGH

        self.select_or_save_label(next_st, next_lbl, force_no_advance=True)

    # -----------------------------------------------------------------
    # RENDERING CURRENT SCENE & MULTI-STUDENT CLASSROOM OVERLAY
    # -----------------------------------------------------------------
    def _update_label_button_styles(self, state: str, label: str) -> None:
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

        self.current_bucket_ms = get_bucket_ms(self.rows[self.current_idx])
        self._refresh_bucket_combobox_labels()
        self._refresh_scene_student_combobox()

        scene_rows = self.bucket_to_rows.get(self.current_bucket_ms, [])
        scene_done = sum(1 for i in scene_rows if is_row_reviewed(self.rows[i]))
        total_done = sum(1 for x in self.rows if is_row_reviewed(x))
        total_pct = total_done / max(1, len(self.rows)) * 100.0

        self.progress_summary_lbl.config(
            text=f"📊 Toàn video: {total_done}/{len(self.rows)} ({total_pct:.1f}%)  |  Khung hình này: {scene_done}/{len(scene_rows)} SV đã duyệt"
        )

        r = self.rows[self.current_idx]
        t_start = int(_safe_float(r.get("t_start_ms", 0), 0))
        t_end = int(_safe_float(r.get("t_end_ms", 4000), 4000))
        track_raw = str(r.get("track_id", "1")).strip()
        track_key = _normalize_track_key(track_raw) or track_raw
        state = r.get("annotation_state", "VALID").strip()
        label = r.get("participation_label", "").strip()
        source = r.get("label_source", "").strip()
        phase = r.get("class_phase", "lecture").strip()
        zone_raw = str(r.get("zone_id", "")).strip()
        zone = zone_raw if zone_raw and zone_raw.lower() not in ("nan", "none") else "M-C"

        disp_lbl = to_display_label(label) if label in CANONICAL_TRAINING_LABELS else state
        reviewed = is_row_reviewed(r)
        src_str = "✓ Đã duyệt" if reviewed else "Máy gợi ý"
        stu_pos_str = f"SV {scene_rows.index(self.current_idx)+1}/{len(scene_rows)}" if self.current_idx in scene_rows else ""
        self.current_status_badge.config(
            text=f"👤 {stu_pos_str}: Track #{track_key} ({zone}) ➔ {disp_lbl} [{src_str}]",
            fg="#166534" if reviewed else "#0F172A",
        )

        self.window_title_lbl.config(
            text=f"🏫 Khung hình {self.current_bucket_ms/1000:.0f}s ➔ {(self.current_bucket_ms+4000)/1000:.0f}s ({len(scene_rows)} sinh viên)"
        )
        self.filmstrip_title_lbl.config(
            text=f"🎞️ 5 khung hình diễn biến 4 giây của Sinh viên đang chọn: Track #{track_key} ({t_start/1000:.1f}s ➔ {t_end/1000:.1f}s):"
        )

        # Sync Form widgets
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

        vid_p = self._get_path(self.video_path_var)
        self._render_visuals(vid_p, t_start, t_end, track_key, phase, label, scene_rows)

    def _on_state_combo_change(self, _event=None) -> None:
        st_val = self.state_combo.get()
        self.edit_state_var.set(st_val)
        if st_val == "VALID" and not self.edit_label_var.get():
            self.edit_label_var.set(MODERATE)
        elif st_val != "VALID":
            self.edit_label_var.set("")
        self._update_label_button_styles(st_val, self.edit_label_var.get())

    def _filter_poses_for_track(self, track_key: str):
        if self.poses_df is None or self.poses_df.empty:
            return None
        if "_track_key" in self.poses_df.columns:
            return self.poses_df[self.poses_df["_track_key"] == str(track_key)]
        return self.poses_df[self.poses_df["track_id"].map(_normalize_track_key) == str(track_key)]

    @staticmethod
    def _read_frame_at_ms(cap: cv2.VideoCapture, ts_ms: int, total_frames: int, fps: float) -> Optional[np.ndarray]:
        """Decode a video frame reliably at ts_ms, falling back to frame-index seek when needed."""
        max_ms = int((max(1, total_frames - 1) / max(1.0, fps)) * 1000.0) if total_frames > 0 else ts_ms
        eff_ms = max(0, min(int(ts_ms), max_ms))

        cap.set(cv2.CAP_PROP_POS_MSEC, float(eff_ms))
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return frame

        # Fallback: seek by exact frame index (handles MP4 files where CAP_PROP_POS_MSEC fails)
        if fps > 0:
            target_f = int(round((eff_ms / 1000.0) * fps))
            if total_frames > 0:
                target_f = max(0, min(target_f, total_frames - 1))
            for back in (0, 2, 5, 15):
                f_idx = max(0, target_f - back)
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(f_idx))
                ret2, frame2 = cap.read()
                if ret2 and frame2 is not None and frame2.size > 0:
                    return frame2
        return None

    def _render_visuals(
        self,
        video_path: Optional[Path],
        t_start_ms: int,
        t_end_ms: int,
        track_key: str,
        class_phase: str,
        canonical_label: str,
        scene_row_indices: List[int],
    ) -> None:
        draw_skel = self.opt_draw_skel.get()
        blur_face = self.opt_blur_face.get()

        # 1. Telemetry & scene summary
        scene_high = sum(1 for i in scene_row_indices if self.rows[i].get("participation_label") == HIGH)
        scene_mod = sum(1 for i in scene_row_indices if self.rows[i].get("participation_label") == MODERATE)
        scene_low = sum(1 for i in scene_row_indices if self.rows[i].get("participation_label") == LOW)
        scene_occ = sum(1 for i in scene_row_indices if self.rows[i].get("annotation_state") != "VALID")

        track_all_poses = self._filter_poses_for_track(track_key)
        if track_all_poses is not None and not track_all_poses.empty:
            w_df = track_all_poses[
                (track_all_poses["timestamp_ms"] >= int(t_start_ms))
                & (track_all_poses["timestamp_ms"] <= int(t_end_ms))
            ]
            if not w_df.empty:
                mean_vis = _safe_float(np.nanmean(w_df["visible_ratio"].to_numpy(dtype=float)), 0.0)
                nose_y = w_df["kpt_00_y"].to_numpy(dtype=float)
                nose_c = w_df["kpt_00_conf"].to_numpy(dtype=float)
                ears_y = (w_df["kpt_03_y"].to_numpy(dtype=float) + w_df["kpt_04_y"].to_numpy(dtype=float)) / 2.0
                l_wr_x = w_df["kpt_09_x"].to_numpy(dtype=float)
                r_wr_x = w_df["kpt_10_x"].to_numpy(dtype=float)
                valid_hd = np.isfinite(nose_y) & np.isfinite(ears_y) & (nose_c > 0.3)
                head_dn = float(np.mean(valid_hd & (nose_y > ears_y + 12)))
                lw_std = float(np.nanstd(l_wr_x)) if np.any(np.isfinite(l_wr_x)) else 0.0
                rw_std = float(np.nanstd(r_wr_x)) if np.any(np.isfinite(r_wr_x)) else 0.0
                wr_mot = lw_std + rw_std
                r_state, r_lbl, _r_behs, r_conf = apply_labeling_rules(w_df, class_phase=class_phase)
                r_disp = to_display_label(r_lbl) if r_lbl in CANONICAL_TRAINING_LABELS else r_state

                self.ai_suggest_lbl.config(
                    text=(
                        f"🏫 Tổng quan cả lớp ở mốc này ({len(scene_row_indices)} SV):  "
                        f"🟢 CAO: {scene_high}   •   🟡 VỪA: {scene_mod}   •   🔴 THẤP: {scene_low}   •   ⚪ BỊ CHE: {scene_occ}\n"
                        f"🤖 Chi tiết Track #{track_key} đang chọn:  AI gợi ý [{r_disp}] ({r_conf*100:.0f}%)  •  "
                        f"Nhìn rõ {mean_vis*100:.0f}% khớp  •  Cúi đầu {head_dn*100:.0f}%  •  Cổ tay dịch chuyển {wr_mot:.1f} px"
                    )
                )

        if video_path is None or not video_path.is_file():
            return

        color_bgr = BOX_COLORS_BGR.get(canonical_label, (46, 204, 113))
        track_poses = track_all_poses.sort_values("timestamp_ms") if (track_all_poses is not None and not track_all_poses.empty) else None

        # 50x SPEEDUP: Cache the 5 raw BGR frames for the current 4-second bucket!
        cache_key = (str(video_path), int(self.current_bucket_ms))
        if self._frame_cache_key != cache_key or not self._frame_cache_frames:
            timestamps = np.linspace(
                self.current_bucket_ms, self.current_bucket_ms + 3900, 5, dtype=int
            )
            cached: List[Tuple[int, Optional[np.ndarray]]] = []
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                try:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    last_valid: Optional[np.ndarray] = None
                    for ts in timestamps:
                        frame = self._read_frame_at_ms(cap, int(ts), total_frames, fps)
                        if frame is not None:
                            last_valid = frame
                        elif last_valid is not None:
                            frame = last_valid
                        cached.append((int(ts), frame))
                finally:
                    cap.release()
            self._frame_cache_key = cache_key
            self._frame_cache_frames = [item for item in cached if item[1] is not None]

        # Build all new filmstrip images cleanly BEFORE swapping references so a partial failure never blanks the strip
        new_film_imgs: List[Optional[ImageTk.PhotoImage]] = []
        mid_full_frame = None
        for idx_t in range(5):
            if idx_t < len(self._frame_cache_frames):
                ts, frame = self._frame_cache_frames[idx_t]
            elif self._frame_cache_frames:
                ts, frame = self._frame_cache_frames[-1]
            else:
                self.film_labels[idx_t].config(image="", text="Không đọc\nđược frame", fg="#94A3B8", width=18, height=9)
                self.film_captions[idx_t].config(text="--")
                new_film_imgs.append(None)
                continue

            if idx_t == 2 or mid_full_frame is None:
                mid_full_frame = frame

            matched_row = None
            if track_poses is not None and not track_poses.empty:
                diffs = (track_poses["timestamp_ms"] - int(ts)).abs()
                min_i = diffs.idxmin()
                if diffs.loc[min_i] <= 1500:
                    matched_row = track_poses.loc[min_i]
                else:
                    # Fall back to nearest pose of this student inside the current 4s bucket
                    in_bucket = track_poses[
                        (track_poses["timestamp_ms"] >= self.current_bucket_ms)
                        & (track_poses["timestamp_ms"] <= self.current_bucket_ms + 4000)
                    ]
                    if not in_bucket.empty:
                        b_diffs = (in_bucket["timestamp_ms"] - int(ts)).abs()
                        matched_row = in_bucket.loc[b_diffs.idxmin()]

            try:
                crop_bgr = render_crop_with_pose(
                    frame,
                    matched_row,
                    draw_skel=draw_skel,
                    blur_face=blur_face,
                    color_bgr=color_bgr,
                    out_size=(145, 150),
                )
            except Exception:
                crop_bgr = cv2.resize(frame, (145, 150), interpolation=cv2.INTER_AREA)

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tk_img = ImageTk.PhotoImage(Image.fromarray(crop_rgb))
            new_film_imgs.append(tk_img)
            self.film_labels[idx_t].config(image=tk_img, text="", width=0, height=0)
            self.film_captions[idx_t].config(text=f"Giây {ts/1000:.1f}s")

        self.filmstrip_imgs = [img for img in new_film_imgs if img is not None]

        if mid_full_frame is not None:
            ctx_bgr = self._draw_multi_student_classroom(
                mid_full_frame,
                self.current_bucket_ms + 2000,
                track_key,
                scene_row_indices,
                draw_skel,
                blur_face,
            )
            ctx_resized = cv2.resize(
                ctx_bgr, (self.CONTEXT_DISP_W, self.CONTEXT_DISP_H), interpolation=cv2.INTER_AREA
            )
            ctx_rgb = cv2.cvtColor(ctx_resized, cv2.COLOR_BGR2RGB)
            self.context_img = ImageTk.PhotoImage(Image.fromarray(ctx_rgb))
            self.context_canvas.config(image=self.context_img)

    def _draw_multi_student_classroom(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        active_track_key: str,
        scene_row_indices: List[int],
        draw_skel: bool,
        blur_face: bool,
    ) -> np.ndarray:
        """Draw ALL students in the current 4s scene with color-coded boxes and build clickable map."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        scale_x = self.CONTEXT_DISP_W / float(w)
        scale_y = self.CONTEXT_DISP_H / float(h)

        # Subtle 3x3 classroom zone grid
        for frac in (0.33, 0.66):
            cv2.line(canvas, (int(w * frac), 0), (int(w * frac), h), (185, 185, 185), 1)
            cv2.line(canvas, (0, int(h * frac)), (w, int(h * frac)), (185, 185, 185), 1)

        # Map normalized track_key -> row_idx in the current 4s scene
        scene_track_to_row: Dict[str, int] = {}
        for r_idx in scene_row_indices:
            t_key = _normalize_track_key(self.rows[r_idx].get("track_id", ""))
            if t_key:
                scene_track_to_row[t_key] = r_idx

        self.clickable_boxes.clear()

        if self.poses_df is not None and not self.poses_df.empty:
            # Find representative pose within the 4s window for each student in the scene
            w_poses = self.poses_df[
                (self.poses_df["timestamp_ms"] >= self.current_bucket_ms)
                & (self.poses_df["timestamp_ms"] <= self.current_bucket_ms + 4000)
            ]

            for t_key, r_idx in scene_track_to_row.items():
                if "_track_key" in w_poses.columns:
                    t_poses = w_poses[w_poses["_track_key"] == t_key]
                else:
                    t_poses = w_poses[w_poses["track_id"].map(_normalize_track_key) == t_key]
                if t_poses.empty:
                    continue
                # Pick pose row closest to the middle of the window
                diffs = (t_poses["timestamp_ms"] - int(timestamp_ms)).abs()
                prow = t_poses.loc[diffs.idxmin()]

                fx1 = _safe_float(prow.get("bbox_x1", np.nan), np.nan)
                fy1 = _safe_float(prow.get("bbox_y1", np.nan), np.nan)
                fx2 = _safe_float(prow.get("bbox_x2", np.nan), np.nan)
                fy2 = _safe_float(prow.get("bbox_y2", np.nan), np.nan)
                if not (np.isfinite(fx1) and np.isfinite(fy1) and np.isfinite(fx2) and np.isfinite(fy2)):
                    continue
                x1, y1, x2, y2 = int(fx1), int(fy1), int(fx2), int(fy2)
                if x2 <= x1 or y2 <= y1:
                    continue

                kpts = row_to_keypoints_array(prow)

                if blur_face:
                    blur_head_roi(canvas, (x1, y1, x2, y2), kpts)

                row_data = self.rows[r_idx]
                st_val = row_data.get("annotation_state", "VALID").strip()
                lbl_val = row_data.get("participation_label", "").strip()
                src_val = row_data.get("label_source", "").strip()
                is_human = src_val in ("HUMAN", "ADJUDICATED")

                color = BOX_COLORS_BGR.get(lbl_val if st_val == "VALID" else st_val, (140, 140, 140))
                short_tag = (
                    "HIGH" if lbl_val == HIGH
                    else ("MOD" if lbl_val == MODERATE else ("LOW" if lbl_val == LOW else "OCC"))
                )
                badge_txt = f"{'v' if is_human else ''}#{t_key}:{short_tag}"

                is_selected = t_key == str(active_track_key)
                if is_selected:
                    if draw_skel:
                        draw_skeleton(canvas, kpts, color=color, conf_threshold=0.25)
                    # Outer white glow + thick colored box for the currently selected student
                    cv2.rectangle(canvas, (max(0, x1 - 3), max(0, y1 - 3)), (min(w, x2 + 3), min(h, y2 + 3)), (255, 255, 255), 6)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
                    sel_txt = f">> #{t_key} {short_tag} <<"
                    (tw, th), _ = cv2.getTextSize(sel_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                    cv2.rectangle(canvas, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, y1), color, -1)
                    cv2.putText(
                        canvas, sel_txt, (x1 + 6, max(th, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2
                    )
                else:
                    # Other students in the same scene: clear colored box + badge
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                    cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
                    cv2.putText(
                        canvas, badge_txt, (x1 + 4, max(th, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2
                    )

                # Store scaled display coordinates for mouse click hit-testing
                tid_num = int(t_key) if t_key.isdigit() else r_idx
                self.clickable_boxes.append((
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                    r_idx,
                    tid_num,
                ))

        return canvas

    def play_4s_clip_live(self) -> None:
        if self.is_playing_clip or not self.rows:
            return
        vid_p = self._get_path(self.video_path_var)
        if vid_p is None or not vid_p.is_file():
            messagebox.showwarning("Chưa chọn video", "Vui lòng chọn file Video ở BƯỚC 1 để phát clip 4 giây.")
            return

        t_start = self.current_bucket_ms
        t_end = self.current_bucket_ms + 4000
        track_key = _normalize_track_key(self.rows[self.current_idx].get("track_id", "1")) or "1"
        scene_rows = list(self.bucket_to_rows.get(self.current_bucket_ms, []))
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
                        ctx_bgr = self._draw_multi_student_classroom(
                            frame, int(cur_ms), track_key, scene_rows, draw_skel, blur_face
                        )
                        ctx_res = cv2.resize(ctx_bgr, (self.CONTEXT_DISP_W, self.CONTEXT_DISP_H), interpolation=cv2.INTER_AREA)
                        cv2.putText(
                            ctx_res,
                            f"PHAT VIDEO CA LOP: {cur_ms/1000:.1f}s / {t_end/1000:.1f}s ({len(scene_rows)} SV)",
                            (14, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
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
    # MULTI-STUDENT & INDIVIDUAL LABEL SAVING
    # -----------------------------------------------------------------
    def select_or_save_label(self, state: str, canonical_label: str, force_no_advance: bool = False) -> None:
        if not self.rows:
            return
        self.edit_state_var.set(state)
        self.state_combo.set(state)
        self.edit_label_var.set(canonical_label)
        self._update_label_button_styles(state, canonical_label)

        if state == "VALID":
            for v in self.beh_vars.values():
                v.set(False)
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

        adv = False if force_no_advance else self.opt_auto_next_student.get()
        self.save_current_window(advance_student=adv)

    def save_current_window(self, advance_student: bool = False) -> None:
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
        tid_saved = r.get("track_id", "")
        disp_saved = to_display_label(lbl_val) if lbl_val in CANONICAL_TRAINING_LABELS else st_val

        self._push_undo([self.current_idx], f"Lưu Track #{tid_saved} ➔ {disp_saved}")

        r["annotation_state"] = st_val
        r["participation_label"] = lbl_val
        r["atomic_behavior"] = ";".join(selected_behs)
        r["label_source"] = src_val
        r["label_rule_version"] = rule_ver
        r["occlusion_level"] = self.edit_occ_cb.get()
        r["class_phase"] = self.edit_phase_cb.get()
        r["zone_id"] = self.edit_zone_cb.get()
        r["annotator_id"] = self.annotator_var.get().strip() or "HUMAN_ANNOTATOR"
        r["notes"] = self.edit_notes_var.get().strip()
        r["annotation_guide_version"] = "v0.2"

        self._write_rows_to_csv()

        reached_dataset_end = False
        toast_msg = f"✓ Đã lưu Track #{tid_saved} ➔ {disp_saved}"
        if advance_student:
            r_indices = self.bucket_to_rows.get(self.current_bucket_ms, [])
            if self.current_idx in r_indices:
                pos = r_indices.index(self.current_idx)
                if pos + 1 < len(r_indices):
                    self.current_idx = r_indices[pos + 1]
                    toast_msg = f"✓ Đã lưu Track #{tid_saved} ({disp_saved}) ➔ Sang SV {pos+2}/{len(r_indices)}"
                elif self.current_bucket_ms in self.time_buckets:
                    b_pos = self.time_buckets.index(self.current_bucket_ms)
                    if b_pos + 1 < len(self.time_buckets):
                        # SEAMLESS ROLLOVER to the next 4s time bucket!
                        next_b = self.time_buckets[b_pos + 1]
                        self.current_bucket_ms = next_b
                        self.current_idx = self.bucket_to_rows[next_b][0]
                        toast_msg = f"🎉 Xong Mốc #{b_pos+1} ➔ Tự động sang Mốc #{b_pos+2}/{len(self.time_buckets)}!"
                    else:
                        reached_dataset_end = True
                        toast_msg = "🏁 Đã lưu sinh viên cuối cùng của toàn bộ video!"

        self.refresh_current_window()
        self.set_toast(toast_msg, "#15803D")

        if reached_dataset_end:
            if messagebox.askyesno(
                "🎉 Hoàn Thành Toàn Bộ Video!",
                "Bạn đã lưu đến sinh viên cuối cùng của mốc thời gian cuối cùng!\n\n"
                "Bạn có muốn chuyển sang BƯỚC 3 để kiểm định hợp lệ 100% và xuất file CSV không?",
            ):
                self.run_qa_validation_report()
                self.notebook.select(self.tab_step3_export)

    def approve_entire_scene(self, advance_bucket: bool = True) -> None:
        """Confirm ALL students in the current 4s classroom time bucket as HUMAN and jump to next time bucket!"""
        scene_rows = self.bucket_to_rows.get(self.current_bucket_ms, [])
        if not scene_rows:
            return
        ann_id = self.annotator_var.get().strip() or "HUMAN_ANNOTATOR"
        cur_b_num = self.time_buckets.index(self.current_bucket_ms) + 1 if self.current_bucket_ms in self.time_buckets else 1

        self._push_undo(scene_rows, f"Duyệt cả lớp Mốc #{cur_b_num} ({len(scene_rows)} SV)")

        for r_idx in scene_rows:
            r = self.rows[r_idx]
            st_val = r.get("annotation_state", "VALID").strip()
            lbl_val = r.get("participation_label", "").strip()
            if st_val == "VALID":
                if lbl_val not in CANONICAL_TRAINING_LABELS:
                    r["participation_label"] = MODERATE
                r["label_source"] = "HUMAN"
                r["label_rule_version"] = ""
            else:
                r["participation_label"] = ""
                r["label_source"] = ""
                r["label_rule_version"] = ""
            r["annotator_id"] = ann_id
            r["annotation_guide_version"] = "v0.2"

        self._write_rows_to_csv()

        reached_end = False
        if advance_bucket and self.current_bucket_ms in self.time_buckets:
            pos = self.time_buckets.index(self.current_bucket_ms)
            if pos + 1 < len(self.time_buckets):
                self.current_bucket_ms = self.time_buckets[pos + 1]
                self.current_idx = self.bucket_to_rows[self.current_bucket_ms][0]
            else:
                reached_end = True

        self.refresh_current_window()
        self.set_toast(f"✅ Đã duyệt cả lớp Mốc #{cur_b_num} ({len(scene_rows)} SV)!", "#15803D")

        if reached_end:
            if messagebox.askyesno(
                "🎉 Hoàn Thành Toàn Bộ Video!",
                "Bạn đã duyệt xong cả mốc thời gian cuối cùng!\n\n"
                "Chuyển sang BƯỚC 3 để xem báo cáo kiểm định và xuất file CSV?",
            ):
                self.run_qa_validation_report()
                self.notebook.select(self.tab_step3_export)

    def set_entire_scene_label(self, canonical_label: str) -> None:
        """Assign canonical_label to all VALID students in the current 4s classroom scene."""
        scene_rows = self.bucket_to_rows.get(self.current_bucket_ms, [])
        if not scene_rows:
            return
        self._push_undo(scene_rows, f"Gán cả lớp ➔ {to_display_label(canonical_label)}")
        ann_id = self.annotator_var.get().strip() or "HUMAN_ANNOTATOR"
        default_beh = (
            "BOARD_OR_INSTRUCTOR_FACING" if canonical_label == HIGH
            else ("READING_WRITING" if canonical_label == MODERATE else "HEAD_DOWN")
        )
        for r_idx in scene_rows:
            r = self.rows[r_idx]
            if r.get("annotation_state") == "VALID":
                r["participation_label"] = canonical_label
                r["atomic_behavior"] = default_beh
                r["label_source"] = "HUMAN"
                r["label_rule_version"] = ""
                r["annotator_id"] = ann_id
        self._write_rows_to_csv()
        self.refresh_current_window()
        self.set_toast(f"🏷️ Đã gán cả lớp ở mốc này thành {to_display_label(canonical_label)}!", "#1E40AF")

    def approve_selected_track_all_time(self) -> None:
        """Approve all windows across the entire 60s video for the currently selected student Track ID."""
        if not self.rows:
            return
        tid = str(self.rows[self.current_idx].get("track_id", ""))
        ann_id = self.annotator_var.get().strip() or "HUMAN_ANNOTATOR"
        cnt = 0
        for r in self.rows:
            if str(r.get("track_id", "")) == tid and r.get("annotation_state") == "VALID":
                if r.get("participation_label") in CANONICAL_TRAINING_LABELS:
                    r["label_source"] = "HUMAN"
                    r["label_rule_version"] = ""
                    r["annotator_id"] = ann_id
                    cnt += 1
        self._write_rows_to_csv()
        self.refresh_current_window()
        messagebox.showinfo("Đã duyệt theo Sinh viên", f"Đã xác nhận HUMAN cho toàn bộ {cnt} cửa sổ của Sinh viên Track #{tid}!")

    def _write_rows_to_csv(self) -> None:
        csv_p = self._get_path(self.csv_path_var)
        if csv_p is None:
            vid_p = self._get_path(self.video_path_var)
            stem = vid_p.stem if vid_p is not None else "session"
            csv_p = ROOT / "data/annotations" / f"{stem}_labels.csv"
            self.csv_path_var.set(csv_p.as_posix())
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
        vid_p = self._get_path(self.video_path_var)
        if vid_p is None or not vid_p.is_file():
            self.trim_start_img = None
            self.trim_end_img = None
            self.trim_start_canvas.config(image="", text="Chưa chọn video — bấm [📂 Chọn Video...] ở trên")
            self.trim_end_canvas.config(image="", text="Khung hình Điểm Kết Thúc sẽ hiện sau khi chọn video")
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
                f1_res = cv2.resize(f1, (540, 260), interpolation=cv2.INTER_AREA)
                cv2.putText(f1_res, f"DIEM BAT DAU: {s_ms/1000:.1f}s", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2)
                self.trim_start_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f1_res, cv2.COLOR_BGR2RGB)))
                self.trim_start_canvas.config(image=self.trim_start_img, text="")

            cap.set(cv2.CAP_PROP_POS_MSEC, float(e_ms))
            ret2, f2 = cap.read()
            if ret2 and f2 is not None:
                f2_res = cv2.resize(f2, (540, 260), interpolation=cv2.INTER_AREA)
                cv2.putText(f2_res, f"DIEM KET THUC: {e_ms/1000:.1f}s", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)
                self.trim_end_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f2_res, cv2.COLOR_BGR2RGB)))
                self.trim_end_canvas.config(image=self.trim_end_img, text="")
        finally:
            cap.release()

    @staticmethod
    def _find_python_with_yolo() -> Optional[str]:
        candidates: List[str] = []
        if not getattr(sys, "frozen", False):
            candidates.append(sys.executable)
        local_py = Path.home() / "AppData/Local/Programs/Python"
        if local_py.exists():
            for p_exe in sorted(local_py.glob("Python3*/python.exe"), reverse=True):
                candidates.append(str(p_exe))
        which_py = shutil.which("python")
        if which_py:
            candidates.append(which_py)

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        seen = set()
        for cand in candidates:
            if not cand or cand in seen or not Path(cand).is_file():
                continue
            seen.add(cand)
            try:
                res = subprocess.run(
                    [cand, "-c", "import ultralytics"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=8,
                    creationflags=creation_flags,
                )
                if res.returncode == 0:
                    return cand
            except Exception:
                continue
        return None

    def _run_pose_extraction_then_label(self, vid_p: Path, s_sec: float, e_sec: float) -> None:
        py_exe = self._find_python_with_yolo()
        if not py_exe:
            messagebox.showwarning(
                "Không thể chạy AI trích xuất tự động",
                "Máy hiện chưa sẵn sàng môi trường Python + Ultralytics YOLOv8.\n"
                "Hãy chọn thủ công file Pose (.parquet / .csv) đã trích xuất sẵn.",
            )
            self._browse_pose_and_refresh()
            if self.poses_df is not None and not self.poses_df.empty:
                self.run_auto_label_pipeline()
            return

        out_dir = ROOT / "data/poses"
        out_dir.mkdir(parents=True, exist_ok=True)
        expected_out = out_dir / f"{vid_p.stem}_poses.parquet"

        self.trim_prog_lbl.config(
            text=f"⏳ Đang chạy AI (YOLOv8-Pose) nhận diện sinh viên & 17 khớp xương cho {vid_p.name} ({s_sec:.0f}s ➔ {e_sec:.0f}s)...",
            fg="#1D4ED8",
        )
        self.root.update_idletasks()

        inline_code = (
            "import sys, cv2, pandas as pd; "
            "from pathlib import Path; "
            "from ultralytics import YOLO; "
            "from src.pose.extractor import extract_frame_poses, assign_zone_id, POSE_COLUMNS, load_pose_config; "
            "vid, out_p, s_sec, e_sec = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]); "
            "cfg = load_pose_config('configs/baseline.yaml'); "
            "model = YOLO(cfg['model']); "
            "cap = cv2.VideoCapture(str(vid)); "
            "fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0); "
            "w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); "
            "s_f, e_f = int(s_sec * fps), int(e_sec * fps); "
            "cap.set(cv2.CAP_PROP_POS_FRAMES, s_f); "
            "step_ms = 100.0; next_ms = s_sec * 1000.0; rows = []; s_idx = 0; f_idx = s_f; "
            "exec_loop = True\n"
            "while cap.isOpened() and f_idx <= e_f:\n"
            "    ret, frame = cap.read()\n"
            "    if not ret: break\n"
            "    cur_ms = f_idx * (1000.0 / fps)\n"
            "    if cur_ms + 1e-4 >= next_ms:\n"
            "        f_rows = extract_frame_poses(frame, model, vid.parent.name or 'session_001', vid.stem or 'cam_A', s_idx, int(round(cur_ms)), conf_threshold=cfg['confidence_threshold'], track=True, tracker=cfg['tracker'])\n"
            "        for r in f_rows:\n"
            "            if r.get('track_id') is None: continue\n"
            "            r['zone_id'] = assign_zone_id((r['bbox_x1'], r['bbox_y1'], r['bbox_x2'], r['bbox_y2']), w, h)\n"
            "            rows.append(r)\n"
            "        s_idx += 1; next_ms += step_ms\n"
            "    f_idx += 1\n"
            "cap.release(); "
            "df = pd.DataFrame(rows, columns=POSE_COLUMNS); "
            "out_p.parent.mkdir(parents=True, exist_ok=True); "
            "df.to_parquet(out_p, index=False)"
        )

        def _worker():
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            cmd = [py_exe, "-c", inline_code, str(vid_p), str(expected_out), str(s_sec), str(e_sec)]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    creationflags=creation_flags,
                )
                if proc.returncode == 0 and expected_out.is_file():
                    def _on_done():
                        self.pose_path_var.set(expected_out.as_posix())
                        self.reload_all_resources()
                        self.run_auto_label_pipeline()
                    self.root.after(0, _on_done)
                else:
                    err_tail = (proc.stderr or proc.stdout or "Lỗi không xác định")[-400:]
                    self.root.after(
                        0,
                        lambda: [
                            self.trim_prog_lbl.config(text="❌ Trích xuất Pose thất bại.", fg="#B91C1C"),
                            messagebox.showerror("Lỗi trích xuất Pose", f"Không thể trích xuất khung xương:\n\n{err_tail}"),
                        ],
                    )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: [
                        self.trim_prog_lbl.config(text="❌ Trích xuất Pose thất bại.", fg="#B91C1C"),
                        messagebox.showerror("Lỗi trích xuất Pose", str(exc)),
                    ],
                )

        threading.Thread(target=_worker, daemon=True).start()

    def run_auto_label_pipeline(self) -> None:
        vid_p = self._get_path(self.video_path_var)
        if vid_p is None or not vid_p.is_file():
            messagebox.showerror("Chưa chọn Video", "Vui lòng bấm [📂 Chọn Video...] ở BƯỚC 1 để chọn file video trước!")
            return

        s_sec = float(self.trim_start_var.get())
        e_sec = float(self.trim_end_var.get())
        if e_sec <= s_sec:
            messagebox.showerror("Lỗi khoảng thời gian", "Giây kết thúc phải lớn hơn giây bắt đầu!")
            return

        # Auto-match Pose file if not yet attached
        if self._get_path(self.pose_path_var) is None:
            matched = self._find_matching_pose_file(vid_p)
            if matched is not None:
                self.pose_path_var.set(matched.as_posix())

        self.reload_all_resources()
        if self.poses_df is None or self.poses_df.empty:
            choice = messagebox.askyesnocancel(
                "Video mới chưa có dữ liệu khung xương (Pose)",
                f"Video '{vid_p.name}' chưa có file Pose (.parquet) chứa tọa độ từng sinh viên & 17 khớp xương.\n\n"
                "• Chọn [Yes] : Tự động chạy AI (YOLOv8-Pose) để nhận diện sinh viên cho đoạn video này.\n"
                "• Chọn [No]  : Chọn thủ công một file Pose (.parquet / .csv) có sẵn trên máy.\n"
                "• Chọn [Cancel]: Hủy.",
            )
            if choice is True:
                self._run_pose_extraction_then_label(vid_p, s_sec, e_sec)
                return
            if choice is False:
                self._browse_pose_and_refresh()
                if self.poses_df is None or self.poses_df.empty:
                    return
            else:
                return

        # Automatically assign a working CSV path based on the selected video
        auto_csv = ROOT / "data/annotations" / f"{vid_p.stem}_labels.csv"
        self.csv_path_var.set(auto_csv.as_posix())

        # Optionally export trimmed mp4 if checkbox is checked
        if getattr(self, "opt_export_trimmed_mp4", None) and self.opt_export_trimmed_mp4.get() and vid_p.is_file():
            trimmed_out = vid_p.parent / f"{vid_p.stem}_trimmed_{int(s_sec)}s_{int(e_sec)}s.mp4"
            self.trim_prog_lbl.config(text=f"⏳ Đang cắt video ra file {trimmed_out.name}...")
            self.root.update_idletasks()
            cap = cv2.VideoCapture(str(vid_p))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            s_f, e_f = int(s_sec * fps), int(e_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, s_f)
            writer = cv2.VideoWriter(str(trimmed_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            cur = s_f
            while cur < e_f:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                cur += 1
            cap.release()
            writer.release()

        s_ms = int(s_sec * 1000)
        e_ms = int(e_sec * 1000)
        win_ms = int(self.win_sec_var.get() * 1000)
        str_ms = int(self.stride_sec_var.get() * 1000)

        sub_df = self.poses_df[
            (self.poses_df["timestamp_ms"] >= s_ms) & (self.poses_df["timestamp_ms"] <= e_ms)
        ]
        if sub_df.empty:
            sub_df = self.poses_df

        meta_p = self._get_path(self.meta_path_var)
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if (meta_p is not None and meta_p.is_file()) else {}
        phases = meta.get("class_phase", [])
        sess_id = meta.get("session_id", vid_p.parent.name or "session_001")
        cam_id = meta.get("camera_id", vid_p.stem or "cam_A")

        def get_phase(ts: int) -> str:
            for p in phases:
                if p.get("start_ms", 0) <= ts < p.get("end_ms", 0):
                    return str(p.get("name", "lecture"))
            return "lecture"

        vid_w, vid_h = 1920, 1080
        cap_dim = cv2.VideoCapture(str(vid_p))
        if cap_dim.isOpened():
            try:
                w_val = int(cap_dim.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
                h_val = int(cap_dim.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
                if w_val > 0 and h_val > 0:
                    vid_w, vid_h = w_val, h_val
            finally:
                cap_dim.release()

        new_rows: List[Dict[str, str]] = []
        group_col = "_track_key" if "_track_key" in sub_df.columns else "track_id"
        for track_val, grp in sub_df.groupby(group_col):
            t_key = _normalize_track_key(track_val)
            if not t_key:
                continue
            grp = grp.sort_values("timestamp_ms")
            min_t, max_t = int(grp["timestamp_ms"].min()), int(grp["timestamp_ms"].max())
            t_cur = min_t
            while t_cur + win_ms <= max_t:
                t_stop = t_cur + win_ms
                w_df = grp[(grp["timestamp_ms"] >= t_cur) & (grp["timestamp_ms"] <= t_stop)]
                if len(w_df) >= 3:
                    raw_z = str(w_df["zone_id"].iloc[-1]).strip() if "zone_id" in w_df.columns else ""
                    if not raw_z or raw_z.lower() in ("nan", "none", "null", "unknown"):
                        last_p = w_df.iloc[-1]
                        bx1 = _safe_float(last_p.get("bbox_x1", np.nan), np.nan)
                        by1 = _safe_float(last_p.get("bbox_y1", np.nan), np.nan)
                        bx2 = _safe_float(last_p.get("bbox_x2", np.nan), np.nan)
                        by2 = _safe_float(last_p.get("bbox_y2", np.nan), np.nan)
                        if np.all(np.isfinite([bx1, by1, bx2, by2])):
                            zone = assign_zone_id((bx1, by1, bx2, by2), vid_w, vid_h)
                        else:
                            zone = "M-C"
                    else:
                        zone = raw_z
                    ph = get_phase(t_cur)
                    m_vis = _safe_float(np.nanmean(w_df["visible_ratio"].to_numpy(dtype=float)), 0.0)
                    st_val, lbl_val, behs, conf = apply_labeling_rules(w_df, class_phase=ph)
                    is_v = st_val == "VALID"
                    occ = "none" if m_vis >= 0.75 else ("moderate" if m_vis >= 0.50 else "heavy")
                    new_rows.append({
                        "session_id": sess_id,
                        "camera_id": cam_id,
                        "track_id": t_key,
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
        self.notebook.select(self.tab_step2_review)

    def run_trim_video_file(self) -> None:
        vid_p = self._get_path(self.video_path_var)
        if vid_p is None or not vid_p.is_file():
            messagebox.showerror("Chưa chọn Video", "Vui lòng bấm [📂 Chọn Video...] ở mục (1) trước khi cắt!")
            return
        out_p = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=f"{vid_p.stem}_trimmed.mp4",
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
        vid_p = self._get_path(self.video_path_var)
        if vid_p is None or not vid_p.is_file() or not self.rows:
            messagebox.showerror("Lỗi", "Cần chọn Video ở mục (1) và có danh sách cửa sổ CSV!")
            return
        out_dir = ROOT / "data/clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        subset = self.rows[:10]
        cap = cv2.VideoCapture(str(vid_p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = 0
        for r in subset:
            ts_s = int(_safe_float(r.get("t_start_ms", 0), 0))
            ts_e = int(_safe_float(r.get("t_end_ms", 4000), 4000))
            tid_key = _normalize_track_key(r.get("track_id", "")) or "0"
            tid_tag = f"trk{int(tid_key):03d}" if tid_key.isdigit() else f"trk_{tid_key}"
            lbl = to_display_label(r["participation_label"]) if r["participation_label"] in CANONICAL_TRAINING_LABELS else r["annotation_state"]
            fname = out_dir / f"{r['session_id']}_{tid_tag}_{ts_s:06d}_{ts_e:06d}_{lbl}.mp4"
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
        if not self.rows:
            messagebox.showwarning("Chưa có dữ liệu", "Chưa có dữ liệu nhãn để xuất CSV! Hãy thực hiện BƯỚC 1 và BƯỚC 2 trước.")
            return
        csv_p = self._get_path(self.csv_path_var)
        vid_p = self._get_path(self.video_path_var)
        default_name = (
            f"{csv_p.stem}_final.csv" if csv_p is not None
            else (f"{vid_p.stem}_annotated_final.csv" if vid_p is not None else "annotated_final.csv")
        )
        out_p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
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
            "   BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG & SCHEMA DATASET (VERSION 2 - T11/T12)",
            "=========================================================================",
            f"File CSV hiện tại      : {self.csv_path_var.get()}",
            f"Tổng số cửa sổ 4s      : {stats['total']} (trên {len(self.time_buckets)} mốc khung hình cả lớp)",
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
    AnnotationStudioV2App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
