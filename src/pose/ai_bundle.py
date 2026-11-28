"""Hybrid AI-assisted pre-annotation bundle builder and suggestion I/O.

Supports the two zero-external-API workflows:
1. Local AI proposer: validates video/pose source alignment, extracts temporal
   features, runs the project's temporal model + Uncertainty Gate + weak-v0.2
   kinematics, and emits structured suggestions (state, label, behaviors,
   rationale, rejection_reason).
2. Web Vision LLM (ChatGPT Plus / Gemini Pro) workflow: renders a composite
   contact-sheet image per window (annotated full classroom frame + 5 cropped
   timestamped frames + skeleton + pose quality + verified phase + v0.2 guide),
   generates copy-paste prompts, and parses/validates returned JSON.

CRITICAL CONTRACT: Suggestions produced here are NEVER written directly as
``HUMAN`` labels. They are stored in a separate suggestions file for a human
annotator to review, accept, or modify inside ``app/annotation_app.py``.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from src.contracts.labels import (
    ANNOTATION_STATES,
    CANONICAL_BY_DISPLAY_LABEL,
    CANONICAL_TRAINING_LABELS,
    HIGH,
    INDEX_TO_LABEL,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
)
from src.pose.studio_renderer import (
    extract_frame_at_ms,
    render_classroom_scene,
    render_progression_strip,
)
from src.preprocessing.cleaner import clean_and_normalize_poses
from src.preprocessing.features import compute_track_features
from src.preprocessing.quality_gate import upper_body_visibility
from src.preprocessing.weak_labels import apply_weak_label_rules
from src.uncertainty.gate import UncertaintyThresholds, explain_uncertainty_gate

logger = logging.getLogger(__name__)

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

REQUIRED_POSE_COLUMNS = (
    "session_id",
    "camera_id",
    "frame_idx",
    "timestamp_ms",
    "track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
)

GUIDE_V02_SUMMARY = (
    "Hướng dẫn v0.2: Đánh giá hành vi quan sát được trong cả cửa sổ 4s theo pha thực tế "
    "(lecture / individual_work / discussion). HIGH = nhiều tín hiệu phù hợp rõ và ổn định; "
    "MODERATE = tín hiệu yếu hoặc không liên tục; LOW = nhiều tín hiệu không phù hợp kéo dài "
    "(không suy diễn cảm xúc/ý định). Dùng OCCLUDED khi bị che khuất, TRANSITION khi chuyển pha, "
    "INVALID khi lỗi track/thiếu dữ liệu."
)


def _fit_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    scale = min(width / max(1, image.shape[1]), height / max(1, image.shape[0]))
    nw = max(1, int(round(image.shape[1] * scale)))
    nh = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def validate_window_source(
    video_path: Path,
    pose_df: pd.DataFrame,
    session_id: str,
    camera_id: str,
    track_id: str,
    t_start_ms: float,
    t_end_ms: float,
    *,
    sample_fps: float = 10.0,
    tolerance_ms: float = 75.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Verify video/pose alignment across session, camera, timestamps, and track_id."""
    issues: List[str] = []
    if not video_path.is_file():
        issues.append(f"Thiếu file video: {video_path}")
    if pose_df.empty:
        issues.append("Pose CSV rỗng.")
        return pd.DataFrame(), {"ok": False, "issues": issues, "detection_count": 0}

    missing_cols = [c for c in REQUIRED_POSE_COLUMNS if c not in pose_df.columns]
    if missing_cols:
        issues.append(f"Pose CSV thiếu cột: {missing_cols}")
        return pd.DataFrame(), {"ok": False, "issues": issues, "detection_count": 0}

    if t_end_ms <= t_start_ms or t_start_ms < 0:
        issues.append(f"Khoảng thời gian không hợp lệ: [{t_start_ms}, {t_end_ms}]")
        return pd.DataFrame(), {"ok": False, "issues": issues, "detection_count": 0}

    # Verify session & camera match
    scoped = pose_df[
        pose_df["session_id"].astype(str).eq(str(session_id))
        & pose_df["camera_id"].astype(str).eq(str(camera_id))
    ]
    if scoped.empty:
        issues.append(f"Không tìm thấy dữ liệu cho {session_id}/{camera_id} trong pose CSV.")
        return pd.DataFrame(), {"ok": False, "issues": issues, "detection_count": 0}

    # Check video duration bounds when readable
    if video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                issues.append(f"Không mở được video: {video_path}")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                if fps > 0 and frame_count > 0:
                    duration_ms = frame_count / fps * 1000.0
                    if t_start_ms > duration_ms + 250.0:
                        issues.append(
                            f"Mốc bắt đầu ({t_start_ms:.0f}ms) vượt quá độ dài video ({duration_ms:.0f}ms)."
                        )
        finally:
            cap.release()

    window_poses = scoped[scoped["timestamp_ms"].between(t_start_ms - tolerance_ms, t_end_ms + tolerance_ms)].copy()
    track_poses = window_poses[window_poses["track_id"].astype(str).eq(str(track_id))].sort_values("timestamp_ms")

    expected_frames = max(1, int(round(((t_end_ms - t_start_ms) / 1000.0) * sample_fps)) + 1)
    det_count = int(len(track_poses))
    if det_count == 0:
        issues.append(f"Track #{track_id} không xuất hiện trong khoảng {t_start_ms/1000:.1f}–{t_end_ms/1000:.1f}s.")

    mean_vis = float(track_poses["visible_ratio"].mean()) if det_count and "visible_ratio" in track_poses else 0.0
    mean_conf = float(track_poses["mean_keypoint_conf"].mean()) if det_count and "mean_keypoint_conf" in track_poses else 0.0
    upper_series = upper_body_visibility(track_poses) if det_count else None
    mean_upper_vis = float(upper_series.mean()) if upper_series is not None and not upper_series.empty else mean_vis
    frame_coverage = min(1.0, det_count / float(expected_frames))

    audit = {
        "ok": len(issues) == 0,
        "issues": issues,
        "session_id": str(session_id),
        "camera_id": str(camera_id),
        "track_id": str(track_id),
        "t_start_ms": float(t_start_ms),
        "t_end_ms": float(t_end_ms),
        "detection_count": det_count,
        "expected_frames": expected_frames,
        "frame_coverage": round(frame_coverage, 4),
        "mean_visible_ratio": round(mean_vis, 4),
        "mean_upper_body_visible_ratio": round(mean_upper_vis, 4),
        "mean_pose_confidence": round(mean_conf, 4),
    }
    return window_poses, audit


def build_window_contact_sheet(
    video_path: Path,
    pose_df: pd.DataFrame,
    window_row: Dict[str, Any],
    output_path: Optional[Path] = None,
    *,
    class_phase: str = "lecture",
    phase_verified: bool = True,
    show_skeleton: bool = True,
    blur_face: bool = False,
    sample_fps: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a single composite RGB image containing classroom scene, 5 crops, pose info, and v0.2 context."""
    window_id = str(window_row["window_id"])
    session_id = str(window_row["session_id"])
    camera_id = str(window_row.get("camera_id", "cam_A"))
    track_id = str(window_row["track_id"])
    t_start_ms = float(window_row["t_start_ms"])
    t_end_ms = float(window_row["t_end_ms"])

    window_poses, audit = validate_window_source(
        video_path, pose_df, session_id, camera_id, track_id, t_start_ms, t_end_ms, sample_fps=sample_fps
    )

    mid_t_ms = (t_start_ms + t_end_ms) / 2.0
    frame_rgb = extract_frame_at_ms(video_path, mid_t_ms)
    if frame_rgb is not None and not window_poses.empty:
        near = window_poses.copy()
        near["time_diff"] = (near["timestamp_ms"] - mid_t_ms).abs()
        near = near[near["time_diff"] <= 350.0].sort_values("time_diff").drop_duplicates(subset=["track_id"])
        scene_rgb = render_classroom_scene(
            frame_rgb,
            near,
            selected_track_id=track_id,
            labels_by_track={},
            show_skeleton=show_skeleton,
            show_all_skeletons=False,
            blur_face=blur_face,
        )
    elif frame_rgb is not None:
        scene_rgb = frame_rgb
    else:
        scene_rgb = np.full((420, 960, 3), 30, dtype=np.uint8)

    strip_frames = (
        render_progression_strip(
            video_path,
            pose_df[pose_df["session_id"].astype(str).eq(session_id) & pose_df["camera_id"].astype(str).eq(camera_id)],
            track_id,
            t_start_ms,
            t_end_ms,
            show_skeleton=show_skeleton,
            blur_face=blur_face,
        )
        if not pose_df.empty
        else []
    )

    canvas_w = 1280
    header_h, scene_h, strip_h, footer_h = 74, 440, 236, 40
    canvas = np.full((header_h + scene_h + strip_h + footer_h, canvas_w, 3), 20, dtype=np.uint8)
    canvas[header_h:header_h + scene_h] = _fit_rgb(scene_rgb, canvas_w, scene_h)

    # Draw 5-frame progression strip across bottom
    y_strip = header_h + scene_h
    cell_w = canvas_w // 5
    for idx in range(5):
        x0 = idx * cell_w
        if idx < len(strip_frames):
            lbl, img = strip_frames[idx]
            fitted = _fit_rgb(img, cell_w - 6, strip_h - 28)
            canvas[y_strip + 4:y_strip + strip_h - 24, x0 + 3:x0 + cell_w - 3] = fitted
            cv2.putText(
                canvas, lbl, (x0 + 8, y_strip + strip_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 220, 120), 1, cv2.LINE_AA,
            )

    # Header text
    phase_tag = f"{class_phase} ({'verified' if phase_verified else 'unverified'})"
    line1 = f"WINDOW: {window_id}  |  {session_id}/{camera_id}  |  TARGET: Track #{track_id}  |  {t_start_ms/1000:.1f}s - {t_end_ms/1000:.1f}s"
    line2 = (
        f"PHASE: {phase_tag}  |  vis={audit['mean_visible_ratio']:.2f}  "
        f"upper_vis={audit['mean_upper_body_visible_ratio']:.2f}  "
        f"pose_conf={audit['mean_pose_confidence']:.2f}  "
        f"frames={audit['detection_count']}/{audit['expected_frames']} ({audit['frame_coverage']*100:.0f}%)"
    )
    cv2.putText(canvas, line1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line2, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    footer = "v0.2 Rubric: HIGH (clear aligned cues) | MODERATE (weak/intermittent) | LOW (sustained off-task) | OCCLUDED/INVALID | No emotion inference"
    cv2.putText(canvas, footer, (16, header_h + scene_h + strip_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 210, 180), 1, cv2.LINE_AA)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)

    bundle_meta = {
        "window_id": window_id,
        "session_id": session_id,
        "camera_id": camera_id,
        "track_id": track_id,
        "t_start_ms": t_start_ms,
        "t_end_ms": t_end_ms,
        "split": str(window_row.get("split", "")),
        "zone_id": str(window_row.get("zone_id", "")),
        "class_phase": class_phase,
        "phase_verified": bool(phase_verified),
        "pose_quality": audit,
        "guide_version": "v0.2",
        "guide_summary": GUIDE_V02_SUMMARY,
        "image_path": str(output_path) if output_path is not None else "",
    }
    return canvas, bundle_meta


def _infer_behaviors_and_rationale(
    w_tensor: np.ndarray,
    class_phase: str,
    audit: Dict[str, Any],
    rule_name: str,
    model_pred: Optional[str],
    gate_reason: str,
    p_max: Optional[float],
) -> Tuple[List[str], str, str]:
    """Derive observable behavior codes, Vietnamese rationale, and rejection reason."""
    hx_abs = float(np.mean(np.abs(w_tensor[:, 69])))
    hy_mean = float(np.mean(w_tensor[:, 70]))
    lhr = w_tensor[:, 71]
    rhr = w_tensor[:, 72]
    hr_steps = int(np.sum((lhr > 0.10) | (rhr > 0.10)))
    motion_mean = float(np.mean(w_tensor[:, 74]))
    wrist_vel = float(np.sqrt(np.mean(w_tensor[:, 52:56] ** 2)))

    behaviors: List[str] = []
    if hr_steps >= 8:
        behaviors.append("HAND_RAISED")
    if hy_mean > 0.02 or (hy_mean > -0.02 and wrist_vel < 0.15):
        behaviors.append("HEAD_DOWN_EXTENDED")
    elif -0.22 <= hy_mean <= 0.08 and wrist_vel >= 0.12 and hx_abs < 0.28:
        behaviors.append("READING_WRITING")
    if hx_abs > 0.30:
        behaviors.append("TURNING_AWAY")
    elif hx_abs <= 0.24 and hy_mean <= -0.10:
        behaviors.append("BOARD_OR_INSTRUCTOR_FACING")

    if audit.get("mean_visible_ratio", 1.0) < 0.45 or audit.get("frame_coverage", 1.0) < 0.40:
        if "OCCLUDED_KEYPOINTS" not in behaviors:
            behaviors.append("OCCLUDED_KEYPOINTS")

    if not behaviors:
        behaviors.append("UNCLEAR" if gate_reason != "ACCEPTED" else "BOARD_OR_INSTRUCTOR_FACING")

    parts = [
        f"Pha={class_phase}",
        f"vis={audit['mean_visible_ratio']:.2f}",
        f"conf={audit['mean_pose_confidence']:.2f}",
        f"|head_x|={hx_abs:.2f}",
        f"head_y={hy_mean:+.2f}",
        f"wrist_vel={wrist_vel:.2f}",
        f"motion={motion_mean:.3f}",
        f"rule={rule_name}",
    ]
    if model_pred and p_max is not None:
        parts.append(f"model={model_pred} (p_max={p_max:.2f}, gate={gate_reason})")

    rationale = (
        f"Quan sát track #{audit.get('track_id')} trong pha {class_phase}: "
        + ", ".join(parts)
        + "."
    )

    rejection_reason = ""
    if not audit.get("ok", True):
        rejection_reason = "; ".join(audit.get("issues", []))
    elif gate_reason != "ACCEPTED":
        p_str = f"{p_max:.2f}" if p_max is not None else "0.00"
        rejection_reason = (
            f"Gate cảnh báo {gate_reason} "
            f"(vis={audit['mean_visible_ratio']:.2f}, conf={audit['mean_pose_confidence']:.2f}, "
            f"p_max={p_str}) — nên xem kỹ clip hoặc thả ảnh vào ChatGPT Plus / Gemini Pro."
        )

    return behaviors, rationale, rejection_reason


def propose_local_suggestion(
    window_row: Dict[str, Any],
    track_pose_df: pd.DataFrame,
    *,
    video_path: Path,
    model: Optional[torch.nn.Module] = None,
    thresholds: Optional[UncertaintyThresholds] = None,
    temperature: float = 1.0,
    input_size: int = 94,
    class_phase: str = "lecture",
    sample_fps: float = 10.0,
    model_id: str = "",
    bundle_image_path: str = "",
) -> Dict[str, Any]:
    """Generate a local AI suggestion (never HUMAN) for a single window."""
    window_id = str(window_row["window_id"])
    session_id = str(window_row["session_id"])
    camera_id = str(window_row.get("camera_id", "cam_A"))
    track_id = str(window_row["track_id"])
    t_start_ms = float(window_row["t_start_ms"])
    t_end_ms = float(window_row["t_end_ms"])

    _, audit = validate_window_source(
        video_path, track_pose_df, session_id, camera_id, track_id, t_start_ms, t_end_ms, sample_fps=sample_fps
    )

    if not audit["ok"] or audit["frame_coverage"] < 0.25:
        return {
            "window_id": window_id,
            "suggested_state": "INVALID" if not audit["ok"] else "OCCLUDED",
            "suggested_label": "",
            "suggested_behaviors": ["OCCLUDED_KEYPOINTS" if audit["detection_count"] else "UNCLEAR"],
            "suggested_phase": class_phase,
            "suggested_occlusion": "heavy",
            "rationale": f"Dữ liệu nguồn/track không đủ phủ cửa sổ ({audit['detection_count']}/{audit['expected_frames']} khung).",
            "rejection_reason": "; ".join(audit["issues"]) or "TRACK_INCOMPLETE",
            "confidence": 0.0,
            "gate_reason": "LOW_VISIBILITY",
            "needs_vision_review": True,
            "suggestion_source": "LOCAL_AI_PROPOSER",
            "model_id": model_id,
            "bundle_image_path": bundle_image_path,
        }

    # Slice around [t_start_ms - 500, t_end_ms + 500] before cleaning so long 15-min tracks run in milliseconds.
    trk_df = track_pose_df[
        track_pose_df["session_id"].astype(str).eq(session_id)
        & track_pose_df["camera_id"].astype(str).eq(camera_id)
        & track_pose_df["track_id"].astype(str).eq(track_id)
        & track_pose_df["timestamp_ms"].between(t_start_ms - 500.0, t_end_ms + 500.0)
    ].copy()
    if "norm_kpt_00_x" not in trk_df.columns and not trk_df.empty:
        trk_df = clean_and_normalize_poses(trk_df, min_track_seconds=0.5)
    trk_win = trk_df[trk_df["timestamp_ms"].between(t_start_ms - 75.0, t_end_ms + 75.0)].sort_values("timestamp_ms")

    if len(trk_win) < 5:
        return {
            "window_id": window_id,
            "suggested_state": "OCCLUDED",
            "suggested_label": "",
            "suggested_behaviors": ["OCCLUDED_KEYPOINTS"],
            "suggested_phase": class_phase,
            "suggested_occlusion": "heavy",
            "rationale": f"Chỉ còn {len(trk_win)} khung hợp lệ sau làm sạch; không đủ dựng chuỗi 4 giây.",
            "rejection_reason": "INSUFFICIENT_CLEAN_FRAMES",
            "confidence": 0.0,
            "gate_reason": "LOW_VISIBILITY",
            "needs_vision_review": True,
            "suggestion_source": "LOCAL_AI_PROPOSER",
            "model_id": model_id,
            "bundle_image_path": bundle_image_path,
        }

    include_mask = input_size == 94
    feats, _ = compute_track_features(trk_win, sample_fps=sample_fps, include_mask=include_mask)
    feats_40 = feats[np.linspace(0, len(feats) - 1, 40).round().astype(int)] if len(feats) != 40 else feats
    probs: Optional[np.ndarray] = None
    if model is not None:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            x_in = torch.from_numpy(feats_40[None, ...]).float().to(device)
            logits = model(x_in) / max(float(temperature), 1e-6)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return _build_suggestion_from_features(
        window_id,
        audit,
        feats_40,
        class_phase=class_phase,
        probs=probs,
        thresholds=thresholds,
        model_id=model_id,
        bundle_image_path=bundle_image_path,
    )


def _build_suggestion_from_features(
    window_id: str,
    audit: Dict[str, Any],
    feats_40: np.ndarray,
    *,
    class_phase: str,
    probs: Optional[np.ndarray],
    thresholds: Optional[UncertaintyThresholds],
    model_id: str,
    bundle_image_path: str = "",
) -> Dict[str, Any]:
    """Build a canonical suggestion dict from aligned [40, D] features and optional model probs."""
    rule_label, rule_name = apply_weak_label_rules(
        {"class_phase": class_phase, "quality_state": "VALID"},
        feats_40,
        rule_version="weak-v0.2",
    )

    model_pred: Optional[str] = None
    p_max: Optional[float] = None
    gate_reason = "ACCEPTED"

    if probs is not None:
        th = thresholds or UncertaintyThresholds()
        pred_idx = int(np.argmax(probs))
        model_pred = INDEX_TO_LABEL[pred_idx]
        p_max = float(probs[pred_idx])
        _, gate_reason = explain_uncertainty_gate(
            probs,
            visible_ratio=audit["mean_visible_ratio"],
            mean_keypoint_confidence=audit["mean_pose_confidence"],
            upper_body_visible_ratio=audit["mean_upper_body_visible_ratio"],
            thresholds=th,
            class_labels=CANONICAL_TRAINING_LABELS,
        )

    behaviors, rationale, rejection_reason = _infer_behaviors_and_rationale(
        feats_40, class_phase, audit, rule_name, model_pred, gate_reason, p_max
    )

    if audit["mean_visible_ratio"] < 0.45 or audit["frame_coverage"] < 0.45:
        suggested_state = "OCCLUDED"
        suggested_label = ""
        suggested_occlusion = "heavy"
        if not rejection_reason:
            rejection_reason = f"LOW_VISIBILITY (vis={audit['mean_visible_ratio']:.2f}, cov={audit['frame_coverage']:.2f})"
    else:
        suggested_state = "VALID"
        suggested_label = model_pred or rule_label or MODERATE
        suggested_occlusion = "none" if audit["mean_visible_ratio"] >= 0.75 else "partial"

    phase_evidence_default = {
        "lecture": "Quan sát thấy hoạt động trình bày bài học trước lớp",
        "individual_work": "Quan sát được yêu cầu làm bài cá nhân",
        "discussion": "Quan sát được yêu cầu thảo luận nhóm",
    }.get(class_phase, "")

    return {
        "window_id": window_id,
        "suggested_state": suggested_state,
        "suggested_label": suggested_label,
        "suggested_behaviors": behaviors,
        "suggested_phase": class_phase,
        "suggested_phase_evidence": phase_evidence_default,
        "suggested_occlusion": suggested_occlusion,
        "rationale": rationale,
        "rejection_reason": rejection_reason,
        "confidence": round(p_max if p_max is not None else 0.55, 4),
        "gate_reason": gate_reason,
        "needs_vision_review": bool(gate_reason != "ACCEPTED" or suggested_state != "VALID"),
        "suggestion_source": "LOCAL_AI_PROPOSER",
        "model_id": model_id,
        "bundle_image_path": bundle_image_path,
    }


def propose_local_suggestions_from_features(
    windows_df: pd.DataFrame,
    features: np.ndarray,
    *,
    raw_dir: Path,
    poses_dir: Path,
    model: Optional[torch.nn.Module] = None,
    thresholds: Optional[UncertaintyThresholds] = None,
    temperature: float = 1.0,
    input_size: int = 94,
    class_phase: str = "lecture",
    sample_fps: float = 10.0,
    model_id: str = "",
    batch_size: int = 512,
) -> List[Dict[str, Any]]:
    """Fast batched Local AI proposer over precomputed [N, T, D] feature tensors."""
    if len(windows_df) != len(features):
        raise ValueError(f"Mismatch windows ({len(windows_df)}) vs features ({len(features)}).")
    if len(windows_df) == 0:
        return []

    probs_all: Optional[np.ndarray] = None
    if model is not None:
        model.eval()
        device = next(model.parameters()).device
        temp = max(float(temperature), 1e-6)
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), max(1, int(batch_size))):
                xb = torch.from_numpy(np.asarray(features[start:start + batch_size, :, :input_size], dtype=np.float32)).to(device)
                logits = model(xb) / temp
                chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs_all = np.concatenate(chunks, axis=0)

    session_source_cache: Dict[Tuple[str, str], Tuple[bool, Optional[float], List[str]]] = {}
    results: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(windows_df.iterrows()):
        session_id = str(row["session_id"])
        camera_id = str(row.get("camera_id", "cam_A")) or "cam_A"
        cache_key = (session_id, camera_id)
        if cache_key not in session_source_cache:
            issues: List[str] = []
            duration_ms: Optional[float] = None
            v_cands = [raw_dir / session_id / f"{camera_id}.mp4", raw_dir / session_id / "cam_A.mp4", raw_dir / f"{session_id}.mp4"]
            v_file = next((c for c in v_cands if c.is_file()), None)
            if v_file is None:
                issues.append(f"Thiếu file video cho {session_id}/{camera_id}")
            else:
                cap = cv2.VideoCapture(str(v_file))
                try:
                    if not cap.isOpened():
                        issues.append(f"Không mở được video: {v_file}")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                        if fps > 0 and fc > 0:
                            duration_ms = fc / fps * 1000.0
                finally:
                    cap.release()
            p_cands = [poses_dir / f"{session_id}_{camera_id}_poses.csv", poses_dir / f"{session_id}_poses.csv"]
            if not any(c.is_file() for c in p_cands):
                issues.append(f"Thiếu file pose cho {session_id}/{camera_id}")
            session_source_cache[cache_key] = (len(issues) == 0, duration_ms, issues)

        src_ok, duration_ms, base_issues = session_source_cache[cache_key]
        t_start_ms = float(row.get("t_start_ms", 0.0))
        t_end_ms = float(row.get("t_end_ms", 0.0))
        row_issues = list(base_issues)
        if duration_ms is not None and t_start_ms > duration_ms + 250.0:
            row_issues.append(f"Mốc bắt đầu ({t_start_ms:.0f}ms) vượt quá độ dài video ({duration_ms:.0f}ms).")
            src_ok = False

        feats = np.asarray(features[idx], dtype=np.float32)
        if feats.shape[1] >= 94:
            frame_cov = float(np.mean(feats[:, 77:94].sum(axis=1) > 0))
        else:
            frame_cov = 1.0

        def _fval(col: str, default: float) -> float:
            raw_v = row.get(col, "")
            try:
                v = float(raw_v)
                return v if np.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        has_vis_col = "mean_visible_ratio" in windows_df.columns and str(row.get("mean_visible_ratio", "")).strip() != ""
        default_vis = float(np.mean(feats[:, 77:94])) if feats.shape[1] >= 94 else 0.0
        default_upper_vis = float(np.mean(feats[:, 77:90])) if feats.shape[1] >= 90 else default_vis
        mean_vis = _fval("mean_visible_ratio", default_vis) if has_vis_col else default_vis
        has_uvis_col = "mean_upper_body_visible_ratio" in windows_df.columns and str(row.get("mean_upper_body_visible_ratio", "")).strip() != ""
        mean_upper_vis = _fval("mean_upper_body_visible_ratio", default_upper_vis) if has_uvis_col else default_upper_vis
        has_conf_col = "mean_pose_confidence" in windows_df.columns and str(row.get("mean_pose_confidence", "")).strip() != ""
        mean_conf = _fval("mean_pose_confidence", 0.55 if default_vis >= 0.45 else 0.0) if has_conf_col else (0.55 if default_vis >= 0.45 else 0.0)
        expected_frames = max(1, int(round(((t_end_ms - t_start_ms) / 1000.0) * sample_fps)) + 1)
        audit = {
            "ok": src_ok and len(row_issues) == 0,
            "issues": row_issues,
            "session_id": session_id,
            "camera_id": camera_id,
            "track_id": str(row["track_id"]),
            "t_start_ms": t_start_ms,
            "t_end_ms": t_end_ms,
            "detection_count": int(round(frame_cov * expected_frames)),
            "expected_frames": expected_frames,
            "frame_coverage": round(frame_cov, 4),
            "mean_visible_ratio": round(mean_vis, 4),
            "mean_upper_body_visible_ratio": round(mean_upper_vis, 4),
            "mean_pose_confidence": round(mean_conf, 4),
        }
        feats_40 = feats[np.linspace(0, len(feats) - 1, 40).round().astype(int)] if len(feats) != 40 else feats
        probs_i = probs_all[idx] if probs_all is not None else None
        results.append(
            _build_suggestion_from_features(
                str(row["window_id"]),
                audit,
                feats_40,
                class_phase=class_phase,
                probs=probs_i,
                thresholds=thresholds,
                model_id=model_id,
            )
        )
    return results


def build_web_llm_prompt(bundle_metas: Sequence[Dict[str, Any]]) -> str:
    """Create a copy-paste Vietnamese prompt for ChatGPT Plus / Gemini Pro web vision."""
    lines = [
        "Bạn là trợ lý gán nhãn hành vi học tập quan sát được theo hướng dẫn v0.2 (KHÔNG suy diễn cảm xúc/ý định).",
        "Hãy xem ảnh ghép của từng cửa sổ (ảnh toàn cảnh có khung trắng/vàng chỉ đúng người + 5 ảnh crop theo timestamp bên dưới) và trả về DUY NHẤT một mảng JSON hợp lệ.",
        "",
        "Quy tắc v0.2:",
        "- Chỉ gán cho đúng người mang track ID ghi ở tiêu đề ảnh.",
        "- state: một trong [VALID, OCCLUDED, TRANSITION, INVALID]. Nếu không đủ căn cứ/bị che -> OCCLUDED hoặc INVALID, để label = \"\".",
        "- label (chỉ khi state=VALID): một trong [HIGH_OBSERVED_PARTICIPATION, MODERATE_OBSERVED_PARTICIPATION, LOW_OBSERVED_PARTICIPATION].",
        "  • HIGH: nhiều tín hiệu phù hợp rõ với pha (vd giơ tay, hướng bảng ổn định, đọc/viết tích cực).",
        "  • MODERATE: tín hiệu yếu, ngồi bình thường hoặc không liên tục.",
        "  • LOW: nhiều tín hiệu không phù hợp kéo dài (gục đầu sâu, quay hẳn đi chỗ khác kéo dài trong lecture).",
        f"- behaviors: chọn từ {list(VALID_BEHAVIOR_CODES)}.",
        "- rationale: mô tả ngắn gọn bằng tiếng Việt hành vi quan sát thấy trong ảnh.",
        "- rejection_reason: lý do từ chối nếu state != VALID hoặc có cảnh báo che khuất/mơ hồ (để trống nếu VALID rõ).",
        "",
        "Danh sách cửa sổ trong lượt ảnh này:",
    ]
    for meta in bundle_metas:
        q = meta.get("pose_quality", {})
        lines.append(
            f"- {meta['window_id']} | track #{meta['track_id']} | phase={meta.get('class_phase', 'lecture')} | "
            f"vis={q.get('mean_visible_ratio', 0):.2f}, conf={q.get('mean_pose_confidence', 0):.2f}, cov={q.get('frame_coverage', 0):.2f}"
        )
    lines.extend([
        "",
        "Định dạng JSON bắt buộc:",
        '[{"window_id": "...", "state": "VALID", "label": "MODERATE_OBSERVED_PARTICIPATION", '
        '"behaviors": ["BOARD_OR_INSTRUCTOR_FACING"], "rationale": "...", "rejection_reason": ""}]',
    ])
    return "\n".join(lines)


def _normalize_label(raw_label: Any) -> str:
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


SUGGESTION_CSV_COLUMNS = [
    "window_id",
    "suggested_state",
    "suggested_label",
    "suggested_behaviors",
    "suggested_phase",
    "suggested_phase_evidence",
    "suggested_occlusion",
    "confidence",
    "gate_reason",
    "needs_vision_review",
    "suggestion_source",
    "rationale",
    "rejection_reason",
    "model_id",
    "bundle_image_path",
]


def _write_suggestions_csv(csv_path: Path, existing: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for wid, rec in existing.items():
        behs = rec.get("suggested_behaviors", [])
        if isinstance(behs, list):
            behs_str = "; ".join(str(b).strip() for b in behs if str(b).strip())
        else:
            behs_str = str(behs or "")
        rows.append({
            "window_id": wid,
            "suggested_state": rec.get("suggested_state", ""),
            "suggested_label": rec.get("suggested_label", ""),
            "suggested_behaviors": behs_str,
            "suggested_phase": rec.get("suggested_phase", "lecture"),
            "suggested_phase_evidence": rec.get("suggested_phase_evidence", ""),
            "suggested_occlusion": rec.get("suggested_occlusion", "none"),
            "confidence": rec.get("confidence", ""),
            "gate_reason": rec.get("gate_reason", ""),
            "needs_vision_review": bool(rec.get("needs_vision_review", False)),
            "suggestion_source": rec.get("suggestion_source", ""),
            "rationale": rec.get("rationale", ""),
            "rejection_reason": rec.get("rejection_reason", ""),
            "model_id": rec.get("model_id", ""),
            "bundle_image_path": rec.get("bundle_image_path", ""),
        })
    df = pd.DataFrame(rows, columns=SUGGESTION_CSV_COLUMNS)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp", dir=csv_path.parent, delete=False) as fh:
            tmp = Path(fh.name)
            df.to_csv(fh, index=False)
        tmp.replace(csv_path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def save_suggestions_map(path: Path, records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge suggestions by window_id and atomically write JSON + sibling CSV."""
    json_path = path.with_suffix(".json") if path.suffix == ".csv" else path
    csv_path = json_path.with_suffix(".csv")
    existing = load_suggestions_map(json_path)
    for rec in records:
        wid = str(rec["window_id"])
        prev = existing.get(wid, {})
        merged = {**prev, **rec}
        if not merged.get("bundle_image_path") and prev.get("bundle_image_path"):
            merged["bundle_image_path"] = prev["bundle_image_path"]
        existing[wid] = merged
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", dir=json_path.parent, delete=False) as fh:
            tmp = Path(fh.name)
            json.dump(existing, fh, ensure_ascii=False, indent=2)
        tmp.replace(json_path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    _write_suggestions_csv(csv_path, existing)
    return existing


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
