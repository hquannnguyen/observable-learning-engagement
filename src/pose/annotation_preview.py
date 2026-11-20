"""Person-specific review clips from existing tracks; no engagement predictions."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.pose.video_codec import ffmpeg_executable
from src.pose.visualization import apply_face_blur

BOX_COLUMNS = ["timestamp_ms", "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
HEAD_COLUMNS = [f"kpt_{i:02d}_{axis}" for i in range(5) for axis in ("x", "y", "conf")]


def read_review_boxes(path):
    return pd.read_csv(path, usecols=lambda name: name in BOX_COLUMNS or name in HEAD_COLUMNS).sort_values("timestamp_ms")


def nearest_frame_boxes(poses, timestamp_ms, tolerance_ms=75):
    times = poses.timestamp_ms.unique()
    if not len(times):
        return poses.iloc[:0]
    nearest = times[np.argmin(np.abs(times - timestamp_ms))]
    return poses[poses.timestamp_ms.eq(nearest)] if abs(nearest - timestamp_ms) <= tolerance_ms else poses.iloc[:0]


def fit_image(image, width, height):
    result = np.full((height, width, 3), 24, np.uint8)
    if image.size:
        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
        y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
        result[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return result


def compose_target_frame(frame, boxes, track_id, crop_box, timestamp_ms, anonymize=True):
    """Yellow box + undimmed target on the left; enlarged same target on the right."""
    frame = frame.copy()
    if anonymize:
        for _, person in boxes.iterrows():
            keypoints = person[HEAD_COLUMNS].to_numpy(dtype=float).reshape(5, 3) if all(c in person.index for c in HEAD_COLUMNS) else None
            apply_face_blur(frame, tuple(float(person[c]) for c in BOX_COLUMNS[2:]), keypoints=keypoints)
    target = boxes[boxes.track_id.astype(str).eq(str(track_id))]
    scene = (frame.astype(np.float32) * 0.28).astype(np.uint8)
    crop = np.full((504, 384, 3), 24, np.uint8)
    found = not target.empty
    if found:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(target.iloc[0][c])) for c in BOX_COLUMNS[2:]]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        found = x2 > x1 and y2 > y1
        if found:
            scene[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 240, 255), max(4, w // 320))
            cx1, cy1, cx2, cy2 = crop_box
            crop = fit_image(frame[cy1:cy2, cx1:cx2], 384, 504)
    canvas = np.full((592, 1280, 3), 24, np.uint8)
    canvas[48:552, :896] = fit_image(scene, 896, 504)
    canvas[48:552, 896:] = crop
    cv2.putText(canvas, f"LABEL THIS PERSON: ID {track_id}", (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 255), 2)
    cv2.putText(canvas, "TARGET CLOSE-UP", (910, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)
    status = f"VIDEO TIME {timestamp_ms / 1000:.2f}s" if found else f"ID {track_id} NOT VISIBLE - DO NOT GUESS"
    cv2.putText(canvas, status, (18, 578), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    return canvas, found


def render_annotation_preview(video_path, pose_path, track_id, start_ms, end_ms, cache_dir, anonymize=True):
    video_path, pose_path, cache_dir = Path(video_path), Path(pose_path), Path(cache_dir)
    signature = ["focus-v2", str(video_path.resolve()), video_path.stat().st_mtime_ns, video_path.stat().st_size,
                 str(pose_path.resolve()), pose_path.stat().st_mtime_ns, str(track_id), float(start_ms), float(end_ms), anonymize]
    cache_key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    output, audit_path = cache_dir / f"{cache_key}.mp4", cache_dir / f"{cache_key}.json"
    if output.is_file() and audit_path.is_file():
        return output, json.loads(audit_path.read_text())
    poses = read_review_boxes(pose_path)
    poses = poses[poses.timestamp_ms.between(float(start_ms) - 75, float(end_ms) + 75)]
    target = poses[poses.track_id.astype(str).eq(str(track_id))]
    if target.empty:
        raise ValueError(f"Không có tọa độ của track {track_id} trong khoảng cần gán.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Không mở được video gốc để định vị người.")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if native_fps <= 0 or end_ms <= start_ms:
        cap.release()
        raise ValueError("FPS hoặc khoảng thời gian không hợp lệ.")
    left, top = target.bbox_x1.min(), target.bbox_y1.min()
    right, bottom = target.bbox_x2.max(), target.bbox_y2.max()
    pad_x, pad_y = (right - left) * .12, (bottom - top) * .08
    crop = [max(0, int(left - pad_x)), max(0, int(top - pad_y)), min(width, int(right + pad_x)), min(height, int(bottom + pad_y))]
    start_frame = round(float(start_ms) * native_fps / 1000)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame, rendered, matched = start_frame, 0, 0
    try:
        with tempfile.TemporaryDirectory(dir=cache_dir) as work:
            raw = Path(work) / "raw.mp4"
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 592))
            if not writer.isOpened():
                raise ValueError("Không tạo được clip xem người cần gán.")
            try:
                for timestamp in np.arange(float(start_ms), float(end_ms) + .01, 100):
                    desired = round(timestamp * native_fps / 1000)
                    while current_frame < desired:
                        if not cap.grab():
                            break
                        current_frame += 1
                    ok, frame = cap.read()
                    current_frame += 1
                    if not ok:
                        break
                    boxes = nearest_frame_boxes(poses, timestamp)
                    result, found = compose_target_frame(frame, boxes, track_id, crop, timestamp, anonymize)
                    writer.write(result)
                    matched += int(found)
                    rendered += 1
            finally:
                writer.release()
            expected_frames = len(np.arange(float(start_ms), float(end_ms) + .01, 100))
            if not matched or rendered != expected_frames:
                raise ValueError("Video không phủ đủ khoảng cần gán hoặc không tìm thấy đúng track.")
            encoded = Path(work) / "encoded.mp4"
            try:
                subprocess.run([ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(raw), "-an", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
                    check=True, capture_output=True)
            except subprocess.SubprocessError as exc:
                raise RuntimeError("Không chuyển được clip sang H.264.") from exc
            encoded.replace(output)
    finally:
        cap.release()
    audit = {"track_id":str(track_id), "t_start_ms":float(start_ms), "t_end_ms":float(end_ms),
             "frames":rendered, "target_visible_frames":matched, "source":str(video_path), "poses":str(pose_path), "anonymize":anonymize}
    audit_path.write_text(json.dumps(audit, indent=2))
    return output, audit
