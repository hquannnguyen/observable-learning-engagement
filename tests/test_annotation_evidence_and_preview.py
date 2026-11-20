import cv2
import numpy as np
import pandas as pd
import pytest

from src.contracts.annotation_rubric import evidence_label
from src.contracts.labels import HIGH, LOW, MODERATE
from src.pose.annotation_preview import compose_target_frame, nearest_frame_boxes, render_annotation_preview


@pytest.mark.parametrize("evidence,label", [
    (["ALIGNED"] * 4, HIGH),
    (["ALIGNED"] * 3 + ["AMBIGUOUS"], HIGH),
    (["CONTRADICTORY"] * 3 + ["UNOBSERVABLE"], LOW),
    (["ALIGNED", "CONTRADICTORY", "ALIGNED", "AMBIGUOUS"], MODERATE),
    (["ALIGNED", "CONTRADICTORY", "AMBIGUOUS", "UNOBSERVABLE"], None),
    (["AMBIGUOUS"] * 4, None),
    ([None] * 4, None),
])
def test_rubric_never_equates_ambiguity_with_low(evidence, label):
    assert evidence_label(evidence)[0] == label


def test_focus_uses_only_requested_track_and_does_not_reuse_stale_boxes():
    frame = np.full((100, 200, 3), 180, np.uint8)
    boxes = pd.DataFrame([dict(timestamp_ms=0, track_id=1, bbox_x1=10, bbox_y1=20, bbox_x2=50, bbox_y2=90),
                          dict(timestamp_ms=0, track_id=2, bbox_x1=130, bbox_y1=20, bbox_x2=170, bbox_y2=90)])
    focused, found = compose_target_frame(frame, boxes, "2", [120, 10, 180, 95], 0, anonymize=False)
    assert found
    # 200x100 scene fits into 896x504, vertically centered with a 28px margin.
    target_pixel = focused[48 + 28 + round(60 * 4.48), round(150 * 4.48)]
    other_pixel = focused[48 + 28 + round(60 * 4.48), round(30 * 4.48)]
    assert target_pixel.mean() > other_pixel.mean() * 2
    missing = nearest_frame_boxes(boxes, 500)
    assert missing.empty
    absent, found = compose_target_frame(frame, missing, "2", [120, 10, 180, 95], 500, anonymize=False)
    assert not found
    assert np.all(absent[100:500, 910:1270] == 24)


def test_real_clip_seek_cache_and_complete_window(tmp_path):
    video, poses = tmp_path / "source.mp4", tmp_path / "poses.csv"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 100))
    rows = []
    for i in range(30):
        frame = np.full((100, 200, 3), 50 + i, np.uint8)
        cv2.rectangle(frame, (100, 15), (150, 90), (0, 0, 210), -1)
        writer.write(frame)
        rows.append(dict(timestamp_ms=i * 100, track_id=2, bbox_x1=100, bbox_y1=15, bbox_x2=150, bbox_y2=90))
    writer.release()
    pd.DataFrame(rows).to_csv(poses, index=False)
    output, meta = render_annotation_preview(video, poses, "2", 1500, 2200, tmp_path / "cache")
    assert meta["frames"] == meta["target_visible_frames"] == 8
    assert meta["t_start_ms"] == 1500
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 8
    capture.release()
    before = output.stat().st_mtime_ns
    repeated, _ = render_annotation_preview(video, poses, "2", 1500, 2200, tmp_path / "cache")
    assert repeated == output and repeated.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="tọa độ"):
        render_annotation_preview(video, poses, "99", 1500, 2200, tmp_path / "cache")
