import numpy as np
import pandas as pd
import pytest

import src.pose.studio_renderer as renderer


@pytest.mark.parametrize("show,all_tracks,expected", [
    (True, False, [False, True]),
    (True, True, [True, True]),
    (False, True, [False, False]),
])
def test_scene_skeleton_toggle_contract(monkeypatch, show, all_tracks, expected):
    calls = []
    monkeypatch.setattr(renderer, "_draw_student_box_and_skeleton",
                        lambda *args, **kwargs: calls.append(kwargs["show_skeleton"]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = renderer.render_classroom_scene(
        frame, pd.DataFrame({"track_id": [1, 2]}), selected_track_id="2",
        show_skeleton=show, show_all_skeletons=all_tracks, blur_face=True,
    )
    assert result.shape == frame.shape
    assert calls == expected
