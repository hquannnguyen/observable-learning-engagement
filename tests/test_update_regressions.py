"""Regressions found when validating the supplied update plan."""
import numpy as np
import pandas as pd
import torch

from app.relabel_app import load_source, save_review
from src.preprocessing.cleaner import interpolate_missing_keypoints, smooth_keypoints
from src.reporting.exporter import export_bundle
from src.training.trainer import compute_accuracy_coverage_curve
from tests.test_reporting_export import make_predictions


def test_only_filled_gaps_contribute_to_window_metadata():
    values = [1, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 8]
    df = pd.DataFrame(dict(track_id=1, frame_idx=range(8), kpt_00_x=values, kpt_00_y=values))
    cleaned = interpolate_missing_keypoints(df, max_gap_frames=2)
    assert cleaned.max_interpolated_gap_frames.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert smooth_keypoints(cleaned).loc[3:6, 'kpt_00_x'].isna().all()


def test_shareable_export_excludes_embedded_track_keys(tmp_path):
    predictions = make_predictions()
    predictions['track_id'] = 123
    predictions['notes'] = 'private'
    paths = export_bundle(predictions, tmp_path)
    output = pd.read_csv(paths['results'])
    assert not {'track_id', 'window_id', 'notes'} & set(output)
    assert len(output) == len(predictions)


def test_quality_rejections_count_in_coverage():
    curve = compute_accuracy_coverage_curve(
        np.array([[.8, .1, .1], [.1, .8, .1]]), np.array([0, 1]),
        thresholds=[0., .9], eligible=np.array([True, False]),
    )
    assert curve.coverage.tolist() == [.5, 0.]
    assert pd.isna(curve.accuracy.iloc[1])


def test_legacy_review_preserves_source_and_extra_rows(tmp_path):
    source = tmp_path / 'input.csv'
    # Use the schema-valid sample, with a stable generated legacy window key.
    from pathlib import Path
    sample = Path(__file__).resolve().parents[1] / 'data/sample/annotation.example.csv'
    source.write_bytes(sample.read_bytes())
    before = source.read_bytes()
    rows = load_source(source)
    row = rows.iloc[0]
    updates = row.to_dict()
    updates.update(annotation_state='OCCLUDED', participation_label='', label_source='', label_rule_version='',
                   annotator_id='ANN_PHONG', annotation_confidence=.8)
    target = tmp_path / 'review.csv'
    save_review(target, row, updates)
    assert source.read_bytes() == before
    saved = pd.read_csv(target, keep_default_na=False)
    assert saved.label_source.tolist() == ['']
