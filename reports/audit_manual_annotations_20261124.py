"""Read-only source audit; generated diagnostics never change supplied labels."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.contracts.labels import validate_annotation_label

OUT = ROOT / 'reports/manual_annotation_audit_20261124'
OUT.mkdir(exist_ok=True)
windows = pd.read_csv(ROOT / 'data/features/participation_windows.csv').fillna('')
summary = {}
for session, path in [
    ('class_1', Path(r'D:\Downloads\data\annotations\class_1_labels_final.csv')),
    ('session_006', Path(r'D:\ThiGiacMayTinh\cam_A_labels_final.csv')),
]:
    df = pd.read_csv(path, keep_default_na=False)
    pose = pd.read_parquet(ROOT / f'data/poses/{session}_cam_A_poses.parquet')
    w = windows[windows.session_id.eq(session)]
    issues = []
    details = []
    for i, row in df.iterrows():
        try:
            validate_annotation_label(**{c: row[c] for c in ['participation_label', 'annotation_state', 'label_source', 'label_rule_version']})
        except ValueError as e:
            issues.append({'csv_line': i+2, 'error': str(e)})
        track = pose[pose.track_id.eq(row.track_id)]
        points = track[track.timestamp_ms.ge(row.t_start_ms) & track.timestamp_ms.lt(row.t_end_ms)]
        candidates = w[w.track_id.eq(row.track_id)]
        exact = candidates[candidates.t_start_ms.eq(row.t_start_ms) & candidates.t_end_ms.eq(row.t_end_ms)]
        # Diagnostic tolerance only: do not use this to auto-assign labels.
        near = candidates[(candidates.t_start_ms-row.t_start_ms).abs().le(150) & (candidates.t_end_ms-row.t_end_ms).abs().le(150)]
        overlap = np.minimum(candidates.t_end_ms.to_numpy(), row.t_end_ms)-np.maximum(candidates.t_start_ms.to_numpy(), row.t_start_ms)
        overlap = np.maximum(0, overlap)
        best = candidates.iloc[int(overlap.argmax())] if len(overlap) and overlap.max()>0 else None
        details.append({
            'csv_line': i+2, 'track_id': row.track_id, 't_start_ms': row.t_start_ms,
            't_end_ms': row.t_end_ms, 'pose_samples_in_interval': points.timestamp_ms.nunique(),
            'track_exists': not track.empty, 'exact_time_matches_after_identity_mapping': len(exact),
            'near_time_matches_150ms_after_identity_mapping': len(near),
            'near_window_id': near.iloc[0].window_id if len(near)==1 else '',
            'near_quality_state': near.iloc[0].quality_state if len(near)==1 else '',
            'best_overlap_ms': float(overlap.max()) if len(overlap) else 0,
            'best_window_id': best.window_id if best is not None else '',
        })
    d = pd.DataFrame(details)
    d.to_csv(OUT/f'{session}_alignment.csv', index=False)
    pd.DataFrame(issues, columns=['csv_line','error']).to_csv(OUT/f'{session}_contract_issues.csv', index=False)
    phases = df.groupby(['t_start_ms','t_end_ms']).class_phase.nunique()
    zones = df.groupby('track_id').zone_id.nunique()
    valid_conf = pd.to_numeric(df.annotation_confidence, errors='coerce')
    summary[session] = {
        'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': len(df), 'identity_values': df[['session_id','camera_id']].drop_duplicates().to_dict('records'),
        'pose_identity_values': pose[['session_id','camera_id']].drop_duplicates().to_dict('records'),
        'tracks': sorted(df.track_id.unique().tolist()),
        'time_range_ms': [int(df.t_start_ms.min()),int(df.t_end_ms.max())],
        'duration_ms_counts': (df.t_end_ms-df.t_start_ms).value_counts().to_dict(),
        'labels': df.participation_label.value_counts().to_dict(),
        'states': df.annotation_state.value_counts().to_dict(), 'sources': df.label_source.value_counts().to_dict(),
        'phases': df.class_phase.value_counts().to_dict(),
        'guide_versions': df.annotation_guide_version.value_counts().to_dict(),
        'annotators': df.annotator_id.value_counts().to_dict(),
        'confidence': df.annotation_confidence.value_counts().to_dict(),
        'contract_error_rows': len(issues), 'contract_error_examples': issues[:3],
        'bad_times': int(((df.t_start_ms<0)|(df.t_end_ms<=df.t_start_ms)).sum()),
        'bad_confidence': int((valid_conf.isna()|~valid_conf.between(0,1)).sum()),
        'duplicate_keys': int(df.duplicated(['session_id','camera_id','track_id','t_start_ms','t_end_ms']).sum()),
        'missing_window_id': 'window_id' not in df,
        'missing_pose_tracks': sorted(df.loc[~df.track_id.isin(pose.track_id),'track_id'].unique().tolist()),
        'zero_pose_intervals': int(d.pose_samples_in_interval.eq(0).sum()),
        'under_20_pose_samples': int(d.pose_samples_in_interval.lt(20).sum()),
        'pose_samples_counts': d.pose_samples_in_interval.value_counts().to_dict(),
        'exact_time_matches_after_identity_mapping': int(d.exact_time_matches_after_identity_mapping.eq(1).sum()),
        'near_time_matches_150ms_after_identity_mapping': int(d.near_time_matches_150ms_after_identity_mapping.eq(1).sum()),
        'near_quality_states': d.loc[d.near_window_id.ne(''),'near_quality_state'].value_counts().to_dict(),
        'same_interval_conflicting_phases': int(phases.gt(1).sum()),
        'same_interval_examples': phases[phases.gt(1)].head(5).index.tolist(),
        'tracks_with_multiple_zones': zones[zones.gt(1)].to_dict(),
        'phase_by_start': df.groupby('t_start_ms').class_phase.unique().apply(list).to_dict(),
        'pose_timestamp_range_ms': [float(pose.timestamp_ms.min()),float(pose.timestamp_ms.max())],
    }
other = Path(r'D:\Downloads\data\annotations\class_1_labels.csv')
summary['class_1_files_identical'] = other.read_bytes() == Path(summary['class_1']['source']).read_bytes()
(OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str), encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
