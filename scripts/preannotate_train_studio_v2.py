#!/usr/bin/env python3
"""Conservative, explicitly WEAK_RULE train draft; never writes HUMAN sheets.

The 2D thresholds below are unvalidated proxies, not direct visual judgments.
All results require review in Studio. Phase confirmation must be supplied
explicitly; metadata and validation/test annotations are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.contracts.annotation_io import write_annotation_csv
from src.contracts.labels import HIGH, MODERATE, LOW
from scripts.dataset_qa import validate_annotation_dataframe

RULE_VERSION = "weak-studio-v2-pose-draft-20261126.1"
GUIDE = ROOT / "docs/HUONG_DAN_GAN_NHAN_STUDIO_V2.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_frames(p):
    """Raw per-frame cues, retaining missing-point masks and ambiguous poses."""
    k = np.stack([p[[f"kpt_{i:02d}_{c}" for c in ("x", "y", "conf")]].to_numpy(float)
                  for i in range(17)], axis=1)
    seen = np.isfinite(k).all(2) & (k[:, :, 2] >= .30) & (k[:, :, :2] != 0).any(2)
    shoulders = (k[:, 5, :2] + k[:, 6, :2]) / 2
    width = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1)
    observed = seen[:, 0] & seen[:, 5] & seen[:, 6] & (width >= 12)
    observed &= seen[:, 9] | seen[:, 10]
    with np.errstate(divide="ignore", invalid="ignore"):
        hx = (k[:, 0, 0] - shoulders[:, 0]) / np.maximum(width, 1)
        hy = (k[:, 0, 1] - shoulders[:, 1]) / np.maximum(width, 1)
    # No claim of gaze, task content, phone use, writing, or emotion from pose.
    facing = observed & (abs(hx) <= .40) & (hy <= -.20)
    head_down = observed & (hy >= .12)
    turned = observed & (abs(hx) >= .85)
    return np.column_stack([observed, facing, head_down, turned])


def decide(cues, samples, expected):
    """Return draft state/label/behavior plus measured temporal evidence."""
    n = max(1, expected)
    fractions = np.minimum(1, cues.sum(axis=0) / n)
    obs, facing, down, turned = fractions
    sample_missing = 5 - len(samples)
    sample_unseen = sample_missing + sum(not bool(c[0]) for c in samples)
    audit = dict(frame_coverage=min(1, len(cues) / n), observable_fraction=float(obs),
                 facing_proxy_fraction=float(facing), head_down_proxy_fraction=float(down),
                 turned_proxy_fraction=float(turned), unobservable_keyframes=int(sample_unseen))
    if len(cues) == 0:
        return "INVALID", "", ["UNCLEAR"], "NO_TRACK_DATA", audit
    if len(samples) < 3:
        return "INVALID", "", ["UNCLEAR"], "INSUFFICIENT_TRACK_COVERAGE", audit
    if sample_unseen >= 2 or obs < .60:
        return "OCCLUDED", "", ["OCCLUDED_KEYPOINTS"], "INSUFFICIENT_HEAD_SHOULDER_HAND_VISIBILITY_PROXY", audit
    if down >= .75:
        return "VALID", LOW, ["HEAD_DOWN_EXTENDED"], "HEAD_DOWN_PROXY_75_PERCENT", audit
    if turned >= .75:
        return "VALID", LOW, ["TURNING_AWAY"], "TURNED_PROXY_75_PERCENT", audit
    if facing >= .75:
        return "VALID", HIGH, ["BOARD_OR_INSTRUCTOR_FACING"], "FRONT_FACING_PROXY_75_PERCENT", audit
    return "VALID", MODERATE, ["UNCLEAR"], "VISIBLE_NEUTRAL_OR_MIXED_PROXY_REVIEW_REQUIRED", audit


def render_samples(frame, out, pick_indices):
    """Five timestamped crops plus classroom context; no model labels on images."""
    image_dir = out / "review_images"
    image_dir.mkdir(exist_ok=True)
    for sess, group in frame.loc[pick_indices].groupby("session_id", sort=False):
        p = pd.read_csv(ROOT / "data/poses" / f"{sess}_cam_A_poses.csv", dtype={"track_id": str})
        cap = cv2.VideoCapture(str(ROOT / group.iloc[0].video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for idx, row in group.iterrows():
            track = p[p.track_id.eq(row.track_id)]
            times = np.linspace(float(row.t_start_ms), float(row.t_end_ms), 5)
            sheet = np.full((440, 1280, 3), 24, np.uint8)
            cv2.putText(sheet, f'{idx}: {row.window_id}', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
            for j, ts in enumerate(times):
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(ts * fps / 1000))
                ok, raw = cap.read()
                if not ok:
                    continue
                nearest = track.iloc[(track.timestamp_ms - ts).abs().argmin()] if len(track) else None
                x = j * 256
                if nearest is not None and abs(nearest.timestamp_ms - ts) <= 150:
                    x1,y1,x2,y2 = [int(nearest[c]) for c in ("bbox_x1","bbox_y1","bbox_x2","bbox_y2")]
                    w,h=x2-x1,y2-y1
                    crop=raw[max(0,y1-int(h*.12)):min(raw.shape[0],y2+int(h*.12)),
                             max(0,x1-int(w*.18)):min(raw.shape[1],x2+int(w*.18))]
                    if crop.size:
                        scale=min(250/crop.shape[1],265/crop.shape[0])
                        crop=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
                        sheet[58:58+crop.shape[0],x+3:x+3+crop.shape[1]]=crop
                    if j==2:
                        cv2.rectangle(raw,(x1,y1),(x2,y2),(0,255,255),4)
                        overview=cv2.resize(raw,(200,112))
                        sheet[326:438,3:203]=overview
                cv2.putText(sheet,f'{ts/1000:.2f}s',(x+5,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            cv2.imwrite(str(image_dir/f'{idx:05d}.jpg'),sheet)
        cap.release()
        print(f'Rendered review crops: {sess} ({len(group)})',flush=True)


def finalize_draft(out, confirmations_path, decisions_path, *, skip_studio_publish=False):
    """Apply only recorded phase/vision decisions; publish suggestions, not HUMAN."""
    from src.pose.ai_bundle import save_suggestions_map
    frame=pd.read_csv(out/"train_annotations_weak.csv",dtype=str,keep_default_na=False)
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    phases=json.loads(confirmations_path.read_text(encoding="utf-8"))
    decisions=json.loads(decisions_path.read_text(encoding="utf-8"))
    assert set(phases["sessions"])==set(frame.session_id)
    assert phases["class_phase"]=="lecture" and phases["source"]=="USER_CONFIRMATION"
    assert frame.split.eq("train").all()
    source=ROOT/"data/annotations/train_full_ai_assist"
    protected=manifest["source_hashes"]
    assert all(sha(ROOT/p)==digest for p,digest in protected.items())
    audit=pd.read_csv(out/"pose_audit.csv").set_index("window_id")
    # Short recordings initially abstained solely because their phase was unknown.
    for idx,row in frame[frame.session_id.str.startswith("class_")].iterrows():
        m=audit.loc[row.window_id]
        if m.frame_coverage < .60:
            state,label,beh,reason="INVALID","","UNCLEAR","INSUFFICIENT_TRACK_COVERAGE"
        elif m.unobservable_keyframes>=2 or m.observable_fraction<.60:
            state,label,beh,reason="OCCLUDED","","OCCLUDED_KEYPOINTS","INSUFFICIENT_HEAD_SHOULDER_HAND_VISIBILITY_PROXY"
        elif m.head_down_proxy_fraction>=.75:
            state,label,beh,reason="VALID",LOW,"HEAD_DOWN_EXTENDED","HEAD_DOWN_PROXY_75_PERCENT"
        elif m.turned_proxy_fraction>=.75:
            state,label,beh,reason="VALID",LOW,"TURNING_AWAY","TURNED_PROXY_75_PERCENT"
        elif m.facing_proxy_fraction>=.75:
            state,label,beh,reason="VALID",HIGH,"BOARD_OR_INSTRUCTOR_FACING","FRONT_FACING_PROXY_75_PERCENT"
        else:
            state,label,beh,reason="VALID",MODERATE,"UNCLEAR","VISIBLE_NEUTRAL_OR_MIXED_PROXY_REVIEW_REQUIRED"
        for key,value in dict(annotation_state=state,participation_label=label,atomic_behavior=beh,
                label_source="WEAK_RULE" if state=="VALID" else "",
                label_rule_version=RULE_VERSION if state=="VALID" else "",
                occlusion_level="heavy" if state=="OCCLUDED" else "none",
                notes=f"AI DRAFT - HUMAN REVIEW REQUIRED. {reason}. Confidence 0 = uncalibrated.").items():
            frame.at[idx,key]=value
    frame["class_phase"]="lecture"
    frame["phase_verified"]="True"
    frame["phase_source"]="USER_CONFIRMATION"
    frame["phase_evidence"]=phases["evidence"]
    by_id={str(w):i for i,w in enumerate(frame.window_id)}
    assert len({d["window_id"] for d in decisions})==len(decisions)
    for d in decisions:
        idx=by_id[d["window_id"]]
        assert (out/"review_images"/f'{idx:05d}.jpg').is_file()
        state=d["state"]
        label=d.get("label","") if state=="VALID" else ""
        evidence=json.loads(frame.at[idx,"annotation_evidence"])
        evidence.update(method="CODEX_VISION_FIVE_FRAME_REVIEW",human_reviewed=False,
                        five_timestamped_crops_reviewed=True,full_window_video_watched=False,
                        original_pose_proposal=frame.at[idx,"participation_label"],
                        rationale=d["rationale"],needs_review=True)
        for key,value in dict(annotation_state=state,participation_label=label,
                atomic_behavior=";".join(d["behaviors"]),label_source="WEAK_RULE" if state=="VALID" else "",
                label_rule_version="weak-studio-v2-vision-20261126.1" if state=="VALID" else "",
                annotator_id="CODEX_AI_VISION_REVIEW",occlusion_level=d.get("occlusion","none"),
                notes="AI FIVE-FRAME REVIEW; HUMAN REVIEW REQUIRED. "+d["rationale"],
                annotation_evidence=json.dumps(evidence,ensure_ascii=False)).items():
            frame.at[idx,key]=value
    assert frame.label_source.isin(["", "WEAK_RULE"]).all()
    human=pd.read_csv(source/"annotator_a.csv",dtype=str,keep_default_na=False)
    done=human.annotator_id.ne("") & human.annotation_state.isin(["VALID","INVALID","OCCLUDED","TRANSITION"])
    preserved=human[done].set_index("window_id")
    combined=frame.set_index("window_id").copy()
    for wid,row in preserved.iterrows():
        assert wid in combined.index
        for col,value in row.items(): combined.at[wid,col]=value
    combined=combined.reset_index()[frame.columns]
    assert combined[done.to_numpy()].reset_index(drop=True).equals(human[done].reset_index(drop=True))
    for name,data in (("train_annotations_ai.csv",frame),("train_annotations_preserving_human.csv",combined)):
        if (out/name).exists(): raise FileExistsError(out/name)
        ok,errors=validate_annotation_dataframe(data)
        if not ok: raise ValueError(errors[:20])
        write_annotation_csv(out/name,data)
    proposals=[]
    for idx,row in frame.iterrows():
        if row.window_id in preserved.index: continue
        vision=row.annotator_id=="CODEX_AI_VISION_REVIEW"
        img=out/"review_images"/f"{idx:05d}.jpg"
        proposals.append(dict(window_id=row.window_id,suggested_state=row.annotation_state,
            suggested_label=row.participation_label,suggested_behaviors=row.atomic_behavior.split(";"),
            suggested_phase="lecture",suggested_phase_evidence=phases["evidence"],
            suggested_occlusion=row.occlusion_level,confidence=0.0,needs_vision_review=True,
            suggestion_source="CODEX_VISION_STUDIO_V2" if vision else "CODEX_POSE_DRAFT_STUDIO_V2",
            rationale=row.notes,rejection_reason="NEEDS_HUMAN_REVIEW_UNCALIBRATED_AI_DRAFT",
            gate_reason="HUMAN_REVIEW_REQUIRED",model_id=row.label_rule_version or RULE_VERSION,
            bundle_image_path=str(img.relative_to(ROOT)) if img.exists() else ""))
    # Store a separate file recognized by the existing desktop suggestion loader.
    suggestion_path=(out if skip_studio_publish else source/"ai_bundles")/"codex_suggestions.json"
    if suggestion_path.exists() or suggestion_path.with_suffix(".csv").exists():
        raise FileExistsError("Existing Codex suggestions must be preserved; refusing overwrite.")
    save_suggestions_map(suggestion_path,proposals)
    assert all(sha(ROOT/p)==digest for p,digest in protected.items())
    queue=combined[~combined.window_id.isin(preserved.index)].copy()
    queue["review_priority"]=queue.annotation_state.map({"INVALID":0,"TRANSITION":1,"OCCLUDED":2,"VALID":4})
    queue.loc[queue.participation_label.eq(LOW),"review_priority"]=1
    queue.loc[queue.participation_label.eq(MODERATE),"review_priority"]=3
    queue=queue.sort_values(["review_priority","session_id","t_start_ms"],kind="stable",
                           key=lambda s:pd.to_numeric(s) if s.name=="t_start_ms" else s)
    queue[["review_priority","window_id","session_id","track_id","t_start_ms","t_end_ms","class_phase",
           "annotation_state","participation_label","label_source","label_rule_version","atomic_behavior","notes"]].to_csv(
        out/"human_review_queue.csv",index=False)
    report=dict(rows=len(frame),split="train",states=frame.annotation_state.value_counts().to_dict(),
        labels=frame.participation_label.value_counts().to_dict(),human_rows_preserved=len(preserved),
        source_human_files_unchanged=True,source_val_test_files_and_split_files_unchanged=True,
        studio_suggestions_published=len(proposals),five_frame_ai_reviews=len(decisions),
        windows_without_direct_visual_review=len(frame)-len(decisions),human_review_required=True,
        schema_qa_passed=True,full_visual_annotation_complete=False,zone_annotation_complete=False,
        confidence="0.0 denotes uncalibrated AI draft, not a measured probability",
        phase_confirmation_sha256=sha(confirmations_path),visual_decisions_sha256=sha(decisions_path),
        combined_draft_states=combined.annotation_state.value_counts().to_dict(),
        combined_draft_labels=combined.participation_label.value_counts().to_dict(),
        combined_draft_label_sources=combined.label_source.value_counts().to_dict(),
        rule_script_sha256=sha(Path(__file__)),
        pose_source_sha256={str(Path("data/poses")/f"{sess}_cam_A_poses.csv"):
                           sha(ROOT/"data/poses"/f"{sess}_cam_A_poses.csv") for sess in frame.session_id.unique()},
        outputs={name:sha(out/name) for name in ("train_annotations_ai.csv","train_annotations_preserving_human.csv")})
    (out/"qa_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lecture-confirmation", help="Verbatim user confirmation; applies only to session_001/002.")
    ap.add_argument("--out-dir", default="data/annotations/train_codex_studio_v2_20261126")
    ap.add_argument("--finalize",action="store_true")
    ap.add_argument("--phase-confirmations",type=Path)
    ap.add_argument("--review-decisions",type=Path)
    ap.add_argument("--skip-studio-publish",action="store_true",help="Write suggestions inside --out-dir for isolated reproduction.")
    args=ap.parse_args()
    out=ROOT/args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    if args.finalize:
        if not args.phase_confirmations or not args.review_decisions:
            ap.error("--finalize requires --phase-confirmations and --review-decisions")
        finalize_draft(out,args.phase_confirmations,args.review_decisions,skip_studio_publish=args.skip_studio_publish)
        return
    if not args.lecture_confirmation: ap.error("--lecture-confirmation is required")
    target=out/"train_annotations_weak.csv"
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}; choose another --out-dir.')
    template=ROOT/"data/annotations/train_full_ai_assist/annotation_template.csv"
    frame=pd.read_csv(template,dtype=str,keep_default_na=False)
    train_sessions=set((ROOT/"data/splits/train_sessions.txt").read_text().split())
    assert frame.split.eq("train").all() and set(frame.session_id)<=train_sessions
    assert frame.window_id.is_unique
    protected=[ROOT/"data/annotations"/f"{s}_full_ai_assist"/f"annotator_{who}.csv"
               for s in ("train","val","test") for who in ("a","b")]
    protected += list((ROOT/"data/splits").glob("*sessions.txt"))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    audit=[]
    for sess,group in frame.groupby("session_id",sort=False):
        path=ROOT/"data/poses"/f"{sess}_cam_A_poses.csv"
        pose=pd.read_csv(path,dtype={"track_id":str,"camera_id":str})
        assert pose.session_id.eq(sess).all() and pose.camera_id.eq("cam_A").all()
        cues=classify_frames(pose)
        cache={}
        for tid,g in pose.groupby("track_id",sort=False):
            order=np.argsort(g.timestamp_ms.to_numpy())
            cache[tid]=(g.timestamp_ms.to_numpy()[order], cues[g.index][order])
        for idx,row in group.iterrows():
            times,cs=cache.get(row.track_id,(np.array([]),np.empty((0,4))))
            start,end=float(row.t_start_ms),float(row.t_end_ms)
            subset=cs[(times>=start-.01)&(times<=end+.01)]
            expected=round((end-start)/100)+1
            samples=[]
            for t in np.linspace(start,end,5):
                if len(times):
                    near=np.argmin(abs(times-t))
                    if abs(times[near]-t)<=75:
                        samples.append(cs[near])
            state,label,behaviors,reason,metrics=decide(subset,samples,expected)
            confirmed=sess in {"session_001","session_002"}
            phase="lecture" if confirmed else "UNKNOWN"
            evidence=args.lecture_confirmation if confirmed else ""
            if not confirmed:
                # Short-video phase is visually reviewed separately; do not assume it.
                state,label,reason="INVALID","","PHASE_REQUIRES_VISUAL_REVIEW"
            updates=dict(class_phase=phase,phase_source="USER_CONFIRMATION" if confirmed else "UNKNOWN",
                         phase_verified=str(confirmed),phase_evidence=evidence,annotation_state=state,
                         participation_label=label,label_source="WEAK_RULE" if state=="VALID" else "",
                         label_rule_version=RULE_VERSION if state=="VALID" else "",atomic_behavior=";".join(behaviors),
                         annotator_id="CODEX_AI_WEAK_PREANNOTATION",annotation_confidence="0.0",
                         occlusion_level="heavy" if state=="OCCLUDED" else "none",
                         annotation_guide_version="v0.2",
                         notes=f"AI DRAFT - HUMAN REVIEW REQUIRED. {reason}. Confidence 0 = uncalibrated, not probability.",
                         annotation_evidence=json.dumps(dict(method="AUTOMATED_POSE_PROXY",rule_version=RULE_VERSION,
                             guide_sha256=sha(GUIDE),human_reviewed=False,full_window_visually_reviewed=False,
                             needs_review=True,confidence_calibrated=False,**metrics),ensure_ascii=False))
            for k,v in updates.items(): frame.at[idx,k]=v
            audit.append(dict(window_id=row.window_id,session_id=sess,rule_reason=reason,**metrics))
        print(f'Processed {sess}: {len(group)} train windows',flush=True)
    ok,errors=validate_annotation_dataframe(frame)
    if not ok: raise ValueError(errors[:20])
    write_annotation_csv(target,frame)
    ad=pd.DataFrame(audit)
    ad.to_csv(out/"pose_audit.csv",index=False)
    # Reproducible review sample: earliest instance of each reason/session plus
    # five evenly spaced windows/session. No sampling is used to claim accuracy.
    picks=set(ad.groupby(["session_id","rule_reason"],sort=False).head(1).index)
    for _,g in frame.groupby("session_id",sort=False):
        picks.update(g.index[np.linspace(0,len(g)-1,5).round().astype(int)])
    picks=sorted(picks)
    frame.loc[picks,["window_id","session_id","track_id","t_start_ms","t_end_ms"]].to_csv(out/"review_sample.csv",index=True,index_label="row_index")
    render_samples(frame,out,picks)
    after={str(p.relative_to(ROOT)):sha(p) for p in protected}
    assert before==after, 'Protected source changed during run; inspect before publication.'
    report=dict(rule_version=RULE_VERSION,rows=len(frame),states=frame.annotation_state.value_counts().to_dict(),
                labels=frame.participation_label.value_counts().to_dict(),source_template_sha256=sha(template),
                guide_sha256=sha(GUIDE),lecture_confirmation=args.lecture_confirmation,
                git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                source_hashes=before,protected_files_unchanged=True,schema_qa_passed=True,
                visual_review_sample_indices=picks,full_visual_annotation_complete=False,
                limitation="Unvalidated pose proxy draft; all windows need human review; no accuracy or Kappa claimed.")
    (out/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("rows","states","labels","schema_qa_passed")},ensure_ascii=False),flush=True)


if __name__=="__main__": main()
