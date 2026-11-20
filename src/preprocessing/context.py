"""Attach recording context without turning generated metadata into verified facts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PHASES = {"lecture", "individual_work", "discussion"}


def attach_session_context(poses: pd.DataFrame, raw_dir: Path, *, allow_unverified: bool = False):
    """Use half-open phase intervals. Unverified declarations stay explicit.

    Legacy split_video metadata alternated phases automatically; it must not be
    treated as human-verified context. Set phase_verified=true only after review.
    """
    out = poses.copy()
    audit = []
    out["declared_class_phase"] = out.get("class_phase", pd.Series("UNKNOWN", index=out.index)).fillna("UNKNOWN")
    out["class_phase"] = "UNKNOWN"
    out["phase_source"] = "MISSING"
    out["phase_verified"] = False
    for session_id, rows in out.groupby("session_id"):
        metadata_path = raw_dir / str(session_id) / "metadata.json"
        if not metadata_path.is_file():
            audit.append({"session_id": str(session_id), "status": "MISSING_METADATA"})
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(meta.get("session_id")) != str(session_id):
            raise ValueError(f"Metadata session mismatch: {metadata_path}")
        intervals = sorted(meta.get("class_phase", []), key=lambda p: p["start_ms"])
        previous_end = -1
        for phase in intervals:
            start, end = int(phase["start_ms"]), int(phase["end_ms"])
            if start < 0 or end <= start or start < previous_end:
                raise ValueError(f"Invalid/overlapping phase intervals: {metadata_path}")
            previous_end = end
            name = phase["name"]
            if name not in PHASES | {"UNKNOWN"}:
                raise ValueError(f"Unknown phase {name!r}: {metadata_path}")
            verified = phase.get("verified", meta.get("phase_verified", False)) is True
            idx = rows.index[(rows["timestamp_ms"] >= start) & (rows["timestamp_ms"] < end)]
            out.loc[idx, "declared_class_phase"] = name
            out.loc[idx, "phase_verified"] = verified
            out.loc[idx, "phase_source"] = "METADATA_VERIFIED" if verified else "METADATA_UNVERIFIED"
            if verified or allow_unverified:
                out.loc[idx, "class_phase"] = name
        audit.append({
            "session_id": str(session_id), "metadata": str(metadata_path),
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "phase_verified": bool(meta.get("phase_verified", False)),
            "status": "REVIEW_REQUIRED" if not meta.get("phase_verified", False) else "VERIFIED_DECLARATION",
            "source_recording_id": meta.get("source_recording_id"),
        })
    return out, audit
