"""CLI tool to validate session metadata and folder layout (Task T09).

Owner: Vũ Văn Phong (Data & Annotation QA)
Usage:
    python scripts/validate_metadata.py --session data/raw/session_001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_METADATA_FIELDS = {
    "session_id",
    "date",
    "camera_id",
    "fps",
    "resolution",
    "camera_view",
    "participants",
    "class_phase",
    "lighting_condition",
    "consent_confirmed",
    "notes",
}

VALID_VIEWS = {"front", "front_oblique", "side", "rear", "overhead"}
VALID_PHASES = {"lecture", "individual_work", "discussion", "break", "presentation"}


def validate_session(session_path: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not session_path.exists() or not session_path.is_dir():
        return False, [f"Session directory does not exist: {session_path}"]

    # 1. Check metadata.json existence
    metadata_file = session_path / "metadata.json"
    if not metadata_file.exists():
        errors.append(f"Missing mandatory file: {metadata_file}")
        return False, errors

    try:
        data: Dict[str, Any] = json.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Invalid JSON format in {metadata_file}: {exc}"]

    # 2. Check required fields
    missing_fields = REQUIRED_METADATA_FIELDS - set(data.keys())
    if missing_fields:
        errors.append(f"Missing required metadata fields: {sorted(missing_fields)}")

    # 3. Check values
    if "session_id" in data and not str(data["session_id"]).strip():
        errors.append("Field 'session_id' must not be empty.")

    if "consent_confirmed" in data and not isinstance(data["consent_confirmed"], bool):
        errors.append("Field 'consent_confirmed' must be a boolean (true/false).")

    if "fps" in data:
        try:
            fps = float(data["fps"])
            if fps <= 0:
                errors.append("Field 'fps' must be > 0.")
        except (ValueError, TypeError):
            errors.append("Field 'fps' must be a number.")

    if "participants" in data:
        try:
            p = int(data["participants"])
            if p < 0:
                errors.append("Field 'participants' must be >= 0.")
        except (ValueError, TypeError):
            errors.append("Field 'participants' must be an integer.")

    # 4. Check class_phase timeline
    phases = data.get("class_phase")
    if not isinstance(phases, list) or len(phases) == 0:
        errors.append("Field 'class_phase' must be a non-empty list of phases.")
    else:
        for idx, phase in enumerate(phases):
            if not isinstance(phase, dict):
                errors.append(f"Phase #{idx} must be a JSON object.")
                continue
            name = phase.get("name")
            start = phase.get("start_ms")
            end = phase.get("end_ms")

            if not name or not isinstance(name, str):
                errors.append(f"Phase #{idx} missing valid 'name'.")
            if start is None or not isinstance(start, (int, float)) or start < 0:
                errors.append(f"Phase #{idx} ({name}) invalid 'start_ms' ({start}). Must be >= 0.")
            if end is None or not isinstance(end, (int, float)) or end <= (start or 0):
                errors.append(f"Phase #{idx} ({name}) invalid 'end_ms' ({end}). Must be > start_ms.")

    # 5. Check video files existence
    video_files = list(session_path.glob("*.mp4")) + list(session_path.glob("*.mov")) + list(session_path.glob("*.avi"))
    if not video_files:
        errors.append(f"Warning: No video files (.mp4, .mov, .avi) found in {session_path}")

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate session metadata and layout.")
    parser.add_argument("--session", "-s", type=str, required=True, help="Path to session directory")
    args = parser.parse_args()

    session_path = Path(args.session)
    print(f"[INFO] Validating session layout and metadata: {session_path} ...")
    valid, errors = validate_session(session_path)

    if valid:
        print(f"[OK] Session '{session_path.name}' conforms to docs/dataset_schema.md!")
        sys.exit(0)
    else:
        print(f"[FAIL] Found {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
