"""Canonical label and annotation provenance contract for CVF25PRJ10.

All persistent dataset/model labels use the long canonical values below.
Short HIGH/MODERATE/LOW values are presentation-only labels.
"""

from __future__ import annotations

from typing import Optional

HIGH = "HIGH_OBSERVED_PARTICIPATION"
MODERATE = "MODERATE_OBSERVED_PARTICIPATION"
LOW = "LOW_OBSERVED_PARTICIPATION"
UNCERTAIN_LABEL = "UNCERTAIN"

CANONICAL_TRAINING_LABELS = (HIGH, MODERATE, LOW)
LABEL_TO_INDEX = {label: index for index, label in enumerate(CANONICAL_TRAINING_LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

DISPLAY_LABEL_BY_CANONICAL = {
    HIGH: "HIGH",
    MODERATE: "MODERATE",
    LOW: "LOW",
    UNCERTAIN_LABEL: "UNCERTAIN",
}
CANONICAL_BY_DISPLAY_LABEL = {
    display: canonical
    for canonical, display in DISPLAY_LABEL_BY_CANONICAL.items()
    if canonical != UNCERTAIN_LABEL
}

ANNOTATION_STATES = ("VALID", "OCCLUDED", "TRANSITION", "INVALID")
LABEL_SOURCES = ("HUMAN", "WEAK_RULE", "ADJUDICATED")


def to_display_label(canonical_label: str) -> str:
    """Map a canonical persisted/model label to the short UI/report label."""
    try:
        return DISPLAY_LABEL_BY_CANONICAL[canonical_label]
    except KeyError as exc:
        raise ValueError(f"Unknown canonical label: {canonical_label!r}") from exc


def validate_annotation_label(
    *,
    participation_label: Optional[str],
    annotation_state: str,
    label_source: Optional[str],
    label_rule_version: Optional[str] = None,
) -> None:
    """Validate state-label-provenance relationships.

    Rules:
    - VALID rows require one canonical 3-class label and a label source.
    - Non-VALID rows must not carry a participation label or label source.
    - WEAK_RULE labels require label_rule_version.
    - HUMAN/ADJUDICATED labels must not claim a weak rule version.
    """
    label = (participation_label or "").strip()
    source = (label_source or "").strip()
    rule_version = (label_rule_version or "").strip()

    if annotation_state not in ANNOTATION_STATES:
        raise ValueError(f"Unknown annotation_state: {annotation_state!r}")

    if annotation_state == "VALID":
        if label not in CANONICAL_TRAINING_LABELS:
            raise ValueError(
                "VALID rows require participation_label to be one of "
                f"{CANONICAL_TRAINING_LABELS}; got {label!r}"
            )
        if source not in LABEL_SOURCES:
            raise ValueError(
                f"VALID rows require label_source in {LABEL_SOURCES}; got {source!r}"
            )
        if source == "WEAK_RULE" and not rule_version:
            raise ValueError("WEAK_RULE labels require label_rule_version")
        if source in {"HUMAN", "ADJUDICATED"} and rule_version:
            raise ValueError(
                f"{source} labels must leave label_rule_version empty; got {rule_version!r}"
            )
    else:
        if label:
            raise ValueError(
                f"{annotation_state} rows must leave participation_label empty; got {label!r}"
            )
        if source:
            raise ValueError(
                f"{annotation_state} rows must leave label_source empty; got {source!r}"
            )
        if rule_version:
            raise ValueError(
                f"{annotation_state} rows must leave label_rule_version empty; got {rule_version!r}"
            )
