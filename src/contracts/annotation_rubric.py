"""Operational pilot rubric, requiring direct human observations, not model output."""
from src.contracts.labels import HIGH, MODERATE, LOW

RUBRIC_VERSION = "v0.3-pilot"
EVIDENCE_OPTIONS = {
    "ALIGNED": "Hành vi rõ phù hợp nhiệm vụ",
    "CONTRADICTORY": "Hành vi rõ không phù hợp nhiệm vụ",
    "AMBIGUOUS": "Nhìn thấy nhưng chưa rõ có phù hợp",
    "UNOBSERVABLE": "Bị che / không xác định đúng người",
}


def evidence_label(observations):
    """Four equal subintervals; pilot thresholds are project choices, not validated norms."""
    if len(observations) != 4 or any(value not in EVIDENCE_OPTIONS for value in observations):
        return None, "Cần đánh dấu đủ bốn khoảng quan sát."
    aligned = observations.count("ALIGNED")
    contradictory = observations.count("CONTRADICTORY")
    if aligned + contradictory < 3:
        return None, "Chưa có bằng chứng rõ trong ít nhất 3/4 khoảng; không ép nhãn tham gia."
    if aligned >= 3:
        return HIGH, "Ít nhất 3/4 khoảng có hành vi rõ phù hợp nhiệm vụ."
    if contradictory >= 3:
        return LOW, "Ít nhất 3/4 khoảng có hành vi rõ không phù hợp nhiệm vụ."
    return MODERATE, "Có bằng chứng rõ ở ít nhất 3/4 khoảng, nhưng hành vi phù hợp/không phù hợp xen kẽ."
