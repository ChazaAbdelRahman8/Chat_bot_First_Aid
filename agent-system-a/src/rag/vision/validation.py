"""Normalize and deterministically accept/reject VLM output."""

from __future__ import annotations

from typing import Any


VISUAL_TYPES = {
    "instructional_diagram", "medical_photo", "anatomy_diagram", "flowchart",
    "table", "chart", "mixed_visual_page", "text_page", "publication_details",
    "decorative_image", "blank_image", "other",
}
LIST_FIELDS = (
    "visible_labels", "demonstrated_actions", "action_sequence",
    "warnings_visible", "ambiguities",
)
REJECT_TYPES = {"publication_details", "decorative_image", "blank_image"}


def _clean_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        if cleaned.lower() in {"", "none", "n/a", "not applicable", "no visible warnings"}:
            return []
        return [cleaned]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array or scalar string")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"every {field} array item must be a JSON string")
    return [_clean_text(item) for item in value if _clean_text(item)]


def _clean_text(value: str) -> str:
    """Repair common UTF-8-as-Windows-1252 artifacts without altering meaning."""
    replacements = {
        "â€“": "–", "â€”": "—", "â€™": "’", "â€œ": "“", "â€\u009d": "”", "Â": "",
    }
    cleaned = value.strip()
    for broken, repaired in replacements.items():
        cleaned = cleaned.replace(broken, repaired)
    return cleaned


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a JSON boolean")
    return value


def normalize_description(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("VLM response must be a JSON object")
    visual_type = str(raw.get("visual_type", "")).strip().lower()
    if "|" in visual_type:
        combined_types = {part.strip() for part in visual_type.split("|") if part.strip()}
        if len(combined_types) >= 2 and combined_types.issubset(VISUAL_TYPES):
            visual_type = "mixed_visual_page"
    if visual_type not in VISUAL_TYPES:
        raise ValueError(f"unsupported visual_type: {visual_type!r}")
    instructional_value = str(raw.get("instructional_value", "")).strip().lower()
    if instructional_value not in {"high", "medium", "low", "none"}:
        raise ValueError("invalid instructional_value")

    result = {
        "visual_type": visual_type,
        "description": _clean_text(str(raw.get("description", ""))),
        **{field: _clean_list(raw.get(field), field) for field in LIST_FIELDS},
        "ambiguous_content": _boolean(raw.get("ambiguous_content"), "ambiguous_content"),
        "review_required": _boolean(raw.get("review_required"), "review_required"),
        "instructional_value": instructional_value,
        "should_include_in_rag": _boolean(raw.get("should_include_in_rag"), "should_include_in_rag"),
    }
    if result["ambiguities"]:
        result["ambiguous_content"] = True
        result["review_required"] = True
    if visual_type in REJECT_TYPES:
        result["should_include_in_rag"] = False
        result["instructional_value"] = "none"
    elif visual_type in {
        "instructional_diagram", "medical_photo", "anatomy_diagram", "flowchart",
        "table", "chart", "mixed_visual_page",
    } and (result["description"] or result["visible_labels"]):
        result["should_include_in_rag"] = True
    return result


def classify_description(description: dict) -> tuple[bool, str]:
    visual_type = description["visual_type"]
    if visual_type in REJECT_TYPES:
        return False, f"rejected_{visual_type}"
    if description["review_required"] or description["ambiguous_content"]:
        return bool(description["should_include_in_rag"]), "manual_review_required"
    if not description["should_include_in_rag"]:
        return False, "rejected_low_value"
    has_evidence = bool(
        description["description"]
        or any(description[field] for field in LIST_FIELDS)
    )
    if not has_evidence:
        return False, "rejected_empty_description"
    return True, "processed"
