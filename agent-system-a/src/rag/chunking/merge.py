"""Merge extracted text with human-reviewed visual supplements by page ID."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Callable, Iterable


ELIGIBLE_VISION_STATUSES = {"processed", "human_approved", "human_edited"}


def vision_is_eligible(record: dict) -> bool:
    return (
        record.get("review_status") in ELIGIBLE_VISION_STATUSES
        and record.get("vision_useful") is True
        and record.get("error") is None
        and isinstance(record.get("description"), dict)
    )


def render_vision_supplement(record: dict) -> str:
    """Render only validated visual evidence; ambiguity notes are audit metadata."""
    description = record["description"]
    fields = (
        ("visual_type", "Visual type"),
        ("description", "Visual description"),
        ("visible_labels", "Visible labels"),
        ("demonstrated_actions", "Visually demonstrated actions"),
        ("action_sequence", "Visually ordered steps"),
        ("warnings_visible", "Visible warnings"),
    )
    #which visual record produced it and which source authority it belongs to
    lines = [
        f'[Validated visual supplement | visual_id={record["visual_id"]} | authority={record["authority"]}]'
    ]
    for key, label in fields:
        value = description.get(key)
        if value in (None, "", []):
            continue
        rendered = "; ".join(value) if isinstance(value, list) else str(value)
        lines.append(f"{label}: {rendered}")
    return "\n".join(lines)


def _footer_page_label(text: str) -> str | None:
    patterns = (
        r"(?i)\bpage\s+(\d{1,4})\s+of\s+\d{1,4}\b",
        r"(?i)\bpage\s+(\d{1,4})\b",
        r"(?i)responding\s+to\s+emergencies\s*\|\s*(\d{1,4})\s*\|",
        r"(?i)first\s+aid\s*\|\s*(\d{1,4})\b",
        r"\|\s*(\d{1,4})\s*(?:\||$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    standalone = re.findall(r"(?m)^\s*(\d{1,4})\s*$", text)
    return standalone[-1] if standalone else None


def load_printed_page_labels(page_records: Iterable[dict], project_root: Path) -> dict[str, str]:
    """Read PDF PageLabels when available, otherwise use the physical page number."""
    import pymupdf

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in page_records:
        grouped[str(record["source_path"])].append(record)

    labels: dict[str, str] = {}
    for source_path, records in grouped.items():
        with pymupdf.open(project_root / source_path) as pdf:
            for record in records:
                page = pdf.load_page(int(record["page_index"]))
                label = str(page.get_label() or "").strip()
                if not label:
                    height = page.rect.height
                    footer_text = "\n".join(
                        str(block[4])
                        for block in page.get_text("blocks")
                        if float(block[1]) >= height * 0.75
                    )
                    label = _footer_page_label(footer_text) or ""
                labels[record["record_id"]] = label or str(record["pdf_page"])
    return labels


def merge_pages(
    page_records: list[dict],
    vision_records: list[dict],
    page_label_for: Callable[[dict], str],
) -> tuple[list[dict], dict]:
    page_ids = [row["record_id"] for row in page_records]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("duplicate text-page record_id")
    visual_ids = [row["visual_id"] for row in vision_records]
    if len(visual_ids) != len(set(visual_ids)):
        raise ValueError("duplicate visual_id")
    unknown_pages = sorted({row["record_id"] for row in vision_records} - set(page_ids))
    if unknown_pages:
        raise ValueError(f"vision records reference unknown pages: {unknown_pages[:10]}")

    vision_by_page: dict[str, list[dict]] = defaultdict(list)
    for visual in vision_records:
        if vision_is_eligible(visual):
            vision_by_page[visual["record_id"]].append(visual)

    merged: list[dict] = []
    used_visual_ids: list[str] = []
    for page in page_records:
        visuals = vision_by_page.get(page["record_id"], [])
        visual_text = [render_vision_supplement(record) for record in visuals]
        selected_text = str(page.get("selected_text") or "").strip()
        knowledge_parts = ([selected_text] if selected_text else []) + visual_text
        used_visual_ids.extend(record["visual_id"] for record in visuals)
        if selected_text and visuals:
            content_type = "text+vision"
        elif visuals:
            content_type = "vision"
        else:
            content_type = "text"
        merged.append(
            {
                "record_id": page["record_id"],
                "doc_id": page["doc_id"],
                "source_id": page.get("source_id", page["doc_id"]),
                "scope": page["scope"],
                "source_path": page["source_path"],
                "source_sha256": page["source_sha256"],
                "page": page_label_for(page),
                "pdf_page": page["pdf_page"],
                "page_index": page["page_index"],
                "language": page["ocr_lang"],
                "content_type": content_type,
                "knowledge_text": "\n\n".join(knowledge_parts),
                "selected_text": selected_text,
                "selected_method": page["selected_method"],
                "quality_pass": page["quality_pass"],
                "quality_reasons": page.get("quality_reasons", []),
                "extraction_manual_review_required": page.get("manual_review_required", False),
                "has_vision": bool(visuals),
                "vision_ids": [record["visual_id"] for record in visuals],
                "vision_statuses": [record["review_status"] for record in visuals],
                "vision_authorities": [record["authority"] for record in visuals],
                "vision_supplements": [record["description"] for record in visuals],
            }
        )
    summary = {
        "text_pages": len(page_records),
        "reviewed_vision_records": len(vision_records),
        "eligible_vision_records": sum(vision_is_eligible(row) for row in vision_records),
        "used_vision_records": len(used_visual_ids),
        "pages_with_vision": sum(row["has_vision"] for row in merged),
        "text_only_pages": sum(row["content_type"] == "text" for row in merged),
        "text_vision_pages": sum(row["content_type"] == "text+vision" for row in merged),
        "vision_only_pages": sum(row["content_type"] == "vision" for row in merged),
        "empty_knowledge_pages": sum(not row["knowledge_text"] for row in merged),
    }
    return merged, summary
