"""Heuristic selection and rendering of pages for the expensive VLM pass."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


VISUAL_KEYWORDS = re.compile(
    r"\b(?:figure|diagram|illustration|shown|see\s+figure|photo|image|"
    r"hand\s+placement|body\s+position|recovery\s+position|"
    r"compression\s+technique|bandaging\s+technique|splint\s+application|"
    r"tourniquet\s+placement)\b",
    re.IGNORECASE,
)


def candidate_decision(record: dict, forced_ids: set[str] | None = None) -> tuple[bool, list[str]]:
    """Apply the frozen strong-candidate logic and expose all supporting signals."""
    forced_ids = forced_ids or set()
    reasons: list[str] = []
    record_id = str(record["record_id"])
    chars = int(record.get("character_count", 0))
    image_count = int(record.get("embedded_image_count", 0))
    block_count = int(record.get("native_block_count", 0))
    method = str(record.get("selected_method", ""))
    text = str(record.get("selected_text") or "")

    embedded = image_count > 0
    keyword = bool(VISUAL_KEYWORDS.search(text))
    extraction_problem = method in {"native_fallback", "extraction_failed"}
    ocr_routed = method == "paddleocr"
    image_heavy = embedded and 100 <= chars < 350
    layout_signal = block_count >= 8 or bool(re.search(r"(?:^|\n)\s*(?:\d+[.)]|[•●▪])\s+", text))
    is_cover = int(record.get("pdf_page", 0)) <= 2 and chars < 300
    probably_blank = chars < 30 and not embedded
    decorative_low_text = chars < 100 and not extraction_problem and not embedded

    if embedded:
        reasons.append("embedded_image_present")
    if keyword:
        reasons.append("instructional_visual_keyword")
    if layout_signal:
        reasons.append("layout_signal")
    if ocr_routed:
        reasons.append("ocr_routed")
    if image_heavy:
        reasons.append("image_heavy_page")
    if extraction_problem:
        reasons.append("extraction_problem")

    strong_candidate = (
        not is_cover
        and not probably_blank
        and not decorative_low_text
        and (
            image_heavy
            or (embedded and keyword)
            or (extraction_problem and (embedded or chars >= 30))
        )
    )

    if record_id in forced_ids:
        reasons.append("manual_review_include")
        strong_candidate = True
    return strong_candidate, reasons


def render_page(record: dict, project_root: Path, output_dir: Path, dpi: int = 180) -> Path:
    """Render a page when extraction did not already produce a usable image."""
    import pymupdf

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f'{record["doc_id"]}__p{int(record["pdf_page"]):04d}.png'
    if target.exists():
        return target.resolve()

    existing = record.get("page_image_path")
    if existing and Path(str(existing)).exists():
        return Path(str(existing)).resolve()

    source = project_root / str(record["source_path"])
    with pymupdf.open(source) as pdf:
        page = pdf.load_page(int(record["page_index"]))
        page.get_pixmap(dpi=dpi, alpha=False).save(target)
    return target.resolve()


def build_candidate_queue(
    records: Iterable[dict],
    project_root: Path,
    forced_ids: set[str] | None = None,
    render_dpi: int = 180,
    output_dir: Path | None = None,
) -> list[dict]:
    output_dir = output_dir or project_root / "data" / "vision" / "page_images"
    queue: list[dict] = []
    for record in records:
        selected, reasons = candidate_decision(record, forced_ids)
        if not selected:
            continue
        image_path = render_page(record, project_root, output_dir, render_dpi)
        queue.append(
            {
                "visual_id": f'{record["record_id"]}:page-render',
                "record_id": record["record_id"],
                "doc_id": record["doc_id"],
                "pdf_page": record["pdf_page"],
                "page_index": record["page_index"],
                "image_path": str(image_path),
                "candidate_reasons": reasons,
                "review_status": "pending",
            }
        )
    return queue
