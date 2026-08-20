"""Native, page-by-page PDF extraction using PyMuPDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, cast

from .quality_gate import assess_text_quality, normalize_text


def _pymupdf():
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        try:
            import fitz

            return fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required. Install requirements.txt."
            ) from exc


def extract_document(document: dict, source_root: Path) -> Iterator[dict]:
    """Yield one provenance-rich native extraction record per PDF page."""
    pymupdf = _pymupdf()
    relative_path = Path(document["path"])
    source_path = source_root / relative_path

    with pymupdf.open(source_path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            raw_text = cast(str, page.get_text("text", sort=True))
            text_payload = cast(dict[str, Any], page.get_text("dict", sort=True))
            native_text = normalize_text(raw_text)
            images = page.get_images(full=True)
            blocks = text_payload.get("blocks", [])
            quality = assess_text_quality(native_text, len(images))
            doc_id = document["doc_id"]
            yield {
                "record_id": f"{doc_id}:p{page_index + 1}",
                "doc_id": doc_id,
                "source_id": doc_id,
                "scope": document["scope"],
                "ocr_lang": document["ocr_lang"],
                "source_path": f"data/source_documents/{relative_path.as_posix()}",
                "source_sha256": document["sha256"],
                "page_index": page_index,
                "pdf_page": page_index + 1,
                "native_text": native_text,
                "native_block_count": len(blocks),
                **quality,
            }


def page_count(path: Path) -> int:
    pymupdf = _pymupdf()
    with pymupdf.open(path) as pdf:
        return pdf.page_count
