"""Structure-aware, page-provenance-preserving chunk creation."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from dataclasses import dataclass
from typing import Protocol


TARGET_TOKENS = 450
MAX_TOKENS = 650
OVERLAP_TOKENS = 80
MIN_DIAGNOSTIC_TOKENS = 40
VARIANT = "structure_aware"
#This regex recognizes words that commonly begin headings.
HEADING_PATTERN = re.compile(
    r"^(?:chapter|section|module|part|lesson|unit|appendix|skill|procedure|"
    r"assessment|management|warning|trauma|breathing|circulation|shock|"
    r"الفصل|القسم|الوحدة|الدرس|الملحق|المهارة|الإجراء|التقييم|التحذير)\b",
    re.IGNORECASE,
)
STRICT_HEADING_PATTERN = re.compile(
    r"^(?:chapter|section|module|part|lesson|unit|appendix)\s+"
    r"(?:\d+|[ivxlcdm]+|[a-z])(?:\b|[:.\-])",
    re.IGNORECASE,
)

IGNORED_HEADINGS = {
    "participant workbook", "first aid", "intro", "abcde", "ams", "skills",
    "glossary", "refs & quick cards", "contents", "table of contents",
    "chapter", "section", "module", "part", "lesson", "unit", "appendix",
}
IGNORED_HEADING_PATTERNS = (
    re.compile(r"^lrc\s+ems\s+.*curriculum$", re.IGNORECASE),
    re.compile(r"^first\s+aid.*\|.*\d+", re.IGNORECASE),
    re.compile(r"^page\s+\d+", re.IGNORECASE),
    re.compile(r"^©|copyright", re.IGNORECASE),
)


class TokenEncoding(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class Unit:
    text: str
    section: str


def normalize_for_chunking(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_count(text: str, encoding: TokenEncoding) -> int:
    return len(encoding.encode(text))


def clamp_to_token_limit(text: str, limit: int, encoding: TokenEncoding) -> str:
    text = text.strip()
    for _ in range(8):
        tokens = encoding.encode(text)
        if len(tokens) <= limit:
            return text
        text = encoding.decode(tokens[: max(1, limit - 1)]).strip()
    while text and token_count(text, encoding) > limit:
        text = text[:-1].rstrip()
    return text


def stable_chunk_id(record_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{VARIANT}:{record_id}:c{index:03d}:{digest}"


def is_heading(line: str) -> bool:
    value = line.strip().strip("#").replace("\u200b", "")
    if not value or value.lower() in IGNORED_HEADINGS or len(value) > 120:
        return False
    if any(pattern.search(value) for pattern in IGNORED_HEADING_PATTERNS):
        return False
    words = value.split()
    if len(words) > 8 or value.startswith("["):
        return False
    if re.search(r"[.!?؟;]", value) or re.search(r"\b(?:19|20)\d{2}\b", value):
        return False
    if STRICT_HEADING_PATTERN.match(value):
        return True
    letters = [char for char in value if char.isalpha()]
    if bool(letters) and value.upper() == value:
        return True
    alpha_words = [word.strip("():/-") for word in words if any(char.isalpha() for char in word)]
    title_words = [
        word for word in alpha_words
        if word and (word[0].isupper() or word.lower() in {"and", "of", "to", "the", "in", "for"})
    ]
    if 2 <= len(alpha_words) <= 8 and len(title_words) / len(alpha_words) >= 0.8:
        return True
    # Manuals frequently use sentence-case topic labels as standalone paragraphs.
    return (
        2 <= len(alpha_words) <= 7
        and len(value) <= 80
        and value[0].isupper()
    )


def infer_page_section(text: str, previous: str | None = None) -> str:
    normalized = normalize_for_chunking(text)
    # Running headers/footers identify these back-matter regions more reliably
    # than glossary table cells or wrapped bibliography lines do.
    for label in ("Glossary", "References", "Index"):
        if re.search(rf"(?im)^\s*{label}(?:\s*\||\s*$)", normalized):
            return label
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    for paragraph in paragraphs[:20]:
        paragraph_lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        first_line = paragraph_lines[0]
        if is_heading(first_line):
            return first_line
        # Some manuals put a repeated running header directly above the real
        # section title without a blank line.
        if any(pattern.search(first_line) for pattern in IGNORED_HEADING_PATTERNS):
            for line in paragraph_lines[1:3]:
                if is_heading(line):
                    return line
    return previous or "Document"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?؟])\s+|(?<=:)\n", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _token_slices(text: str, limit: int, encoding: TokenEncoding) -> list[str]:
    tokens = encoding.encode(text)
    return [
        encoding.decode(tokens[index : index + limit]).strip()
        for index in range(0, len(tokens), limit)
        if tokens[index : index + limit]
    ]


def split_oversized_unit(unit: Unit, limit: int, encoding: TokenEncoding) -> list[Unit]:
    if token_count(unit.text, encoding) <= limit:
        return [unit]
    sentences = split_sentences(unit.text)
    if len(sentences) <= 1:
        return [Unit(piece, unit.section) for piece in _token_slices(unit.text, limit, encoding) if piece]

    pieces: list[Unit] = []
    current: list[str] = []
    for sentence in sentences:
        if token_count(sentence, encoding) > limit:
            if current:
                pieces.append(Unit(" ".join(current), unit.section))
                current = []
            pieces.extend(Unit(piece, unit.section) for piece in _token_slices(sentence, limit, encoding) if piece)
            continue
        candidate = " ".join(current + [sentence])
        if current and token_count(candidate, encoding) > limit:
            pieces.append(Unit(" ".join(current), unit.section))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(Unit(" ".join(current), unit.section))
    return pieces


def structural_units(text: str, initial_section: str) -> list[Unit]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[Unit] = []
    current_section = initial_section
    for paragraph in paragraphs:
        first_line = paragraph.splitlines()[0].strip()
        if is_heading(first_line):
            current_section = first_line
        list_parts = re.split(r"(?m)(?=^(?:[-*•▪●]|\d+[.)]|[A-Za-z][.)])\s+)", paragraph)
        list_parts = [part.strip() for part in list_parts if part.strip()]
        for part in list_parts if len(list_parts) > 1 else [paragraph]:
            units.append(Unit(part, current_section))
    return units


def trailing_overlap(units: list[Unit], limit: int, encoding: TokenEncoding) -> list[Unit]:
    kept: list[Unit] = []
    total = 0
    for unit in reversed(units):
        size = token_count(unit.text, encoding)
        if kept and total + size > limit:
            break
        kept.insert(0, unit)
        total += size
    return kept


def _chunk_content_type(text: str) -> str:
    marker = "[Validated visual supplement |"
    has_visual = marker in text
    if not has_visual:
        return "text"
    prefix = text.split(marker, 1)[0].strip()
    return "text+vision" if prefix else "vision"


def chunk_page(page: dict, encoding: TokenEncoding, initial_section: str) -> list[dict]:
    text = normalize_for_chunking(page.get("knowledge_text", ""))
    if not text:
        return []
    raw_units = structural_units(text, initial_section)
    units: list[Unit] = []
    for unit in raw_units:
        units.extend(split_oversized_unit(unit, MAX_TOKENS, encoding))

    packed: list[list[Unit]] = []
    current: list[Unit] = []
    for unit in units:
        candidate = "\n\n".join(item.text for item in current + [unit])
        if current and token_count(candidate, encoding) > TARGET_TOKENS:
            packed.append(current)
            current = trailing_overlap(current, OVERLAP_TOKENS, encoding) + [unit]
            if token_count("\n\n".join(item.text for item in current), encoding) > MAX_TOKENS:
                current = [unit]
        else:
            current.append(unit)
    if current:
        packed.append(current)

    chunks: list[dict] = []
    for index, group in enumerate(packed):
        chunk_text = clamp_to_token_limit(
            "\n\n".join(item.text for item in group), MAX_TOKENS, encoding
        )
        if not chunk_text:
            continue
        section = next((item.section for item in group if item.section), initial_section)
        chunks.append(
            {
                "chunk_id": stable_chunk_id(page["record_id"], index, chunk_text),
                "variant": VARIANT,
                "chunk_index": index,
                "text": chunk_text,
                "token_count": token_count(chunk_text, encoding),
                "record_id": page["record_id"],
                "doc_id": page["doc_id"],
                "source_id": page["source_id"],
                "scope": page["scope"],
                "source_path": page["source_path"],
                "source_sha256": page["source_sha256"],
                "page": page["page"],
                "pdf_page": page["pdf_page"],
                "page_index": page["page_index"],
                "section": section,
                "language": page["language"],
                "content_type": _chunk_content_type(chunk_text),
                "selected_method": page["selected_method"],
                "has_vision": "[Validated visual supplement |" in chunk_text,
                "vision_ids": page.get("vision_ids", []) if "[Validated visual supplement |" in chunk_text else [],
            }
        )
    return chunks


def chunk_pages(pages: list[dict], encoding: TokenEncoding) -> tuple[list[dict], dict]:
    chunks: list[dict] = []
    current_sections: dict[str, str] = {}
    for page in pages:
        doc_id = page["doc_id"]
        section = infer_page_section(page.get("selected_text", ""), current_sections.get(doc_id))
        current_sections[doc_id] = section
        chunks.extend(chunk_page(page, encoding, section))
    sizes = [row["token_count"] for row in chunks]
    sorted_sizes = sorted(sizes)
    p95_index = max(0, math.ceil(len(sorted_sizes) * 0.95) - 1) if sorted_sizes else 0
    summary = {
        "variant": VARIANT,
        "pages": len(pages),
        "nonempty_pages": sum(bool(page.get("knowledge_text", "").strip()) for page in pages),
        "chunks": len(chunks),
        "mean_tokens": round(statistics.mean(sizes), 1) if sizes else 0,
        "median_tokens": round(statistics.median(sizes), 1) if sizes else 0,
        "p95_tokens": sorted_sizes[p95_index] if sizes else 0,
        "max_tokens": max(sizes, default=0),
        "under_diagnostic_minimum": sum(size < MIN_DIAGNOSTIC_TOKENS for size in sizes),
        "chunks_with_vision": sum(row["has_vision"] for row in chunks),
        "documents": len({row["doc_id"] for row in chunks}),
    }
    return chunks, summary
