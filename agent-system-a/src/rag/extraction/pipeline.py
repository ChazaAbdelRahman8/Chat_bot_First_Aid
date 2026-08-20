"""Orchestration and validation for Phase 2 extraction."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .native_pdf import extract_document, page_count
from .paddle_ocr import make_ocr_engine, render_page, run_ocr


EXPECTED_METHOD_COUNTS = {
    "native": 2049,
    "paddleocr": 230,
    "native_fallback": 4,
    "extraction_failed": 6,
}
EXPECTED_TOTAL_PAGES = 2289


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_verify_registry(registry_path: Path, source_root: Path) -> list[dict]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    if len(documents) != 7:
        raise ValueError(f"Expected exactly 7 source documents, found {len(documents)}")
    seen: set[str] = set()
    for document in documents:
        doc_id = document["doc_id"]
        if doc_id in seen:
            raise ValueError(f"Duplicate doc_id in registry: {doc_id}")
        seen.add(doc_id)
        source_path = source_root / document["path"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing registered PDF: {source_path}")
        actual_hash = sha256_file(source_path)
        if actual_hash.lower() != document["sha256"].lower():
            raise ValueError(
                f"SHA256 mismatch for {doc_id}: expected {document['sha256']}, "
                f"found {actual_hash}"
            )
    return documents


def native_extract(
    documents: list[dict], source_root: Path, native_output_path: Path
) -> list[dict]:
    records: list[dict] = []
    for document in documents:
        print(f"Native extraction: {document['doc_id']}", flush=True)
        records.extend(extract_document(document, source_root))
    write_jsonl_atomic(native_output_path, records)
    return records


def _base_selected_record(record: dict) -> dict:
    selected = dict(record)
    selected.update(
        {
            "page_image_path": None,
            "ocr_text": "",
            "ocr_mean_confidence": None,
            "ocr_error": None,
            "selected_method": "native",
            "selected_text": record["native_text"].strip(),
            "manual_review_required": False,
        }
    )
    return selected


def selective_ocr(
    native_records: list[dict],
    source_root: Path,
    output_path: Path,
    image_dir: Path,
    dpi: int = 200,
    force: bool = False,
) -> list[dict]:
    checkpoint_path = output_path.with_name(output_path.stem + ".checkpoint.jsonl")
    completed: dict[str, dict] = {}
    if not force:
        for record in read_jsonl(checkpoint_path):
            completed[record["record_id"]] = record

    engines: dict[str, object] = {}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "a"
    with checkpoint_path.open(mode, encoding="utf-8", newline="\n") as checkpoint:
        for index, native_record in enumerate(native_records, start=1):
            record_id = native_record["record_id"]
            if record_id in completed:
                continue

            record = _base_selected_record(native_record)
            if not native_record["quality_pass"]:
                relative_source = Path(native_record["source_path"]).relative_to(
                    Path("data/source_documents")
                )
                source_path = source_root / relative_source
                image_filename = (
                    f"{native_record['doc_id']}__p{native_record['pdf_page']:04d}.png"
                )
                image_path = image_dir / image_filename
                record["page_image_path"] = (
                    f"data/extraction/page_images/{image_filename}"
                )
                ocr_text = ""
                confidence: float | None = None
                try:
                    render_page(source_path, native_record["page_index"], image_path, dpi)
                    lang = native_record["ocr_lang"]
                    if lang not in engines:
                        print(f"Loading PaddleOCR model: {lang}", flush=True)
                        engines[lang] = make_ocr_engine(lang)
                    ocr_text, confidence = run_ocr(engines[lang], image_path)
                    record["ocr_text"] = ocr_text
                    record["ocr_mean_confidence"] = confidence
                except Exception as exc:  # Preserve the page and surface the failure.
                    record["ocr_error"] = f"{type(exc).__name__}: {exc}"

                if ocr_text:
                    record["selected_method"] = "paddleocr"
                    record["selected_text"] = ocr_text
                    if confidence is None or confidence < 0.70:
                        record["manual_review_required"] = True
                elif native_record["native_text"].strip():
                    record["selected_method"] = "native_fallback"
                    record["manual_review_required"] = True
                else:
                    record["selected_method"] = "extraction_failed"
                    record["selected_text"] = ""
                    record["manual_review_required"] = True

            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()
            completed[record_id] = record
            if index % 100 == 0 or not native_record["quality_pass"]:
                print(
                    f"Resolved {len(completed):,}/{len(native_records):,}: "
                    f"{record_id} -> {record['selected_method']}",
                    flush=True,
                )

    ordered = [completed[record["record_id"]] for record in native_records]
    write_jsonl_atomic(output_path, ordered)
    checkpoint_path.unlink(missing_ok=True)
    return ordered


def validate_records(records: list[dict], documents: list[dict], source_root: Path) -> None:
    expected_pages = sum(page_count(source_root / document["path"]) for document in documents)
    if len(records) != expected_pages:
        raise ValueError(f"Page count mismatch: expected {expected_pages}, found {len(records)}")
    record_ids = [record["record_id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate record_id values found")
    for record in records:
        if record["pdf_page"] != record["page_index"] + 1:
            raise ValueError(f"Page provenance mismatch: {record['record_id']}")


def build_summary(records: list[dict], native_only: bool) -> dict:
    if native_only:
        routed_count = sum(not record["quality_pass"] for record in records)
        return {
            "status": "native_extraction_complete_ocr_pending",
            "total_pages": len(records),
            "native_gate_pass": len(records) - routed_count,
            "routed_to_ocr": routed_count,
            "expected_total_pages": EXPECTED_TOTAL_PAGES,
            "expected_native_gate_failures_v1": 240,
            "large_deviation": abs(routed_count - 240) > 25,
        }

    method_counts = Counter(record["selected_method"] for record in records)
    by_document: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        by_document[record["doc_id"]][record["selected_method"]] += 1
    deviations = {
        method: method_counts.get(method, 0) - expected
        for method, expected in EXPECTED_METHOD_COUNTS.items()
    }
    return {
        "status": "complete",
        "total_pages": len(records),
        "method_counts": dict(method_counts),
        "expected_v1_method_counts": EXPECTED_METHOD_COUNTS,
        "deviation_from_v1": deviations,
        "large_deviation": any(abs(value) > 25 for value in deviations.values()),
        "manual_review_pages": sum(
            bool(record["manual_review_required"]) for record in records
        ),
        "by_document": {key: dict(value) for key, value in by_document.items()},
    }


def run_pipeline(
    project_root: Path,
    native_only: bool = False,
    force: bool = False,
    dpi: int = 200,
) -> dict:
    source_root = project_root / "data" / "source_documents"
    registry_path = project_root / "data" / "source_registry.json"
    output_dir = project_root / "data" / "extraction"
    native_path = output_dir / "native_page_records.jsonl"
    output_path = output_dir / "page_records.jsonl"
    summary_path = output_dir / "extraction_summary.json"
    documents = load_and_verify_registry(registry_path, source_root)

    native_records = [] if force else read_jsonl(native_path)
    if not native_records:
        native_records = native_extract(documents, source_root, native_path)
    validate_records(native_records, documents, source_root)

    if native_only:
        records = native_records
    else:
        checkpoint_path = output_path.with_name(
            output_path.stem + ".checkpoint.jsonl"
        )
        in_progress_summary = {
            "status": "ocr_in_progress",
            "total_pages": len(native_records),
            "completed_records": len(read_jsonl(checkpoint_path)),
            "routed_to_ocr": sum(
                not record["quality_pass"] for record in native_records
            ),
            "source_hashes_verified": True,
            "final_page_records_ready": False,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(in_progress_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        records = selective_ocr(
            native_records,
            source_root,
            output_path,
            output_dir / "page_images",
            dpi=dpi,
            force=force,
        )
        validate_records(records, documents, source_root)

    summary = build_summary(records, native_only=native_only)
    summary["source_hashes_verified"] = True
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
