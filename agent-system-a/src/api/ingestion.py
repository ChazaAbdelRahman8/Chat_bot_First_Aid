"""Per-document ingestion and deletion for the live API corpus."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tiktoken
from qdrant_client import models

from rag.chunking.merge import merge_pages
from rag.chunking.structure_aware import chunk_pages
from rag.embedding.bge_m3 import normalize_rows, sha256_file
from rag.extraction.native_pdf import extract_document
from rag.extraction.paddle_ocr import make_ocr_engine, render_page, run_ocr
from rag.indexing.qdrant_store import ensure_collection, payload_for_chunk, point_id_for_chunk
from rag.vision.candidate_detection import build_candidate_queue
from rag.vision.ollama_vlm import describe_page
from rag.vision.prompt import PROMPT_VERSION as VISION_PROMPT_VERSION
from rag.vision.validation import classify_description

from .runtime import RagRuntime


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_doc_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_").upper()
    if not normalized or len(normalized) > 80:
        raise ValueError("doc_id must contain 1-80 letters, digits, hyphens, or underscores")
    return normalized


class DocumentIngestor:
    def __init__(
        self, runtime: RagRuntime, *, ollama_url: str, vision_model: str,
        vision_enabled: bool = True,
    ) -> None:
        self.runtime = runtime
        self.store = runtime.store
        self.ollama_url = ollama_url
        self.vision_model = vision_model
        self.vision_enabled = vision_enabled

    def _extract(self, document: dict[str, Any], source_root: Path, work: Path) -> list[dict]:
        records = list(extract_document(document, source_root))
        engines: dict[str, object] = {}
        for record in records:
            record["source_path"] = str((source_root / document["path"]).resolve())
            record.update(
                page_image_path=None,
                ocr_text="",
                ocr_mean_confidence=None,
                ocr_error=None,
                selected_method="native",
                selected_text=str(record["native_text"]).strip(),
                manual_review_required=False,
            )
            if record["quality_pass"]:
                continue
            image = work / "ocr_images" / f'{record["doc_id"]}__p{record["pdf_page"]:04d}.png'
            try:
                render_page(source_root / document["path"], record["page_index"], image, 200)
                lang = record["ocr_lang"]
                if lang not in engines:
                    engines[lang] = make_ocr_engine(lang)
                engine = engines[lang]
                text, confidence = run_ocr(engine, image)
                record["ocr_text"] = text
                record["ocr_mean_confidence"] = confidence
                if text:
                    record["selected_method"] = "paddleocr"
                    record["selected_text"] = text
                    record["manual_review_required"] = confidence is None or confidence < 0.70
                elif record["native_text"].strip():
                    record["selected_method"] = "native_fallback"
                    record["manual_review_required"] = True
                else:
                    record["selected_method"] = "extraction_failed"
                    record["selected_text"] = ""
                    record["manual_review_required"] = True
            except Exception as exc:
                record["ocr_error"] = f"{type(exc).__name__}: {exc}"
                record["selected_method"] = "native_fallback" if record["native_text"].strip() else "extraction_failed"
                record["manual_review_required"] = True
        return records

    def _vision(self, records: list[dict], work: Path) -> list[dict]:
        if not self.vision_enabled:
            return []
        queue = build_candidate_queue(records, self.runtime.project_root, output_dir=work / "vision_images")
        results = []
        for candidate in queue:
            try:
                description = describe_page(
                    Path(candidate["image_path"]), self.vision_model, self.ollama_url, 300
                )
                useful, status = classify_description(description)
                error = None
            except Exception as exc:
                description, useful, status = None, False, "error"
                error = f"{type(exc).__name__}: {exc}"
            results.append({
                **candidate,
                "model": self.vision_model,
                "provider": "ollama_local",
                "prompt_version": VISION_PROMPT_VERSION,
                "authority": "model_generated_supplement",
                "review_status": status,
                "vision_useful": useful,
                "description": description,
                "error": error,
            })
        return results

    def ingest(
        self, *, doc_id: str, filename: str, pdf_path: Path, ocr_lang: str,
        scope: str, progress: Callable[[str, dict[str, Any]], None],
    ) -> dict[str, Any]:
        doc_id = normalize_doc_id(doc_id)
        work = self.store.work_dir / doc_id
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        document = {
            "doc_id": doc_id,
            "scope": scope,
            "ocr_lang": ocr_lang,
            "path": pdf_path.name,
            "sha256": sha256_file(pdf_path),
        }
        progress("extracting", {})
        records = self._extract(document, pdf_path.parent, work)
        progress("vision", {"pages": len(records)})
        visions = self._vision(records, work)
        merged, _ = merge_pages(records, visions, lambda page: str(page["pdf_page"]))
        progress("chunking", {"vision_candidates": len(visions)})
        chunks, _ = chunk_pages(merged, tiktoken.get_encoding("cl100k_base"))
        if not chunks:
            raise ValueError("document produced no searchable chunks")
        progress("embedding", {"chunks": len(chunks)})
        vectors = normalize_rows(np.asarray(self.runtime.encoder.encode(
            [row["text"] for row in chunks],
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float32))
        dimension = vectors.shape[1]
        ensure_collection(self.runtime.qdrant, self.runtime.collection, dimension)
        points = [models.PointStruct(
            id=point_id_for_chunk(chunk["chunk_id"]),
            vector=vectors[index].tolist(),
            payload=payload_for_chunk(chunk),
        ) for index, chunk in enumerate(chunks)]
        # Keep the dense and lexical indexes consistent from the perspective of
        # concurrent chat requests. RagRuntime uses an RLock, so the nested
        # refresh_retriever call is safe.
        with self.runtime.lock:
            self.runtime.qdrant.delete(
                collection_name=self.runtime.collection,
                points_selector=models.FilterSelector(filter=models.Filter(must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                ])),
                wait=True,
            )
            for offset in range(0, len(points), 64):
                self.runtime.qdrant.upsert(
                    collection_name=self.runtime.collection,
                    points=points[offset:offset + 64],
                    wait=True,
                )
            progress("updating_bm25", {"qdrant_points": len(points)})
            self.store.replace_chunks(doc_id, chunks)
            self.runtime.refresh_retriever()
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "path": str(pdf_path),
            "sha256": document["sha256"],
            "scope": scope,
            "ocr_lang": ocr_lang,
            "status": "ready",
            "pages": len(records),
            "chunks": len(chunks),
            "vision_candidates": len(visions),
            "created_at": utc_now(),
        }
        self.store.set_document(doc_id, metadata)
        return metadata

    def delete(self, doc_id: str) -> dict[str, Any]:
        doc_id = normalize_doc_id(doc_id)
        metadata = self.store.registry().get(doc_id)
        if metadata is None:
            raise KeyError(doc_id)
        with self.runtime.lock:
            self.runtime.qdrant.delete(
                collection_name=self.runtime.collection,
                points_selector=models.FilterSelector(filter=models.Filter(must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                ])),
                wait=True,
            )
            self.store.replace_chunks(doc_id, [])
            self.store.remove_document(doc_id)
            self.runtime.refresh_retriever()
        pdf = Path(str(metadata.get("path", "")))
        if pdf.is_file() and pdf.parent.resolve() == self.store.upload_dir.resolve():
            pdf.unlink()
        work = self.store.work_dir / doc_id
        if work.exists():
            shutil.rmtree(work)
        return {"doc_id": doc_id, "deleted": True}
