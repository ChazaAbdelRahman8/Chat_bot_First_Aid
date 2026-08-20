"""Render retrieved manual pages as bounded base64 visual evidence."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image


class ManualVisualRenderer:
    """Resolve a retrieved ``doc_id`` and render its cited PDF page.

    The full page is returned rather than an inferred crop so captions, labels,
    warnings, and surrounding procedural context remain visible.
    """

    def __init__(self, project_root: Path, *, max_bytes: int = 2_500_000) -> None:
        self.project_root = project_root
        self.max_bytes = max_bytes

    def _source_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        registry_path = self.project_root / "data" / "source_registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            for document in registry.get("documents", []):
                paths[str(document["doc_id"])] = (
                    self.project_root / "data" / "source_documents" / str(document["path"])
                ).resolve()

        managed_path = self.project_root / "data" / "managed_documents" / "registry.json"
        if managed_path.exists():
            for doc_id, document in json.loads(
                managed_path.read_text(encoding="utf-8")
            ).items():
                if document.get("path"):
                    paths[str(doc_id)] = Path(str(document["path"])).resolve()
        return paths

    def render(self, retrieval: dict[str, Any], *, limit: int = 1) -> list[dict[str, Any]]:
        results = list(retrieval.get("results") or [])
        # Prefer chunks explicitly carrying validated visual content, while
        # retaining retrieval rank within each group.
        ranked = sorted(
            enumerate(results),
            key=lambda pair: (
                0
                if pair[0] < 3
                and "vision" in str(pair[1].get("content_type", "")).lower()
                else 1,
                pair[0],
            ),
        )
        source_paths = self._source_paths()
        visuals: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for _, result in ranked:
            doc_id = str(result.get("doc_id") or "")
            raw_page = result.get("pdf_page", result.get("page"))
            try:
                pdf_page = int(raw_page)
            except (TypeError, ValueError):
                continue
            key = (doc_id, pdf_page)
            path = source_paths.get(doc_id)
            if not doc_id or pdf_page < 1 or path is None or not path.is_file() or key in seen:
                continue
            seen.add(key)
            visuals.append(self._render_page(path, result, pdf_page))
            if len(visuals) >= limit:
                break
        return visuals

    def _render_page(
        self, path: Path, result: dict[str, Any], pdf_page: int,
    ) -> dict[str, Any]:
        with pymupdf.open(path) as document:
            if pdf_page > document.page_count:
                raise ValueError(
                    f"retrieved pdf_page {pdf_page} exceeds {document.page_count} pages"
                )
            page = document.load_page(pdf_page - 1)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.35, 1.35), alpha=False)
            content = pixmap.tobytes("png")

        if len(content) > self.max_bytes:
            with Image.open(io.BytesIO(content)) as image:
                image.thumbnail((1200, 1600), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG", optimize=True)
                content = buffer.getvalue()
        if len(content) > self.max_bytes:
            raise ValueError("rendered manual page exceeds the visual response size limit")

        return {
            "visual_id": f"{result.get('doc_id')}:p{pdf_page}:manual-page",
            "doc_id": result.get("doc_id"),
            "pdf_page": pdf_page,
            "section": result.get("section"),
            "chunk_id": result.get("chunk_id"),
            "mime_type": "image/png",
            "encoding": "base64",
            "data_base64": base64.b64encode(content).decode("ascii"),
            "alt_text": (
                f"Rendered page {pdf_page} from first-aid manual {result.get('doc_id')}"
            ),
        }
