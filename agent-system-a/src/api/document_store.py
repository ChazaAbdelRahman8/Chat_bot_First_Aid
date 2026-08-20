"""Atomic registry and active-chunk storage for API-managed documents."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from rag.embedding.bge_m3 import read_jsonl
from rag.retrieval.bm25 import write_jsonl_atomic


class DocumentStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.root = project_root / "data" / "managed_documents"
        self.registry_path = self.root / "registry.json"
        self.chunks_path = self.root / "managed_chunks.jsonl"
        self.jobs_path = self.root / "jobs.json"
        self.upload_dir = self.root / "uploads"
        self.work_dir = self.root / "work"
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def registry(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def jobs(self) -> dict[str, dict[str, Any]]:
        if not self.jobs_path.exists():
            return {}
        return json.loads(self.jobs_path.read_text(encoding="utf-8"))

    def set_job(self, job_id: str, value: dict[str, Any]) -> None:
        with self.lock:
            jobs = self.jobs()
            jobs[job_id] = value
            self._write_json(self.jobs_path, jobs)

    def set_document(self, doc_id: str, value: dict[str, Any]) -> None:
        with self.lock:
            registry = self.registry()
            registry[doc_id] = value
            self._write_json(self.registry_path, registry)

    def remove_document(self, doc_id: str) -> dict[str, Any] | None:
        with self.lock:
            registry = self.registry()
            removed = registry.pop(doc_id, None)
            self._write_json(self.registry_path, registry)
            return removed

    def managed_chunks(self) -> list[dict[str, Any]]:
        return read_jsonl(self.chunks_path) if self.chunks_path.exists() else []

    def replace_chunks(self, doc_id: str, replacement: list[dict[str, Any]]) -> None:
        with self.lock:
            existing = [
                row for row in self.managed_chunks() if str(row.get("doc_id")) != doc_id
            ]
            write_jsonl_atomic(self.chunks_path, existing + replacement)

    def active_chunks(self) -> list[dict[str, Any]]:
        base = read_jsonl(
            self.project_root
            / "data" / "processed" / "chunk" / "chunks_structure_aware.jsonl"
        )
        return base + self.managed_chunks()
