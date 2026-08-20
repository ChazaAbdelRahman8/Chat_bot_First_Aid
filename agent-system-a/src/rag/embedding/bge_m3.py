"""Resumable, normalized BGE-M3 embedding checkpoint creation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


MODEL_NAME = "BAAI/bge-m3"
EXPECTED_DIMENSION = 1024


class SentenceEncoder(Protocol):
    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _replace_with_retry(source: Path, destination: Path, *, attempts: int = 8) -> None:
    """Tolerate short Windows antivirus/indexer locks on checkpoint files."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (2 ** min(attempt, 4)))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _replace_with_retry(temporary, path)


def write_ids_atomic(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, chunk in enumerate(chunks):
            handle.write(json.dumps({"row": index, "chunk_id": chunk["chunk_id"]}))
            handle.write("\n")
    _replace_with_retry(temporary, path)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embedding model returned a zero-length vector")
    return matrix / norms


def checkpoint_matches(
    manifest: dict[str, Any], *, chunk_sha256: str, model_name: str,
    rows: int, dimension: int,
) -> bool:
    return all(
        (
            manifest.get("chunk_sha256") == chunk_sha256,
            manifest.get("model_name") == model_name,
            manifest.get("rows") == rows,
            manifest.get("dimension") == dimension,
            manifest.get("dtype") == "float32",
            manifest.get("normalized") is True,
        )
    )


def create_embeddings(
    *,
    chunks_path: Path,
    output_dir: Path,
    encoder: SentenceEncoder,
    model_name: str = MODEL_NAME,
    batch_size: int = 8,
    force: bool = False,
    device: str = "cpu",
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError("chunk input is empty")
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("every chunk must have a nonempty chunk_id")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_id values must be unique")
    if any(not str(chunk.get("text", "")).strip() for chunk in chunks):
        raise ValueError("every chunk must have nonempty text")

    current_dimension_getter = getattr(encoder, "get_embedding_dimension", None)
    dimension = (
        current_dimension_getter()
        if callable(current_dimension_getter)
        else encoder.get_sentence_embedding_dimension()
    )
    if not dimension:
        raise ValueError("embedding model did not report its vector dimension")
    dimension = int(dimension) #type: ignore 
    if model_name == MODEL_NAME and dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"{MODEL_NAME} must produce {EXPECTED_DIMENSION} dimensions, got {dimension}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "bge_m3_embeddings.npy"
    ids_path = output_dir / "embedding_ids.jsonl"
    manifest_path = output_dir / "embedding_manifest.json"
    chunk_sha256 = sha256_file(chunks_path)
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not checkpoint_matches(
            existing,
            chunk_sha256=chunk_sha256,
            model_name=model_name,
            rows=len(chunks),
            dimension=dimension,
        ):
            if not force:
                raise ValueError(
                    "embedding checkpoint does not match the current chunks/model; "
                    "rerun with --force to replace it"
                )
            existing = {}

    if (
        existing.get("status") == "complete"
        and vectors_path.exists()
        and ids_path.exists()
        and sha256_file(vectors_path) == existing.get("embeddings_sha256")
    ):
        return existing

    completed = int(existing.get("completed_rows", 0)) if existing else 0
    if completed and not vectors_path.exists():
        completed = 0
    mode = "r+" if completed else "w+"
    vectors = np.lib.format.open_memmap(
        vectors_path, mode=mode, dtype=np.float32, shape=(len(chunks), dimension)
    )
    if not completed:
        write_ids_atomic(ids_path, chunks)

    started_at = existing.get("started_at", utc_now())
    base_manifest = {
        "status": "in_progress",
        "model_name": model_name,
        "dimension": dimension,
        "normalized": True,
        "dtype": "float32",
        "rows": len(chunks),
        "completed_rows": completed,
        "batch_size": batch_size,
        "device": device,
        "chunk_source": str(chunks_path.resolve()),
        "chunk_sha256": chunk_sha256,
        "vectors_file": str(vectors_path.resolve()),
        "ids_file": str(ids_path.resolve()),
        "started_at": started_at,
        "updated_at": utc_now(),
    }
    write_json_atomic(manifest_path, base_manifest)

    for start in range(completed, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        texts = [str(chunk["text"]) for chunk in chunks[start:end]]
        encoded = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        batch = normalize_rows(encoded)
        if batch.shape != (end - start, dimension):
            raise ValueError(
                f"unexpected embedding shape {batch.shape}; expected {(end - start, dimension)}"
            )
        vectors[start:end] = batch
        vectors.flush()
        completed = end
        base_manifest.update({"completed_rows": completed, "updated_at": utc_now()})
        write_json_atomic(manifest_path, base_manifest)
        if progress:
            progress(completed, len(chunks))

    del vectors
    complete_manifest = {
        **base_manifest,
        "status": "complete",
        "completed_rows": len(chunks),
        "embeddings_sha256": sha256_file(vectors_path),
        "completed_at": utc_now(),
        "updated_at": utc_now(),
    }
    write_json_atomic(manifest_path, complete_manifest)
    return complete_manifest
