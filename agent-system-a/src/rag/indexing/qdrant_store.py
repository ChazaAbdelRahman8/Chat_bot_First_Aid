"""Qdrant collection creation, document replacement, and validation."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rag.embedding.bge_m3 import checkpoint_matches, read_jsonl, sha256_file


COLLECTION_NAME = "first_aid_chunks"
POINT_NAMESPACE = uuid.UUID("4be11a08-3084-49a7-9e3a-653f838ec3c9")
REQUIRED_PAYLOAD_FIELDS = (
    "doc_id", "page", "pdf_page", "section", "chunk_id", "language", "content_type",
)


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def payload_for_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_PAYLOAD_FIELDS if chunk.get(field) in (None, "")]
    if missing:
        raise ValueError(f"{chunk.get('chunk_id')}: missing payload fields {missing}")
    return {
        "doc_id": chunk["doc_id"],
        "page": chunk["page"],
        "pdf_page": chunk["pdf_page"],
        "section": chunk["section"],
        "chunk_id": chunk["chunk_id"],
        "language": chunk["language"],
        "content_type": chunk["content_type"],
        "text": chunk["text"],
        "record_id": chunk.get("record_id"),
        "chunk_index": chunk.get("chunk_index"),
        "token_count": chunk.get("token_count"),
        "source_id": chunk.get("source_id"),
        "has_vision": bool(chunk.get("has_vision")),
        "vision_ids": chunk.get("vision_ids", []),
    }


def load_embedding_checkpoint(
    chunks_path: Path, embedding_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    chunks = read_jsonl(chunks_path)
    manifest_path = embedding_dir / "embedding_manifest.json"
    ids_path = embedding_dir / "embedding_ids.jsonl"
    vectors_path = embedding_dir / "bge_m3_embeddings.npy"
    if not all(path.exists() for path in (manifest_path, ids_path, vectors_path)):
        raise FileNotFoundError("embedding checkpoint is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("embedding checkpoint status is not complete")
    if not checkpoint_matches(
        manifest,
        chunk_sha256=sha256_file(chunks_path),
        model_name=str(manifest.get("model_name")),
        rows=len(chunks),
        dimension=int(manifest.get("dimension", 0)),
    ):
        raise ValueError("embedding manifest does not match the chunk artifact")
    if sha256_file(vectors_path) != manifest.get("embeddings_sha256"):
        raise ValueError("embedding file SHA256 does not match its manifest")
    ids = read_jsonl(ids_path)
    expected_ids = [chunk["chunk_id"] for chunk in chunks]
    actual_ids = [row.get("chunk_id") for row in ids]
    if actual_ids != expected_ids:
        raise ValueError("embedding row order does not match chunk order")
    vectors = np.load(vectors_path, mmap_mode="r")
    expected_shape = (len(chunks), int(manifest["dimension"]))
    if vectors.shape != expected_shape:
        raise ValueError(f"embedding shape {vectors.shape} does not match {expected_shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("embedding checkpoint contains non-normalized vectors")
    return chunks, vectors, manifest


def ensure_collection(client: Any, collection_name: str, dimension: int) -> None:
    from qdrant_client import models

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
    else:
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors
        actual_size = getattr(vectors, "size", None)
        actual_distance = getattr(vectors, "distance", None)
        if actual_size != dimension or actual_distance != models.Distance.COSINE:
            raise ValueError(
                f"existing collection vector config is size={actual_size}, "
                f"distance={actual_distance}; expected size={dimension}, cosine"
            )
    # This must happen before the first payload upsert for efficient filtered deletes.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )


def index_checkpoint(
    *, client: Any, collection_name: str, chunks: list[dict[str, Any]],
    vectors: np.ndarray, dimension: int, batch_size: int = 64,
    replace_documents: bool = True,
) -> dict[str, Any]:
    from qdrant_client import models

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    ensure_collection(client, collection_name, dimension)
    by_document: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        by_document.setdefault(str(chunk["doc_id"]), []).append(index)

    uploaded = 0
    for doc_id, indexes in by_document.items():
        if replace_documents:
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                    )
                ),
                wait=True,
            )
        for offset in range(0, len(indexes), batch_size):
            batch_indexes = indexes[offset : offset + batch_size]
            points = [
                models.PointStruct(
                    id=point_id_for_chunk(chunks[index]["chunk_id"]),
                    vector=np.asarray(vectors[index], dtype=np.float32).tolist(),
                    payload=payload_for_chunk(chunks[index]),
                )
                for index in batch_indexes
            ]
            client.upsert(collection_name=collection_name, points=points, wait=True)
            uploaded += len(points)
    return {
        "collection": collection_name,
        "uploaded": uploaded,
        "documents": len(by_document),
        "document_chunk_counts": dict(Counter(chunk["doc_id"] for chunk in chunks)),
    }


def validate_collection(
    *, client: Any, collection_name: str, chunks: list[dict[str, Any]], dimension: int,
) -> dict[str, Any]:
    from qdrant_client import models

    errors: list[str] = []
    info = client.get_collection(collection_name)
    exact_count = client.count(collection_name=collection_name, exact=True).count
    if exact_count != len(chunks):
        errors.append(f"Qdrant point count {exact_count} != chunk count {len(chunks)}")
    schema = getattr(info, "payload_schema", {}) or {}
    if "doc_id" not in schema:
        errors.append("doc_id payload index is missing")
    vectors_config = info.config.params.vectors
    if getattr(vectors_config, "size", None) != dimension:
        errors.append("collection vector dimension mismatch")

    expected_by_doc = Counter(str(chunk["doc_id"]) for chunk in chunks)
    actual_by_doc: dict[str, int] = {}
    for doc_id, expected in expected_by_doc.items():
        condition = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        actual = client.count(
            collection_name=collection_name, count_filter=condition, exact=True
        ).count
        actual_by_doc[doc_id] = actual
        if actual != expected:
            errors.append(f"{doc_id}: Qdrant count {actual} != chunk count {expected}")

    sample_ids = [point_id_for_chunk(chunk["chunk_id"]) for chunk in chunks[:3]]
    samples = client.retrieve(
        collection_name=collection_name,
        ids=sample_ids,
        with_payload=True,
        with_vectors=False,
    )
    if len(samples) != len(sample_ids):
        errors.append("could not retrieve all sample points")
    for point in samples:
        payload = point.payload or {}
        missing = [field for field in REQUIRED_PAYLOAD_FIELDS if payload.get(field) in (None, "")]
        if missing:
            errors.append(f"sample point {point.id} missing payload fields {missing}")

    return {
        "valid": not errors,
        "collection": collection_name,
        "chunks": len(chunks),
        "qdrant_points": exact_count,
        "documents": len(expected_by_doc),
        "vector_dimension": dimension,
        "doc_id_payload_index": "doc_id" in schema,
        "document_counts": actual_by_doc,
        "errors": errors,
    }

