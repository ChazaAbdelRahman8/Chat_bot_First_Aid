"""BGE-M3 query encoding and dense Qdrant retrieval."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class QueryEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


def encode_query(query: str, encoder: QueryEncoder) -> np.ndarray:
    if not query.strip():
        raise ValueError("query must not be empty")
    encoded = np.asarray(
        encoder.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if encoded.ndim != 2 or encoded.shape[0] != 1:
        raise ValueError(f"query encoder returned unexpected shape {encoded.shape}")
    vector = encoded[0]
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("query encoder returned a zero vector")
    return vector / norm


def result_from_payload(payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "chunk_id": payload.get("chunk_id"),
        "doc_id": payload.get("doc_id"),
        "page": payload.get("page"),
        "pdf_page": payload.get("pdf_page"),
        "section": payload.get("section"),
        "language": payload.get("language"),
        "content_type": payload.get("content_type"),
        "text": payload.get("text", ""),
        "record_id": payload.get("record_id"),
        **extra,
    }


class DenseRetriever:
    def __init__(self, *, client: Any, encoder: QueryEncoder, collection_name: str) -> None:
        self.client = client
        self.encoder = encoder
        self.collection_name = collection_name

    def search(
        self, query: str, *, limit: int = 5, doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        vector = encode_query(query, self.encoder)
        return self.search_vector(vector, limit=limit, doc_id=doc_id)

    def search_vector(
        self, vector: np.ndarray, *, limit: int = 5, doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client import models

        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"query vector must be one-dimensional, got {vector.shape}")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("query vector must not be zero")
        vector = vector / norm
        query_filter = None
        if doc_id:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for rank, point in enumerate(response.points, start=1):
            results.append(
                result_from_payload(
                    point.payload or {},
                    score=float(point.score),
                    dense_score=float(point.score),
                    dense_rank=rank,
                    retrieval_method="dense",
                    retrieval_methods=["dense"],
                )
            )
        return results
