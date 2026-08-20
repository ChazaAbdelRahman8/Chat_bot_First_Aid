"""Lazy, thread-safe runtime shared by chat and ingestion endpoints."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.generation.providers import build_text_generator
from rag.generation.service import GenerationService
from rag.indexing.qdrant_store import COLLECTION_NAME
from rag.retrieval.bm25 import BM25Index, tokenize
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import LanguageAwareRetriever
from rag.retrieval.translation import build_query_translator
from system_limits import GENERATION_VALIDATION_ATTEMPTS

from .document_store import DocumentStore


class RagRuntime:
    def __init__(
        self, project_root: Path, *, qdrant_url: str, qdrant_api_key: str | None,
        collection: str = COLLECTION_NAME, embedding_model: str = "BAAI/bge-m3",
        generation_model: str = "qwen3:4b", ollama_url: str = "http://localhost:11434",
        generation_provider: str = "ollama", groq_api_key: str | None = None,
        generation_timeout: float = 60, generation_fallback_model: str = "qwen3:4b",
        arabic_translation_enabled: bool = True,
        arabic_translation_provider: str = "auto",
        arabic_translation_model: str = "qwen/qwen3.6-27b",
        arabic_translation_fallback_model: str = "qwen3:4b",
        arabic_translation_timeout: float = 20,
        device: str | None = None,
    ) -> None:
        import torch

        self.project_root = project_root
        self.store = DocumentStore(project_root)
        self.collection = collection
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(
            embedding_model,
            device=selected_device,
            local_files_only=os.getenv("ALLOW_MODEL_DOWNLOAD", "false").lower() != "true",
        )
        self.generator = GenerationService(
            build_text_generator(
                provider=generation_provider,
                model=generation_model,
                ollama_url=ollama_url,
                groq_api_key=groq_api_key,
                timeout_seconds=generation_timeout,
                fallback_model=generation_fallback_model,
            ),
            max_validation_attempts=GENERATION_VALIDATION_ATTEMPTS,
        )
        self.embedding_model = embedding_model
        self.device = selected_device
        self.query_translator = build_query_translator(
            enabled=arabic_translation_enabled,
            provider=arabic_translation_provider,
            groq_api_key=groq_api_key,
            groq_model=arabic_translation_model,
            ollama_model=arabic_translation_fallback_model,
            ollama_url=ollama_url,
            timeout_seconds=arabic_translation_timeout,
        )
        self.lock = threading.RLock()
        self._retriever: LanguageAwareRetriever | None = None
        self.refresh_retriever()

    def refresh_retriever(self) -> None:
        with self.lock:
            chunks = self.store.active_chunks()
            bm25 = BM25Index(
                chunks=chunks,
                token_rows=[tokenize(str(row["text"])) for row in chunks],
            )
            self._retriever = LanguageAwareRetriever(
                dense=DenseRetriever(
                    client=self.qdrant,
                    encoder=self.encoder,
                    collection_name=self.collection,
                ),
                bm25=bm25,
                translator=self.query_translator,
            )

    def answer(
        self, query: str, doc_id: str | None = None,
        *, retrieval_query: str | None = None,
    ) -> dict[str, Any]:
        from guardrails.input_guard import evaluate_input
        from rag.generation.service import policy_gate_response

        decision = evaluate_input(query)
        if decision.blocked:
            generation = policy_gate_response(query, self.generator.client.model, decision)
            return {"query": query, "retrieval": None, "generation": generation}
        retrieval_payload = self.retrieve(retrieval_query or query, doc_id=doc_id)
        retrieval = {**retrieval_payload}
        generation = self.generator.generate(query, retrieval)
        return {
            "query": query,
            "retrieval_query": retrieval_query or query,
            "retrieval": retrieval_payload,
            "generation": generation,
        }

    def retrieve(self, query: str, doc_id: str | None = None) -> dict[str, Any]:
        """Return retrieval evidence without invoking the generation model."""
        with self.lock:
            assert self._retriever is not None
            retrieval = self._retriever.retrieve(query, doc_id=doc_id)
        return {
            "route": retrieval["route"],
            "language": retrieval["language"],
            "translation": retrieval["translation"],
            "result_count": retrieval["result_count"],
            "doc_id_filter": retrieval["doc_id_filter"],
            "results": retrieval["results"],
        }
