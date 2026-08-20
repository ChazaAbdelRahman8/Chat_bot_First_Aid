"""Persistent BM25 index with multilingual token normalization."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.embedding.bge_m3 import read_jsonl, sha256_file, write_json_atomic


TOKENIZER_VERSION = "unicode-arabic-v1"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_lexical_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = ARABIC_DIACRITICS.sub("", value).replace("ـ", "")
    value = re.sub(r"[أإآٱ]", "ا", value)
    value = value.replace("ى", "ي")
    return value


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(normalize_lexical_text(text))
    return [token for token in tokens if len(token) > 1 and not token.isdigit()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def replace_document_chunks(
    chunks: list[dict[str, Any]], doc_id: str, replacement: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if any(str(row.get("doc_id")) != doc_id for row in replacement):
        raise ValueError("every replacement chunk must have the requested doc_id")
    positions = [index for index, row in enumerate(chunks) if str(row.get("doc_id")) == doc_id]
    insert_at = positions[0] if positions else len(chunks)
    retained = [row for row in chunks if str(row.get("doc_id")) != doc_id]
    return retained[:insert_at] + replacement + retained[insert_at:]


def build_bm25_checkpoint(
    *, chunks_path: Path, output_dir: Path, k1: float = DEFAULT_K1, b: float = DEFAULT_B,
) -> dict[str, Any]:
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError("chunk input is empty")
    chunk_ids = [str(row.get("chunk_id", "")) for row in chunks]
    if any(not value for value in chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_id values must be nonempty and unique")
    rows = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if not text.strip():
            raise ValueError(f"{chunk['chunk_id']}: text is empty")
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "tokens": tokenize(text),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "bm25_corpus.jsonl"
    manifest_path = output_dir / "bm25_manifest.json"
    write_jsonl_atomic(corpus_path, rows)
    lengths = [len(row["tokens"]) for row in rows]
    manifest = {
        "status": "complete",
        "tokenizer_version": TOKENIZER_VERSION,
        "k1": k1,
        "b": b,
        "chunks": len(chunks),
        "documents": len({row["doc_id"] for row in chunks}),
        "average_document_length": sum(lengths) / len(lengths),
        "chunk_source": str(chunks_path.resolve()),
        "chunk_sha256": sha256_file(chunks_path),
        "corpus_file": str(corpus_path.resolve()),
        "corpus_sha256": sha256_file(corpus_path),
        "created_at": utc_now(),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_bm25_checkpoint(
    chunks_path: Path, checkpoint_dir: Path,
) -> tuple[list[dict[str, Any]], "BM25Index", dict[str, Any]]:
    chunks = read_jsonl(chunks_path)
    corpus_path = checkpoint_dir / "bm25_corpus.jsonl"
    manifest_path = checkpoint_dir / "bm25_manifest.json"
    if not corpus_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("BM25 checkpoint is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("status") != "complete":
        errors.append("status is not complete")
    if manifest.get("tokenizer_version") != TOKENIZER_VERSION:
        errors.append("tokenizer version mismatch")
    if manifest.get("chunk_sha256") != sha256_file(chunks_path):
        errors.append("chunk SHA256 mismatch")
    if manifest.get("corpus_sha256") != sha256_file(corpus_path):
        errors.append("corpus SHA256 mismatch")
    rows = read_jsonl(corpus_path)
    if [row.get("chunk_id") for row in rows] != [row.get("chunk_id") for row in chunks]:
        errors.append("BM25 rows do not match chunk order")
    if errors:
        raise ValueError("invalid BM25 checkpoint: " + "; ".join(errors))
    index = BM25Index(
        chunks=chunks,
        token_rows=[list(row.get("tokens", [])) for row in rows],
        k1=float(manifest["k1"]),
        b=float(manifest["b"]),
    )
    return chunks, index, manifest


class BM25Index:
    def __init__(
        self, *, chunks: list[dict[str, Any]], token_rows: list[list[str]],
        k1: float = DEFAULT_K1, b: float = DEFAULT_B,
    ) -> None:
        if len(chunks) != len(token_rows):
            raise ValueError("chunk and token row counts differ")
        self.chunks = chunks
        self.token_rows = token_rows
        self.k1 = k1
        self.b = b
        self.lengths = [len(tokens) for tokens in token_rows]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in token_rows]
        document_frequency: Counter[str] = Counter()
        for terms in self.term_frequencies:
            document_frequency.update(terms.keys())
        count = len(token_rows)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, limit: int = 10, doc_id: str | None = None) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not self.chunks:
            return []
        scores: list[tuple[float, int]] = []
        for index, (chunk, frequencies, length) in enumerate(
            zip(self.chunks, self.term_frequencies, self.lengths)
        ):
            if doc_id is not None and str(chunk.get("doc_id")) != doc_id:
                continue
            score = 0.0
            length_norm = 1.0 - self.b
            if self.average_length:
                length_norm += self.b * length / self.average_length
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * length_norm
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"chunk": self.chunks[index], "score": score, "rank": rank}
            for rank, (score, index) in enumerate(scores[:limit], start=1)
        ]

