"""Language-aware routing and reciprocal-rank fusion."""

from __future__ import annotations

import re
from typing import Any, Callable, Protocol, Sequence

from rag.retrieval.dense import result_from_payload


ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


class Searcher(Protocol):
    def search(self, query: str, *, limit: int, doc_id: str | None = None) -> list[dict[str, Any]]: ...


def detect_query_language(query: str) -> str:
    if not query.strip():
        raise ValueError("query must not be empty")
    return "ar" if ARABIC_PATTERN.search(query) else "en"


def _bm25_result(item: dict[str, Any]) -> dict[str, Any]:
    chunk = item["chunk"]
    return result_from_payload(
        chunk,
        score=float(item["score"]),
        bm25_score=float(item["score"]),
        bm25_rank=int(item["rank"]),
        retrieval_method="bm25",
        retrieval_methods=["bm25"],
    )


def english_dense_plus_bm25(
    dense_results: list[dict[str, Any]], bm25_results: list[dict[str, Any]],
    *, max_bm25_additions: int = 3,
) -> list[dict[str, Any]]:
    combined = [dict(result) for result in dense_results]
    positions = {result["chunk_id"]: index for index, result in enumerate(combined)}
    additions = 0
    for raw in bm25_results:
        result = _bm25_result(raw)
        chunk_id = result["chunk_id"]
        if chunk_id in positions:
            existing = combined[positions[chunk_id]]
            existing["bm25_score"] = result["bm25_score"]
            existing["bm25_rank"] = result["bm25_rank"]
            existing["retrieval_methods"] = ["dense", "bm25"]
            continue
        if additions >= max_bm25_additions:
            continue
        positions[chunk_id] = len(combined)
        combined.append(result)
        additions += 1
    return combined


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]], bm25_results: list[dict[str, Any]],
    *, limit: int = 5, rrf_k: int = 60,
    dense_weight: float = 1.0, bm25_weight: float = 1.0,
) -> list[dict[str, Any]]:
    if dense_weight < 0 or bm25_weight < 0 or dense_weight + bm25_weight <= 0:
        raise ValueError("RRF weights must be non-negative with a positive sum")
    fused: dict[str, dict[str, Any]] = {}
    for rank, source in enumerate(dense_results, start=1):
        chunk_id = source["chunk_id"]
        item = fused.setdefault(chunk_id, dict(source))
        item["dense_rank"] = rank
        item["dense_score"] = source.get("dense_score", source.get("score"))
        item.setdefault("retrieval_methods", [])
        if "dense" not in item["retrieval_methods"]:
            item["retrieval_methods"].append("dense")
        item["rrf_score"] = item.get("rrf_score", 0.0) + dense_weight / (rrf_k + rank)
    for rank, raw in enumerate(bm25_results, start=1):
        source = _bm25_result(raw)
        chunk_id = source["chunk_id"]
        item = fused.setdefault(chunk_id, dict(source))
        item["bm25_rank"] = rank
        item["bm25_score"] = source["bm25_score"]
        item.setdefault("retrieval_methods", [])
        if "bm25" not in item["retrieval_methods"]:
            item["retrieval_methods"].append("bm25")
        item["rrf_score"] = item.get("rrf_score", 0.0) + bm25_weight / (rrf_k + rank)
    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -float(item.get("rrf_score", 0.0)),
            int(item.get("dense_rank", 10**9)),
            int(item.get("bm25_rank", 10**9)),
            str(item.get("chunk_id")),
        ),
    )
    for item in ordered:
        item["score"] = item["rrf_score"]
        item["retrieval_method"] = "hybrid_rrf"
    return ordered[:limit]


def weighted_reciprocal_rank_fusion(
    rankings: Sequence[tuple[str, list[dict[str, Any]], float]],
    *,
    limit: int = 8,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse any number of normalized rankings without comparing raw scores.

    Each result must already use the common dense-result payload shape. This is
    useful for bilingual retrieval, where original and translated dense/BM25
    rankings are four independent signals.
    """
    if limit <= 0 or rrf_k < 0:
        raise ValueError("RRF limit must be positive and rrf_k non-negative")
    if not rankings or any(weight < 0 for _, _, weight in rankings):
        raise ValueError("RRF weights must be non-negative with a positive sum")
    weight_sum = sum(weight for _, _, weight in rankings)
    if weight_sum <= 0:
        raise ValueError("RRF weights must be non-negative with a positive sum")

    fused: dict[str, dict[str, Any]] = {}
    for signal_name, rows, weight in rankings:
        if not signal_name:
            raise ValueError("RRF signal names must not be empty")
        if weight == 0:
            continue
        for rank, source in enumerate(rows, start=1):
            chunk_id = str(source["chunk_id"])
            item = fused.setdefault(chunk_id, dict(source))
            item[f"{signal_name}_rank"] = rank
            item.setdefault("retrieval_signals", [])
            if signal_name not in item["retrieval_signals"]:
                item["retrieval_signals"].append(signal_name)
            item["rrf_score"] = item.get("rrf_score", 0.0) + weight / (rrf_k + rank)

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -float(item.get("rrf_score", 0.0)),
            min(
                (
                    int(item.get(f"{name}_rank", 10**9))
                    for name, _, _ in rankings
                ),
                default=10**9,
            ),
            str(item.get("chunk_id")),
        ),
    )
    for item in ordered:
        item["score"] = item["rrf_score"]
        item["retrieval_method"] = "weighted_multi_rrf"
        item["retrieval_methods"] = list(item.get("retrieval_signals", []))
    return ordered[:limit]


def diversify_results(
    results: list[dict[str, Any]],
    *,
    limit: int = 8,
    max_per_page: int = 2,
    max_per_section: int = 3,
) -> list[dict[str, Any]]:
    """Conservatively reduce page/section redundancy while preserving rank.

    A second pass fills any remaining slots, so this never shortens a sufficiently
    large ranking merely because the corpus has limited diversity.
    """
    if limit <= 0 or max_per_page <= 0 or max_per_section <= 0:
        raise ValueError("diversity limits must be positive")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    page_counts: dict[tuple[str, str], int] = {}
    section_counts: dict[tuple[str, str], int] = {}
    for row in results:
        chunk_id = str(row.get("chunk_id", ""))
        page_key = (str(row.get("doc_id", "")), str(row.get("pdf_page", row.get("page", ""))))
        section_key = (str(row.get("doc_id", "")), str(row.get("section", "")))
        if page_counts.get(page_key, 0) >= max_per_page:
            continue
        if section_counts.get(section_key, 0) >= max_per_section:
            continue
        selected.append(row)
        selected_ids.add(chunk_id)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        section_counts[section_key] = section_counts.get(section_key, 0) + 1
        if len(selected) == limit:
            return selected
    for row in results:
        if str(row.get("chunk_id", "")) in selected_ids:
            continue
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


class LanguageAwareRetriever:
    def __init__(
        self, *, dense: Searcher, bm25: Searcher, dense_top_k: int = 8,
        bm25_additions: int = 0, hybrid_candidates: int = 30,
        hybrid_top_k: int = 8, rrf_k: int = 60,
        arabic_dense_weight: float = 0.70,
        arabic_bm25_weight: float = 0.30,
        translator: Callable[[str], str] | None = None,
        original_dense_weight: float = 0.40,
        original_bm25_weight: float = 0.05,
        translated_dense_weight: float = 0.35,
        translated_bm25_weight: float = 0.20,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.dense_top_k = dense_top_k
        self.bm25_additions = bm25_additions
        self.hybrid_candidates = hybrid_candidates
        self.hybrid_top_k = hybrid_top_k
        self.rrf_k = rrf_k
        self.arabic_dense_weight = arabic_dense_weight
        self.arabic_bm25_weight = arabic_bm25_weight
        self.translator = translator
        self.bilingual_weights = {
            "original_dense": original_dense_weight,
            "original_bm25": original_bm25_weight,
            "translated_dense": translated_dense_weight,
            "translated_bm25": translated_bm25_weight,
        }
        if any(weight < 0 for weight in self.bilingual_weights.values()):
            raise ValueError("bilingual RRF weights must be non-negative")
        if sum(self.bilingual_weights.values()) <= 0:
            raise ValueError("bilingual RRF weights must have a positive sum")

    @staticmethod
    def _translation_metadata(
        translator: Callable[[str], str] | None,
        *, attempted: bool,
        succeeded: bool,
        translated_query: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "enabled": translator is not None,
            "attempted": attempted,
            "succeeded": succeeded,
            "provider": getattr(translator, "provider", "custom") if translator else None,
            "model": getattr(translator, "model", None) if translator else None,
            "translated_query": translated_query,
            "error": error,
        }

    def retrieve(self, query: str, *, doc_id: str | None = None) -> dict[str, Any]:
        language = detect_query_language(query)
        if language == "ar":
            dense_results = self.dense.search(
                query, limit=self.hybrid_candidates, doc_id=doc_id
            )
            bm25_results = self.bm25.search(
                query, limit=self.hybrid_candidates, doc_id=doc_id
            )
            translation = self._translation_metadata(
                self.translator, attempted=False, succeeded=False,
            )
            if self.translator is not None:
                try:
                    translated_query = self.translator(query).strip()
                    if not translated_query:
                        raise ValueError("translation was empty")
                    if ARABIC_PATTERN.search(translated_query):
                        raise ValueError("translation still contains Arabic script")
                    translated_dense = self.dense.search(
                        translated_query, limit=self.hybrid_candidates, doc_id=doc_id
                    )
                    translated_bm25 = self.bm25.search(
                        translated_query, limit=self.hybrid_candidates, doc_id=doc_id
                    )
                    results = weighted_reciprocal_rank_fusion(
                        [
                            ("original_dense", dense_results, self.bilingual_weights["original_dense"]),
                            (
                                "original_bm25",
                                [_bm25_result(row) for row in bm25_results],
                                self.bilingual_weights["original_bm25"],
                            ),
                            (
                                "translated_dense",
                                translated_dense,
                                self.bilingual_weights["translated_dense"],
                            ),
                            (
                                "translated_bm25",
                                [_bm25_result(row) for row in translated_bm25],
                                self.bilingual_weights["translated_bm25"],
                            ),
                        ],
                        limit=self.hybrid_top_k,
                        rrf_k=self.rrf_k,
                    )
                    route = "arabic_dual_query_weighted_rrf"
                    translation = self._translation_metadata(
                        self.translator,
                        attempted=True,
                        succeeded=True,
                        translated_query=translated_query,
                    )
                except Exception as exc:
                    results = reciprocal_rank_fusion(
                        dense_results,
                        bm25_results,
                        limit=self.hybrid_top_k,
                        rrf_k=self.rrf_k,
                        dense_weight=self.arabic_dense_weight,
                        bm25_weight=self.arabic_bm25_weight,
                    )
                    route = "arabic_weighted_rrf_translation_fallback"
                    translation = self._translation_metadata(
                        self.translator,
                        attempted=True,
                        succeeded=False,
                        error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
            else:
                results = reciprocal_rank_fusion(
                    dense_results,
                    bm25_results,
                    limit=self.hybrid_top_k,
                    rrf_k=self.rrf_k,
                    dense_weight=self.arabic_dense_weight,
                    bm25_weight=self.arabic_bm25_weight,
                )
                route = "arabic_weighted_rrf"
        else:
            dense_results = self.dense.search(query, limit=self.dense_top_k, doc_id=doc_id)
            results = dense_results
            route = "english_dense_top8"
            translation = self._translation_metadata(
                self.translator, attempted=False, succeeded=False,
            )
        return {
            "query": query,
            "language": language,
            "route": route,
            "doc_id_filter": doc_id,
            "translation": translation,
            "result_count": len(results),
            "results": results,
        }
