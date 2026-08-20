"""Deterministic validation for final multi-agent responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from guardrails.reference_validation import validate_references
from rag.retrieval.hybrid import detect_query_language


ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


@dataclass(frozen=True)
class OutputGuardDecision:
    valid: bool
    errors: tuple[str, ...] = ()


def _language_matches(query: str, answer: str) -> bool:
    if not answer.strip():
        return False
    arabic_chars = len(ARABIC_PATTERN.findall(answer))
    letters = sum(character.isalpha() for character in answer)
    if not letters:
        return True
    arabic_ratio = arabic_chars / letters
    if detect_query_language(query) == "ar":
        return arabic_ratio >= 0.5
    return arabic_ratio < 0.5


def validate_agent_output(
    *, query: str, response: dict[str, Any], answer_mode: str,
    specialists: Sequence[Mapping[str, Any]],
) -> OutputGuardDecision:
    """Validate response shape and evidence references before returning it."""
    generation = response.get("generation") or {}
    answer = str(generation.get("answer") or "")
    status = generation.get("status")
    errors: list[str] = []

    if status not in {"answered", "abstained"}:
        errors.append("generation status must be answered or abstained")
    if status == "answered" and not answer.strip():
        errors.append("answered response is empty")
    if status == "abstained" and not generation.get("abstention_category"):
        errors.append("abstention category is missing")
    if answer and not _language_matches(query, answer):
        errors.append("response language does not match the query")

    expected_route = list(dict.fromkeys(
        item["agent"] for item in specialists if item.get("ok", True)
    ))
    actual_route = response.get("route") or generation.get("agent_route") or []
    if expected_route and list(actual_route) != expected_route:
        errors.append("response route does not match successful specialists")

    if answer_mode == "synthesis" and status == "answered":
        evidence = "\n\n".join(
            f"SPECIALIST={item['agent']}\n{item.get('output', '')}"
            for item in specialists if item.get("ok", True)
        )
        reference_check = validate_references(
            answer,
            evidence,
            web_used=any(
                item.get("agent") == "web_search" and item.get("ok", True)
                for item in specialists
            ),
        )
        errors.extend(reference_check.errors)

    return OutputGuardDecision(valid=not errors, errors=tuple(dict.fromkeys(errors)))
