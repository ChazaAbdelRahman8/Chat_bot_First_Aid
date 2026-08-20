"""Deterministic schema, citation-label, and response-language validation."""

from __future__ import annotations

import json
import re
from typing import Any

from rag.retrieval.hybrid import detect_query_language


CITATION_PATTERN = re.compile(r"\[(S[1-9]\d*)\]")
CITATION_GROUP_PATTERN = re.compile(r"\[((?:S[1-9]\d*)(?:\s*,\s*S[1-9]\d*)+)\]")
ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
FACT_UNIT_SPLIT = re.compile(r"(?<=[.!?\u061f])\s+(?!\[S[1-9]\d*\])|\n+")


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("generation model did not return a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("generation response JSON must be an object")
    return value


def citation_labels(answer: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(
        r"\[(S[1-9]\d*(?:\s*,\s*S[1-9]\d*)*)\]", answer
    ):
        labels.extend(part.strip() for part in match.group(1).split(","))
    return list(dict.fromkeys(labels))


def _citation_present(text: str) -> bool:
    return bool(CITATION_PATTERN.search(text) or CITATION_GROUP_PATTERN.search(text))


def _protect_abbreviations(text: str) -> str:
    """Protect common medical/manual abbreviations from sentence splitting."""
    protected = text
    for abbreviation in ("St.", "e.g.", "i.e.", "vs.", "Dr.", "Mr.", "Mrs."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<DOT>"))
    return protected


def uncited_factual_units(answer: str) -> list[str]:
    """Return nonempty prose sentences/bullets that have no inline citation."""
    units = []
    for raw in FACT_UNIT_SPLIT.split(_protect_abbreviations(answer)):
        unit = raw.replace("<DOT>", ".").strip().lstrip("-\u2022* ").strip()
        if not unit or not any(character.isalpha() for character in unit):
            continue
        # A bare numbered/list lead-in is formatting, not a factual claim.
        if re.search(r":\s*\d+\.?$", unit):
            continue
        if not _citation_present(unit):
            units.append(unit)
    return units


def _language_matches(query: str, response_text: str) -> bool:
    if not response_text.strip():
        return False
    query_language = detect_query_language(query)
    arabic_chars = len(ARABIC_PATTERN.findall(response_text))
    letters = sum(character.isalpha() for character in response_text)
    if not letters:
        return True
    ratio = arabic_chars / letters
    # Both directions require the response to be *dominantly* in the query's
    # language - a single untranslated term (Arabic in an English answer, or
    # vice versa) must not be enough to pass.
    return ratio >= 0.5 if query_language == "ar" else ratio < 0.5


def validate_generation(
    raw: dict[str, Any], *, query: str, allowed_labels: set[str],
) -> dict[str, Any]:
    required = {"answer", "cited_sources", "abstain", "insufficient_evidence_reason"}
    if set(raw) != required:
        raise ValueError(f"generation JSON keys must be exactly {sorted(required)}")
    answer = raw["answer"]
    cited = raw["cited_sources"]
    abstain = raw["abstain"]
    reason = raw["insufficient_evidence_reason"]
    if not isinstance(answer, str) or not isinstance(reason, str):
        raise ValueError("answer and insufficient_evidence_reason must be strings")
    if not isinstance(abstain, bool):
        raise ValueError("abstain must be a boolean")
    if not isinstance(cited, list) or any(not isinstance(label, str) for label in cited):
        raise ValueError("cited_sources must be a list of strings")
    cited = list(dict.fromkeys(cited))
    inline = citation_labels(answer)
    invalid = (set(cited) | set(inline)) - allowed_labels
    if invalid:
        raise ValueError(f"unknown citation labels: {sorted(invalid)}")
    # Inline labels are authoritative; cited_sources is redundant metadata.
    cited = inline
    if abstain:
        if cited or inline:
            raise ValueError("an abstention must not include citations")
        if not reason.strip():
            raise ValueError("an abstention requires insufficient_evidence_reason")
    else:
        if not answer.strip():
            raise ValueError("a non-abstaining answer must not be empty")
        if not cited:
            raise ValueError("a non-abstaining answer requires at least one citation")
        uncited = uncited_factual_units(answer)
        if uncited:
            raise ValueError(f"every factual sentence or bullet requires a citation: {uncited[:2]}")
        if reason.strip():
            raise ValueError("a non-abstaining answer must have an empty reason")
    language_text = answer if not abstain else f"{answer} {reason}"
    if not _language_matches(query, language_text):
        raise ValueError("response language does not match the question")
    return {
        "answer": answer.strip(),
        "cited_sources": cited,
        "abstain": abstain,
        "insufficient_evidence_reason": reason.strip(),
    }
