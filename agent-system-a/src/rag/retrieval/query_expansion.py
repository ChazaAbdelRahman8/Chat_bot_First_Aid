"""Schema-constrained, answer-free query expansion for retrieval."""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "medical_rewrite": {"type": "string"},
        "synonym_variant": {"type": "string"},
    },
    "required": ["medical_rewrite", "synonym_variant"],
    "additionalProperties": False,
}

EXPANSION_SYSTEM_PROMPT = """You produce search queries for a first-aid manual corpus.
Given one user question, return exactly two short query variants in the same
language as the input (preserve a mixed-language input as mixed):
1. medical_rewrite: concise clinical and procedural keywords expressing the
   same information need.
2. synonym_variant: a natural alternative phrasing with useful first-aid
   synonyms or common terminology.

Preserve all symptoms, body sites, ages, numbers, units, comparisons, negation,
and named manuals. Do not add facts or assumptions. Do not answer the question,
give instructions, diagnose, cite sources, or translate it into another language.
Return only the JSON object required by the schema."""

ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


def validate_expansion(query: str, expansion: dict[str, str]) -> None:
    values = [str(expansion.get(name, "")).strip() for name in EXPANSION_SCHEMA["required"]]
    if any(not value or len(value) > 500 for value in values):
        raise ValueError("query variants must contain 1-500 characters")
    if values[0].casefold() == values[1].casefold():
        raise ValueError("query variants must be distinct")
    original_is_arabic = bool(ARABIC_PATTERN.search(query))
    if any(bool(ARABIC_PATTERN.search(value)) != original_is_arabic for value in values):
        raise ValueError("query variants must preserve the routed language")
    required_numbers = set(NUMBER_PATTERN.findall(query))
    for value in values:
        if not required_numbers.issubset(set(NUMBER_PATTERN.findall(value))):
            raise ValueError("query variants must preserve every number")


def expand_query(
    query: str,
    *,
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    timeout_seconds: int = 180,
) -> dict[str, str]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if ARABIC_PATTERN.search(query):
        routing_constraint = (
            "ROUTING LANGUAGE: Arabic or Arabic-English code-switched. "
            "Both output variants MUST contain Arabic script. Do not translate to English."
        )
    else:
        routing_constraint = (
            "ROUTING LANGUAGE: English. Both output variants MUST remain English "
            "and must not introduce Arabic script."
        )
    routed_query = f"{routing_constraint}\n\nQUERY:\n{query}"
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": routed_query},
        ],
        "format": EXPANSION_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 384},
        "keep_alive": "10m",
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {exc.reason}") from exc
    content = body.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        output = {
            "medical_rewrite": str(parsed["medical_rewrite"]).strip(),
            "synonym_variant": str(parsed["synonym_variant"]).strip(),
        }
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("expansion model did not return the required JSON") from exc
    validate_expansion(query, output)
    return output
