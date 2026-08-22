"""Deterministic, schema-constrained Arabic query translation for retrieval."""

from __future__ import annotations

import json
from functools import partial
from typing import Any, Callable, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from telemetry import record_usage


TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
    "additionalProperties": False,
}

TRANSLATION_SYSTEM_PROMPT = """You translate first-aid search queries into English.
Return only the JSON object required by the schema. Preserve all medical terms,
proper names, abbreviations, numbers, units, comparisons, and negation. Do not
answer the query, explain it, add advice, or remove constraints. For mixed Arabic
and English, translate the Arabic while preserving already-English technical terms.
Verify anatomy carefully: كاحل means ankle, كتف means shoulder, رسغ means wrist,
ركبة means knee, مرفق means elbow, and ورك means hip."""

ARABIC_RETRIEVAL_CONCEPTS = (
    (("كاحل",), "ankle"),
    (("التواء", "التوى", "ملتو"), "sprain"),
    (("تورم", "متورم", "منتفخ"), "swelling"),
    (("ألم", "مؤلم", "وجع"), "pain"),
    (("كتف",), "shoulder"),
    (("رسغ",), "wrist"),
    (("ركبة",), "knee"),
    (("مرفق",), "elbow"),
    (("ورك",), "hip"),
    (("نزيف",), "bleeding"),
    (("حرق",), "burn"),
    (("اختناق",), "choking"),
)


def preserve_arabic_retrieval_concepts(query: str, translation: str) -> str:
    """Append deterministic English anchors for critical Arabic medical terms."""
    concepts = [
        english
        for arabic_variants, english in ARABIC_RETRIEVAL_CONCEPTS
        if any(variant in query for variant in arabic_variants)
    ]
    missing = [term for term in concepts if term not in translation.lower()]
    if not missing:
        return translation.strip()
    return f"{translation.strip()} Retrieval concepts: {' '.join(missing)}."


def translate_query_to_english(
    query: str,
    *,
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    timeout_seconds: int = 180,
) -> str:
    if not query.strip():
        raise ValueError("query must not be empty")
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "format": TRANSLATION_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 256},
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
        translation = str(json.loads(content)["translation"]).strip()
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("translation model did not return the required JSON") from exc
    if not translation:
        raise ValueError("translation model returned an empty translation")
    return preserve_arabic_retrieval_concepts(query, translation)


def translate_query_to_english_groq(
    query: str,
    *,
    api_key: str,
    model: str = "qwen/qwen3.6-27b",
    timeout_seconds: float = 20,
    client: Any | None = None,
) -> str:
    """Translate through Groq while preserving the same strict JSON contract."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not api_key.strip():
        raise ValueError("Groq query translation requires GROQ_API_KEY")
    if client is None:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    try:
        completion = client.chat.completions.create(
            model=cast(Any, model),
            messages=cast(Any, [
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]),
            temperature=0,
            max_completion_tokens=256,
            response_format={"type": "json_object"},
            reasoning_effort="none",
            reasoning_format="hidden",
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Groq query translation failed ({type(exc).__name__}): {str(exc)[:300]}"
        ) from exc
    content = completion.choices[0].message.content or ""
    usage = getattr(completion, "usage", None)
    if usage is not None:
        record_usage(
            model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            provider="groq",
        )
    try:
        translation = str(json.loads(content)["translation"]).strip()
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Groq translation model did not return the required JSON") from exc
    if not translation:
        raise ValueError("Groq translation model returned an empty translation")
    return preserve_arabic_retrieval_concepts(query, translation)


def translate_query_to_english_openrouter(
    query: str,
    *,
    api_key: str,
    model: str = "qwen/qwen3.5-flash-02-23",
    base_url: str = "https://openrouter.ai/api/v1",
    timeout_seconds: float = 20,
) -> str:
    """Translate through OpenRouter using its OpenAI-compatible endpoint."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not api_key.strip():
        raise ValueError("OpenRouter query translation requires OPENROUTER_API_KEY")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "First Aid Companion",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach OpenRouter at {base_url}: {exc.reason}"
        ) from exc
    try:
        content = body["choices"][0]["message"]["content"] or ""
        translation = str(json.loads(content)["translation"]).strip()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "OpenRouter translation model did not return the required JSON"
        ) from exc
    if not translation:
        raise ValueError("OpenRouter translation model returned an empty translation")
    usage = body.get("usage") or {}
    record_usage(
        model,
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        provider="openrouter",
    )
    return preserve_arabic_retrieval_concepts(query, translation)


def build_query_translator(
    *,
    enabled: bool = True,
    provider: str = "auto",
    openrouter_api_key: str | None = None,
    openrouter_model: str = "qwen/qwen3.5-flash-02-23",
    openrouter_url: str = "https://openrouter.ai/api/v1",
    groq_api_key: str | None = None,
    groq_model: str = "qwen/qwen3.6-27b",
    ollama_model: str = "qwen3:4b",
    ollama_url: str = "http://localhost:11434",
    timeout_seconds: float = 20,
) -> Callable[[str], str] | None:
    """Build an OpenRouter/Groq translator with a local Ollama fallback."""
    if not enabled:
        return None
    selected = provider.strip().lower()
    if selected not in {"auto", "openrouter", "groq", "ollama"}:
        raise ValueError(
            "Arabic translation provider must be auto, openrouter, groq, or ollama"
        )
    if selected == "openrouter" and not openrouter_api_key:
        raise ValueError("OpenRouter Arabic translation requires OPENROUTER_API_KEY")
    if selected == "groq" and not groq_api_key:
        raise ValueError("Groq Arabic translation requires GROQ_API_KEY")
    if selected == "groq":
        translator = partial(
            translate_query_to_english_groq,
            api_key=str(groq_api_key),
            model=groq_model,
            timeout_seconds=timeout_seconds,
        )
        translator.provider = "groq"  # type: ignore[attr-defined]
        translator.model = groq_model  # type: ignore[attr-defined]
        return translator
    if selected == "openrouter":
        translator = partial(
            translate_query_to_english_openrouter,
            api_key=str(openrouter_api_key),
            model=openrouter_model,
            base_url=openrouter_url,
            timeout_seconds=timeout_seconds,
        )
        translator.provider = "openrouter"  # type: ignore[attr-defined]
        translator.model = openrouter_model  # type: ignore[attr-defined]
        return translator
    ollama_translator = partial(
        translate_query_to_english,
        model=ollama_model,
        base_url=ollama_url,
        timeout_seconds=max(1, int(timeout_seconds)),
    )
    if selected == "auto":
        candidates: list[tuple[str, str, Callable[[str], str]]] = []
        if openrouter_api_key:
            candidates.append((
                "openrouter",
                openrouter_model,
                partial(
                    translate_query_to_english_openrouter,
                    api_key=str(openrouter_api_key),
                    model=openrouter_model,
                    base_url=openrouter_url,
                    timeout_seconds=timeout_seconds,
                ),
            ))
        if groq_api_key:
            candidates.append((
                "groq",
                groq_model,
                partial(
                    translate_query_to_english_groq,
                    api_key=str(groq_api_key),
                    model=groq_model,
                    timeout_seconds=timeout_seconds,
                ),
            ))
        candidates.append(("ollama", ollama_model, ollama_translator))

        def translator_with_fallback(query: str) -> str:
            errors: list[str] = []
            for candidate_provider, _, candidate in candidates:
                try:
                    return candidate(query)
                except Exception as exc:
                    errors.append(
                        f"{candidate_provider}: {type(exc).__name__}: {str(exc)[:200]}"
                    )
            raise RuntimeError(
                "Arabic query translation failed through all providers: "
                + " | ".join(errors)
            )

        translator_with_fallback.provider = "_then_".join(  # type: ignore[attr-defined]
            candidate[0] for candidate in candidates
        )
        translator_with_fallback.model = "|".join(  # type: ignore[attr-defined]
            candidate[1] for candidate in candidates
        )
        return translator_with_fallback
    ollama_translator.provider = "ollama"  # type: ignore[attr-defined]
    ollama_translator.model = ollama_model  # type: ignore[attr-defined]
    return ollama_translator
