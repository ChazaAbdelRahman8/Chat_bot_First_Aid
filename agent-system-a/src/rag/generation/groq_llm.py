"""Groq-hosted text generation with the same JSON contract as Ollama."""

from __future__ import annotations

from typing import Any, cast

from .prompt import SYSTEM_PROMPT
from .validation import extract_json_object
from telemetry import record_usage


class GroqTextGenerator:
    provider = "groq"

    def __init__(
        self, *, api_key: str, model: str = "qwen/qwen3.6-27b",
        timeout_seconds: float = 60, client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq generation requires GROQ_API_KEY")
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self.client = client
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, user_prompt: str, correction: str | None = None) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if correction:
            messages.append({"role": "user", "content": correction})
        try:
            completion = self.client.chat.completions.create(
                model=cast(Any, self.model),
                messages=cast(Any, messages),
                temperature=0,
                max_completion_tokens=768,
                response_format={"type": "json_object"},
                reasoning_effort="none",
                reasoning_format="hidden",
                stream=False,
            )
        except Exception as exc:
            # Normalize provider/network failures so the fallback wrapper can
            # distinguish them from a repairable malformed model response.
            raise RuntimeError(
                f"Groq generation failed ({type(exc).__name__}): {str(exc)[:500]}"
            ) from exc
        content = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        if usage is not None:
            record_usage(
                self.model,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                provider="groq",
            )
        if not content:
            raise ValueError("Groq returned an empty generation response")
        return extract_json_object(content)
