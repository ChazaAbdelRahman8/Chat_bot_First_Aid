"""Dependency-light Ollama client for Qwen3 text generation."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompt import SYSTEM_PROMPT
from .validation import extract_json_object


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_sources": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
        "insufficient_evidence_reason": {"type": "string"},
    },
    "required": ["answer", "cited_sources", "abstain", "insufficient_evidence_reason"],
    "additionalProperties": False,
}


class OllamaTextGenerator:
    provider = "ollama"

    def __init__(
        self, *, model: str = "qwen3:4b", base_url: str = "http://localhost:11434",
        timeout_seconds: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(self, user_prompt: str, correction: str | None = None) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if correction:
            messages.append({"role": "user", "content": correction})
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": messages,
            "format": OUTPUT_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 768},
            "keep_alive": "10m",
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc.reason}") from exc
        content = result.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Ollama returned an empty generation response")
        return extract_json_object(content)
