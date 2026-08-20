"""Provider selection and failover for evidence-grounded generation."""

from __future__ import annotations

from typing import Any

from .groq_llm import GroqTextGenerator
from .ollama_llm import OllamaTextGenerator


class FallbackTextGenerator:
    """Try the hosted provider and use Ollama only for provider failures."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback
        self._active = primary

    @property
    def model(self) -> str:
        return str(self._active.model)

    @property
    def provider(self) -> str:
        return str(getattr(self._active, "provider", "unknown"))

    def complete(self, user_prompt: str, correction: str | None = None) -> dict:
        self._active = self.primary
        try:
            return self.primary.complete(user_prompt, correction)
        except RuntimeError:
            self._active = self.fallback
            return self.fallback.complete(user_prompt, correction)


def build_text_generator(
    *, provider: str, model: str, ollama_url: str,
    groq_api_key: str | None = None, timeout_seconds: float = 60,
    fallback_model: str = "qwen3:4b",
) -> Any:
    """Build the requested generator while retaining a local safe fallback."""
    selected = provider.strip().lower()
    if selected not in {"auto", "groq", "ollama"}:
        raise ValueError("generation provider must be auto, groq, or ollama")
    fallback = OllamaTextGenerator(
        model=fallback_model if selected != "ollama" else model,
        base_url=ollama_url,
        # A local fallback is a resilience path, not an unbounded queue. Keep
        # it below the API request budget so a slow workstation cannot hold a
        # multi-agent fan-out open for five minutes.
        timeout_seconds=max(5, min(int(timeout_seconds), 20)),
    )
    fallback.provider = "ollama"
    use_groq = selected == "groq" or (selected == "auto" and bool(groq_api_key))
    if not use_groq:
        return fallback
    if not groq_api_key:
        raise ValueError("GENERATION_PROVIDER=groq requires GROQ_API_KEY")
    primary = GroqTextGenerator(
        api_key=groq_api_key, model=model, timeout_seconds=timeout_seconds,
    )
    return FallbackTextGenerator(primary, fallback)
