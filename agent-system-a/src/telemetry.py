"""Request-scoped, thread-safe telemetry for agent evaluation and operations."""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator, Mapping

from langchain_core.callbacks.usage import UsageMetadataCallbackHandler


GROQ_PRICES_PER_MILLION = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.6-27b": (0.60, 3.00),
}


class TelemetryCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tool_events: list[dict[str, Any]] = []
        self.direct_usage: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "input_tokens": 0, "output_tokens": 0,
                "provider": "unknown", "model_calls": 0,
            }
        )

    def tool(
        self, name: str, arguments: dict[str, Any], *, success: bool,
        error: str | None = None,
    ) -> None:
        event = {"tool": name, "arguments": arguments, "success": success}
        if error:
            event["error"] = error[:300]
        with self._lock:
            self.tool_events.append(event)

    def usage(
        self, model: str, *, input_tokens: int, output_tokens: int,
        provider: str, model_calls: int = 1,
    ) -> None:
        with self._lock:
            row = self.direct_usage[model]
            row["input_tokens"] += max(0, int(input_tokens))
            row["output_tokens"] += max(0, int(output_tokens))
            row["provider"] = provider
            row["model_calls"] += max(0, int(model_calls))


class CountingUsageCallback(UsageMetadataCallbackHandler):
    """LangChain usage callback that also counts completed model calls."""

    def __init__(self) -> None:
        super().__init__()
        self.model_calls = 0
        self._call_lock = threading.Lock()

    def on_llm_end(self, response, **kwargs: Any) -> None:
        with self._call_lock:
            self.model_calls += 1
        super().on_llm_end(response, **kwargs)


_CURRENT: ContextVar[TelemetryCollector | None] = ContextVar(
    "agent_telemetry_collector", default=None,
)


@contextmanager
def telemetry_context() -> Generator[TelemetryCollector, None, None]:
    collector = TelemetryCollector()
    token = _CURRENT.set(collector)
    try:
        yield collector
    finally:
        _CURRENT.reset(token)


def record_tool(
    name: str, arguments: dict[str, Any], *, success: bool,
    error: str | None = None,
) -> None:
    collector = _CURRENT.get()
    if collector is not None:
        collector.tool(name, arguments, success=success, error=error)


def record_usage(
    model: str, *, input_tokens: int, output_tokens: int,
    provider: str, model_calls: int = 1,
) -> None:
    collector = _CURRENT.get()
    if collector is not None:
        collector.usage(
            model, input_tokens=input_tokens, output_tokens=output_tokens,
            provider=provider, model_calls=model_calls,
        )


def summarize_usage(
    callback_usage: Mapping[str, Mapping[str, Any]], collector: TelemetryCollector,
    *, callback_model_calls: int = 0,
) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    for model, usage in callback_usage.items():
        merged[model] = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "provider": "groq" if model in GROQ_PRICES_PER_MILLION else "unknown",
            "model_calls": 0,
        }
    for model, usage in collector.direct_usage.items():
        target = merged.setdefault(
            model, {
                "input_tokens": 0, "output_tokens": 0,
                "provider": usage["provider"], "model_calls": 0,
            },
        )
        target["input_tokens"] += int(usage["input_tokens"])
        target["output_tokens"] += int(usage["output_tokens"])
        if target["provider"] == "unknown":
            target["provider"] = usage["provider"]
        target["model_calls"] += int(usage.get("model_calls", 0))
    total_input = total_output = 0
    total_cost = 0.0
    unknown_pricing: list[str] = []
    models: dict[str, dict[str, Any]] = {}
    for model, usage in sorted(merged.items()):
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
        total_input += input_tokens
        total_output += output_tokens
        prices = GROQ_PRICES_PER_MILLION.get(model)
        if prices:
            cost = input_tokens * prices[0] / 1_000_000 + output_tokens * prices[1] / 1_000_000
        elif usage.get("provider") == "ollama":
            cost = 0.0
        else:
            cost = None
            unknown_pricing.append(model)
        if cost is not None:
            total_cost += cost
        models[model] = {
            **usage, "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": round(cost, 8) if cost is not None else None,
        }
    return {
        "models": models,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "model_calls": callback_model_calls + sum(
            int(row.get("model_calls", 0)) for row in collector.direct_usage.values()
        ),
        "estimated_cost_usd": round(total_cost, 8),
        "unknown_pricing_models": unknown_pricing,
        "pricing_basis": "Groq on-demand USD per 1M tokens; local Ollama API cost is $0",
    }
