"""Shared helpers for compiled LangGraph agents."""

from __future__ import annotations

from typing import Any


def final_text(result: dict[str, Any]) -> str:
    """Return user-visible text from the final message of an agent invocation."""
    messages = result.get("messages") or []
    if not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()
