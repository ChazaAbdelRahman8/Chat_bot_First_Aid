"""this is the exact client Agent System A uses to communicate with Agent System B over HTTP."""
from __future__ import annotations

import uuid
from typing import Any

import httpx
"""Agent A uses AgentSystemBClient and the Python httpx library to send an HTTP POST request to Agent B’s /v1/chat endpoint. The request contains the query, conversation ID, unique request ID, and source, and Agent B returns a JSON response."""

class AgentSystemBClient:
    def __init__(self, base_url: str, *, timeout: float = 20, retries: int = 1) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def chat(self, query: str, conversation_id: str) -> dict[str, Any]:
        payload = {
            "query": query, "conversation_id": conversation_id,
            "request_id": str(uuid.uuid4()), "source": "agent-system-a",
        }
        last_error: Exception | None = None
        for _attempt in range(self.retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/v1/chat", json=payload, timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
        raise RuntimeError("Agent System B request failed") from last_error
