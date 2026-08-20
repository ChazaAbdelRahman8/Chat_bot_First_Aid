"""API and domain schemas for the synthetic appointment agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=100)
    request_id: str | None = Field(default=None, max_length=100)
    source: str = Field(default="agent-system-a", max_length=100)


class ChatResponse(BaseModel):
    request_id: str
    conversation_id: str
    status: Literal[
        "needs_information", "matches_found", "awaiting_confirmation",
        "booked", "no_matches", "crisis_escalation", "error",
    ]
    message: str
    matches: list[dict[str, Any]] = Field(default_factory=list)
    appointment: dict[str, Any] | None = None
    missing_fields: list[str] = Field(default_factory=list)
    framework: str = "google-adk"
    execution_mode: str = "adk"
    synthetic: bool = True
    usage: dict[str, Any] | None = None
