"""FastAPI request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    doc_id: str | None = None
    conversation_id: str | None = Field(default=None, min_length=1, max_length=100)


class ChatResponse(BaseModel):
    conversation_id: str
    query: str
    route: list[str] = Field(default_factory=list)
    specialists: list[dict[str, Any]] = Field(default_factory=list)
    retrieval: dict[str, Any] | None
    generation: dict[str, Any]
    answer_agent: dict[str, Any] | None = None
    orchestration: dict[str, Any] | None = None
    visuals: list[dict[str, Any]] = Field(default_factory=list)
    appointment: dict[str, Any] | None = None


class RouteEvaluationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    image_attached: bool = False
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    pages: int | None = None
    chunks: int | None = None
    error: str | None = None


class JobResponse(BaseModel):
    job_id: str
    doc_id: str
    status: Literal["queued", "running", "complete", "failed"]
    detail: dict[str, Any] = Field(default_factory=dict)
