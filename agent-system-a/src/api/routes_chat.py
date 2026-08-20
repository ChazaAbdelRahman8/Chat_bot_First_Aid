"""Chat endpoint using the production retrieval and generation path."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from .schemas import ChatRequest, ChatResponse, RouteEvaluationRequest
from .agent_logging import log_agent_event, log_agent_exception


router = APIRouter(prefix="/v1", tags=["chat"])
SSE_HEARTBEAT_SECONDS = 10.0


def runtime_dependency():  # replaced by main.py dependency override
    raise RuntimeError("RAG runtime is not configured")


def agent_system_dependency():
    raise RuntimeError("System A supervisor is not configured")


def conversation_store_dependency():
    raise RuntimeError("Conversation store is not configured")


def _sse_event(event: str, data: dict) -> str:
    """Serialize one named server-sent event without exposing Python errors."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _answer_and_store(system, store, query: str, conversation_id: str | None, image_path=None):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    selected_id = conversation_id
    log_agent_event(
        "request_received",
        request_id=request_id,
        conversation_id=conversation_id,
        input=query,
        input_length=len(query),
        image_attached=image_path is not None,
    )
    try:
        selected_id = conversation_id or store.create()
        history = store.recent_messages(selected_id, limit=12)
        result = system.answer(
            query, image_path=image_path, history=history, conversation_id=selected_id,
        )
        generation = result.get("generation", {})
        orchestration = result.get("orchestration", {})
        answer_agent = result.get("answer_agent", {})
        answer = str(generation.get("answer", ""))
        store.append_turn(
            selected_id,
            query,
            answer,
            metadata={
                "route": result.get("route", []),
                "status": generation.get("status"),
                "abstention_category": generation.get("abstention_category"),
                "citations": generation.get("citations", []),
                "orchestration": orchestration,
            },
        )
        log_agent_event(
            "request_completed",
            request_id=request_id,
            conversation_id=selected_id,
            input=query,
            output=answer,
            output_length=len(answer),
            status=generation.get("status"),
            abstention_category=generation.get("abstention_category"),
            route=result.get("route", []),
            citation_labels=generation.get("citation_labels", []),
            citations=[{
                "label": citation.get("label"),
                "doc_id": citation.get("doc_id"),
                "pdf_page": citation.get("pdf_page", citation.get("page")),
            } for citation in generation.get("citations", [])],
            supervisor_provider=orchestration.get("supervisor_provider"),
            supervisor_model=orchestration.get("supervisor_model"),
            supervisor_latency_ms=orchestration.get("supervisor_latency_ms"),
            selected_routes=orchestration.get("selected_routes", []),
            input_guard=orchestration.get("input_guard"),
            router_errors=orchestration.get("router_errors", []),
            specialist_failures=orchestration.get("specialist_failures", []),
            specialist_diagnostics=orchestration.get("specialist_diagnostics", []),
            answer_mode=orchestration.get("answer_mode"),
            output_guard=orchestration.get("output_guard"),
            output_guard_errors=orchestration.get("output_guard_errors", []),
            answer_agent_provider=answer_agent.get("provider_used"),
            answer_agent_fallback=answer_agent.get("provider_fallback", False),
            answer_agent_errors=answer_agent.get("provider_errors", []),
            visual_references=[{
                "visual_id": visual.get("visual_id"),
                "doc_id": visual.get("doc_id"),
                "pdf_page": visual.get("pdf_page"),
            } for visual in result.get("visuals", [])],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return {"conversation_id": selected_id, **result}
    except Exception as exc:
        log_agent_exception(
            "request_failed",
            request_id=request_id,
            conversation_id=selected_id,
            input=query,
            image_attached=image_path is not None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    system=Depends(agent_system_dependency),
    store=Depends(conversation_store_dependency),
) -> dict:
    return await run_in_threadpool(
        _answer_and_store, system, store, request.query, request.conversation_id,
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    system=Depends(agent_system_dependency),
    store=Depends(conversation_store_dependency),
) -> StreamingResponse:
    """Run the normal chat path and deliver lifecycle/result events over SSE.

    The agent workflow currently produces a validated answer atomically rather
    than token by token. Heartbeats keep the HTTP stream alive while that work
    runs, and the final ``result`` event has the same payload as ``POST /chat``.
    """

    async def events():
        yield _sse_event("status", {"status": "processing"})
        task = asyncio.create_task(run_in_threadpool(
            _answer_and_store,
            system,
            store,
            payload.query,
            payload.conversation_id,
        ))
        try:
            while True:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(task), timeout=SSE_HEARTBEAT_SECONDS,
                    )
                    break
                except TimeoutError:
                    if await request.is_disconnected():
                        return
                    yield _sse_event("heartbeat", {"status": "processing"})
            yield _sse_event("result", result)
            yield _sse_event("done", {})
        except asyncio.CancelledError:
            raise
        except Exception:
            yield _sse_event("error", {
                "status": "error",
                "message": "The request could not be completed safely.",
            })
            yield _sse_event("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/route")
async def evaluate_route(
    request: RouteEvaluationRequest,
    system=Depends(agent_system_dependency),
) -> dict:
    """Evaluate the real guard/supervisor decision without running specialists."""
    return await run_in_threadpool(
        system.route_only,
        request.query,
        image_attached=request.image_attached,
        history=request.history,
    )


@router.post("/chat/visual", response_model=ChatResponse)
async def visual_chat(
    query: str = Form(..., min_length=1, max_length=4000),
    image: UploadFile = File(...),
    conversation_id: str | None = Form(default=None, max_length=100),
    system=Depends(agent_system_dependency),
    store=Depends(conversation_store_dependency),
) -> dict:
    allowed = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    suffix = allowed.get(image.content_type or "")
    if suffix is None:
        raise HTTPException(400, "image must be JPEG, PNG, or WebP")
    limit = int(os.getenv("MAX_IMAGE_UPLOAD_BYTES", str(15 * 1024 * 1024)))
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(handle.name)
    total = 0
    try:
        while chunk := await image.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise HTTPException(413, f"image exceeds {limit} bytes")
            handle.write(chunk)
        handle.close()
        try:
            with Image.open(path) as opened:
                opened.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(400, "uploaded file is not a valid image") from exc
        return await run_in_threadpool(
            _answer_and_store, system, store, query, conversation_id, path,
        )
    finally:
        handle.close()
        await image.close()
        path.unlink(missing_ok=True)
