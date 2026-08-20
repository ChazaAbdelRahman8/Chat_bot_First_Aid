"""Independent FastAPI service for the Google ADK appointment agent."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from adk_agent import AdkAppointmentRuntime
from models import ChatRequest, ChatResponse
from reminders import ReminderScheduler
from service import AppointmentService
from store import MemoryAppointmentStore, MongoAppointmentStore


def create_app(*, store_override=None, runtime_override=None, start_scheduler: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = store_override or MongoAppointmentStore(
            os.getenv("MONGODB_URL", "mongodb://localhost:27018"),
            os.getenv("AGENT_B_MONGODB_DATABASE", "mental_health_appointments"),
        )
        service = AppointmentService(store, timezone_name=os.getenv("APPOINTMENT_TIMEZONE", "Asia/Beirut"))
        runtime = runtime_override or AdkAppointmentRuntime(
            service, timeout=float(os.getenv("AGENT_B_TIMEOUT_SECONDS", "20")),
        )
        scheduler = ReminderScheduler(
            store,
            webhook_url=os.getenv(
                "AGENT_A_REMINDER_WEBHOOK_URL",
                "http://agent-system-a:8000/internal/v1/appointment-reminders",
            ),
            service_token=os.getenv("INTERNAL_SERVICE_TOKEN", ""),
            interval_seconds=float(os.getenv("REMINDER_POLL_SECONDS", "10")),
            timeout_seconds=float(os.getenv("REMINDER_HTTP_TIMEOUT_SECONDS", "10")),
        )
        app.state.store = store
        app.state.service = service
        app.state.runtime = runtime
        app.state.scheduler = scheduler
        if start_scheduler:
            scheduler.start()
        yield
        scheduler.stop()

    app = FastAPI(
        title="Agent System B - Synthetic Mental-Health Appointments",
        version="1.0.0",
        description="Independent Google ADK service for synthetic appointment matching and booking.",
        lifespan=lifespan,
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "agent-system-b", "framework": "google-adk", "docs": "/docs"}

    @app.get("/health")
    def health(request: Request) -> dict:
        runtime = request.app.state.runtime
        return {
            "status": "ok", "service": "agent-system-b", "framework": "google-adk",
            "adk_available": bool(getattr(runtime, "available", True)),
            "adk_error": getattr(runtime, "error", None),
        }

    async def execute(payload: ChatRequest, request: Request) -> dict:
        request_id = payload.request_id or str(uuid.uuid4())
        try:
            result = await request.app.state.runtime.run(
                payload.query, payload.conversation_id, request_id,
            )
            return ChatResponse.model_validate(result).model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(503, "appointment service is temporarily unavailable") from exc

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> dict:
        return await execute(payload, request)

    @app.post("/v1/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        async def events():
            yield "event: status\ndata: {\"status\":\"processing\"}\n\n"
            result = await execute(payload, request)
            yield f"event: result\ndata: {json.dumps(result)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/psychologists")
    def psychologists(request: Request) -> list[dict]:
        return request.app.state.store.list_profiles()

    @app.get("/v1/appointments/{appointment_id}")
    def appointment(appointment_id: str, request: Request) -> dict:
        row = request.app.state.store.get_appointment(appointment_id)
        if not row:
            raise HTTPException(404, "appointment not found")
        return {key: value.isoformat() if hasattr(value, "isoformat") else value for key, value in row.items()}

    return app


app = create_app()
