"""FastAPI application for querying and managing the first-aid RAG."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from rag.indexing.qdrant_store import COLLECTION_NAME

from .ingestion import DocumentIngestor
from .routes_chat import router as chat_router
from .routes_chat import agent_system_dependency
from .routes_chat import conversation_store_dependency as chat_conversation_store_dependency
from .routes_conversations import conversation_store_dependency
from .routes_conversations import router as conversation_router
from .routes_documents import ingestor_dependency, router as document_router
from .routes_documents import runtime_dependency as document_runtime_dependency
from .runtime import RagRuntime
from .session_store import MemoryConversationStore, build_conversation_store
from agents import AgentSystemA
from system_b_client import AgentSystemBClient
from .notifications import (
    MemoryNotificationStore, MongoNotificationStore, NotificationHub,
    internal_token_dependency, notification_hub_dependency,
    notification_store_dependency, router as notification_router,
)


PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
).resolve()


def create_app(
    runtime_factory=RagRuntime, *, runtime_override=None, ingestor_override=None,
    agent_system_override=None, conversation_store_override=None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime_override is not None:
            app.state.runtime = runtime_override
            app.state.ingestor = ingestor_override
            app.state.agent_system = agent_system_override or runtime_override
            app.state.conversation_store = conversation_store_override or MemoryConversationStore()
            app.state.notification_store = MemoryNotificationStore()
            app.state.notification_hub = NotificationHub()
            app.state.internal_service_token = os.getenv("INTERNAL_SERVICE_TOKEN", "test-token")
            yield
            return
        runtime = runtime_factory(
            PROJECT_ROOT,
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            collection=os.getenv("QDRANT_COLLECTION", COLLECTION_NAME),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            generation_model=os.getenv("GENERATION_MODEL", "qwen3:4b"),
            generation_provider=os.getenv("GENERATION_PROVIDER", "ollama"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            generation_timeout=float(os.getenv("GENERATION_TIMEOUT_SECONDS", "60")),
            generation_fallback_model=os.getenv("GENERATION_FALLBACK_MODEL", "qwen3:4b"),
            arabic_translation_enabled=(
                os.getenv("ARABIC_QUERY_TRANSLATION_ENABLED", "true").lower() == "true"
            ),
            arabic_translation_provider=os.getenv(
                "ARABIC_QUERY_TRANSLATION_PROVIDER", "auto"
            ),
            arabic_translation_model=os.getenv(
                "ARABIC_QUERY_TRANSLATION_MODEL", "qwen/qwen3.6-27b"
            ),
            arabic_translation_openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            arabic_translation_openrouter_model=os.getenv(
                "ARABIC_QUERY_TRANSLATION_OPENROUTER_MODEL",
                "qwen/qwen3.5-flash-02-23",
            ),
            arabic_translation_openrouter_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            arabic_translation_fallback_model=os.getenv(
                "ARABIC_QUERY_TRANSLATION_FALLBACK_MODEL", "qwen3:4b"
            ),
            arabic_translation_timeout=float(os.getenv(
                "ARABIC_QUERY_TRANSLATION_TIMEOUT_SECONDS", "20"
            )),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            device=os.getenv("EMBEDDING_DEVICE") or None,
        )
        ingestor = DocumentIngestor(
            runtime,
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            vision_model=os.getenv(
                "INGEST_VISION_MODEL",
                os.getenv("VISION_MODEL", "qwen2.5vl:7b"),
            ),
            vision_enabled=os.getenv("INGEST_VISION", "true").lower() == "true",
        )
        app.state.runtime = runtime
        app.state.ingestor = ingestor
        app.state.agent_system = AgentSystemA(
            runtime,
            agent_model=os.getenv("AGENT_MODEL", "qwen/qwen3.6-27b"),
            agent_provider=os.getenv("AGENT_PROVIDER", "auto"),
            agent_fallback_model=os.getenv("AGENT_FALLBACK_MODEL", "qwen3:4b"),
            agent_timeout=float(os.getenv("AGENT_TIMEOUT_SECONDS", "60")),
            vision_model=os.getenv(
                "UPLOAD_VISION_MODEL", "qwen/qwen3-vl-32b-instruct"
            ),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            upload_vision_provider=os.getenv("UPLOAD_VISION_PROVIDER", "auto"),
            upload_vision_fallback_model=os.getenv(
                "UPLOAD_VISION_FALLBACK_MODEL", "qwen2.5vl:7b"
            ),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            upload_vision_timeout=int(os.getenv(
                "UPLOAD_VISION_TIMEOUT_SECONDS", "30"
            )),
            web_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
            supervisor_provider=os.getenv("SUPERVISOR_PROVIDER", "auto"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_SUPERVISOR_MODEL", "openai/gpt-oss-120b"),
            supervisor_timeout=float(os.getenv("SUPERVISOR_TIMEOUT_SECONDS", "20")),
            appointment_client=AgentSystemBClient(
                os.getenv("AGENT_SYSTEM_B_URL", "http://agent-system-b:8001"),
                timeout=float(os.getenv("AGENT_SYSTEM_B_TIMEOUT_SECONDS", "20")),
                retries=int(os.getenv("AGENT_SYSTEM_B_RETRIES", "1")),
            ),
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/mcp"),
            mcp_timeout=float(os.getenv("MCP_TIMEOUT_SECONDS", "20")),
        )
        app.state.conversation_store = build_conversation_store(
            os.getenv("MONGODB_URL", "mongodb://localhost:27018"),
            os.getenv("MONGODB_DATABASE", "first_aid"),
        )
        app.state.notification_store = MongoNotificationStore(
            os.getenv("MONGODB_URL", "mongodb://localhost:27018"),
            os.getenv("MONGODB_DATABASE", "first_aid"),
        )
        app.state.notification_hub = NotificationHub()
        app.state.internal_service_token = os.getenv("INTERNAL_SERVICE_TOKEN", "")
        yield

    app = FastAPI(
        title="First-Aid RAG API",
        version="1.0.0",
        description="Grounded first-aid chat plus managed PDF ingestion and deletion.",
        lifespan=lifespan,
    )

    def runtime_from_request(request: Request):
        return request.app.state.runtime

    def ingestor_from_request(request: Request):
        return request.app.state.ingestor

    def agent_system_from_request(request: Request):
        return request.app.state.agent_system

    def conversation_store_from_request(request: Request):
        return request.app.state.conversation_store

    def notification_store_from_request(request: Request):
        return request.app.state.notification_store

    def notification_hub_from_request(request: Request):
        return request.app.state.notification_hub

    def internal_token_from_request(request: Request):
        return request.app.state.internal_service_token

    app.dependency_overrides[agent_system_dependency] = agent_system_from_request
    app.dependency_overrides[chat_conversation_store_dependency] = conversation_store_from_request
    app.dependency_overrides[conversation_store_dependency] = conversation_store_from_request
    app.dependency_overrides[document_runtime_dependency] = runtime_from_request
    app.dependency_overrides[ingestor_dependency] = ingestor_from_request
    app.dependency_overrides[notification_store_dependency] = notification_store_from_request
    app.dependency_overrides[notification_hub_dependency] = notification_hub_from_request
    app.dependency_overrides[internal_token_dependency] = internal_token_from_request
    app.include_router(chat_router)
    app.include_router(conversation_router)
    app.include_router(document_router)
    app.include_router(notification_router)
    @app.get("/", include_in_schema=False)
    def service_root() -> dict[str, str]:
        return {
            "service": "agent-system-a",
            "status": "running",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/health")
    def health(request: Request) -> dict:
        runtime = request.app.state.runtime
        info = runtime.qdrant.get_collection(runtime.collection)
        return {
            "status": "ok",
            "qdrant_collection": runtime.collection,
            "qdrant_points": info.points_count,
            "managed_documents": len(runtime.store.registry()),
            "generation_model": runtime.generator.client.model,
            "generation_provider": runtime.generator.client.provider,
            "embedding_model": runtime.embedding_model,
        }

    return app


app = create_app()
