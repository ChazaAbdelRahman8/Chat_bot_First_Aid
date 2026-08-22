"""Three specialist ReAct agents, each compiled and executed by LangGraph."""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from .common import final_text
from .manual_visual import ManualVisualRenderer
from .prompts import (
    MANUAL_VISUAL_PROMPT,
    PROTOCOL_TOOLS_PROMPT,
    RAG_PROMPT,
    VISUAL_PROMPT,
    WEB_PROMPT,
)
from .visual_tool import inspect_uploaded_image
from system_limits import (
    IMAGE_INSPECTION_MAX_CALLS,
    MANUAL_VISUAL_AGENT_RECURSION_LIMIT,
    MANUAL_VISUAL_MAX_CALLS,
    MCP_TOOL_MAX_CALLS,
    PROTOCOL_AGENT_RECURSION_LIMIT,
    PROVIDER_FALLBACK_ATTEMPTS,
    RAG_AGENT_RECURSION_LIMIT,
    RAG_RETRIEVAL_MAX_CALLS,
    TOOL_RETRY_ATTEMPTS,
    VISUAL_AGENT_RECURSION_LIMIT,
    WEB_AGENT_RECURSION_LIMIT,
    WEB_SEARCH_MAX_CALLS,
)
from telemetry import record_tool


def _model(
    name: str, base_url: str, *, provider: str = "ollama",
    groq_api_key: str | None = None, timeout: float = 60,
    fallback_model: str | None = None,
) -> BaseChatModel:
    selected = provider.strip().lower()
    if selected not in {"auto", "groq", "ollama"}:
        raise ValueError("agent provider must be auto, groq, or ollama")
    use_groq = selected == "groq" or (selected == "auto" and bool(groq_api_key))
    if use_groq:
        if not groq_api_key:
            raise ValueError("AGENT_PROVIDER=groq requires GROQ_API_KEY")
        return ChatGroq(
            model=name,
            api_key=SecretStr(groq_api_key),
            temperature=0,
            timeout=timeout,
            max_retries=0,
            max_tokens=1024,
            reasoning_effort="none",
            reasoning_format="hidden",
        )
    return ChatOllama(
        model=fallback_model or name,
        base_url=base_url,
        temperature=0,
        num_ctx=8192,
        validate_model_on_init=False,
    )


class RagReActAgent:
    def __init__(
        self, runtime: Any, *, model: str, ollama_url: str,
        provider: str = "ollama", groq_api_key: str | None = None,
        timeout: float = 60, fallback_model: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.model = model
        self.ollama_url = ollama_url
        self.provider = provider
        self.groq_api_key = groq_api_key
        self.timeout = timeout
        self.fallback_model = fallback_model

    def run(
        self, question: str, *, fallback_retrieval_query: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        user_question = question
        # Despite the legacy parameter name, this is now the primary retrieval
        # query. The original question remains the generation/language contract.
        retrieval_query = (
            fallback_retrieval_query.strip()
            if fallback_retrieval_query and fallback_retrieval_query.strip()
            else question
        )
        captured: list[dict[str, Any]] = []
        tool_calls = 0

        @tool("retrieve_first_aid_evidence")
        def retrieve_first_aid_evidence(question: str) -> str:
            """Retrieve validated manual evidence and produce the grounded RAG answer."""
            nonlocal tool_calls
            if tool_calls >= RAG_RETRIEVAL_MAX_CALLS:
                raise RuntimeError("RAG retrieval tool-call limit reached")
            tool_calls += 1
            try:
                result = self.runtime.answer(
                    user_question,
                    retrieval_query=(
                        retrieval_query if retrieval_query != user_question else None
                    ),
                )
            except Exception as exc:
                record_tool(
                    "retrieve_first_aid_evidence", {
                        "question": user_question,
                        "tool_question": question,
                        "retrieval_query": retrieval_query,
                    },
                    success=False, error=f"{type(exc).__name__}: {exc}",
                )
                raise
            record_tool(
                "retrieve_first_aid_evidence", {
                    "question": user_question,
                    "tool_question": question,
                    "retrieval_query": retrieval_query,
                }, success=True,
            )
            captured.append(result)
            generation = result["generation"]
            compact = {
                "status": generation.get("status"),
                "answer": generation.get("answer"),
                "citations": generation.get("citations", []),
                "abstention_category": generation.get("abstention_category"),
                "insufficient_evidence_reason": generation.get("insufficient_evidence_reason"),
                "retrieval_route": generation.get("retrieval_route"),
            }
            return json.dumps(compact, ensure_ascii=False)

        self._retrieve_tool = retrieve_first_aid_evidence

        graph = create_agent(
            model=_model(
                self.model, self.ollama_url, provider=self.provider,
                groq_api_key=self.groq_api_key, timeout=self.timeout,
                fallback_model=self.fallback_model,
            ),
            tools=[retrieve_first_aid_evidence],
            system_prompt=RAG_PROMPT,
            name="first_aid_rag_agent",
        )

        output = ""
        try:
            output = final_text(graph.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": RAG_AGENT_RECURSION_LIMIT},
            ))
        except Exception:
            # The deterministic RAG service remains authoritative when the
            # optional ReAct wrapper fails before making its required call.
            pass
        if not captured:
            # The tool is mandatory. Provider formatting failures must not turn
            # a RAG branch into ungrounded model prose.
            retrieve_first_aid_evidence.invoke({"question": question})
        payload = captured[-1] if captured else {}
        if not output:
            output = str(payload.get("generation", {}).get("answer", ""))
        return output, payload


class WebSearchReActAgent:
    def __init__(
        self, *, model: str, ollama_url: str, max_results: int = 5,
        provider: str = "ollama", groq_api_key: str | None = None,
        timeout: float = 60, fallback_model: str | None = None,
    ) -> None:
        self.max_results = max_results
        # Live search must not inherit the long local-generation budget. The
        # final answer agent performs synthesis; if this ReAct wrapper exceeds
        # 20 seconds, raw search evidence is a safer and faster fallback.
        web_model_timeout = min(float(timeout), 20.0)
        self._search_calls: ContextVar[int] = ContextVar("web_search_calls", default=0)
        self._search_results: ContextVar[list[str]] = ContextVar(
            "web_search_results", default=[],
        )

        @tool("web_search")
        def web_search(query: str) -> str:
            """Search the public web for current information and return titles, snippets, and URLs."""
            try:
                from ddgs import DDGS
            except ImportError as exc:
                record_tool(
                    "web_search", {"query": query, "max_results": self.max_results},
                    success=False, error=f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError("Web search requires the 'ddgs' package") from exc
            last_error: Exception | None = None
            rows: list[dict[str, Any]] = []
            for _attempt in range(TOOL_RETRY_ATTEMPTS + 1):
                calls = self._search_calls.get()
                if calls >= WEB_SEARCH_MAX_CALLS:
                    raise RuntimeError("web-search external-call limit reached")
                self._search_calls.set(calls + 1)
                try:
                    rows = list(DDGS(timeout=8).text(
                        query, max_results=self.max_results,
                    ))
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                record_tool(
                    "web_search", {"query": query, "max_results": self.max_results},
                    success=False, error=f"{type(last_error).__name__}: {last_error}",
                )
                raise RuntimeError("web search failed after one retry") from last_error
            record_tool(
                "web_search", {"query": query, "max_results": self.max_results},
                success=True,
            )
            results = [{
                "title": str(row.get("title", "")),
                "snippet": str(row.get("body", "")),
                "url": str(row.get("href", "")),
            } for row in rows]
            serialized = json.dumps(results, ensure_ascii=False)
            # Mutate the request-scoped list in place. LangGraph copies the
            # ContextVar context for tool execution, but both contexts retain
            # this same list object, so the caller can see the exact URLs.
            self._search_results.get().append(serialized)
            return serialized

        self._web_search_tool = web_search

        self.graph = create_agent(
            model=_model(
                model, ollama_url, provider=provider, groq_api_key=groq_api_key,
                timeout=web_model_timeout, fallback_model=fallback_model,
            ),
            tools=[web_search],
            system_prompt=WEB_PROMPT,
            name="web_search_agent",
        )
        uses_remote = provider == "groq" or (provider == "auto" and bool(groq_api_key))
        self.fallback_graph = None
        # Do not cascade a failed remote web wrapper into a slow local agent.
        # `_ensure_search_evidence` below invokes the bounded search tool
        # directly, retaining real URLs without another model round trip.
        self.uses_remote = uses_remote

    def run(self, question: str) -> tuple[str, dict[str, Any]]:
        # Lazily creating the counter also keeps lightweight test doubles made
        # with ``__new__`` compatible, while normal instances create it in init.
        if not hasattr(self, "_search_calls"):
            self._search_calls = ContextVar("web_search_calls", default=0)
        if not hasattr(self, "_search_results"):
            self._search_results = ContextVar("web_search_results", default=[])
        counter_token = self._search_calls.set(0)
        results_token = self._search_results.set([])
        try:
            errors: list[str] = []
            try:
                output = final_text(
                    self.graph.invoke(
                        {"messages": [{"role": "user", "content": question}]},
                        config={"recursion_limit": WEB_AGENT_RECURSION_LIMIT},
                    )
                )
                output = self._ensure_search_evidence(question, output)
                return output, {
                    "provider_used": "primary", "provider_fallback": False,
                    "provider_errors": [],
                }
            except Exception as exc:
                errors.append(f"primary: {type(exc).__name__}: {exc}")
            if self.fallback_graph is None or PROVIDER_FALLBACK_ATTEMPTS < 1:
                output = self._ensure_search_evidence(question, "")
                if output.strip():
                    return output, {
                        "provider_used": "deterministic_search",
                        "provider_fallback": True, "provider_errors": errors,
                    }
                raise RuntimeError(errors[0])
            try:
                output = final_text(
                    self.fallback_graph.invoke(
                        {"messages": [{"role": "user", "content": question}]},
                        config={"recursion_limit": WEB_AGENT_RECURSION_LIMIT},
                    )
                )
                output = self._ensure_search_evidence(question, output)
                return output, {
                    "provider_used": "ollama", "provider_fallback": True,
                    "provider_errors": errors,
                }
            except Exception as exc:
                errors.append(f"ollama: {type(exc).__name__}: {exc}")
                output = self._ensure_search_evidence(question, "")
                if output.strip():
                    return output, {
                        "provider_used": "deterministic_search",
                        "provider_fallback": True, "provider_errors": errors,
                    }
                raise RuntimeError("; ".join(errors)) from exc
        finally:
            self._search_calls.reset(counter_token)
            self._search_results.reset(results_token)

    def _ensure_search_evidence(self, question: str, output: str) -> str:
        results = self._search_results.get()
        if not results and hasattr(self, "_web_search_tool"):
            self._web_search_tool.invoke({"query": question})
            results = self._search_results.get()
        if results:
            # Preserve the exact retrieved URLs for final reference validation,
            # even if the ReAct model summarizes without printing them.
            return f"{output}\n\nVERBATIM_SEARCH_RESULTS:\n{results[-1]}"
        return output


class VisualReActAgent:
    """A request-scoped visual agent so its tool can only access one validated image."""

    def __init__(
        self, image_path: Path, *, model: str, agent_model: str, ollama_url: str,
        agent_provider: str = "ollama", groq_api_key: str | None = None,
        agent_timeout: float = 60, agent_fallback_model: str | None = None,
        vision_provider: str = "ollama",
        vision_fallback_model: str = "qwen2.5vl:7b",
        openrouter_api_key: str | None = None,
        openrouter_url: str = "https://openrouter.ai/api/v1",
        vision_timeout: int = 30,
    ) -> None:
        tool_calls = 0

        @tool("inspect_image")
        def inspect_image(question: str) -> str:
            """Inspect the attached image and return visible evidence with safety limitations."""
            nonlocal tool_calls
            if tool_calls >= IMAGE_INSPECTION_MAX_CALLS:
                raise RuntimeError("image-inspection tool-call limit reached")
            tool_calls += 1
            try:
                result = inspect_uploaded_image(
                    image_path,
                    question,
                    provider=vision_provider,
                    model=model,
                    ollama_url=ollama_url,
                    fallback_model=vision_fallback_model,
                    openrouter_api_key=openrouter_api_key,
                    openrouter_url=openrouter_url,
                    timeout=vision_timeout,
                )
            except Exception as exc:
                record_tool(
                    "inspect_image", {
                        "question": question,
                        "image_attached": True,
                        "provider": vision_provider,
                    },
                    success=False, error=f"{type(exc).__name__}: {exc}",
                )
                raise
            record_tool(
                "inspect_image", {
                    "question": question,
                    "image_attached": True,
                    "provider": result.get("analysis_provider", vision_provider),
                    "provider_fallback": result.get("provider_fallback", False),
                },
                success=True,
            )
            return json.dumps(result, ensure_ascii=False)

        self.graph = create_agent(
            model=_model(
                agent_model, ollama_url, provider=agent_provider,
                groq_api_key=groq_api_key, timeout=agent_timeout,
                fallback_model=agent_fallback_model,
            ),
            tools=[inspect_image],
            system_prompt=VISUAL_PROMPT,
            name="visual_agent",
        )

    def run(self, question: str) -> str:
        return final_text(self.graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": VISUAL_AGENT_RECURSION_LIMIT},
        ))


class ManualVisualReActAgent:
    """ReAct specialist that retrieves and renders cited manual pages."""

    def __init__(
        self, runtime: Any, project_root: Path, *, agent_model: str,
        ollama_url: str, agent_provider: str = "ollama",
        groq_api_key: str | None = None, agent_timeout: float = 60,
        agent_fallback_model: str | None = None,
    ) -> None:
        self.captured: list[dict[str, Any]] = []
        self.tool_calls = 0
        renderer = ManualVisualRenderer(project_root)

        def execute(question: str) -> dict[str, Any]:
            try:
                retrieval = runtime.retrieve(question)
                visuals = renderer.render(retrieval, limit=1)
            except Exception as exc:
                record_tool(
                    "retrieve_and_render_manual_page", {"question": question, "limit": 1},
                    success=False, error=f"{type(exc).__name__}: {exc}",
                )
                raise
            record_tool(
                "retrieve_and_render_manual_page", {"question": question, "limit": 1},
                success=bool(visuals), error=None if visuals else "no renderable page",
            )
            payload = {"mode": "manual_render", "retrieval": retrieval, "visuals": visuals}
            self.captured.append(payload)
            return payload

        @tool("retrieve_and_render_manual_page")
        def retrieve_and_render_manual_page(question: str) -> str:
            """Retrieve first-aid evidence and render the best cited manual page."""
            if self.tool_calls >= MANUAL_VISUAL_MAX_CALLS:
                raise RuntimeError("manual-visual tool-call limit reached")
            self.tool_calls += 1
            payload = execute(question)
            compact = [{
                "visual_id": item["visual_id"], "doc_id": item["doc_id"],
                "pdf_page": item["pdf_page"], "section": item.get("section"),
            } for item in payload["visuals"]]
            return json.dumps({"rendered_pages": compact}, ensure_ascii=False)

        self.graph = create_agent(
            model=_model(
                agent_model, ollama_url, provider=agent_provider,
                groq_api_key=groq_api_key, timeout=agent_timeout,
                fallback_model=agent_fallback_model,
            ),
            tools=[retrieve_and_render_manual_page],
            system_prompt=MANUAL_VISUAL_PROMPT,
            name="manual_visual_agent",
        )
        self._execute = execute

    def run(self, question: str) -> tuple[str, dict[str, Any]]:
        try:
            output = final_text(
                self.graph.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config={"recursion_limit": MANUAL_VISUAL_AGENT_RECURSION_LIMIT},
                )
            )
        except Exception:
            # Rendering is deterministic and remains safe if the optional
            # reasoning model is rate-limited after the supervisor has routed
            # this request. Never let provider availability hide source evidence.
            payload = self.captured[-1] if self.captured else self._execute_once(question)
            output = self._fallback_output(payload)
        if not self.captured:
            payload = self._execute_once(question)
            output = self._fallback_output(payload)
        return output, self.captured[-1]

    def _execute_once(self, question: str) -> dict[str, Any]:
        if self.tool_calls >= MANUAL_VISUAL_MAX_CALLS:
            raise RuntimeError("manual-visual tool-call limit reached")
        self.tool_calls += 1
        return self._execute(question)

    @staticmethod
    def _fallback_output(payload: dict[str, Any]) -> str:
        visuals = payload.get("visuals", [])
        if not visuals:
            return "No renderable manual page was found for this request."
        item = visuals[0]
        return f"Rendered {item['doc_id']} PDF page {item['pdf_page']} from retrieved evidence."


def _mcp_result_to_text(result: Any) -> str:
    """Normalize an MCP tool result (often a list of text content blocks) to a string."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = [
            str(item.get("text", ""))
            for item in result
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if parts:
            return "\n".join(parts)
    return json.dumps(result, ensure_ascii=False, default=str)


class ProtocolToolsReActAgent:
    """ReAct specialist that calls the standalone MCP first-aid tools server.

    Unlike the other specialists, its tools live in a separate service
    (``mcp-server``) reached over the network via the MCP Streamable HTTP
    transport, never imported as a Python module. Tool discovery is lazy and
    cached so a not-yet-ready MCP server at process startup cannot break
    Agent System A's own startup.
    """

    def __init__(
        self, *, mcp_server_url: str, model: str, ollama_url: str,
        provider: str = "ollama", groq_api_key: str | None = None,
        timeout: float = 60, fallback_model: str | None = None,
        mcp_timeout: float = 20,
    ) -> None:
        self.mcp_server_url = mcp_server_url
        self.model = model
        self.ollama_url = ollama_url
        self.provider = provider
        self.groq_api_key = groq_api_key
        self.timeout = timeout
        self.fallback_model = fallback_model
        self.mcp_timeout = mcp_timeout
        self._cached_tools: dict[str, Any] | None = None

    def _load_tools(self) -> dict[str, Any]:
        if self._cached_tools is not None:
            return self._cached_tools
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "first_aid_tools": {
                "url": self.mcp_server_url,
                "transport": "streamable_http",
            },
        })

        async def _fetch() -> list[Any]:
            return await client.get_tools()

        tools = asyncio.run(asyncio.wait_for(_fetch(), timeout=self.mcp_timeout))
        by_name = {getattr(t, "name", ""): t for t in tools}
        if "get_first_aid_protocol" not in by_name or "assess_burn_severity" not in by_name:
            raise RuntimeError(
                f"MCP server at {self.mcp_server_url} did not expose the expected tools"
            )
        self._cached_tools = by_name
        return by_name

    def run(self, question: str) -> tuple[str, dict[str, Any]]:
        remote_tools = self._load_tools()
        call_count = 0
        captured: list[dict[str, Any]] = []

        def call_remote(name: str, kwargs: dict[str, Any]) -> Any:
            nonlocal call_count
            if call_count >= MCP_TOOL_MAX_CALLS:
                raise RuntimeError("MCP tool-call limit reached")
            call_count += 1
            raw_result = asyncio.run(
                asyncio.wait_for(remote_tools[name].ainvoke(kwargs), timeout=self.mcp_timeout)
            )
            result = _mcp_result_to_text(raw_result)
            captured.append({"tool": name, "args": kwargs, "result": result})
            return result

        @tool("get_first_aid_protocol")
        def get_first_aid_protocol(condition: str) -> str:
            """Look up ordered first-aid steps and warnings for a named condition, with a source citation."""
            args = {"condition": condition}
            try:
                result = call_remote("get_first_aid_protocol", args)
            except Exception as exc:
                record_tool(
                    "get_first_aid_protocol", args, success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            record_tool("get_first_aid_protocol", args, success=True)
            return _mcp_result_to_text(result)

        @tool("assess_burn_severity")
        def assess_burn_severity(percent_body_surface_area: float, degree: str) -> str:
            """Classify burn severity and recommended action from BSA percentage and burn depth."""
            args = {"percent_body_surface_area": percent_body_surface_area, "degree": degree}
            try:
                result = call_remote("assess_burn_severity", args)
            except Exception as exc:
                record_tool(
                    "assess_burn_severity", args, success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            record_tool("assess_burn_severity", args, success=True)
            return _mcp_result_to_text(result)

        graph = create_agent(
            model=_model(
                self.model, self.ollama_url, provider=self.provider,
                groq_api_key=self.groq_api_key, timeout=self.timeout,
                fallback_model=self.fallback_model,
            ),
            tools=[get_first_aid_protocol, assess_burn_severity],
            system_prompt=PROTOCOL_TOOLS_PROMPT,
            name="protocol_tools_agent",
        )
        output = final_text(graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": PROTOCOL_AGENT_RECURSION_LIMIT},
        ))
        if not captured:
            # At least one MCP call is mandatory, mirroring the RAG specialist's
            # required-call guarantee, so a formatting-only model failure can
            # never silently skip the network round trip to mcp-server.
            get_first_aid_protocol.invoke({"condition": question})
        payload = {"mode": "mcp_protocol_tools", "calls": captured}
        return output, payload
