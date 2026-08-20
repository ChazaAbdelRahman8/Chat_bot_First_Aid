"""Explicit top-level LangGraph workflow for Agent System A."""

from __future__ import annotations

import json
import operator
import os
import re
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping, Sequence, cast

from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import NotRequired, TypedDict

from guardrails import (
    conversational_response, evaluate_input, greeting_response, has_first_aid_intent,
)
from guardrails.output_guard import validate_agent_output
from rag.generation.service import policy_gate_response
from rag.retrieval.hybrid import detect_query_language
from system_limits import OUTPUT_REPAIR_ATTEMPTS, TOP_LEVEL_RECURSION_LIMIT
from telemetry import (
    CountingUsageCallback, record_tool, record_usage, summarize_usage, telemetry_context,
)

from .answer_agent import FinalAnswerReActAgent
from .prompts import SUPERVISOR_ROUTING_PROMPT
from .specialists import (
    ManualVisualReActAgent,
    ProtocolToolsReActAgent,
    RagReActAgent,
    VisualReActAgent,
    WebSearchReActAgent,
    _model,
)


AgentName = Literal[
    "rag", "web_search", "visual", "appointment", "protocol_tools", "scope_guard",
]
CURRENT_INFORMATION = re.compile(
    r"\b(?:current|latest|today|now|updated|emergency (?:number|phone)|nearest)\b|"
    r"(?:حالي|الأحدث|اليوم|الآن|رقم الطوارئ|أقرب)",
    re.IGNORECASE,
)
FIRST_AID_SIGNAL = re.compile(
    r"\b(?:first aid|injur|bleed|burn|chok|fracture|wound|breath|unconscious|"
    r"emergency|pain|poison|seizure|shock|cpr)\w*\b|"
    r"(?:إسعاف|إصابة|نزيف|حرق|اختناق|كسر|جرح|تنفس|فاقد الوعي|طوارئ)",
    re.IGNORECASE,
)
MANUAL_VISUAL_SIGNAL = re.compile(
    r"\b(?:show|display|render|give|provide|see|need|find)\b.{0,45}"
    r"\b(?:image|picture|diagram|illustration|flowchart|visual|figure)\b|"
    r"\b(?:image|picture|diagram|illustration|flowchart|visual|figure)\b.{0,45}"
    r"\b(?:manual|guide|cpr|first aid|procedure|technique)\b|"
    r"(?:اعرض|أظهر|ارني|أرني).{0,55}(?:رسم|رسماً|صورة|مخطط|توضيح).{0,55}(?:دليل|كتيب)|"
    r"(?:رسم|رسماً|صورة|مخطط|توضيح).{0,55}(?:دليل|كتيب)|"
    r"\b(?:montrez|affichez|montrer|voir)\b.{0,55}"
    r"\b(?:illustration|image|schéma|schema|figure|diagramme)\b.{0,55}"
    r"\b(?:manuel|guide)\b|"
    r"\b(?:illustration|image|schéma|schema|figure|diagramme)\b.{0,55}"
    r"\b(?:manuel|guide)\b",
    re.IGNORECASE,
)
MIXED_CURRENT_SCOPE_SIGNAL = re.compile(
    r"\b(?:bitcoin|cryptocurrency|stock price|weather)\b|"
    r"\b(?:official\s+)?(?:heat\s+)?(?:emergency\s+)?alerts?\b",
    re.IGNORECASE,
)
MIXED_CLAUSE_SEPARATOR = re.compile(
    r"\s*(?:,?\s+and\s+also\s+|,?\s+and\s+|;|\bplus\b)\s*",
    re.IGNORECASE,
)
APPOINTMENT_SIGNAL = re.compile(
    r"\b(?:psychologist|therapist|mental[- ]health|counsell?or|appointment|therapy)\b|"
    r"\b(?:book|schedule|reserve|confirm)\b.{0,60}\b(?:appointment|therapist|psychologist)\b|"
    r"\b(?:book|confirm|reserve)\b.{0,30}\b(?:first|second|third|1st|2nd|3rd|one)\b|"
    r"(?:Ø§Ø®ØµØ§Ø¦ÙŠ Ù†ÙØ³ÙŠ|Ø·Ø¨ÙŠØ¨ Ù†ÙØ³ÙŠ|Ù…ÙˆØ¹Ø¯ Ù†ÙØ³ÙŠ|Ø§Ø­Ø¬Ø² Ù…ÙˆØ¹Ø¯)",
    re.IGNORECASE,
)
PROTOCOL_TOOL_SIGNAL = re.compile(
    r"\b(?:steps?|checklist|protocol)\b.{0,10}\b(?:for|to)\b.{0,60}"
    r"\b(?:choking|bleeding|burns?|cpr|fracture|seizure|shock|unconscious)\w*\b|"
    r"\bhow do i\b.{0,20}\b(?:treat|handle)\b.{0,60}"
    r"\b(?:choking|bleeding|burns?|cpr|fracture|seizure|shock|unconscious)\w*\b|"
    r"\b(?:burn (?:severity|percentage)|body surface area|shock index|"
    r"triage calculat\w*|severity calculat\w*)\b",
    re.IGNORECASE,
)


class RouteItem(BaseModel):
    agent: AgentName
    request: str = Field(min_length=1, max_length=4000)


def _first_aid_only_subquery(query: str) -> str | None:
    """Extract first-aid clauses when a request also asks for excluded current data."""
    clauses = [item.strip(" ,.;") for item in MIXED_CLAUSE_SEPARATOR.split(query)]
    if len(clauses) < 2 or not any(MIXED_CURRENT_SCOPE_SIGNAL.search(item) for item in clauses):
        return None
    first_aid = [
        item for item in clauses
        if has_first_aid_intent(item) and not MIXED_CURRENT_SCOPE_SIGNAL.search(item)
    ]
    return " ".join(first_aid).strip() or None


class RoutingDecision(BaseModel):
    routes: list[RouteItem] = Field(min_length=1, max_length=3)


class SpecialistResult(TypedDict):
    agent: AgentName
    request: str
    output: str
    ok: bool
    error: NotRequired[str]
    payload: NotRequired[dict[str, Any]]


class AgentState(TypedDict):
    query: str
    history: NotRequired[list[dict[str, str]]]
    image_path: NotRequired[Path | None]
    conversation_id: NotRequired[str | None]
    routing: NotRequired[list[dict[str, str]]]
    specialist_results: NotRequired[Annotated[list[SpecialistResult], operator.add]]
    response: NotRequired[dict[str, Any]]
    guard_decision: NotRequired[str]
    answer_mode: NotRequired[str]
    answer_attempts: NotRequired[int]
    output_guard_errors: NotRequired[list[str]]
    output_guard_status: NotRequired[str]
    supervisor_provider: NotRequired[str]
    supervisor_model: NotRequired[str]
    supervisor_latency_ms: NotRequired[float]
    router_errors: NotRequired[list[str]]
    trace_events: NotRequired[Annotated[list[dict[str, Any]], operator.add]]


class AgentSystemA:
    """Compiled orchestration graph with ReAct specialists and deterministic guards."""

    def __init__(
        self, runtime: Any, *, agent_model: str, vision_model: str,
        ollama_url: str, web_max_results: int = 5,
        agent_provider: str = "ollama", agent_fallback_model: str = "qwen3:4b",
        agent_timeout: float = 60,
        supervisor_provider: str = "auto", groq_api_key: str | None = None,
        groq_model: str = "openai/gpt-oss-120b", supervisor_timeout: float = 20,
        appointment_client: Any | None = None,
        mcp_server_url: str = "http://mcp-server:8002/mcp", mcp_timeout: float = 20,
    ) -> None:
        self.agent_model = agent_model
        self.runtime = runtime
        self.project_root = Path(getattr(runtime, "project_root", Path.cwd()))
        self.vision_model = vision_model
        self.ollama_url = ollama_url
        self.agent_provider = agent_provider
        self.agent_fallback_model = agent_fallback_model
        self.agent_timeout = agent_timeout
        self.supervisor_provider = supervisor_provider.lower().strip()
        if self.supervisor_provider not in {"auto", "groq", "ollama"}:
            raise ValueError("supervisor_provider must be auto, groq, or ollama")
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.supervisor_timeout = supervisor_timeout
        self.appointment_client = appointment_client
        self.rag = RagReActAgent(
            runtime, model=agent_model, ollama_url=ollama_url,
            provider=agent_provider, groq_api_key=groq_api_key,
            timeout=agent_timeout, fallback_model=agent_fallback_model,
        )
        self.web = WebSearchReActAgent(
            model=agent_model, ollama_url=ollama_url, max_results=web_max_results,
            provider=agent_provider, groq_api_key=groq_api_key,
            timeout=agent_timeout, fallback_model=agent_fallback_model,
        )
        self.visual_factory = VisualReActAgent
        self.manual_visual_factory = ManualVisualReActAgent
        self.protocol_tools = ProtocolToolsReActAgent(
            mcp_server_url=mcp_server_url, model=agent_model, ollama_url=ollama_url,
            provider=agent_provider, groq_api_key=groq_api_key,
            timeout=agent_timeout, fallback_model=agent_fallback_model,
            mcp_timeout=mcp_timeout,
        )
        self.answer_agent = FinalAnswerReActAgent(
            model=agent_model, ollama_url=ollama_url, provider=agent_provider,
            groq_api_key=groq_api_key, timeout=agent_timeout,
            fallback_model=agent_fallback_model,
        )
        self.router_models = self._build_router_models()
        self.graph = self._build_graph()

    def _build_router_models(self) -> list[tuple[str, str, Any]]:
        models: list[tuple[str, str, Any]] = []
        if self.supervisor_provider in {"auto", "groq"} and self.groq_api_key:
            try:
                from langchain_groq import ChatGroq

                groq = ChatGroq(
                    model=self.groq_model,
                    api_key=SecretStr(self.groq_api_key),
                    temperature=0,
                    timeout=self.supervisor_timeout,
                    max_retries=0,
                ).with_structured_output(
                    RoutingDecision, method="json_schema", strict=True,
                )
                models.append(("groq", self.groq_model, groq))
            except Exception:
                # The local router remains available when the optional provider
                # dependency or configuration is unavailable.
                pass
        ollama = _model(
            self.agent_fallback_model, self.ollama_url, provider="ollama",
        ).with_structured_output(
            RoutingDecision
        )
        models.append(("ollama", self.agent_fallback_model, ollama))
        return models

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("input_guard", self._traced("input_guard", self._input_guard_node))
        builder.add_node("supervisor", self._traced("supervisor", self._supervisor_node))
        builder.add_node("rag_agent", self._traced("rag_agent", self._rag_node))
        builder.add_node("web_agent", self._traced("web_agent", self._web_node))
        builder.add_node("visual_agent", self._traced("visual_agent", self._visual_node))
        builder.add_node(
            "appointment_agent", self._traced("appointment_agent", self._appointment_node),
        )
        builder.add_node(
            "protocol_agent", self._traced("protocol_agent", self._protocol_node),
        )
        builder.add_node("scope_guard", self._traced("scope_guard", self._scope_node))
        builder.add_node("answer_agent", self._traced("answer_agent", self._answer_node))
        builder.add_node("output_guard", self._traced("output_guard", self._output_guard_node))
        builder.add_edge(START, "input_guard")
        builder.add_conditional_edges(
            "input_guard", self._after_input_guard,
            {"supervisor": "supervisor", "output_guard": "output_guard"},
        )
        builder.add_conditional_edges(
            "supervisor", self._specialist_nodes,
            [
                "rag_agent", "web_agent", "visual_agent", "appointment_agent",
                "protocol_agent", "scope_guard",
            ],
        )
        for node in (
            "rag_agent", "web_agent", "visual_agent", "appointment_agent",
            "protocol_agent", "scope_guard",
        ):
            builder.add_edge(node, "answer_agent")
        builder.add_edge("answer_agent", "output_guard")
        builder.add_conditional_edges(
            "output_guard", self._after_output_guard,
            {"repair": "answer_agent", "end": END},
        )
        return builder.compile()

    @staticmethod
    def _traced(
        name: str, node: Callable[[AgentState], dict[str, Any]],
    ) -> Runnable[AgentState, dict[str, Any]]:
        def execute(state: AgentState) -> dict[str, Any]:
            started = time.perf_counter()
            result = node(state)
            return {
                **result,
                "trace_events": [{
                    "node": name,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }],
            }
        execute.__name__ = name
        # Returning LangChain's typed Runnable satisfies LangGraph's StateNode
        # contract and avoids the callable positional-parameter ambiguity in
        # recent Pylance/LangGraph type definitions.
        return RunnableLambda(execute, name=name)

    def mermaid(self) -> str:
        return self.graph.get_graph(xray=True).draw_mermaid()

    def route_only(
        self, question: str, *, image_attached: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run the production guard and supervisor without specialist execution."""
        started = time.perf_counter()
        state: AgentState = {
            "query": question,
            "history": list(history or [])[-12:],
            "image_path": Path("_routing_evaluation_image_") if image_attached else None,
            "routing": [],
            "router_errors": [],
        }
        guard_result = self._input_guard_node(state)
        state = cast(AgentState, {**state, **guard_result})
        if guard_result.get("response") is not None:
            return {
                "query": question,
                "image_attached": image_attached,
                "input_guard": guard_result.get("guard_decision", "unknown"),
                "selected_routes": [],
                "supervisor_provider": "not_used",
                "supervisor_model": "not_used",
                "router_errors": [],
                "supervisor_latency_ms": 0.0,
                "decision_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        supervisor_result = self._supervisor_node(state)
        return {
            "query": question,
            "image_attached": image_attached,
            "input_guard": guard_result.get("guard_decision", "allowed"),
            "selected_routes": [
                item["agent"] for item in supervisor_result.get("routing", [])
            ],
            "supervisor_provider": supervisor_result.get("supervisor_provider"),
            "supervisor_model": supervisor_result.get("supervisor_model"),
            "router_errors": supervisor_result.get("router_errors", []),
            "supervisor_latency_ms": supervisor_result.get("supervisor_latency_ms", 0.0),
            "decision_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def answer(
        self, question: str, *, image_path: Path | None = None,
        history: list[dict[str, str]] | None = None, conversation_id: str | None = None,
    ) -> dict[str, Any]:
        initial_state: AgentState = {
            "query": question,
            "history": list(history or [])[-12:],
            "image_path": image_path,
            "conversation_id": conversation_id,
            "routing": [],
            "specialist_results": [],
            "answer_attempts": 0,
            "router_errors": [],
            "trace_events": [],
        }
        usage_callback = CountingUsageCallback()
        try:
            with telemetry_context() as telemetry:
                final_state = self.graph.invoke(
                    initial_state,
                    config={
                        "recursion_limit": TOP_LEVEL_RECURSION_LIMIT,
                        "callbacks": [usage_callback],
                    },
                )
        except GraphRecursionError as exc:
            initial_state["answer_mode"] = "safe_abstention"
            initial_state["router_errors"] = [
                f"workflow iteration limit reached: {type(exc).__name__}"
            ]
            response = self._abstention_response(
                question, [], detect_query_language(question),
                "The workflow reached its safe iteration limit.", [],
            )
            return self._add_orchestration(
                response, initial_state, "failed_closed",
                ["workflow iteration limit reached"],
            )
        response = final_state["response"]
        orchestration = response.setdefault("orchestration", {})
        trace_events = list(final_state.get("trace_events", []))
        orchestration.update({
            "trajectory": [event["node"] for event in trace_events],
            "node_trace": trace_events,
            "node_steps": len(trace_events),
            "tool_trace": list(telemetry.tool_events),
            "tool_calls": len(telemetry.tool_events),
            "usage": summarize_usage(
                usage_callback.usage_metadata,
                telemetry,
                callback_model_calls=usage_callback.model_calls,
            ),
        })
        return response

    def _input_guard_node(self, state: AgentState) -> dict[str, Any]:
        query = state["query"]
        conversation = conversational_response(query, state.get("history"))
        if conversation is not None and state.get("image_path") is None:
            message, language = conversation
            return {
                "guard_decision": "conversation",
                "answer_mode": "conversation",
                "response": self._direct_response(
                    query, message, language=language, mode="conversation",
                ),
            }
        greeting = greeting_response(query)
        if greeting is not None and state.get("image_path") is None:
            message, language = greeting
            return {
                "guard_decision": "greeting",
                "answer_mode": "greeting",
                "response": self._direct_response(query, message, language=language),
            }
        decision = evaluate_input(query)
        if decision.blocked:
            generation = policy_gate_response(query, self.agent_model, decision)
            return {
                "guard_decision": decision.reason_code or "blocked",
                "answer_mode": "policy_gate",
                "response": {
                    "query": query,
                    "route": ["input_guard"],
                    "specialists": [],
                    "retrieval": None,
                    "generation": {**generation, "agent_route": ["input_guard"]},
                    "answer_agent": {"used": False, "mode": "policy_gate", "valid": True},
                },
            }
        return {"guard_decision": "allowed"}

    @staticmethod
    def _after_input_guard(state: AgentState) -> str:
        return "output_guard" if state.get("response") else "supervisor"

    def _supervisor_node(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        payload = {
            "query": state["query"],
            "history": state.get("history", []),
            "image_attached": state.get("image_path") is not None,
            "manual_visual_requested": bool(MANUAL_VISUAL_SIGNAL.search(state["query"])),
        }
        errors: list[str] = []
        for provider, model_name, router in self.router_models:
            try:
                raw = router.invoke([
                    ("system", SUPERVISOR_ROUTING_PROMPT),
                    ("human", json.dumps(payload, ensure_ascii=False)),
                ])
                decision = raw if isinstance(raw, RoutingDecision) else RoutingDecision.model_validate(raw)
                routes = self._normalize_routes(decision.routes, state)
                if routes:
                    return {
                        "routing": routes,
                        "supervisor_provider": provider,
                        "supervisor_model": model_name,
                        "router_errors": errors,
                        "supervisor_latency_ms": round(
                            (time.perf_counter() - started) * 1000, 3
                        ),
                    }
            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
        return {
            "routing": self._deterministic_routes(state),
            "supervisor_provider": "deterministic_fallback",
            "supervisor_model": "rules-v1",
            "router_errors": errors,
            "supervisor_latency_ms": round(
                (time.perf_counter() - started) * 1000, 3
            ),
        }

    def _normalize_routes(
        self, items: list[RouteItem], state: AgentState,
    ) -> list[dict[str, str]]:
        scoped_query = _first_aid_only_subquery(state["query"])
        selected: dict[str, str] = {}
        for item in items:
            if (
                item.agent == "visual"
                and state.get("image_path") is None
                and not MANUAL_VISUAL_SIGNAL.search(state["query"])
            ):
                continue
            selected.setdefault(item.agent, item.request.strip())
        if "appointment" in selected or APPOINTMENT_SIGNAL.search(state["query"]):
            selected = {"appointment": state["query"]}
        if "scope_guard" in selected:
            if scoped_query:
                selected = {"rag": scoped_query}
            elif not (
                has_first_aid_intent(state["query"])
                and MIXED_CURRENT_SCOPE_SIGNAL.search(state["query"])
            ):
                return [{"agent": "scope_guard", "request": selected["scope_guard"]}]
            else:
                selected = {"rag": state["query"]}
        if scoped_query:
            selected.pop("web_search", None)
            if "protocol_tools" in selected and not PROTOCOL_TOOL_SIGNAL.search(scoped_query):
                selected.pop("protocol_tools", None)
            selected.setdefault("rag", scoped_query)
        rag_search_request = selected.get("rag")
        # Preserve the original utterance unless the deterministic scope policy
        # extracted a narrower first-aid-only clause from a mixed request.
        for specialist in ("rag", "visual", "appointment", "protocol_tools"):
            if specialist in selected:
                selected[specialist] = (
                    scoped_query
                    if scoped_query and specialist in {"rag", "protocol_tools"}
                    else state["query"]
                )
        if (
            not scoped_query
            and
            has_first_aid_intent(state["query"])
            and MIXED_CURRENT_SCOPE_SIGNAL.search(state["query"])
            and CURRENT_INFORMATION.search(state["query"])
        ):
            selected.setdefault("rag", state["query"])
            selected.setdefault("web_search", state["query"])
        if state.get("image_path") is not None and FIRST_AID_SIGNAL.search(state["query"]):
            selected.setdefault("visual", state["query"])
            selected.setdefault("rag", state["query"])
        if state.get("image_path") is None and MANUAL_VISUAL_SIGNAL.search(state["query"]):
            selected.pop("web_search", None)
            selected.setdefault("visual", state["query"])
            selected.setdefault("rag", state["query"])
        order = ("rag", "web_search", "visual", "appointment", "protocol_tools")
        routes = [
            {"agent": name, "request": selected[name]}
            for name in order if name in selected
        ]
        if rag_search_request and rag_search_request.strip() != state["query"].strip():
            for route in routes:
                if route["agent"] == "rag":
                    route["search_request"] = rag_search_request.strip()
                    break
        return routes

    def _deterministic_routes(self, state: AgentState) -> list[dict[str, str]]:
        query = state["query"]
        scoped_query = _first_aid_only_subquery(query)
        if scoped_query:
            agent = "protocol_tools" if PROTOCOL_TOOL_SIGNAL.search(scoped_query) else "rag"
            return [{"agent": agent, "request": scoped_query}]
        if APPOINTMENT_SIGNAL.search(query):
            return [{"agent": "appointment", "request": query}]
        if PROTOCOL_TOOL_SIGNAL.search(query):
            return [{"agent": "protocol_tools", "request": query}]
        routes: list[dict[str, str]] = []
        if CURRENT_INFORMATION.search(query):
            routes.append({"agent": "web_search", "request": query})
        else:
            routes.append({"agent": "rag", "request": query})
        if state.get("image_path") is not None:
            routes.append({"agent": "visual", "request": query})
            if FIRST_AID_SIGNAL.search(query) and not any(row["agent"] == "rag" for row in routes):
                routes.insert(0, {"agent": "rag", "request": query})
        elif MANUAL_VISUAL_SIGNAL.search(query):
            if not any(row["agent"] == "rag" for row in routes):
                routes.insert(0, {"agent": "rag", "request": query})
            routes.append({"agent": "visual", "request": query})
        return routes

    @staticmethod
    def _specialist_nodes(state: AgentState) -> list[str]:
        mapping = {
            "rag": "rag_agent", "web_search": "web_agent",
            "visual": "visual_agent", "appointment": "appointment_agent",
            "protocol_tools": "protocol_agent", "scope_guard": "scope_guard",
        }
        return [mapping[item["agent"]] for item in state.get("routing", [])]

    @staticmethod
    def _request_for(state: AgentState, agent: str) -> str:
        for item in state.get("routing", []):
            if item["agent"] == agent:
                return item["request"]
        return state["query"]

    @staticmethod
    def _search_request_for(state: AgentState, agent: str) -> str | None:
        for item in state.get("routing", []):
            if item["agent"] == agent:
                return item.get("search_request")
        return None

    def _rag_node(self, state: AgentState) -> dict[str, Any]:
        request = self._request_for(state, "rag")
        try:
            output, payload = self.rag.run(
                request,
                fallback_retrieval_query=self._search_request_for(state, "rag"),
            )
            generation = payload.get("generation", {}) if payload else {}
            # The canonical RAG generation is the evidence contract. The outer
            # ReAct prose is orchestration text and may accidentally omit its
            # citation labels.
            output = str(generation.get("answer") or output)
            return {"specialist_results": [{
                "agent": "rag", "request": request, "output": output,
                "payload": payload, "ok": bool(payload),
            }]}
        except Exception as exc:
            return {"specialist_results": [self._failed_result("rag", request, exc)]}

    def _web_node(self, state: AgentState) -> dict[str, Any]:
        request = self._request_for(state, "web_search")
        try:
            web_result = self.web.run(request)
            if isinstance(web_result, tuple):
                output, payload = web_result
            else:
                output, payload = web_result, {}
            return {"specialist_results": [{
                "agent": "web_search", "request": request,
                "output": output, "payload": payload,
                "ok": bool(output.strip()),
            }]}
        except Exception as exc:
            return {"specialist_results": [self._failed_result("web_search", request, exc)]}

    def _visual_node(self, state: AgentState) -> dict[str, Any]:
        request = self._request_for(state, "visual")
        try:
            image_path = state.get("image_path")
            if image_path is None:
                visual = self.manual_visual_factory(
                    self.runtime, self.project_root,
                    agent_model=self.agent_model, ollama_url=self.ollama_url,
                    agent_provider=self.agent_provider, groq_api_key=self.groq_api_key,
                    agent_timeout=self.agent_timeout,
                    agent_fallback_model=self.agent_fallback_model,
                )
                output, payload = visual.run(request)
            else:
                visual = self.visual_factory(
                    image_path, model=self.vision_model,
                    agent_model=self.agent_model, ollama_url=self.ollama_url,
                    agent_provider=self.agent_provider, groq_api_key=self.groq_api_key,
                    agent_timeout=self.agent_timeout,
                    agent_fallback_model=self.agent_fallback_model,
                )
                output = visual.run(request)
                payload = {"mode": "attachment_analysis", "visuals": []}
            return {"specialist_results": [{
                "agent": "visual", "request": request,
                "output": output, "payload": payload,
                "ok": bool(output.strip()) and (
                    image_path is not None or bool(payload.get("visuals"))
                ),
            }]}
        except Exception as exc:
            return {"specialist_results": [self._failed_result("visual", request, exc)]}

    def _appointment_node(self, state: AgentState) -> dict[str, Any]:
        request = self._request_for(state, "appointment")
        try:
            if self.appointment_client is None:
                raise RuntimeError("Agent System B client is not configured")
            payload = self.appointment_client.chat(
                request, str(state.get("conversation_id") or "anonymous"),
            )
            record_tool(
                "agent_system_b_http",
                {
                    "query": request,
                    "conversation_id": str(state.get("conversation_id") or "anonymous"),
                },
                success=True,
            )
            usage = payload.get("usage") or {}
            if usage.get("model"):
                record_usage(
                    str(usage["model"]),
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    provider=str(usage.get("provider", "unknown")),
                    model_calls=int(usage.get("model_calls", 0)),
                )
            output = str(payload.get("message", ""))
            return {"specialist_results": [{
                "agent": "appointment", "request": request, "output": output,
                "payload": payload, "ok": bool(output.strip()),
            }]}
        except Exception as exc:
            record_tool(
                "agent_system_b_http",
                {
                    "query": request,
                    "conversation_id": str(state.get("conversation_id") or "anonymous"),
                },
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {"specialist_results": [
                self._failed_result("appointment", request, exc)
            ]}

    def _protocol_node(self, state: AgentState) -> dict[str, Any]:
        request = self._request_for(state, "protocol_tools")
        try:
            output, payload = self.protocol_tools.run(request)
            return {"specialist_results": [{
                "agent": "protocol_tools", "request": request, "output": output,
                "payload": payload, "ok": bool(output.strip()),
            }]}
        except Exception as exc:
            return {"specialist_results": [
                self._failed_result("protocol_tools", request, exc)
            ]}

    def _scope_node(self, state: AgentState) -> dict[str, Any]:
        request = self._request_for(state, "scope_guard")
        arabic = detect_query_language(state["query"]) == "ar"
        output = (
            "يمكنني المساعدة في الإسعافات الأولية والإرشادات الطارئة فقط. "
            "اسألني عن إصابة أو عَرَض أو إجراء إسعافي."
            if arabic else
            "I can help with first aid and immediate emergency guidance only. "
            "Ask me about an injury, symptom, or first-aid action."
        )
        return {"specialist_results": [{
            "agent": "scope_guard", "request": request, "output": output, "ok": True,
        }]}

    @staticmethod
    def _failed_result(agent: AgentName, request: str, exc: Exception) -> SpecialistResult:
        return {
            "agent": agent, "request": request, "output": "", "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    def _answer_node(self, state: AgentState) -> dict[str, Any]:
        successful = [item for item in state.get("specialist_results", []) if item.get("ok")]
        routes = list(dict.fromkeys(item["agent"] for item in successful))
        failures = [item for item in state.get("specialist_results", []) if not item.get("ok")]
        language = detect_query_language(state["query"])

        if not successful:
            return {
                "answer_mode": "safe_abstention",
                "response": self._abstention_response(
                    state["query"], routes, language,
                    "No specialist returned usable evidence.", failures,
                ),
            }
        if routes == ["scope_guard"]:
            return {
                "answer_mode": "scope_refusal",
                "response": self._scope_response(state["query"], successful[0]["output"], language),
            }
        if routes == ["appointment"]:
            payload = successful[0].get("payload", {})
            return {
                "answer_mode": "appointment_passthrough",
                "response": {
                    "query": state["query"], "route": routes,
                    "specialists": successful + failures, "retrieval": None,
                    "generation": {
                        "status": "answered", "answer": payload.get("message", ""),
                        "citations": [], "citation_labels": [], "abstention_category": None,
                        "language": language, "model": "agent-system-b/google-adk",
                        "agent_route": routes,
                    },
                    "appointment": payload,
                    "answer_agent": {
                        "used": False, "mode": "appointment_passthrough", "valid": True,
                    },
                },
            }
        rag_payload = successful[0].get("payload") if routes == ["rag"] else None
        if rag_payload:
            payload = rag_payload
            return {
                "answer_mode": "rag_passthrough",
                "response": {
                    "query": state["query"],
                    "route": routes,
                    "specialists": successful + failures,
                    "retrieval": payload.get("retrieval"),
                    "generation": {**payload.get("generation", {}), "agent_route": routes},
                    "answer_agent": {"used": False, "mode": "rag_passthrough", "valid": True},
                },
            }

        manual_visual_payload = next((
            item.get("payload", {}) for item in successful
            if item["agent"] == "visual"
            and item.get("payload", {}).get("mode") == "manual_render"
        ), None)
        combined_rag_payload = next((
            item.get("payload", {}) for item in successful if item["agent"] == "rag"
        ), None)
        if manual_visual_payload and combined_rag_payload:
            generation = combined_rag_payload.get("generation", {})
            if (
                generation.get("status") != "answered"
                or not generation.get("citation_labels")
            ):
                generation = self._manual_visual_generation(
                    state["query"], manual_visual_payload,
                )
            return {
                "answer_mode": "rag_visual_passthrough",
                "response": {
                    "query": state["query"],
                    "route": routes,
                    "specialists": successful + failures,
                    "retrieval": combined_rag_payload.get("retrieval"),
                    "generation": {**generation, "agent_route": routes},
                    "visuals": manual_visual_payload.get("visuals", []),
                    "answer_agent": {
                        "used": False, "mode": "rag_visual_passthrough", "valid": True,
                    },
                },
            }

        attempts = int(state.get("answer_attempts", 0)) + 1
        traces = [
            {"agent": item["agent"], "output": item["output"]} for item in successful
        ]
        repair = "\n".join(state.get("output_guard_errors", []))
        draft = "Synthesize the successful specialist evidence."
        if repair:
            draft += f" Previous output failed validation: {repair}"
        try:
            finalized = self.answer_agent.run(
                question=state["query"], supervisor_draft=draft,
                specialists=traces, language=language,
            )
        except Exception as exc:
            finalized = {
                "answer": (
                    "I cannot produce a sufficiently grounded final answer "
                    "because the answer service is temporarily unavailable."
                ),
                "valid": False,
                "citation_labels": [],
                "urls": [],
                "errors": [f"answer service failure: {type(exc).__name__}: {exc}"],
                "provider_used": "unavailable",
                "provider_fallback": False,
                "provider_errors": [f"{type(exc).__name__}: {exc}"],
            }
        answer = finalized.get("answer", "")
        response = {
            "query": state["query"],
            "route": routes,
            "specialists": successful + failures,
            "retrieval": next(
                (item.get("payload", {}).get("retrieval") for item in successful if item["agent"] == "rag"),
                None,
            ),
            "generation": {
                "status": "answered" if finalized.get("valid") else "abstained",
                "answer": answer,
                "citations": next(
                    (item.get("payload", {}).get("generation", {}).get("citations", []) for item in successful if item["agent"] == "rag"),
                    [],
                ),
                "citation_labels": finalized.get("citation_labels", []),
                "abstention_category": None if finalized.get("valid") else "model_generated",
                "language": language,
                "model": self.agent_model,
                "agent_route": routes,
            },
            "answer_agent": {"used": True, "mode": "synthesis", **finalized},
            "visuals": next(
                (item.get("payload", {}).get("visuals", []) for item in successful if item["agent"] == "visual"),
                [],
            ),
        }
        return {"answer_mode": "synthesis", "answer_attempts": attempts, "response": response}

    @staticmethod
    def _manual_visual_generation(
        query: str, payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create a cited, non-interpretive answer for a rendered source page."""
        visual = next(iter(payload.get("visuals", [])), {})
        doc_id = str(visual.get("doc_id") or "manual")
        pdf_page = visual.get("pdf_page")
        language = detect_query_language(query)
        if language == "ar":
            answer = (
                f"تم عرض الصفحة {pdf_page} من دليل {doc_id}، وهي الصفحة المرئية "
                "المسترجعة من المصدر المطلوب [S1]."
            )
        elif re.search(r"\b(?:manuel|illustration|montrez|affichez)\b", query, re.I):
            answer = (
                f"La page {pdf_page} du manuel {doc_id} est affichée; il s'agit "
                "de la page visuelle extraite de la source demandée [S1]."
            )
        else:
            answer = (
                f"Page {pdf_page} from manual {doc_id} is displayed as the "
                "retrieved authoritative source page [S1]."
            )
        citation = {
            "label": "S1", "doc_id": doc_id,
            "pdf_page": pdf_page, "page": pdf_page,
            "section": visual.get("section"),
            "chunk_id": visual.get("chunk_id"),
        }
        return {
            "status": "answered", "answer": answer,
            "citations": [citation], "citation_labels": ["S1"],
            "abstention_category": None, "language": language,
            "model": "deterministic-manual-visual",
        }

    def _output_guard_node(self, state: AgentState) -> dict[str, Any]:
        response = state.get("response") or self._abstention_response(
            state["query"], [], detect_query_language(state["query"]),
            "The workflow did not produce a response.", [],
        )
        decision = validate_agent_output(
            query=state["query"], response=response,
            answer_mode=state.get("answer_mode", "unknown"),
            specialists=state.get("specialist_results", []),
        )
        if decision.valid:
            return {
                "response": self._add_orchestration(response, state, "passed", []),
                "output_guard_status": "passed", "output_guard_errors": [],
            }
        errors = list(decision.errors)
        if (
            state.get("answer_mode") == "synthesis"
            and int(state.get("answer_attempts", 0)) <= OUTPUT_REPAIR_ATTEMPTS
        ):
            return {"output_guard_status": "repair", "output_guard_errors": errors}
        failed = self._abstention_response(
            state["query"], response.get("route", []), detect_query_language(state["query"]),
            "The final answer failed deterministic safety validation.",
            state.get("specialist_results", []),
        )
        return {
            "response": self._add_orchestration(failed, state, "failed_closed", errors),
            "output_guard_status": "failed_closed", "output_guard_errors": errors,
        }

    @staticmethod
    def _after_output_guard(state: AgentState) -> str:
        return "repair" if state.get("output_guard_status") == "repair" else "end"

    def _add_orchestration(
        self, response: dict[str, Any], state: AgentState,
        output_status: str, output_errors: list[str],
    ) -> dict[str, Any]:
        failures = [
            {"agent": item.get("agent"), "error": item.get("error", "unknown error")}
            for item in state.get("specialist_results", []) if not item.get("ok")
        ]
        diagnostics = [
            {
                "agent": item.get("agent"),
                "provider_used": item.get("payload", {}).get("provider_used"),
                "provider_fallback": item.get("payload", {}).get("provider_fallback", False),
                "provider_errors": item.get("payload", {}).get("provider_errors", []),
            }
            for item in state.get("specialist_results", [])
            if item.get("ok") and item.get("payload", {}).get("provider_used")
        ]
        return {
            **response,
            "orchestration": {
                "supervisor_provider": state.get("supervisor_provider", "not_used"),
                "supervisor_model": state.get("supervisor_model", "not_used"),
                "supervisor_latency_ms": float(state.get("supervisor_latency_ms", 0.0)),
                "selected_routes": [item["agent"] for item in state.get("routing", [])],
                "input_guard": state.get("guard_decision", "unknown"),
                "router_errors": state.get("router_errors", []),
                "specialist_failures": failures,
                "specialist_diagnostics": diagnostics,
                "answer_mode": state.get("answer_mode", "unknown"),
                "answer_attempts": int(state.get("answer_attempts", 0)),
                "output_guard": output_status,
                "output_guard_errors": output_errors,
            },
        }

    @staticmethod
    def _direct_response(
        query: str, answer: str, *, language: str, mode: str = "greeting",
    ) -> dict[str, Any]:
        return {
            "query": query, "route": ["supervisor"], "specialists": [],
            "retrieval": None,
            "generation": {
                "status": "answered", "answer": answer, "citations": [],
                "citation_labels": [], "abstention_category": None,
                "language": language, "model": "deterministic",
                "agent_route": ["supervisor"],
            },
            "answer_agent": {"used": False, "mode": mode, "valid": True},
        }

    def _scope_response(self, query: str, answer: str, language: str) -> dict[str, Any]:
        return {
            "query": query, "route": ["scope_guard"],
            "specialists": [{"agent": "scope_guard", "output": answer, "ok": True}],
            "retrieval": None,
            "generation": {
                "status": "abstained", "answer": answer, "citations": [],
                "citation_labels": [], "abstention_category": "policy_gate",
                "policy_reason": "out_of_scope", "language": language,
                "model": self.agent_model, "agent_route": ["scope_guard"],
            },
            "answer_agent": {"used": False, "mode": "scope_refusal", "valid": True},
        }

    def _abstention_response(
        self, query: str, routes: list[str], language: str, reason: str,
        specialists: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "query": query, "route": routes,
            "specialists": [dict(item) for item in specialists],
            "retrieval": None,
            "generation": {
                "status": "abstained", "answer": reason, "citations": [],
                "citation_labels": [], "abstention_category": "model_generated",
                "insufficient_evidence_reason": reason, "language": language,
                "model": self.agent_model, "agent_route": routes,
            },
            "answer_agent": {"used": False, "mode": "safe_abstention", "valid": True},
        }
