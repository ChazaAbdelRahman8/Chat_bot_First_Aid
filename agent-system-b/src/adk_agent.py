"""Google ADK runtime: the model picks between a search tool and a book tool."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any


class AdkAppointmentRuntime:
    def __init__(self, service: Any, *, timeout: float = 20) -> None:
        self.service = service
        self.timeout = timeout
        self.results: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.available = False
        self.error: str | None = None
        try:
            from google.adk.agents import Agent
            from google.adk.models.lite_llm import LiteLlm
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService

            def _dispatch(handler: Any, query: str | int, conversation_id: str, request_id: str, slots: dict[str, Any]) -> dict[str, Any]:
                with self.lock:
                    existing = self.results.get(request_id)
                if existing is not None:
                    return existing
                # query accepts int too: when the user's whole message is a bare
                # number (e.g. replying "3" to pick an option), Groq's strict
                # tool-call schema rejects the call outright if it emits an
                # unquoted number for a string-typed field.
                result = handler(str(query), conversation_id, request_id, slots)
                with self.lock:
                    self.results[request_id] = result
                return result

            def search_psychologists(
                query: str | int,
                conversation_id: str,
                request_id: str,
                language: str = "",
                consultation_mode: str = "",
                specialization: str = "",
                max_budget_usd: int = 0,
                relative_minutes: int = 0,
            ) -> dict[str, Any]:
                """Search or refine synthetic psychologist matches.

                Call this when the user is describing what they want, changing a
                preference, or asking to see options again - not when they are
                picking one of the previously shown options to book.

                Args:
                    query: The user's original message, unmodified.
                    conversation_id: The conversation id supplied by the caller.
                    request_id: The request id supplied by the caller.
                    language: "Arabic", "English", or "French" if the user stated a
                        preferred spoken language, else "".
                    consultation_mode: "online" or "in_person" if stated, else "".
                    specialization: One of "anxiety", "stress", "grief", "trauma",
                        "family", "sleep" if stated, else "".
                    max_budget_usd: The user's stated budget ceiling in USD, or 0 if
                        none was stated.
                    relative_minutes: Minutes from now if the user asked for a
                        specific relative start time (e.g. "in 10 minutes"), else 0.

                Never invent a language, mode, specialization, or budget the user
                did not state - leave the slot empty instead.
                """
                slots = {
                    "language": language, "consultation_mode": consultation_mode,
                    "specialization": specialization, "max_budget_usd": max_budget_usd,
                    "relative_minutes": relative_minutes, "selection": "",
                }
                return _dispatch(self.service.handle_search, query, conversation_id, request_id, slots)

            def book_appointment(
                query: str | int,
                conversation_id: str,
                request_id: str,
                selection: str | int = "",
                relative_minutes: int = 0,
            ) -> dict[str, Any]:
                """Book one of the synthetic psychologist matches shown earlier.

                Call this only when the user is confirming, picking a numbered or
                named option, or explicitly asking to book - after
                search_psychologists has already shown matches in this
                conversation.

                Args:
                    query: The user's original message, unmodified.
                    conversation_id: The conversation id supplied by the caller.
                    request_id: The request id supplied by the caller.
                    selection: Which listed option the user is picking - an ordinal
                        ("1"/"2"/"3" or "first"/"second"/"third") or a psychologist id
                        (e.g. "PSY-002"). Empty if no option was named.
                    relative_minutes: Minutes from now if the user asked for a
                        specific relative start time (e.g. "in 10 minutes"), else 0.

                Never invent a psychologist id or selection the user did not state.
                """
                selection = "" if selection in ("", 0) else str(selection)
                slots = {
                    "selection": selection, "relative_minutes": relative_minutes,
                    "language": "", "consultation_mode": "", "specialization": "", "max_budget_usd": 0,
                }
                return _dispatch(self.service.handle_book, query, conversation_id, request_id, slots)

            model_name = os.getenv("AGENT_B_MODEL", "qwen/qwen3.6-27b")
            model = LiteLlm(
                model=f"groq/{model_name}", timeout=timeout, num_retries=0,
                reasoning_effort="none", reasoning_format="hidden",
            )
            self.root_agent = Agent(
                name="mental_health_appointment_agent",
                model=model,
                instruction=(
                    "You manage synthetic mental-health appointments only. You have two "
                    "tools: search_psychologists to find or refine matches from stated "
                    "preferences, and book_appointment to reserve one of the options "
                    "already shown in this conversation. Call exactly one of them per "
                    "message, choosing whichever matches what the user is doing right "
                    "now, and pass the conversation_id and request_id supplied by the "
                    "caller. Return the tool message without inventing psychologists, "
                    "availability, diagnoses, or treatment advice - the tools own all "
                    "of that."
                ),
                tools=[search_psychologists, book_appointment],
            )
            self.runner = Runner(
                app_name="agent_system_b", agent=self.root_agent,
                session_service=InMemorySessionService(), auto_create_session=True,
            )
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    async def run(self, query: str, conversation_id: str, request_id: str) -> dict[str, Any]:
        if not self.available:
            result = self.service.handle(query, conversation_id, request_id)
            result["execution_mode"] = "deterministic_fallback"
            return result
        try:
            from google.genai import types
            prompt = (
                f"query={query}\nconversation_id={conversation_id}\nrequest_id={request_id}\n"
                "Call the appointment tool exactly once."
            )
            async with asyncio.timeout(self.timeout):
                usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
                async for event in self.runner.run_async(
                    user_id="agent-system-a", session_id=conversation_id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                ):
                    metadata = getattr(event, "usage_metadata", None)
                    if metadata is not None:
                        input_tokens = int(
                            getattr(metadata, "prompt_token_count", 0)
                            or getattr(metadata, "input_token_count", 0) or 0
                        )
                        output_tokens = int(
                            getattr(metadata, "candidates_token_count", 0)
                            or getattr(metadata, "output_token_count", 0) or 0
                        )
                        if input_tokens or output_tokens:
                            usage["input_tokens"] += input_tokens
                            usage["output_tokens"] += output_tokens
                            usage["model_calls"] += 1
            with self.lock:
                result = self.results.pop(request_id, None)
            if result is None:
                raise RuntimeError("ADK agent did not call its appointment tool")
            result["execution_mode"] = "adk"
            result["usage"] = {
                **usage,
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                "provider": "groq",
                "model": os.getenv("AGENT_B_MODEL", "qwen/qwen3.6-27b"),
            }
            return result
        except Exception as exc:
            result = self.service.store.get_request(request_id) or self.service.handle(
                query, conversation_id, request_id,
            )
            result["execution_mode"] = "deterministic_fallback"
            result["adk_error"] = f"{type(exc).__name__}: {exc}"
            return result
