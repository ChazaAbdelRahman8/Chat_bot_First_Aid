"""Final LangGraph ReAct agent with deterministic evidence-reference validation."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from .prompts import ANSWER_PROMPT
from .specialists import _model
from guardrails.reference_validation import ReferenceValidation, validate_references
from system_limits import ANSWER_AGENT_RECURSION_LIMIT, PROVIDER_FALLBACK_ATTEMPTS
from telemetry import record_tool


class FinalAnswerReActAgent:
    """Synthesizes specialist outputs but cannot retrieve additional evidence."""

    def __init__(
        self, *, model: str, ollama_url: str, provider: str = "ollama",
        groq_api_key: str | None = None, timeout: float = 60,
        fallback_model: str | None = None,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.provider = provider
        self.groq_api_key = groq_api_key
        self.timeout = timeout
        self.fallback_model = fallback_model

    def _invoke_graph(
        self, *, provider: str, model_name: str,
        submit_tool: Any, prompt: str,
    ) -> None:
        graph = create_agent(
            model=_model(
                model_name,
                self.ollama_url,
                provider=provider,
                groq_api_key=self.groq_api_key,
                timeout=self.timeout,
                fallback_model=self.fallback_model,
            ),
            tools=[submit_tool],
            system_prompt=ANSWER_PROMPT,
            name="final_answer_agent",
        )
        graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": ANSWER_AGENT_RECURSION_LIMIT},
        )

    def run(
        self, *, question: str, supervisor_draft: str,
        specialists: list[dict[str, str]], language: str,
    ) -> dict[str, Any]:
        evidence = "\n\n".join(
            f"SPECIALIST={item['agent']}\n{item['output']}" for item in specialists
        )
        accepted: list[tuple[str, ReferenceValidation]] = []
        web_used = any(item["agent"] == "web_search" for item in specialists)

        @tool("submit_final_answer")
        def submit_final_answer(answer: str) -> str:
            """Validate and submit the complete final answer against allowed evidence references."""
            validation = validate_references(answer, evidence, web_used=web_used)
            record_tool(
                "submit_final_answer",
                {
                    "answer_length": len(answer),
                    "citation_labels": list(validation.citation_labels),
                    "urls": list(validation.urls),
                },
                success=validation.valid,
                error="; ".join(validation.errors) if validation.errors else None,
            )
            if validation.valid:
                accepted.append((answer.strip(), validation))
            return json.dumps({
                "accepted": validation.valid,
                "errors": list(validation.errors),
                "instruction": "Finish now." if validation.valid else "Correct the answer and resubmit.",
            })

        prompt = (
            f"USER_LANGUAGE={language}\nUSER_QUESTION:\n{question}\n\n"
            f"SUPERVISOR_DRAFT:\n{supervisor_draft}\n\n"
            f"SPECIALIST_EVIDENCE:\n{evidence}"
        )
        provider_errors: list[str] = []
        primary_is_groq = self.provider == "groq" or (
            self.provider == "auto" and bool(self.groq_api_key)
        )
        primary_provider = "groq" if primary_is_groq else "ollama"
        provider_used = primary_provider
        fallback_used = False
        try:
            self._invoke_graph(
                provider=self.provider,
                model_name=self.model,
                submit_tool=submit_final_answer,
                prompt=prompt,
            )
        except Exception as exc:
            provider_errors.append(
                f"{primary_provider}: {type(exc).__name__}: {exc}"
            )

        if not accepted and primary_is_groq and PROVIDER_FALLBACK_ATTEMPTS > 0:
            # The deterministic evidence-only composition below is the bounded
            # fallback. Launching a local answer model here duplicated synthesis
            # and could add another 60+ seconds after Groq rate limiting.
            fallback_used = True
            provider_used = "deterministic_evidence"

        if not accepted:
            # Fail over to a deterministic evidence-preserving composition.
            # This adds no new medical claim: it concatenates only successful
            # specialist evidence and still applies the same reference guard.
            deterministic = "\n\n".join(
                item["output"].strip() for item in specialists if item["output"].strip()
            )
            validation = validate_references(
                deterministic, evidence, web_used=web_used,
            )
            record_tool(
                "submit_final_answer",
                {
                    "answer_length": len(deterministic),
                    "citation_labels": list(validation.citation_labels),
                    "urls": list(validation.urls),
                    "mode": "deterministic_evidence_fallback",
                },
                success=validation.valid,
                error="; ".join(validation.errors) if validation.errors else None,
            )
            if validation.valid:
                accepted.append((deterministic, validation))
        if not accepted:
            return {
                "answer": "I cannot produce a sufficiently grounded final answer from the available evidence.",
                "valid": False,
                "citation_labels": [],
                "urls": [],
                "errors": [
                    "answer agent did not submit a valid evidence-referenced answer",
                    *provider_errors,
                ],
                "provider_used": provider_used,
                "provider_fallback": fallback_used,
                "provider_errors": provider_errors,
            }
        answer, validation = accepted[-1]
        return {
            "answer": answer,
            "valid": True,
            "citation_labels": list(validation.citation_labels),
            "urls": list(validation.urls),
            "errors": [],
            "provider_used": provider_used,
            "provider_fallback": fallback_used,
            "provider_errors": provider_errors,
        }
