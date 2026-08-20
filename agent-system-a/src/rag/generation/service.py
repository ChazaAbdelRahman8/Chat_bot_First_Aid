"""Generation orchestration with deterministic validation and abstention labels."""

from __future__ import annotations

import json
from typing import Any, Protocol

from guardrails.input_guard import GuardDecision, evaluate_input
from rag.retrieval.hybrid import detect_query_language

from .prompt import PROMPT_VERSION, build_evidence, build_user_prompt
from .validation import validate_generation


class ModelClient(Protocol):
    model: str

    def complete(self, user_prompt: str, correction: str | None = None) -> dict: ...


def _build_correction(
    raw: dict[str, Any], error: ValueError, allowed_labels: set[str], query: str,
    *, final_repair: bool = False,
) -> str:
    """Give the model enough concrete state to repair only formatting defects."""
    previous = json.dumps(raw, ensure_ascii=False, indent=2)
    labels = ", ".join(sorted(allowed_labels))
    expected_language = "Arabic" if detect_query_language(query) == "ar" else "English"
    targeted = ""
    error_text = str(error)
    if "every factual sentence or bullet" in error_text:
        targeted = (
            "Rewrite as short bullets. Put a genuinely supporting [S#] citation in "
            "EVERY bullet before its final punctuation. Remove unsupported bullets.\n"
        )
    elif "response language" in error_text:
        targeted = (
            f"Rewrite the entire answer and reason in {expected_language}. Preserve valid "
            "citation labels and do not leave explanatory prose in another language.\n"
        )
    if final_repair:
        targeted += (
            "FINAL REPAIR: produce at most three short sentences in one paragraph. "
            "Each sentence must end in the form '[S1].' using only a label that supports "
            "that sentence. Do not use headings, numbered lists, abbreviations, or an "
            "uncited introduction. If the evidence cannot support a direct answer, set "
            "abstain to true instead of inventing content.\n"
        )
    return (
        "REPAIR THE PREVIOUS JSON. Do not draft an unrelated new answer.\n"
        f"Validation error: {error}\n"
        f"Allowed citation labels: {labels}\n"
        f"Required response language: {expected_language}\n"
        f"{targeted}"
        "Repair rules:\n"
        "- Every factual sentence and every bullet must contain at least one supporting "
        "[S#] label before that sentence or bullet ends.\n"
        "- If no supplied evidence supports a claim, delete that claim; never guess a label.\n"
        "- Recompute cited_sources from the repaired answer. It must contain each inline "
        "label exactly once and no label absent from the answer.\n"
        "- Preserve the required four JSON keys and return JSON only.\n"
        "PREVIOUS JSON:\n"
        f"{previous}"
    )


def policy_gate_response(query: str, model: str, decision: GuardDecision) -> dict[str, Any]:
    arabic = detect_query_language(query) == "ar"
    answer = decision.message_ar if arabic else decision.message_en
    return {
        "status": "abstained",
        "answer": answer,
        "citations": [],
        "citation_labels": [],
        "abstention_category": "policy_gate",
        "policy_reason": decision.reason_code,
        "insufficient_evidence_reason": answer,
        "language": "ar" if arabic else "en",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_attempts": 0,
        "retrieval_route": None,
    }


def _build_output_error_correction(error: ValueError) -> str:
    """Request a clean schema response when model output cannot be parsed."""
    return (
        "Your previous response could not be parsed as complete JSON. "
        f"Parser error: {error}. Return one complete JSON object only, with exactly "
        "these keys: answer, cited_sources, abstain, insufficient_evidence_reason. "
        "Do not use Markdown fences or add commentary. Keep the answer concise so the "
        "JSON is not truncated. Every factual sentence or bullet must contain a valid "
        "inline [S#] citation before it ends."
    )


def _empty_evidence_response(query: str, model: str) -> dict[str, Any]:
    arabic = detect_query_language(query) == "ar"
    answer = (
        "لا تتوفر أدلة كافية في المصادر المسترجعة للإجابة بأمان."
        if arabic else
        "The retrieved sources do not contain enough evidence to answer safely."
    )
    return {
        "status": "abstained",
        "answer": answer,
        "citations": [],
        "citation_labels": [],
        "abstention_category": "retrieval_induced",
        "insufficient_evidence_reason": answer,
        "language": "ar" if arabic else "en",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_attempts": 0,
    }


def _validation_failure_abstention(
    query: str, model: str, attempts: int, error: ValueError | None,
    retrieval_route: str | None,
) -> dict[str, Any]:
    arabic = detect_query_language(query) == "ar"
    answer = (
        "\u0644\u0627 \u064a\u0645\u0643\u0646 \u0625\u0646\u0634\u0627\u0621 \u0625\u062c\u0627\u0628\u0629 \u0645\u062f\u0639\u0648\u0645\u0629 \u0628\u0627\u0644\u0623\u062f\u0644\u0629 \u0628\u0623\u0645\u0627\u0646."
        if arabic else
        "A safely evidence-supported answer could not be generated."
    )
    reason = (
        "\u0641\u0634\u0644 \u0627\u0644\u062c\u0648\u0627\u0628 \u0627\u0644\u0645\u0648\u0644\u062f \u0641\u064a \u0627\u0644\u062a\u062d\u0642\u0642 \u0645\u0646 \u0627\u0644\u0627\u0633\u062a\u0634\u0647\u0627\u062f \u0623\u0648 \u0627\u0644\u0623\u062f\u0644\u0629."
        if arabic else
        "The generated response failed citation or evidence validation."
    )
    return {
        "status": "abstained",
        "answer": answer,
        "citations": [],
        "citation_labels": [],
        "abstention_category": "model_generated",
        "insufficient_evidence_reason": reason,
        "language": "ar" if arabic else "en",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_attempts": attempts,
        "retrieval_route": retrieval_route,
        "validation_error": str(error) if error else "unknown validation error",
    }


class GenerationService:
    def __init__(self, client: ModelClient, *, max_validation_attempts: int = 2) -> None:
        if max_validation_attempts < 1:
            raise ValueError("max_validation_attempts must be at least one")
        self.client = client
        self.max_validation_attempts = max_validation_attempts

    def generate(self, query: str, retrieval: dict[str, Any]) -> dict[str, Any]:
        decision = evaluate_input(query)
        if decision.blocked:
            return policy_gate_response(query, self.client.model, decision)
        evidence, sources = build_evidence(list(retrieval.get("results", [])))
        if not sources:
            return _empty_evidence_response(query, self.client.model)
        user_prompt = build_user_prompt(query, evidence)
        correction = None
        last_error: ValueError | None = None
        for attempt in range(1, self.max_validation_attempts + 1):
            try:
                raw = self.client.complete(user_prompt, correction)
            except ValueError as exc:
                last_error = exc
                correction = _build_output_error_correction(exc)
                continue
            try:
                validated = validate_generation(raw, query=query, allowed_labels=set(sources))
            except ValueError as exc:
                last_error = exc
                correction = _build_correction(
                    raw,
                    exc,
                    set(sources),
                    query,
                    final_repair=attempt >= self.max_validation_attempts - 1,
                )
                continue
            labels = validated["cited_sources"]
            abstained = validated["abstain"]
            return {
                "status": "abstained" if abstained else "answered",
                "answer": validated["answer"],
                "citations": [sources[label] for label in labels],
                "citation_labels": labels,
                "abstention_category": "model_generated" if abstained else None,
                "insufficient_evidence_reason": validated["insufficient_evidence_reason"],
                "language": detect_query_language(query),
                "model": self.client.model,
                "provider": str(getattr(self.client, "provider", "unknown")),
                "prompt_version": PROMPT_VERSION,
                "validation_attempts": attempt,
                "retrieval_route": retrieval.get("route"),
            }
        return _validation_failure_abstention(
            query,
            self.client.model,
            self.max_validation_attempts,
            last_error,
            retrieval.get("route"),
        )
