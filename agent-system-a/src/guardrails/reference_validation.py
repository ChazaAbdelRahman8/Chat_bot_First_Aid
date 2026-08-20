"""Shared evidence-reference validation with no agent-package dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass


LABEL_PATTERN = re.compile(r"\[(S\d+)\]")
URL_PATTERN = re.compile(r"https?://[^\s<>\])]+")


@dataclass(frozen=True)
class ReferenceValidation:
    valid: bool
    errors: tuple[str, ...]
    citation_labels: tuple[str, ...]
    urls: tuple[str, ...]


def validate_references(answer: str, evidence: str, *, web_used: bool) -> ReferenceValidation:
    """Ensure the final answer cannot invent citation labels or source URLs."""
    answer_labels = tuple(dict.fromkeys(LABEL_PATTERN.findall(answer)))
    evidence_labels = set(LABEL_PATTERN.findall(evidence))
    answer_urls = tuple(dict.fromkeys(URL_PATTERN.findall(answer)))
    evidence_urls = set(URL_PATTERN.findall(evidence))
    errors: list[str] = []
    unknown_labels = sorted(set(answer_labels) - evidence_labels)
    unknown_urls = sorted(set(answer_urls) - evidence_urls)
    if unknown_labels:
        errors.append(f"unknown citation labels: {', '.join(unknown_labels)}")
    if unknown_urls:
        errors.append(f"unknown source URLs: {', '.join(unknown_urls)}")
    if evidence_labels and not answer_labels:
        errors.append("manual evidence is available but the answer has no citation label")
    if web_used and evidence_urls and not answer_urls:
        errors.append("web evidence is available but the answer has no source URL")
    if not answer.strip():
        errors.append("answer is empty")
    return ReferenceValidation(
        valid=not errors,
        errors=tuple(errors),
        citation_labels=answer_labels,
        urls=answer_urls,
    )
