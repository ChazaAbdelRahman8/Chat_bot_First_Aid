"""Deterministic first-aid protocol lookup and triage calculators.

Backed by a small curated JSON knowledge base (``data/first_aid_protocols.json``),
not by live retrieval. This is what makes these MCP tools a distinct capability
from the RAG specialist in Agent System A rather than a re-implementation of it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "first_aid_protocols.json"


@lru_cache(maxsize=1)
def _load_conditions() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["conditions"]


def _match_condition(condition: str) -> dict[str, Any] | None:
    query = condition.strip().lower()
    if not query:
        return None
    for entry in _load_conditions():
        if query == entry["condition_id"].lower():
            return entry
        if any(query == alias.lower() for alias in entry["aliases"]):
            return entry
    for entry in _load_conditions():
        haystacks = [entry["condition_id"].lower(), entry["title"].lower(), *[
            alias.lower() for alias in entry["aliases"]
        ]]
        if any(query in haystack or haystack in query for haystack in haystacks):
            return entry
    return None


def list_conditions() -> list[str]:
    return [entry["condition_id"] for entry in _load_conditions()]


def get_first_aid_protocol(condition: str) -> dict[str, Any]:
    """Look up the ordered first-aid steps for a named condition.

    Returns a dict with ``found``, and on a match: ``condition_id``, ``title``,
    ``steps``, ``warnings``, and ``source``. On a miss, returns ``found: False``
    plus the list of ``available_conditions`` so a caller can retry.
    """
    match = _match_condition(condition)
    if match is None:
        return {
            "found": False,
            "query": condition,
            "available_conditions": list_conditions(),
        }
    return {
        "found": True,
        "condition_id": match["condition_id"],
        "title": match["title"],
        "steps": match["steps"],
        "warnings": match["warnings"],
        "source": match["source"],
    }


_BURN_DEGREES = {"first", "second", "third"}


def assess_burn_severity(percent_body_surface_area: float, degree: str) -> dict[str, Any]:
    """Classify burn severity from affected body-surface-area percentage and depth.

    ``percent_body_surface_area`` is the estimated percentage of total body
    surface area affected (0-100), typically estimated with the rule of nines.
    ``degree`` is one of ``first``, ``second``, or ``third``.

    Returns a dict with ``valid``, and on valid input: ``severity``
    (``minor``, ``moderate``, or ``critical``), ``recommended_action``, and the
    normalized inputs. This is a deterministic triage calculation, not a
    diagnosis, and does not replace emergency medical assessment.
    """
    degree_normalized = degree.strip().lower()
    if degree_normalized not in _BURN_DEGREES:
        return {
            "valid": False,
            "error": f"degree must be one of {sorted(_BURN_DEGREES)}, got {degree!r}",
        }
    if not (0 <= percent_body_surface_area <= 100):
        return {
            "valid": False,
            "error": "percent_body_surface_area must be between 0 and 100",
        }

    if degree_normalized == "third" or percent_body_surface_area >= 20:
        severity = "critical"
        action = (
            "Seek emergency care immediately (call emergency services). "
            "Cool the burn, cover loosely with a sterile non-fluffy dressing, "
            "and monitor for shock while awaiting help."
        )
    elif degree_normalized == "second" and percent_body_surface_area >= 5:
        severity = "moderate"
        action = (
            "Seek medical attention promptly. Cool the burn for 10-20 minutes "
            "under cool running water and cover loosely with a sterile dressing."
        )
    else:
        severity = "minor"
        action = (
            "Cool the burn for 10-20 minutes under cool running water, cover "
            "loosely, and monitor. Seek medical care if pain worsens, signs of "
            "infection appear, or the burn is on the face, hands, or genitals."
        )

    return {
        "valid": True,
        "percent_body_surface_area": percent_body_surface_area,
        "degree": degree_normalized,
        "severity": severity,
        "recommended_action": action,
    }
