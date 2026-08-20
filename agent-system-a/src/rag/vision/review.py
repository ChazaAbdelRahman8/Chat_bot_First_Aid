"""Auditable human-review workflow for Phase 3 vision records."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validation import normalize_description


DECISIONS = {"approve", "edit", "exclude", "retry"}
ABSENCE_ONLY = re.compile(
    r"\b(?:not specified|not provided|not shown|not visible|not fully visible|"
    r"not clearly visible|not defined|not detailed|not explained|not labeled|"
    r"not explicitly labeled|not quantified|not mentioned|not included|"
    r"not indicated|not depicted|missing|unclear|cannot be determined|"
    r"does not specify|does not show|partially obscured|not fully legible)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def is_low_information(record: dict) -> bool:
    description = record.get("description") or {}
    labels = {str(value).strip().lower() for value in description.get("visible_labels", [])}
    evidence = any(
        description.get(field)
        for field in ("description", "demonstrated_actions", "action_sequence", "warnings_visible", "ambiguities")
    )
    return not evidence and labels.issubset({"forward", "foreword"})


def is_absence_only(record: dict) -> bool:
    ambiguities = (record.get("description") or {}).get("ambiguities") or []
    return bool(ambiguities) and all(ABSENCE_ONLY.search(str(item)) for item in ambiguities)


def build_review_queue(records: list[dict]) -> list[dict]:
    queue: list[dict] = []
    for record in records:
        if record.get("review_status") == "error" or record.get("error") is not None:
            category, recommendation = "model_error", "edit_or_exclude"
        elif is_low_information(record):
            category, recommendation = "low_information", "exclude"
        elif record.get("review_status") == "manual_review_required":
            if is_absence_only(record):
                category, recommendation = "absence_only", "approve_after_check"
            else:
                category, recommendation = "visual_ambiguity", "inspect"
        else:
            continue
        queue.append(
            {
                "visual_id": record["visual_id"],
                "record_id": record["record_id"],
                "doc_id": record["doc_id"],
                "pdf_page": record["pdf_page"],
                "image_path": record["image_path"],
                "review_category": category,
                "recommended_decision": recommendation,
                "review_status": record.get("review_status"),
                "vision_useful": record.get("vision_useful"),
                "model_error": record.get("error"),
                "description": record.get("description"),
            }
        )
    return queue


def validate_decisions(payload: Any, queue_ids: set[str]) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions", {}), dict):
        raise ValueError("decision payload must contain a decisions object")
    clean: dict[str, dict] = {}
    for visual_id, value in payload.get("decisions", {}).items():
        if visual_id not in queue_ids:
            raise ValueError(f"unknown visual_id: {visual_id}")
        if not isinstance(value, dict) or value.get("decision") not in DECISIONS:
            raise ValueError(f"invalid decision for {visual_id}")
        item = {
            "decision": value["decision"],
            "notes": str(value.get("notes", "")).strip(),
            "reviewed_at": str(value.get("reviewed_at") or datetime.now(timezone.utc).isoformat()),
        }
        if value["decision"] == "edit":
            item["edited_description"] = normalize_description(value.get("edited_description"))
        clean[visual_id] = item
    return {
        "decision_version": "1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "decisions": clean,
    }


def apply_decisions(records: list[dict], queue: list[dict], payload: dict) -> tuple[list[dict], list[str]]:
    queue_ids = {row["visual_id"] for row in queue}
    validated = validate_decisions(payload, queue_ids)
    decisions = validated["decisions"]
    missing = sorted(queue_ids - decisions.keys())
    output: list[dict] = []
    for raw in records:
        record = deepcopy(raw)
        decision = decisions.get(record["visual_id"])
        if not decision:
            output.append(record)
            continue
        action = decision["decision"]
        original_error = record.get("error")
        record["human_review"] = {
            **decision,
            "original_review_status": record.get("review_status"),
            "original_model_error": original_error,
        }
        record["error"] = None
        if action == "approve":
            if not isinstance(record.get("description"), dict):
                raise ValueError(f'{record["record_id"]}: an error record cannot be approved without an edit')
            record["description"]["review_required"] = False
            record["description"]["ambiguous_content"] = False
            record["review_status"] = "human_approved"
            record["vision_useful"] = bool(record["description"].get("should_include_in_rag", True))
            record["authority"] = "human_validated_supplement"
        elif action == "edit":
            record["description"] = decision["edited_description"]
            record["description"]["review_required"] = False
            record["description"]["ambiguous_content"] = False
            record["review_status"] = "human_edited"
            record["vision_useful"] = bool(record["description"].get("should_include_in_rag", True))
            record["authority"] = "human_validated_supplement"
        elif action == "exclude":
            record["review_status"] = "human_excluded"
            record["vision_useful"] = False
            record["authority"] = "human_reviewed_exclusion"
        else:
            record["review_status"] = "review_retry_required"
            record["vision_useful"] = False
            record["error"] = original_error or "Human reviewer requested model retry"
        output.append(record)
    return output, missing
