"""Resumable Qwen2.5-VL processing for the prepared candidate queue."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .ollama_vlm import describe_page
from .prompt import PROMPT_VERSION
from .validation import classify_description


LOGGER = logging.getLogger("rag.vision.pipeline")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_vision(
    queue_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
    timeout_seconds: int = 300,
    limit: int | None = None,
) -> dict:
    queue = _read_jsonl(queue_path)
    existing = {row["visual_id"]: row for row in _read_jsonl(output_path)}
    successful = {
        key for key, row in existing.items()
        if (
            row.get("error") is None
            and isinstance(row.get("description"), dict)
            and row.get("model") == model
            and row.get("prompt_version") == PROMPT_VERSION
        )
    }
    remaining = [row for row in queue if row["visual_id"] not in successful]
    if limit is not None:
        remaining = remaining[:limit]

    LOGGER.info(
        "Vision run started model=%s candidates=%d existing=%d remaining=%d limit=%s",
        model, len(queue), len(existing), len(remaining), limit,
    )

    for index, candidate in enumerate(remaining, start=1):
        started = time.perf_counter()
        LOGGER.info(
            "Processing page position=%d/%d record_id=%s visual_id=%s image=%s",
            index, len(remaining), candidate["record_id"], candidate["visual_id"],
            candidate["image_path"],
        )
        try:
            description = describe_page(
                Path(candidate["image_path"]), model, base_url, timeout_seconds
            )
            useful, status = classify_description(description)
            error = None
        except Exception as exc:  # checkpoint failures and allow a later retry
            description, useful, status = None, False, "error"
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception(
                "Vision page failed record_id=%s visual_id=%s",
                candidate["record_id"], candidate["visual_id"],
            )
        existing[candidate["visual_id"]] = {
            **candidate,
            "model": model,
            "provider": "ollama_local",
            "prompt_version": PROMPT_VERSION,
            "authority": "model_generated_supplement",
            "review_status": status,
            "vision_useful": useful,
            "description": description,
            "error": error,
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        write_jsonl_atomic(output_path, list(existing.values()))
        LOGGER.info(
            "Page checkpointed record_id=%s status=%s useful=%s latency_seconds=%.3f error=%s",
            candidate["record_id"], status, useful,
            time.perf_counter() - started, error,
        )
        print(f"[{index}/{len(remaining)}] {candidate['record_id']}: {status}", flush=True)

    values = list(existing.values())
    summary = {
        "candidates": len(queue),
        "saved": len(values),
        "successful": sum(row.get("error") is None for row in values),
        "useful": sum(bool(row.get("vision_useful")) for row in values),
        "manual_review": sum(row.get("review_status") == "manual_review_required" for row in values),
        "errors": sum(row.get("review_status") == "error" for row in values),
        "remaining": len(queue) - sum(row.get("error") is None for row in values),
    }
    LOGGER.info("Vision run completed summary=%s", json.dumps(summary, sort_keys=True))
    return summary
