"""Structured, rotating request/response logs for Agent System A."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


LOGGER_NAME = "first_aid.agent_io"
_lock = threading.RLock()
_configured_path: str | None = None


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "agent_event"),
        }
        fields = getattr(record, "event_fields", {})
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_agent_logger() -> logging.Logger:
    """Configure the file handler lazily from environment variables.

    No file is created when ``AGENT_LOG_PATH`` is unset, which keeps unit tests
    and library imports side-effect free.
    """
    global _configured_path
    selected_path = os.getenv("AGENT_LOG_PATH", "").strip()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    with _lock:
        if _configured_path == selected_path and logger.handlers:
            return logger
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        if selected_path:
            path = Path(selected_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = RotatingFileHandler(
                path,
                maxBytes=int(os.getenv("AGENT_LOG_MAX_BYTES", str(10 * 1024 * 1024))),
                backupCount=int(os.getenv("AGENT_LOG_BACKUP_COUNT", "5")),
                encoding="utf-8",
            )
            handler.setFormatter(JsonLineFormatter())
            logger.addHandler(handler)
        else:
            logger.addHandler(logging.NullHandler())
        _configured_path = selected_path
    return logger


def log_agent_event(event: str, **fields: Any) -> None:
    get_agent_logger().info(
        "agent event",
        extra={"event": event, "event_fields": fields},
    )


def log_agent_exception(event: str, **fields: Any) -> None:
    get_agent_logger().exception(
        "agent exception",
        extra={"event": event, "event_fields": fields},
    )


def reset_agent_logger_for_tests() -> None:
    """Close the active handler so tests can select an isolated path."""
    global _configured_path
    logger = logging.getLogger(LOGGER_NAME)
    with _lock:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        _configured_path = None
