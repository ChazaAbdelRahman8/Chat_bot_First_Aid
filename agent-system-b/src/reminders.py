"""Durable five-minute appointment reminder delivery."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

import httpx

from store import utcnow


LOGGER = logging.getLogger(__name__)


class ReminderScheduler:
    def __init__(
        self, store: Any, *, webhook_url: str, service_token: str,
        interval_seconds: float = 10, timeout_seconds: float = 10,
    ) -> None:
        self.store = store
        self.webhook_url = webhook_url
        self.service_token = service_token
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, name="appointment-reminders", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            try:
                self.tick()
            except Exception:
                LOGGER.exception("reminder scheduler tick failed")

    def tick(self, now: datetime | None = None) -> int:
        rows = self.store.claim_due_reminders(now or utcnow())
        for row in rows:
            payload = {
                "event_id": f"REM-{row['appointment_id']}",
                "conversation_id": row["conversation_id"],
                "appointment_id": row["appointment_id"],
                "title": "Appointment reminder",
                "message": (
                    f"Your appointment with {row['psychologist_name']} "
                    "starts in five minutes."
                ),
                "appointment_time": row["start_time"],
                "synthetic": True,
            }
            try:
                response = httpx.post(
                    self.webhook_url, json=payload,
                    headers={"X-Internal-Service-Token": self.service_token},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                self.store.finish_reminder(row["appointment_id"], True)
            except Exception as exc:
                self.store.finish_reminder(
                    row["appointment_id"], False, f"{type(exc).__name__}: {exc}",
                )
        return len(rows)
