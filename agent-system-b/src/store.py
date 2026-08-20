"""Mongo and in-memory persistence for Agent System B."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryAppointmentStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.profiles: dict[str, dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.appointments: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, dict[str, Any]] = {}

    def seed_profiles(self, rows: list[dict[str, Any]]) -> None:
        with self.lock:
            self.profiles.update({row["psychologist_id"]: dict(row) for row in rows})

    def list_profiles(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(row) for row in self.profiles.values()]

    def get_state(self, conversation_id: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.states.get(conversation_id, {}))

    def save_state(self, conversation_id: str, state: dict[str, Any]) -> None:
        with self.lock:
            self.states[conversation_id] = {**state, "updated_at": utcnow()}

    def save_request(self, request_id: str, result: dict[str, Any]) -> None:
        with self.lock:
            self.requests[request_id] = dict(result)

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.requests.get(request_id)
            return dict(row) if row else None

    def book(self, appointment: dict[str, Any]) -> bool:
        with self.lock:
            duplicate = any(
                row["psychologist_id"] == appointment["psychologist_id"]
                and row["start_time_utc"] == appointment["start_time_utc"]
                and row.get("status") == "booked"
                for row in self.appointments.values()
            )
            if duplicate:
                return False
            self.appointments[appointment["appointment_id"]] = dict(appointment)
            return True

    def get_appointment(self, appointment_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.appointments.get(appointment_id)
            return dict(row) if row else None

    def claim_due_reminders(self, now: datetime, limit: int = 20) -> list[dict[str, Any]]:
        claimed: list[dict[str, Any]] = []
        with self.lock:
            for row in self.appointments.values():
                if len(claimed) >= limit:
                    break
                if row.get("reminder_status") != "scheduled":
                    continue
                if row["reminder_at_utc"] <= now:
                    row["reminder_status"] = "processing"
                    row["reminder_attempts"] = int(row.get("reminder_attempts", 0)) + 1
                    claimed.append(dict(row))
        return claimed

    def finish_reminder(self, appointment_id: str, delivered: bool, error: str | None = None) -> None:
        with self.lock:
            row = self.appointments[appointment_id]
            attempts = int(row.get("reminder_attempts", 0))
            row["reminder_status"] = "delivered" if delivered else (
                "scheduled" if attempts < 3 else "failed"
            )
            row["reminder_error"] = error


class MongoAppointmentStore:
    def __init__(self, uri: str, database: str) -> None:
        from pymongo import ASCENDING, MongoClient

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)
        self.client.admin.command("ping")
        db = self.client[database]
        self.profiles = db["psychologists"]
        self.states = db["conversations"]
        self.appointments = db["appointments"]
        self.requests = db["requests"]
        self.appointments.create_index(
            [("psychologist_id", ASCENDING), ("start_time_utc", ASCENDING)], unique=True,
        )
        self.appointments.create_index([("reminder_status", ASCENDING), ("reminder_at_utc", ASCENDING)])

    def seed_profiles(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.profiles.update_one(
                {"psychologist_id": row["psychologist_id"]}, {"$set": row}, upsert=True,
            )

    def list_profiles(self) -> list[dict[str, Any]]:
        return list(self.profiles.find({}, {"_id": 0}))

    def get_state(self, conversation_id: str) -> dict[str, Any]:
        return self.states.find_one({"_id": conversation_id}, {"_id": 0}) or {}

    def save_state(self, conversation_id: str, state: dict[str, Any]) -> None:
        self.states.update_one(
            {"_id": conversation_id}, {"$set": {**state, "updated_at": utcnow()}}, upsert=True,
        )

    def save_request(self, request_id: str, result: dict[str, Any]) -> None:
        self.requests.update_one({"_id": request_id}, {"$set": result}, upsert=True)

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        return self.requests.find_one({"_id": request_id}, {"_id": 0})

    def book(self, appointment: dict[str, Any]) -> bool:
        from pymongo.errors import DuplicateKeyError
        try:
            self.appointments.insert_one(dict(appointment))
            return True
        except DuplicateKeyError:
            return False

    def get_appointment(self, appointment_id: str) -> dict[str, Any] | None:
        return self.appointments.find_one({"appointment_id": appointment_id}, {"_id": 0})

    def claim_due_reminders(self, now: datetime, limit: int = 20) -> list[dict[str, Any]]:
        from pymongo import ReturnDocument
        due = list(self.appointments.find(
            {"reminder_status": "scheduled", "reminder_at_utc": {"$lte": now}},
            {"appointment_id": 1},
        ).limit(limit))
        claimed = []
        for candidate in due:
            row = self.appointments.find_one_and_update(
                {"appointment_id": candidate["appointment_id"], "reminder_status": "scheduled"},
                {"$set": {"reminder_status": "processing"}, "$inc": {"reminder_attempts": 1}},
                projection={"_id": 0}, return_document=ReturnDocument.AFTER,
            )
            if row:
                claimed.append(row)
        return claimed

    def finish_reminder(self, appointment_id: str, delivered: bool, error: str | None = None) -> None:
        row = self.get_appointment(appointment_id) or {}
        attempts = int(row.get("reminder_attempts", 0))
        status = "delivered" if delivered else ("scheduled" if attempts < 3 else "failed")
        self.appointments.update_one(
            {"appointment_id": appointment_id},
            {"$set": {"reminder_status": status, "reminder_error": error}},
        )


def new_appointment_id() -> str:
    return f"APT-{uuid.uuid4().hex[:8].upper()}"
