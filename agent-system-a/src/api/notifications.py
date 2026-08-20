"""Persistent appointment notifications and live SSE delivery."""

from __future__ import annotations

import asyncio
import hmac
import json
import queue
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class ReminderEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=100)
    conversation_id: str = Field(min_length=1, max_length=100)
    appointment_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=1000)
    appointment_time: str
    synthetic: bool = True


class MemoryNotificationStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.rows: dict[str, dict[str, Any]] = {}

    def save(self, payload: dict[str, Any]) -> bool:
        with self.lock:
            if payload["event_id"] in self.rows:
                return False
            self.rows[payload["event_id"]] = {
                **payload, "created_at": datetime.now(timezone.utc), "read": False,
            }
            return True

    def list(self, conversation_id: str, unread_only: bool = False) -> list[dict[str, Any]]:
        with self.lock:
            rows = [
                dict(row) for row in self.rows.values()
                if row["conversation_id"] == conversation_id
                and (not unread_only or not row["read"])
            ]
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    def mark_read(self, event_id: str) -> bool:
        with self.lock:
            if event_id not in self.rows:
                return False
            self.rows[event_id]["read"] = True
            return True


class MongoNotificationStore:
    def __init__(self, uri: str, database: str) -> None:
        from pymongo import ASCENDING, DESCENDING, MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)
        client.admin.command("ping")
        self.rows = client[database]["notifications"]
        self.rows.create_index("event_id", unique=True)
        self.rows.create_index([("conversation_id", ASCENDING), ("created_at", DESCENDING)])

    def save(self, payload: dict[str, Any]) -> bool:
        result = self.rows.update_one(
            {"event_id": payload["event_id"]},
            {"$setOnInsert": {**payload, "created_at": datetime.now(timezone.utc), "read": False}},
            upsert=True,
        )
        return result.upserted_id is not None

    def list(self, conversation_id: str, unread_only: bool = False) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"conversation_id": conversation_id}
        if unread_only:
            query["read"] = False
        return list(self.rows.find(query, {"_id": 0}).sort("created_at", -1).limit(100))

    def mark_read(self, event_id: str) -> bool:
        return self.rows.update_one({"event_id": event_id}, {"$set": {"read": True}}).matched_count > 0


class NotificationHub:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.subscribers: dict[str, list[queue.Queue]] = {}

    def subscribe(self, conversation_id: str) -> queue.Queue:
        channel: queue.Queue = queue.Queue(maxsize=20)
        with self.lock:
            self.subscribers.setdefault(conversation_id, []).append(channel)
        return channel

    def unsubscribe(self, conversation_id: str, channel: queue.Queue) -> None:
        with self.lock:
            channels = self.subscribers.get(conversation_id, [])
            if channel in channels:
                channels.remove(channel)

    def publish(self, conversation_id: str, payload: dict[str, Any]) -> None:
        with self.lock:
            channels = list(self.subscribers.get(conversation_id, []))
        for channel in channels:
            try:
                channel.put_nowait(payload)
            except queue.Full:
                pass


router = APIRouter(tags=["notifications"])


def notification_store_dependency():
    raise RuntimeError("notification store is not configured")


def notification_hub_dependency():
    raise RuntimeError("notification hub is not configured")


def internal_token_dependency():
    raise RuntimeError("internal token is not configured")

#Agent System B → Agent System A
@router.post("/internal/v1/appointment-reminders")
def receive_reminder(
    event: ReminderEvent,
    x_internal_service_token: str = Header(default=""),
    store=Depends(notification_store_dependency), hub=Depends(notification_hub_dependency),
    expected_token: str = Depends(internal_token_dependency),
) -> dict:
    if not expected_token or not hmac.compare_digest(x_internal_service_token, expected_token):
        raise HTTPException(401, "invalid internal service token")
    payload = event.model_dump(mode="json")
    created = store.save(payload)
    if created:
        hub.publish(event.conversation_id, payload)
    return {"accepted": True, "duplicate": not created}

#Frontend → Agent System A
#retrieve notifications for a conversation
@router.get("/v1/notifications")
def list_notifications(
    conversation_id: str = Query(min_length=1, max_length=100), unread_only: bool = False,
    store=Depends(notification_store_dependency),
) -> list[dict]:
    return store.list(conversation_id, unread_only)

#mark a notification as read
@router.patch("/v1/notifications/{event_id}/read")
def mark_notification_read(event_id: str, store=Depends(notification_store_dependency)) -> dict:
    if not store.mark_read(event_id):
        raise HTTPException(404, "notification not found")
    return {"event_id": event_id, "read": True}


#Frontend → Agent System A
#Keep a live connection open between Agent System A and the frontend.
@router.get("/v1/notifications/stream")
def notification_stream(
    request: Request, conversation_id: str = Query(min_length=1, max_length=100),
    hub=Depends(notification_hub_dependency),
) -> StreamingResponse:
    channel = hub.subscribe(conversation_id)

    async def events():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.to_thread(channel.get, True, 15)
                    yield f"event: appointment_reminder\ndata: {json.dumps(payload)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            hub.unsubscribe(conversation_id, channel)

    return StreamingResponse(events(), media_type="text/event-stream")
