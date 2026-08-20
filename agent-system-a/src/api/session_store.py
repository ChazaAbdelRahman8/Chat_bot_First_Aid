"""Conversation history stores for the chat API."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryConversationStore:
    """Thread-safe test/development fallback with the Mongo store interface."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conversations: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}

    def create(self) -> str:
        conversation_id = str(uuid.uuid4())
        now = _now()
        with self._lock:
            self._conversations[conversation_id] = {
                "conversation_id": conversation_id, "title": "New conversation",
                "created_at": now, "updated_at": now, "message_count": 0,
            }
            self._messages[conversation_id] = []
        return conversation_id

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._lock:
            rows = self._messages.get(conversation_id, [])[-limit:]
            return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_turn(
        self, conversation_id: str, user_content: str, assistant_content: str,
        *, metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = {
                    "conversation_id": conversation_id, "title": user_content[:80],
                    "created_at": now, "updated_at": now, "message_count": 0,
                }
                self._messages[conversation_id] = []
            conversation = self._conversations[conversation_id]
            if conversation["message_count"] == 0:
                conversation["title"] = user_content[:80]
            self._messages[conversation_id].extend([
                {"role": "user", "content": user_content, "created_at": now, "metadata": {}},
                {"role": "assistant", "content": assistant_content, "created_at": now, "metadata": metadata or {}},
            ])
            conversation["updated_at"] = now
            conversation["message_count"] += 2

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                return None
            return {**conversation, "messages": list(self._messages[conversation_id])}

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(
                self._conversations.values(), key=lambda row: row["updated_at"], reverse=True,
            )[:limit]
            return [dict(row) for row in rows]

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            existed = conversation_id in self._conversations
            self._conversations.pop(conversation_id, None)
            self._messages.pop(conversation_id, None)
            return existed


class MongoConversationStore:
    """Durable conversation metadata and ordered messages in MongoDB."""

    def __init__(self, uri: str, database: str = "first_aid") -> None:
        from pymongo import ASCENDING, DESCENDING, MongoClient

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)
        self.client.admin.command("ping")
        db = self.client[database]
        self.conversations = db["conversations"]
        self.messages = db["messages"]
        self.conversations.create_index([("updated_at", DESCENDING)])
        self.messages.create_index([("conversation_id", ASCENDING), ("created_at", ASCENDING)])

    def create(self) -> str:
        conversation_id = str(uuid.uuid4())
        now = _now()
        self.conversations.insert_one({
            "_id": conversation_id, "title": "New conversation",
            "created_at": now, "updated_at": now, "message_count": 0,
        })
        return conversation_id

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
        rows = list(self.messages.find(
            {"conversation_id": conversation_id}, {"_id": 0, "role": 1, "content": 1},
        ).sort("created_at", -1).limit(limit))
        rows.reverse()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_turn(
        self, conversation_id: str, user_content: str, assistant_content: str,
        *, metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        existing = self.conversations.find_one({"_id": conversation_id}, {"message_count": 1})
        self.conversations.update_one(
            {"_id": conversation_id},
            {
                "$set": {"updated_at": now},
                "$setOnInsert": {"created_at": now, "title": user_content[:80]},
                "$inc": {"message_count": 2},
            },
            upsert=True,
        )
        if not existing or existing.get("message_count", 0) == 0:
            self.conversations.update_one({"_id": conversation_id}, {"$set": {"title": user_content[:80]}})
        self.messages.insert_many([
            {"conversation_id": conversation_id, "role": "user", "content": user_content, "created_at": now, "metadata": {}},
            {"conversation_id": conversation_id, "role": "assistant", "content": assistant_content, "created_at": now, "metadata": metadata or {}},
        ])

    @staticmethod
    def _conversation(row: dict[str, Any]) -> dict[str, Any]:
        return {"conversation_id": str(row.pop("_id")), **row}

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        row = self.conversations.find_one({"_id": conversation_id})
        if row is None:
            return None
        messages = list(self.messages.find(
            {"conversation_id": conversation_id}, {"_id": 0},
        ).sort("created_at", 1))
        return {**self._conversation(row), "messages": messages}

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            self._conversation(row)
            for row in self.conversations.find().sort("updated_at", -1).limit(limit)
        ]

    def delete(self, conversation_id: str) -> bool:
        deleted = self.conversations.delete_one({"_id": conversation_id}).deleted_count > 0
        self.messages.delete_many({"conversation_id": conversation_id})
        return deleted


def build_conversation_store(uri: str | None, database: str = "first_aid"):
    """Use MongoDB when configured; otherwise use a process-local test fallback."""
    if uri:
        return MongoConversationStore(uri, database)
    return MemoryConversationStore()
