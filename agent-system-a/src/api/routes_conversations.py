"""Conversation history management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query


router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


def conversation_store_dependency():
    raise RuntimeError("Conversation store is not configured")


@router.get("")
def list_conversations(
    limit: int = Query(50, ge=1, le=200), store=Depends(conversation_store_dependency),
) -> list[dict]:
    return store.list(limit)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, store=Depends(conversation_store_dependency)) -> dict:
    conversation = store.get(conversation_id)
    if conversation is None:
        raise HTTPException(404, "conversation not found")
    return conversation


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, store=Depends(conversation_store_dependency)) -> dict:
    if not store.delete(conversation_id):
        raise HTTPException(404, "conversation not found")
    return {"conversation_id": conversation_id, "deleted": True}
