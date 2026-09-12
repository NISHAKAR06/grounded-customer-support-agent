"""Conversations and Inbox API routes."""

from typing import Any, Dict, List

from fastapi import APIRouter, Query

from app.repositories.conversation_repository import ConversationRepository

router = APIRouter(prefix="/conversations", tags=["Conversations"])
_repo = ConversationRepository()


@router.get("", response_model=List[Dict[str, Any]])
def list_conversations(
    status: str = Query("all", description="Status filter: all, auto_ready, needs_human, resolved")
):
    """List reconstructed support conversations."""
    return _repo.list_conversations(status_filter=status)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str):
    """Retrieve single conversation record by ID."""
    conv = _repo.get_by_id(conversation_id)
    if not conv:
        return {"error": "Conversation not found"}
    return conv
