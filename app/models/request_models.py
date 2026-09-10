"""API request models and schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    customer_message: str = Field(
        ..., min_length=2, max_length=2000, description="Incoming customer text"
    )
    conversation_id: Optional[str] = Field(
        None, description="Optional thread/conversation identifier"
    )
    brand: Optional[str] = Field(None, description="Target brand identifier")


class ConversationFilterRequest(BaseModel):
    status_filter: Optional[str] = Field(
        "all", description="Filter by status: all, auto_ready, needs_human, resolved"
    )
    limit: int = Field(50, ge=1, le=100)
    offset: int = Field(0, ge=0)
