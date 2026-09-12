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
    customer_handle: Optional[str] = Field(
        None, description="Customer handle or username (e.g. @alex)"
    )
    provider: Optional[str] = Field(
        None,
        description="Optional LLM provider: groq, ollama, openai, gemini, claude",
    )
