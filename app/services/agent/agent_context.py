"""Agent context and working memory."""

import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentRunContext(BaseModel):
    """Encapsulates context, telemetry, and working state during an agent pipeline execution."""

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    customer_message: str
    conversation_id: Optional[str] = None
    brand: Optional[str] = None
    start_time: float = Field(default_factory=time.time)
    events: list = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def log_event(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        elapsed = round((time.time() - self.start_time) * 1000, 2)
        self.events.append(
            {
                "event": event_name,
                "elapsed_ms": elapsed,
                "payload": payload or {},
            }
        )
