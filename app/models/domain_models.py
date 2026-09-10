"""Domain entities and business models."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    ESCALATED = "ESCALATED"


class RoutingDecision(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"


class MessageRole(str, Enum):
    CUSTOMER = "CUSTOMER"
    BRAND = "BRAND"
    AGENT = "AGENT"


class ConversationMessage(BaseModel):
    message_id: str
    role: MessageRole
    text: str
    timestamp: Optional[str] = None
    author_id: Optional[str] = None


class HistoricalCase(BaseModel):
    case_id: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    customer_text: str
    brand_response: str
    resolution_status: ResolutionStatus = ResolutionStatus.RESOLVED
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntentPrediction(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    signals: List[str] = Field(default_factory=list)
    alternatives: List[Dict[str, float]] = Field(default_factory=list)


class ValidationResult(BaseModel):
    all_passed: bool
    checks: Dict[str, bool] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class EscalationDecision(BaseModel):
    decision: RoutingDecision
    confidence: float
    reasons: List[str] = Field(default_factory=list)
