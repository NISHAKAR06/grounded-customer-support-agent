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


class SupportIntent(str, Enum):
    OPERATING_SYSTEM_UPDATES = "OPERATING_SYSTEM_UPDATES"
    BATTERY_POWER_HARDWARE = "BATTERY_POWER_HARDWARE"
    ACCOUNT_APPLE_ID = "ACCOUNT_APPLE_ID"
    CONNECTIVITY_NETWORKING = "CONNECTIVITY_NETWORKING"
    AUDIO_ACCESSORIES = "AUDIO_ACCESSORIES"
    SUBSCRIPTIONS_BILLING = "SUBSCRIPTIONS_BILLING"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"


INTENT_LABELS: Dict[SupportIntent, str] = {
    SupportIntent.OPERATING_SYSTEM_UPDATES: "OS & iOS Updates",
    SupportIntent.BATTERY_POWER_HARDWARE: "Battery & Hardware",
    SupportIntent.ACCOUNT_APPLE_ID: "Apple ID & Account",
    SupportIntent.CONNECTIVITY_NETWORKING: "Connectivity & Wi-Fi",
    SupportIntent.AUDIO_ACCESSORIES: "Audio & Accessories",
    SupportIntent.SUBSCRIPTIONS_BILLING: "Billing & Subscriptions",
    SupportIntent.GENERAL_INQUIRY: "General Support",
}


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
    code: Optional[SupportIntent] = None
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
