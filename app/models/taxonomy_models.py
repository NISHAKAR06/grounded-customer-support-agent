"""Domain schemas and models for the 7-class Intent Taxonomy."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.domain_models import RoutingDecision, SupportIntent


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IntentDefinition(BaseModel):
    """Formal definition of a single intent class."""

    code: SupportIntent
    label: str
    description: str
    default_routing: RoutingDecision
    risk_level: RiskLevel
    priority_rank: int = Field(
        ...,
        description="Priority rank used in tie-breaking disambiguation (1 = highest priority)",
    )
    key_signals: List[str] = Field(default_factory=list)
    negative_signals: List[str] = Field(default_factory=list)
    canonical_examples: List[str] = Field(default_factory=list)
    empirical_frequency_pct: Optional[float] = None


class DisambiguationRule(BaseModel):
    """Rule defining precedence when a message spans multiple intent signals."""

    rule_id: str
    title: str
    higher_priority_intent: SupportIntent
    lower_priority_intent: SupportIntent
    condition_description: str
    example: str


class IntentTaxonomySchema(BaseModel):
    """Complete machine-readable intent taxonomy specification."""

    brand: str = "AppleSupport"
    version: str = "1.0.0"
    num_classes: int = 7
    classes: List[IntentDefinition]
    disambiguation_rules: List[DisambiguationRule] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
