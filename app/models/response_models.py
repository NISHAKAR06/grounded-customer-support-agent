"""API response models and schemas."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.models.domain_models import (
    EscalationDecision,
    HistoricalCase,
    IntentPrediction,
    ValidationResult,
)


class RetrievalResult(BaseModel):
    top_k: int
    evidence: List[HistoricalCase]


class GenerationResult(BaseModel):
    draft_reply: str
    provider: str
    grounded_evidence_ids: List[str]


class LatencyBreakdown(BaseModel):
    intent_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0


class AgentRunResult(BaseModel):
    run_id: str
    timestamp: str
    brand: str = "AppleSupport"
    customer_message: str
    intent: IntentPrediction
    retrieval: RetrievalResult
    generation: GenerationResult
    validation: ValidationResult
    routing: EscalationDecision
    latency_ms: LatencyBreakdown
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    llm_provider: str
    models: Dict[str, bool]
    timestamp: str


class BenchmarkMetricsResponse(BaseModel):
    golden_set_size: int
    metrics: Dict[str, Any]
