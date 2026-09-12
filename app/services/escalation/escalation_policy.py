"""Deterministic escalation routing policy engine."""

from typing import List

from app.core.config import get_settings
from app.models.domain_models import (
    EscalationDecision,
    HistoricalCase,
    IntentPrediction,
    RoutingDecision,
    ValidationResult,
)


class EscalationPolicy:
    """Evaluates multi-factor signals to route between automated handling and human escalation."""

    def __init__(
        self,
        min_confidence: float = None,
        min_retrieval_score: float = None,
    ):
        settings = get_settings()
        self.min_confidence = min_confidence or settings.ESCALATION_MIN_CONFIDENCE
        self.min_retrieval_score = min_retrieval_score or settings.ESCALATION_MIN_RETRIEVAL_SCORE

        # High-risk trigger phrases that demand human attention
        self.human_escalation_triggers = [
            "speak to human",
            "talk to representative",
            "manager",
            "lawyer",
            "legal action",
            "sue you",
            "bank dispute",
            "unacceptable fraud",
            "stolen card",
        ]

    def evaluate(
        self,
        customer_message: str,
        intent: IntentPrediction,
        evidence: List[HistoricalCase],
        validation: ValidationResult,
    ) -> EscalationDecision:
        """Deterministically determine routing with human-auditable reasons."""
        reasons = []
        is_escalation = False

        # 1. Customer explicit human request / high-risk phrase
        msg_lower = customer_message.lower()
        if any(trigger in msg_lower for trigger in self.human_escalation_triggers):
            is_escalation = True
            reasons.append(
                "Customer explicitly requested a human agent or used legal/risk trigger language."
            )

        # 2. Intent classifier confidence check
        if intent.confidence < self.min_confidence:
            is_escalation = True
            reasons.append(
                f"Intent classification confidence ({intent.confidence:.2f}) below threshold ({self.min_confidence:.2f})."
            )

        # 3. Retrieval evidence sufficiency check
        if not evidence:
            is_escalation = True
            reasons.append("No historical resolved support cases found to ground response.")
        else:
            top_similarity = max(c.similarity for c in evidence)
            if top_similarity < self.min_retrieval_score:
                is_escalation = True
                reasons.append(
                    f"Top retrieval similarity ({top_similarity:.2f}) below threshold ({self.min_retrieval_score:.2f})."
                )

        # 4. Response validation check
        if not validation.all_passed:
            is_escalation = True
            reasons.append("Response validation failed (hallucination or unverified claim check).")

        if is_escalation:
            return EscalationDecision(
                decision=RoutingDecision.HUMAN_ESCALATION,
                confidence=intent.confidence,
                reasons=reasons,
            )

        # Approved for automation
        auto_reasons = [
            f"High intent confidence ({intent.confidence:.2f} >= {self.min_confidence:.2f})",
            "Sufficient historical evidence retrieved and verified",
            "No human escalation triggers or risk keywords detected",
            "All grounding and policy checks successfully passed",
        ]
        return EscalationDecision(
            decision=RoutingDecision.AUTO_HANDLE,
            confidence=intent.confidence,
            reasons=auto_reasons,
        )
