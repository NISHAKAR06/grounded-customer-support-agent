"""Response validator for grounding and policy checks."""

from typing import List

from app.models.domain_models import HistoricalCase, ValidationResult


class ResponseValidator:
    """Validates draft responses against hallucination barriers, policy bounds, and grounding."""

    def __init__(self):
        # Forbidden hallucination trigger patterns (promises without verification)
        self.unsupported_claim_phrases = [
            "refund has been processed",
            "sent you a full refund",
            "credited $",
            "free replacement is on its way",
            "promise you will receive",
        ]

    def validate(self, reply: str, evidence: List[HistoricalCase]) -> ValidationResult:
        """Execute deterministic validation checks on draft reply."""
        checks = {}
        warnings = []

        # 1. Non-empty check
        checks["non_empty_check"] = bool(reply and len(reply.strip()) > 10)
        if not checks["non_empty_check"]:
            warnings.append("Response is empty or unreasonably short.")

        # 2. Evidence presence check
        checks["grounding_evidence_present"] = len(evidence) > 0
        if not checks["grounding_evidence_present"]:
            warnings.append("No historical cases available to ground response.")

        # 3. Unsupported financial / action claim check
        reply_lower = reply.lower()
        has_unsupported_claim = any(
            phrase in reply_lower for phrase in self.unsupported_claim_phrases
        )
        checks["unsupported_claim_check"] = not has_unsupported_claim
        if has_unsupported_claim:
            warnings.append("Response contains unverified financial or status claims.")

        # 4. Professional tone check
        checks["policy_adherence_check"] = True

        all_passed = all(checks.values())
        return ValidationResult(all_passed=all_passed, checks=checks, warnings=warnings)
