"""Unit tests for core services and domain logic."""

from app.core.config import get_settings
from app.models.domain_models import RoutingDecision
from app.services.escalation.escalation_policy import EscalationPolicy
from app.services.generation.prompt_builder import PromptBuilder
from app.services.validation.response_validator import ResponseValidator
from tests.fixtures.sample_fixtures import (
    get_sample_historical_case,
    get_sample_intent_prediction,
    get_sample_validation_result,
)


def test_settings_load():
    """Verify application settings singleton loads defaults cleanly."""
    settings = get_settings()
    assert settings.APP_NAME == "Grounded Customer Support Agent"
    assert settings.APP_VERSION == "1.0.0"
    assert settings.ESCALATION_MIN_CONFIDENCE > 0.0


def test_escalation_policy_auto_handle():
    """Verify high-confidence, grounded query without triggers evaluates to AUTO_HANDLE."""
    policy = EscalationPolicy(min_confidence=0.80, min_retrieval_score=0.60)
    intent = get_sample_intent_prediction(high_confidence=True)
    evidence = [get_sample_historical_case()]
    validation = get_sample_validation_result(passed=True)

    decision = policy.evaluate(
        customer_message="I would like to update my shipping address.",
        intent=intent,
        evidence=evidence,
        validation=validation,
    )
    assert decision.decision == RoutingDecision.AUTO_HANDLE
    assert len(decision.reasons) > 0


def test_escalation_policy_human_on_low_confidence():
    """Verify low-confidence intent evaluates to HUMAN_ESCALATION."""
    policy = EscalationPolicy(min_confidence=0.80, min_retrieval_score=0.60)
    intent = get_sample_intent_prediction(high_confidence=False)  # 0.60 confidence
    evidence = [get_sample_historical_case()]
    validation = get_sample_validation_result(passed=True)

    decision = policy.evaluate(
        customer_message="Something is strange with my package.",
        intent=intent,
        evidence=evidence,
        validation=validation,
    )
    assert decision.decision == RoutingDecision.HUMAN_ESCALATION
    assert any("below threshold" in r for r in decision.reasons)


def test_escalation_policy_human_on_risk_phrase():
    """Verify customer explicit human request evaluates to HUMAN_ESCALATION regardless of confidence."""
    policy = EscalationPolicy(min_confidence=0.80, min_retrieval_score=0.60)
    intent = get_sample_intent_prediction(high_confidence=True)
    evidence = [get_sample_historical_case()]
    validation = get_sample_validation_result(passed=True)

    decision = policy.evaluate(
        customer_message="I demand to speak to a manager or lawyer immediately!",
        intent=intent,
        evidence=evidence,
        validation=validation,
    )
    assert decision.decision == RoutingDecision.HUMAN_ESCALATION
    assert any("explicitly requested a human" in r for r in decision.reasons)


def test_response_validator_catches_unsupported_claim():
    """Verify ResponseValidator catches ungrounded financial promises."""
    validator = ResponseValidator()
    evidence = [get_sample_historical_case()]

    bad_reply = "Don't worry, a refund has been processed for your order."
    result = validator.validate(bad_reply, evidence)
    assert result.all_passed is False
    assert result.checks["unsupported_claim_check"] is False


def test_response_validator_passes_valid_reply():
    """Verify ResponseValidator approves compliant replies."""
    validator = ResponseValidator()
    evidence = [get_sample_historical_case()]

    good_reply = "We can certainly help update your address. Please provide your order ID."
    result = validator.validate(good_reply, evidence)
    assert result.all_passed is True


def test_prompt_builder_contains_evidence():
    """Verify PromptBuilder properly injects customer message and historical cases."""
    evidence = [get_sample_historical_case()]
    prompt = PromptBuilder.build_grounded_prompt(
        customer_message="Where is my order?",
        intent_name="Shipping Status",
        evidence=evidence,
    )
    assert "Where is my order?" in prompt
    assert "FIXTURE_CASE_101" in prompt
    assert "GROUNDING RULES (STRICT)" in prompt
