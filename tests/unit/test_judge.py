"""Unit tests for Phase 9 LLM-as-a-Judge and Cohen's Kappa agreement metrics."""

import pytest

from app.models.domain_models import HistoricalCase, ResolutionStatus
from app.services.evaluation.judge_service import LLMJudgeService


@pytest.fixture
def sample_evidence():
    return [
        HistoricalCase(
            case_id="case_apple_201",
            similarity=0.85,
            customer_text="How do I reset my Apple ID password?",
            brand_response="You can reset your Apple ID password at iforgot.apple.com. Let us know if you need help.",
            resolution_status=ResolutionStatus.RESOLVED,
        )
    ]


def test_judge_rubric_evaluation_structure(sample_evidence):
    """Verify judge evaluates response and returns all 4 rubric criteria plus overall score."""
    judge = LLMJudgeService()
    result = judge.evaluate_response(
        customer_message="Forgot my Apple ID password, need reset.",
        draft_reply="Please visit iforgot.apple.com to reset your credentials. Send us a DM if you need more help.",
        evidence=sample_evidence,
        routing_decision="AUTO_HANDLE",
        expected_routing="AUTO_HANDLE",
    )

    assert "groundedness" in result
    assert "answer_relevance" in result
    assert "brand_tone" in result
    assert "safety_compliance" in result
    assert "overall_score" in result
    assert "routing_agreement" in result

    assert 1.0 <= result["groundedness"] <= 5.0
    assert 1.0 <= result["answer_relevance"] <= 5.0
    assert 1.0 <= result["brand_tone"] <= 5.0
    assert 1.0 <= result["safety_compliance"] <= 5.0
    assert 1.0 <= result["overall_score"] <= 5.0
    assert result["routing_agreement"] is True


def test_judge_penalizes_ungrounded_pricing(sample_evidence):
    """Verify judge penalizes groundedness score when ungrounded pricing ($199) is present."""
    judge = LLMJudgeService()
    result_clean = judge.evaluate_response(
        customer_message="How do I reset my password?",
        draft_reply="Go to iforgot.apple.com to reset your password.",
        evidence=sample_evidence,
        routing_decision="AUTO_HANDLE",
    )

    result_with_price = judge.evaluate_response(
        customer_message="How do I reset my password?",
        draft_reply="Go to iforgot.apple.com to reset your password, or pay $199 for expedited unlock.",
        evidence=sample_evidence,
        routing_decision="AUTO_HANDLE",
    )

    assert result_with_price["groundedness"] < result_clean["groundedness"]


def test_judge_penalizes_unsafe_hazard():
    """Verify judge sets safety compliance to 1 when hazardous query lacks safety actions."""
    judge = LLMJudgeService()
    result = judge.evaluate_response(
        customer_message="My iPhone battery is swollen and hot!",
        draft_reply="Try restarting your phone and installing iOS 11 update.",
        evidence=[],
        routing_decision="AUTO_HANDLE",
    )

    assert result["safety_compliance"] == 1


def test_cohens_kappa_perfect_agreement():
    """Verify Cohen's Kappa is exactly 1.0 when human and system agree 100%."""
    labels = ["AUTO_HANDLE", "HUMAN_ESCALATION", "AUTO_HANDLE", "HUMAN_ESCALATION"]
    metrics = LLMJudgeService.calculate_cohens_kappa(labels, labels)

    assert metrics["observed_agreement"] == 1.0
    assert metrics["cohens_kappa"] == 1.0


def test_cohens_kappa_known_distribution():
    """Verify Cohen's Kappa produces valid range [-1, 1] on mixed distributions."""
    human = [
        "AUTO_HANDLE",
        "AUTO_HANDLE",
        "HUMAN_ESCALATION",
        "HUMAN_ESCALATION",
        "AUTO_HANDLE",
    ]
    system = [
        "AUTO_HANDLE",
        "HUMAN_ESCALATION",
        "HUMAN_ESCALATION",
        "HUMAN_ESCALATION",
        "AUTO_HANDLE",
    ]
    metrics = LLMJudgeService.calculate_cohens_kappa(human, system)

    assert metrics["observed_agreement"] == 0.80
    assert 0.0 < metrics["cohens_kappa"] <= 1.0
    assert "expected_agreement" in metrics
