"""Test fixtures for deterministic testing."""

from app.models.domain_models import (
    HistoricalCase,
    IntentPrediction,
    ResolutionStatus,
    ValidationResult,
)


def get_sample_historical_case() -> HistoricalCase:
    return HistoricalCase(
        case_id="FIXTURE_CASE_101",
        similarity=0.92,
        customer_text="I need to change my shipping address.",
        brand_response="We can update your address before dispatch.",
        resolution_status=ResolutionStatus.RESOLVED,
        metadata={"brand": "TestBrand"},
    )


def get_sample_intent_prediction(high_confidence: bool = True) -> IntentPrediction:
    return IntentPrediction(
        name="Shipping / Address Change",
        confidence=0.95 if high_confidence else 0.60,
        signals=["shipping address", "dispatch"],
        alternatives=[{"Other": 0.05}],
    )


def get_sample_validation_result(passed: bool = True) -> ValidationResult:
    return ValidationResult(
        all_passed=passed,
        checks={"grounding_check": passed, "claim_check": passed},
        warnings=[] if passed else ["Unsupported claim detected"],
    )
