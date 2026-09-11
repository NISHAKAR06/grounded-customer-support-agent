"""Unit tests for Phase 8 Grounded Generation and Multi-Barrier Response Validation."""

import pytest

from app.models.domain_models import HistoricalCase, ResolutionStatus
from app.services.generation.mock_provider import MockProvider
from app.services.generation.prompt_builder import PromptBuilder
from app.services.validation.response_validator import ResponseValidator


@pytest.fixture
def sample_evidence():
    return [
        HistoricalCase(
            case_id="case_101",
            similarity=0.88,
            customer_text="My iPhone screen is frozen after updating iOS 11. How do I restart?",
            brand_response="Thanks for reaching out. Try a force restart by holding the power and volume down buttons. Let us know if you need more help in DM.",
            resolution_status=ResolutionStatus.RESOLVED,
            metadata={"intent_code": "OPERATING_SYSTEM_UPDATES"},
        ),
        HistoricalCase(
            case_id="case_102",
            similarity=0.79,
            customer_text="iPhone won't update to the new version.",
            brand_response="Please check your Wi-Fi connection and ensure at least 5GB of free storage in Settings > General > iPhone Storage.",
            resolution_status=ResolutionStatus.RESOLVED,
            metadata={"intent_code": "OPERATING_SYSTEM_UPDATES"},
        ),
    ]


def test_prompt_builder_includes_historical_evidence(sample_evidence):
    """Verify prompt builder formats evidence, brand tone, and grounding constraints."""
    prompt = PromptBuilder.build_grounded_prompt(
        customer_message="My iPhone is stuck on Apple logo.",
        intent_name="OS & iOS Updates",
        evidence=sample_evidence,
        brand="AppleSupport",
    )

    assert "@AppleSupport" in prompt
    assert "OS & iOS Updates" in prompt
    assert "Historical Case #1" in prompt
    assert "case_101" in prompt
    assert "force restart" in prompt
    assert "GROUNDING RULES (STRICT)" in prompt


def test_response_validator_passes_grounded_response(sample_evidence):
    """Verify legitimate grounded AppleSupport response passes all checks."""
    validator = ResponseValidator()
    reply = (
        "Thanks for reaching out. We can help with your update issue. Please try a force restart, "
        "and ensure you have adequate storage. If the issue persists, send us a DM or check "
        "https://support.apple.com/kb/HT201412."
    )
    result = validator.validate(
        reply=reply,
        evidence=sample_evidence,
        customer_message="My iPhone is frozen after update.",
    )

    assert result.all_passed is True
    assert result.checks["non_empty_check"] is True
    assert result.checks["grounding_evidence_present"] is True
    assert result.checks["unsupported_claim_check"] is True
    assert result.checks["url_whitelist_check"] is True
    assert result.checks["pii_security_check"] is True
    assert result.checks["hazardous_safety_check"] is True
    assert result.grounding_score > 0.0
    assert len(result.warnings) == 0


def test_response_validator_blocks_hallucinated_pricing(sample_evidence):
    """Verify validator intercepts ungrounded numeric pricing claims not in evidence."""
    validator = ResponseValidator()
    reply = "We can replace your battery for $99 at any Apple Store."
    result = validator.validate(
        reply=reply,
        evidence=sample_evidence,
        customer_message="How much for a battery?",
    )

    assert result.all_passed is False
    assert result.checks["unsupported_claim_check"] is False
    assert any("pricing" in w.lower() for w in result.warnings)


def test_response_validator_blocks_unauthorized_external_urls(sample_evidence):
    """Verify validator blocks third-party or phishing links outside official whitelist."""
    validator = ResponseValidator()
    reply = (
        "Please visit https://apple-repair-discount-center.com/claim to book service."
    )
    result = validator.validate(
        reply=reply,
        evidence=sample_evidence,
        customer_message="Need repair.",
    )

    assert result.all_passed is False
    assert result.checks["url_whitelist_check"] is False
    assert any("unauthorized" in w.lower() for w in result.warnings)


def test_response_validator_allows_official_apple_urls(sample_evidence):
    """Verify official Apple support domains are whitelisted."""
    validator = ResponseValidator()
    official_urls = [
        "https://support.apple.com/iphone",
        "https://appleid.apple.com",
        "https://locate.apple.com/find-service",
        "https://getsupport.apple.com",
    ]
    for url in official_urls:
        reply = f"You can review official instructions at {url}. Let us know in DM if you need more help."
        result = validator.validate(reply=reply, evidence=sample_evidence)
        assert (
            result.checks["url_whitelist_check"] is True
        ), f"Failed on valid URL: {url}"


def test_response_validator_blocks_public_pii_solicitation(sample_evidence):
    """Verify validator blocks prompts asking users to tweet passwords publicly."""
    validator = ResponseValidator()
    reply = (
        "Please reply with your password and Apple ID so we can verify your account."
    )
    result = validator.validate(
        reply=reply,
        evidence=sample_evidence,
        customer_message="Locked out.",
    )

    assert result.all_passed is False
    assert result.checks["pii_security_check"] is False
    assert any(
        "credentials" in w.lower() or "pii" in w.lower() for w in result.warnings
    )


def test_response_validator_enforces_hazardous_battery_caution(sample_evidence):
    """Verify validator intercepts responses to physical hazards that omit safety instructions."""
    validator = ResponseValidator()
    # Inquiry has swollen battery, but reply only suggests a generic software update
    reply = "Make sure to install the latest software update to fix performance."
    result = validator.validate(
        reply=reply,
        evidence=sample_evidence,
        customer_message="My iPhone battery is swollen and bulging out of the screen.",
    )

    assert result.all_passed is False
    assert result.checks["hazardous_safety_check"] is False
    assert any("hazard" in w.lower() or "safety" in w.lower() for w in result.warnings)


def test_response_validator_passes_hazardous_safety_with_caution(sample_evidence):
    """Verify validator passes responses to hazards when safety actions are included."""
    validator = ResponseValidator()
    reply = (
        "Please immediately stop using and stop charging your device. Disconnect it from power "
        "and bring it to an authorized service provider or Genius Bar for inspection."
    )
    result = validator.validate(
        reply=reply,
        evidence=sample_evidence,
        customer_message="My battery is swollen and hot.",
    )

    assert result.checks["hazardous_safety_check"] is True
    assert result.checks["non_empty_check"] is True


def test_mock_provider_extracts_evidence_resolution(sample_evidence):
    """Verify deterministic mock provider extracts top historical resolution from prompt."""
    provider = MockProvider()
    prompt = PromptBuilder.build_grounded_prompt(
        customer_message="How do I restart?",
        intent_name="OS & iOS Updates",
        evidence=sample_evidence,
        brand="AppleSupport",
    )
    reply = provider.generate(prompt)

    assert "force restart" in reply
    assert len(reply) > 20
