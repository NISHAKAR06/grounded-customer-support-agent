"""Unit tests for Phase 2: Brand Selection & Scoping."""

import json
from pathlib import Path

from app.core.config import get_settings
from app.services.agent.agent_orchestrator import AgentOrchestrator


def test_target_brand_configuration():
    """Verify default brand configuration is set to AppleSupport."""
    settings = get_settings()
    assert settings.TARGET_BRAND == "AppleSupport"
    assert settings.TARGET_BRAND_NAME == "Apple Support"
    assert settings.TARGET_BRAND_HANDLE == "@AppleSupport"


def test_applesupport_profile_artifact():
    """Verify the empirical AppleSupport profile artifact has exact metrics."""
    profile_path = Path("experiments/brand_profile_applesupport.json")
    assert profile_path.exists(), "experiments/brand_profile_applesupport.json should exist"

    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["brand"] == "AppleSupport"
    assert data["outbound_replies"] == 106860
    assert data["inbound_mentions"] == 97895
    assert data["total_brand_interactions"] == 204755
    assert data["unique_customers"] == 58578


def test_brand_selection_documentation():
    """Verify that BRAND_SELECTION.md thoroughly documents the 6 criteria and taxonomy."""
    doc_path = Path("docs/BRAND_SELECTION.md")
    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    assert "@AppleSupport" in content
    assert "106,860" in content
    assert "Interaction Volume" in content
    assert "SOFTWARE_OS_UPDATE" in content
    assert "HARDWARE_BATTERY_POWER" in content
    assert "ACCOUNT_APPLE_ID_ICLOUD" in content
    assert "APP_CRASH_PERFORMANCE" in content
    assert "BILLING_SUBSCRIPTION_PURCHASE" in content
    assert "NETWORK_CONNECTIVITY_BLUETOOTH" in content
    assert "GENERAL_INQUIRY_FEEDBACK" in content


def test_orchestrator_defaults_to_target_brand():
    """Verify orchestrator applies default brand when none is supplied in request."""
    orchestrator = AgentOrchestrator()

    result = orchestrator.run(
        customer_message="My iPhone battery drains within two hours after updating to iOS 11."
    )
    assert result.brand == "AppleSupport"
