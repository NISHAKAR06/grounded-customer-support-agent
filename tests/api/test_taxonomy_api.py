"""API tests for the Intent Taxonomy endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_get_taxonomy_endpoint():
    """Verify GET /api/v1/taxonomy returns the formal 7-class taxonomy schema."""
    response = client.get("/api/v1/taxonomy")
    assert response.status_code == 200
    data = response.json()

    assert "taxonomy" in data
    tax = data["taxonomy"]
    assert tax["brand"] == "AppleSupport"
    assert tax["version"] == "1.0.0"
    assert tax["num_classes"] == 7
    assert len(tax["classes"]) == 7

    class_codes = [c["code"] for c in tax["classes"]]
    assert "OPERATING_SYSTEM_UPDATES" in class_codes
    assert "BATTERY_POWER_HARDWARE" in class_codes
    assert "ACCOUNT_APPLE_ID" in class_codes
    assert "CONNECTIVITY_NETWORKING" in class_codes
    assert "AUDIO_ACCESSORIES" in class_codes
    assert "SUBSCRIPTIONS_BILLING" in class_codes
    assert "GENERAL_INQUIRY" in class_codes

    assert "disambiguation_rules" in tax
    assert len(tax["disambiguation_rules"]) >= 4


def test_classify_text_endpoint_canonical_os():
    """Verify POST /api/v1/taxonomy/classify identifies OS update inquiries."""
    payload = {"text": "My iPhone is frozen and stuck on 'Verifying update' for iOS 11.0.3."}
    response = client.post("/api/v1/taxonomy/classify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["code"] == "OPERATING_SYSTEM_UPDATES"
    assert data["name"] == "OS & iOS Updates"
    assert data["confidence"] >= 0.85
    assert len(data["signals"]) > 0


def test_classify_text_endpoint_hardware_safety_precedence():
    """Verify POST /api/v1/taxonomy/classify correctly applies safety disambiguation."""
    payload = {"text": "After the update the iPhone battery became burning hot and swollen."}
    response = client.post("/api/v1/taxonomy/classify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["code"] == "BATTERY_POWER_HARDWARE"
    assert data["name"] == "Battery & Hardware"
    assert data["confidence"] >= 0.85
