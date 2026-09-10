"""Tests for Agent execution API routes."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_agent_run_endpoint():
    """Verify POST /api/agent/run executes pipeline and returns complete schema."""
    payload = {
        "customer_message": "Can I update my shipping address for order #412?",
    }
    response = client.post("/api/agent/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "run_id" in data
    assert "intent" in data
    assert "retrieval" in data
    assert "generation" in data
    assert "validation" in data
    assert "routing" in data
    assert "latency_ms" in data
    assert data["routing"]["decision"] in ["AUTO_HANDLE", "HUMAN_ESCALATION"]


def test_agent_run_invalid_empty_input():
    """Verify validation error on empty message."""
    response = client.post("/api/agent/run", json={"customer_message": ""})
    assert response.status_code == 422
