"""Tests for HTML UI routes."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_redirects_to_simulate():
    """Verify root URL redirects to /simulate."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/simulate"


def test_simulate_page_renders():
    """Verify /simulate renders HTML with correct title and composer."""
    response = client.get("/simulate")
    assert response.status_code == 200
    assert "Simulate Incoming Message" in response.text
    assert "Run AI Agent" in response.text


def test_inbox_page_renders():
    """Verify /inbox renders HTML table."""
    response = client.get("/inbox")
    assert response.status_code == 200
    assert "Support Inbox" in response.text


def test_evaluation_page_renders():
    """Verify /evaluation renders benchmark summary."""
    response = client.get("/evaluation")
    assert response.status_code == 200
    assert (
        "Evaluation &amp; Model Benchmarks" in response.text
        or "Evaluation & Model Benchmarks" in response.text
    )


def test_failures_page_renders():
    """Verify /failures renders failure modes and headline audit."""
    response = client.get("/failures")
    assert response.status_code == 200
    assert "Failure Analysis" in response.text
    assert "What is Misleading About My Headline Number?" in response.text


def test_decisions_page_renders():
    """Verify /decisions renders decision log."""
    response = client.get("/decisions")
    assert response.status_code == 200
    assert "Engineering Decision Log" in response.text


def test_methodology_page_renders():
    """Verify /methodology renders formal specifications."""
    response = client.get("/methodology")
    assert response.status_code == 200
    assert (
        "Methodology &amp; System Specifications" in response.text
        or "Methodology & System Specifications" in response.text
    )
