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


def test_simulate_page_preloads_message_from_query_param():
    """Verify /simulate?msg=... pre-populates the customer message textarea."""
    test_msg = "My iPhone 7 battery is draining very fast after iOS 11.1"
    response = client.get(f"/simulate?msg={test_msg}")
    assert response.status_code == 200
    assert test_msg in response.text


def test_inbox_page_renders():
    """Verify /inbox renders HTML table."""
    response = client.get("/inbox")
    assert response.status_code == 200
    assert "Support Inbox" in response.text
    assert "Showing" in response.text
    assert "records" in response.text


def test_inbox_page_with_filters_sorting_and_pagination():
    """Verify /inbox handles filter, decision, turns, sort, and pagination query params."""
    response = client.get(
        "/inbox?filter=needs_human&decision=human_escalation&turns=deep&sort=turns_desc&page=1&page_size=10"
    )
    assert response.status_code == 200
    assert "Needs Human" in response.text
    assert "Support Inbox" in response.text
    assert "Per page:" in response.text


def test_inbox_page_with_intent_filter():
    """Verify /inbox filters tickets by domain intent."""
    response = client.get("/inbox?intent=OPERATING_SYSTEM_UPDATES")
    assert response.status_code == 200
    assert (
        "OS &amp; iOS Updates" in response.text or "OS & iOS Updates" in response.text
    )
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
