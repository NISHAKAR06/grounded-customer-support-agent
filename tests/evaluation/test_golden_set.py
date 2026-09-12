"""Unit and integration tests for the Golden Evaluation Set (Phase 5)."""

import json

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.repositories.golden_set_repository import GoldenSetRepository

client = TestClient(app)
settings = get_settings()

GOLDEN_PATH = settings.DATA_DIR / "golden" / "golden_set.jsonl"
SUMMARY_PATH = settings.DATA_DIR / "golden" / "golden_set_summary.json"

EXPECTED_INTENTS = {
    "OPERATING_SYSTEM_UPDATES",
    "BATTERY_POWER_HARDWARE",
    "ACCOUNT_APPLE_ID",
    "CONNECTIVITY_NETWORKING",
    "AUDIO_ACCESSORIES",
    "SUBSCRIPTIONS_BILLING",
    "GENERAL_INQUIRY",
}


def test_golden_set_file_exists():
    """Verify golden_set.jsonl and golden_set_summary.json exist on disk."""
    assert GOLDEN_PATH.exists(), f"Missing golden set file at {GOLDEN_PATH}"
    assert SUMMARY_PATH.exists(), f"Missing summary file at {SUMMARY_PATH}"


def test_golden_set_sample_count_in_specification_range():
    """Verify total sample count is strictly within the 150-250 range."""
    count = 0
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    assert 150 <= count <= 250, f"Expected sample count between 150 and 250, but got {count}"
    assert count == 200, f"Expected exactly 200 curated samples, got {count}"


def test_golden_set_all_intents_represented():
    """Verify every one of the 7 MECE intent classes is represented in the golden set."""
    found_intents = set()
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            found_intents.add(record.get("gold_intent_code"))

    assert (
        found_intents == EXPECTED_INTENTS
    ), f"Mismatch in intent classes: {EXPECTED_INTENTS - found_intents}"


def test_golden_set_routing_decisions_valid():
    """Verify routing decisions are valid and escalations have justified reasons."""
    auto_count = 0
    escalate_count = 0

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            routing = record.get("gold_routing")
            assert routing in [
                "AUTO_HANDLE",
                "HUMAN_ESCALATION",
            ], f"Invalid routing: {routing}"

            if routing == "HUMAN_ESCALATION":
                escalate_count += 1
                reason = record.get("gold_escalation_reason")
                assert (
                    reason is not None and len(reason) > 5
                ), f"Sample {record.get('sample_id')} has HUMAN_ESCALATION but empty reason: {reason}"
            else:
                auto_count += 1

    assert auto_count > 0, "Golden set should have Auto-Handle candidates"
    assert escalate_count > 0, "Golden set should have Human Escalation candidates"


def test_golden_set_no_duplicate_sample_ids():
    """Verify all sample IDs are unique and sequentially formatted."""
    seen_ids = set()
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            sid = record.get("sample_id")
            assert sid not in seen_ids, f"Duplicate sample_id found: {sid}"
            seen_ids.add(sid)


def test_golden_set_record_schema_and_text_integrity():
    """Verify that all records have non-empty text, turns, resolutions, and notes."""
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assert len(r.get("customer_message", "").strip()) >= 10
            assert len(r.get("gold_resolution", "").strip()) >= 5
            assert len(r.get("annotator_notes", "").strip()) >= 20
            assert r.get("turn_count", 0) >= 1
            assert len(r.get("conversation_turns", [])) >= 1


def test_golden_set_summary_consistency():
    """Verify summary artifact matches the raw JSONL contents."""
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary.get("total_samples") == 200
    assert summary.get("compliance") == "PASS"
    assert sum(summary["intent_distribution"].values()) == 200
    assert (
        summary["routing_distribution"]["AUTO_HANDLE"]
        + summary["routing_distribution"]["HUMAN_ESCALATION"]
        == 200
    )


def test_golden_set_repository_methods():
    """Verify GoldenSetRepository functions correctly."""
    repo = GoldenSetRepository()
    assert repo.count() == 200
    all_samples = repo.get_all()
    assert len(all_samples) == 200

    first = repo.get_by_sample_id("gold_001")
    assert first is not None
    assert first["sample_id"] == "gold_001"

    # Filtering
    os_samples = repo.filter(intent_code="OPERATING_SYSTEM_UPDATES")
    assert len(os_samples) == 40

    auto_samples = repo.filter(routing="AUTO_HANDLE")
    assert len(auto_samples) == 42

    # Pagination
    page_1, total = repo.paginate(limit=10, offset=0)
    assert len(page_1) == 10
    assert total == 200


def test_golden_set_api_endpoints():
    """Verify GET /api/v1/evaluation/golden-set and /golden-set/summary endpoints."""
    res_summary = client.get("/api/v1/evaluation/golden-set/summary")
    assert res_summary.status_code == 200
    data_summary = res_summary.json()
    assert data_summary["total_samples"] == 200

    res_page = client.get("/api/v1/evaluation/golden-set?limit=25&offset=0")
    assert res_page.status_code == 200
    data_page = res_page.json()
    assert len(data_page["samples"]) == 25
    assert data_page["total"] == 200
