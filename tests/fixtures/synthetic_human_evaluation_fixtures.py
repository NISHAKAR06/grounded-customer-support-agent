"""SYNTHETIC TEST FIXTURES FOR UNIT TESTING ONLY.

CRITICAL NOTICE:
The fixtures in this module are purely synthetic toy examples constructed solely
to test unit validation routines, schema enforcement, and calculation math in pytest.
THEY DO NOT REPRESENT REAL HUMAN ANNOTATIONS OR BENCHMARK EVALUATION RESULTS.
"""

from typing import Any, Dict, List


def get_synthetic_blank_annotations(count: int = 5) -> List[Dict[str, Any]]:
    """Return synthetic unannotated records matching the template schema."""
    return [
        {
            "example_id": f"synthetic_test_{i:03d}",
            "conversation_id": f"conv_synth_{i}",
            "customer_message": f"Synthetic customer query test issue {i}",
            "conversation": f"[CUSTOMER]: Synthetic customer query test issue {i}",
            "turn_count": 2,
            "human_intent": None,
            "human_routing": None,
            "human_notes": "",
            "annotator_id": "",
            "annotated": False,
        }
        for i in range(1, count + 1)
    ]


def get_synthetic_valid_annotations(count: int = 5) -> List[Dict[str, Any]]:
    """Return synthetic complete annotations with valid intents and routings for schema tests."""
    intents = [
        "OPERATING_SYSTEM_UPDATES",
        "BATTERY_POWER_HARDWARE",
        "ACCOUNT_APPLE_ID",
        "CONNECTIVITY_NETWORKING",
        "AUDIO_ACCESSORIES",
    ]
    routings = ["AUTO_HANDLE", "HUMAN_ESCALATION"]
    return [
        {
            "example_id": f"synthetic_test_{i:03d}",
            "conversation_id": f"conv_synth_{i}",
            "customer_message": f"Synthetic customer query test issue {i}",
            "conversation": f"[CUSTOMER]: Synthetic customer query test issue {i}",
            "turn_count": 2,
            "human_intent": intents[(i - 1) % len(intents)],
            "human_routing": routings[(i - 1) % len(routings)],
            "human_notes": "Unit test synthetic verification note",
            "annotator_id": "annotator_test_unit",
            "annotated": True,
        }
        for i in range(1, count + 1)
    ]


def get_synthetic_invalid_annotations() -> List[Dict[str, Any]]:
    """Return synthetic records with deliberate errors to test validator assertions."""
    return [
        {
            "example_id": "bad_001",
            "customer_message": "Query with invalid intent",
            "human_intent": "NOT_A_REAL_INTENT",
            "human_routing": "AUTO_HANDLE",
            "annotated": True,
        },
        {
            "example_id": "bad_002",
            "customer_message": "Query with invalid routing",
            "human_intent": "OPERATING_SYSTEM_UPDATES",
            "human_routing": "MAYBE_ESCALATE",
            "annotated": True,
        },
        {
            "example_id": "bad_003",
            "customer_message": "Query marked annotated=True but null fields",
            "human_intent": None,
            "human_routing": None,
            "annotated": True,
        },
        {
            "example_id": "bad_001",  # duplicate ID
            "customer_message": "Duplicate ID record",
            "human_intent": "BATTERY_POWER_HARDWARE",
            "human_routing": "HUMAN_ESCALATION",
            "annotated": True,
        },
    ]


def get_synthetic_response_ratings(count: int = 5) -> List[Dict[str, Any]]:
    """Return synthetic response ratings on the 1-5 scale for mathematical agreement tests."""
    return [
        {
            "example_id": f"synthetic_test_{i:03d}",
            "customer_message": f"Customer query {i}",
            "generated_response": f"Generated diagnostic response {i}",
            "human_groundedness": (i % 5) + 1,
            "human_relevance": ((i + 1) % 5) + 1,
            "human_brand_voice": ((i + 2) % 5) + 1,
            "human_safety": 5,
            "human_notes": "Test synthetic rating",
            "annotator_id": "test_raters",
            "rated": True,
        }
        for i in range(1, count + 1)
    ]
