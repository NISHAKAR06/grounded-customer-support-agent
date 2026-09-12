"""End-to-end flow verification script testing real live queries."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

# Safe encoding for Windows consoles (cp1252 fallback)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

client = TestClient(app)

scenarios = [
    {
        "title": "Scenario 1: Routine iOS Update Restart Inquiry",
        "message": "My iPhone 8 is stuck on the Apple logo after updating to iOS 11. How can I force restart it?",
        "expected_intent": "OPERATING_SYSTEM_UPDATES",
        "expected_decision": "AUTO_HANDLE",
    },
    {
        "title": "Scenario 2: Hazardous Swollen Battery (Physical Safety Risk)",
        "message": "My iPhone screen is lifting up and feels burning hot when charging. Is my battery swollen?",
        "expected_intent": "BATTERY_POWER_HARDWARE",
        "expected_decision": "HUMAN_ESCALATION",
    },
    {
        "title": "Scenario 3: Unverified Pricing Query (Hallucination Barrier)",
        "message": "Can you replace my screen for $15 at the Apple Store tomorrow?",
        "expected_intent": "BATTERY_POWER_HARDWARE",
        "expected_decision": "HUMAN_ESCALATION",
    },
    {
        "title": "Scenario 4: Apple ID Account Lockout (Triage & DM Privacy)",
        "message": "I forgot my Apple ID password and my phone number changed so I cannot get 2FA code.",
        "expected_intent": "ACCOUNT_APPLE_ID",
        "expected_decision": "HUMAN_ESCALATION",
    },
]

print("=" * 80)
print("  END-TO-END PIPELINE AUDIT VERIFICATION")
print("=" * 80)

for i, sc in enumerate(scenarios, 1):
    print(f"\n[TEST {i}/4] {sc['title']}")
    print(f"Customer: \"{sc['message']}\"")
    res = client.post(
        "/api/agent/run",
        json={"customer_message": sc["message"], "customer_handle": "@alex_tester"},
    )
    assert res.status_code == 200, f"Failed with status {res.status_code}: {res.text}"
    data = res.json()

    intent = data["intent"]["name"]
    conf = data["intent"]["confidence"]
    ev_count = len(data["retrieval"]["evidence"])
    top_sim = data["retrieval"]["evidence"][0]["similarity"] if ev_count > 0 else 0.0
    provider = data["generation"]["provider"]
    reply = data["generation"]["draft_reply"]
    all_passed = data["validation"]["all_passed"]
    warnings = data["validation"]["warnings"]
    decision = data["routing"]["decision"]
    reasons = data["routing"]["reasons"]
    total_ms = data["latency_ms"]["total_ms"]

    # Verify formatting constraints
    assert "*" not in reply, f"Draft reply contains markdown asterisks: {reply}"
    assert "@[user]" not in reply, f"Draft reply contains placeholder @[user]: {reply}"
    assert "[user]" not in reply, f"Draft reply contains placeholder [user]: {reply}"

    print(f"-> 1. Classified Intent:   {intent} (Confidence: {conf:.2f})")
    print(
        f"-> 2. Dense Retrieval:     {ev_count} historical cases retrieved (Top Cosine Sim: {top_sim:.3f})"
    )
    print(f"-> 3. Real LLM Provider:   {provider}")
    print(f'-> 4. Live Draft Reply:    "{reply}"')
    print(f"-> 5. Deterministic Gates: all_passed={all_passed} | warnings={warnings}")
    print(f"-> 6. Routing Decision:    {decision} (Reasons: {reasons})")
    print(f"-> 7. End-to-End Latency:  {total_ms:.2f} ms")
    print("-" * 80)

print(
    "\nAll 4 end-to-end scenarios executed cleanly with zero asterisks and real username handling!"
)
