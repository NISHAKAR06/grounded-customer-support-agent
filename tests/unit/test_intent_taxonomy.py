"""Unit tests for the 7-class domain-specific intent taxonomy and classifier."""

from pathlib import Path

from app.models.domain_models import INTENT_LABELS, SupportIntent
from app.repositories.conversation_repository import ConversationRepository
from app.services.intent.intent_classifier import (
    IntentClassifier,
    RuleBasedIntentClassifier,
)
from scripts.data.classify_intents import classify_text_intent


def test_intent_taxonomy_canonical_examples():
    """Verify each of the 7 canonical intents is correctly classified with high confidence."""
    test_cases = [
        (
            "My iPhone has been stuck on 'Verifying update' for iOS 11 for over 3 hours. How do I fix this?",
            SupportIntent.OPERATING_SYSTEM_UPDATES,
        ),
        (
            "My iPhone 7 battery percentage drops from 80% to 15% in less than an hour and the device gets burning hot.",
            SupportIntent.BATTERY_POWER_HARDWARE,
        ),
        (
            "I am locked out of my Apple ID because I changed my phone number and cannot receive the two-factor code.",
            SupportIntent.ACCOUNT_APPLE_ID,
        ),
        (
            "Why does my Wi-Fi keep disconnecting every time my phone locks on my home wireless network?",
            SupportIntent.CONNECTIVITY_NETWORKING,
        ),
        (
            "My right AirPod won't connect or charge in the case, only the left earbud works.",
            SupportIntent.AUDIO_ACCESSORIES,
        ),
        (
            "I was charged $9.99 on my bank statement from itunes.com/bill for an app I never bought, need a refund.",
            SupportIntent.SUBSCRIPTIONS_BILLING,
        ),
        (
            "Hello, what are the Apple Store Regent Street opening hours on Sunday?",
            SupportIntent.GENERAL_INQUIRY,
        ),
    ]

    for text, expected_intent in test_cases:
        intent_enum, conf, signals = classify_text_intent(text)
        assert (
            intent_enum == expected_intent
        ), f"Expected {expected_intent} for '{text}', got {intent_enum}"
        assert conf >= 0.85, f"Expected confidence >= 0.85, got {conf} for '{text}'"
        assert len(signals) > 0, f"Expected matched signals for '{text}'"


def test_intent_taxonomy_disambiguation_safety_priority():
    """Verify safety (swollen battery) takes priority over software update."""
    text = "After installing the iOS 11 update my iPhone battery became swollen and pushed the screen out."
    intent_enum, conf, signals = classify_text_intent(text)
    assert intent_enum == SupportIntent.BATTERY_POWER_HARDWARE
    assert "swollen" in signals


def test_intent_taxonomy_disambiguation_billing_priority():
    """Verify unauthorized charge takes priority over general store inquiry."""
    text = "I checked my receipt and noticed an unauthorized subscription charge from iTunes billing, need refund."
    intent_enum, conf, signals = classify_text_intent(text)
    assert intent_enum == SupportIntent.SUBSCRIPTIONS_BILLING


def test_rule_based_intent_classifier_service():
    """Verify RuleBasedIntentClassifier returns well-formed IntentPrediction."""
    classifier = RuleBasedIntentClassifier()
    prediction = classifier.classify(
        "My lightning headphone adapter is producing static noise and crackling audio."
    )

    assert prediction.code == SupportIntent.AUDIO_ACCESSORIES
    assert prediction.name == INTENT_LABELS[SupportIntent.AUDIO_ACCESSORIES]
    assert prediction.confidence >= 0.85
    assert len(prediction.signals) > 0


def test_intent_classifier_defaults_to_rule_based():
    """Verify IntentClassifier delegates to RuleBasedIntentClassifier out-of-the-box."""
    coordinator = IntentClassifier()
    prediction = coordinator.classify(
        "Cannot log in to my Apple ID, forgot password and 2fa verification code."
    )

    assert prediction.code == SupportIntent.ACCOUNT_APPLE_ID
    assert "Apple ID" in prediction.name
    assert prediction.confidence >= 0.90


def test_conversations_in_sample_have_valid_intents():
    """Verify all conversations in applesupport_sample.jsonl have classified intents matching the 7 classes."""
    sample_file = Path("data/processed/applesupport_sample.jsonl")
    assert sample_file.exists()

    repo = ConversationRepository(data_path=sample_file)
    conversations = repo.list_conversations()
    assert len(conversations) == 100

    valid_labels = set(INTENT_LABELS.values())
    for conv in conversations:
        assert (
            conv.get("intent") in valid_labels
        ), f"Unknown intent: {conv.get('intent')}"
        assert conv.get("intent_code") in [i.value for i in SupportIntent]
        assert "confidence" in conv


def test_conversation_repository_filters_by_intent():
    """Verify repo can filter conversations by intent code or intent display label."""
    sample_file = Path("data/processed/applesupport_sample.jsonl")
    repo = ConversationRepository(data_path=sample_file)

    os_threads = repo.list_conversations(intent_filter="OPERATING_SYSTEM_UPDATES")
    assert len(os_threads) > 0
    assert all(c["intent"] == "OS & iOS Updates" for c in os_threads)

    battery_threads = repo.list_conversations(intent_filter="BATTERY_POWER_HARDWARE")
    assert len(battery_threads) > 0
    assert all(c["intent"] == "Battery & Hardware" for c in battery_threads)


def test_intent_taxonomy_schema_validation():
    """Verify machine-readable intent taxonomy artifact conforms to IntentTaxonomySchema."""
    from scripts.data.classify_intents import load_intent_taxonomy

    schema = load_intent_taxonomy()
    assert schema.brand == "AppleSupport"
    assert schema.version == "1.0.0"
    assert schema.num_classes == 7
    assert len(schema.classes) == 7
    assert len(schema.disambiguation_rules) >= 4
    assert schema.metadata.get("mece_verified") is True

    # Verify every SupportIntent has a registered definition
    defined_codes = {cls.code for cls in schema.classes}
    assert defined_codes == set(SupportIntent)


def test_intent_taxonomy_audio_over_bluetooth_priority():
    """Verify audio device specificity takes precedence over generic wireless connection."""
    text = "My AirPods won't connect or pair via Bluetooth to my device."
    intent_enum, conf, signals = classify_text_intent(text)
    assert intent_enum == SupportIntent.AUDIO_ACCESSORIES
    assert "airpods" in signals


def test_intent_taxonomy_account_over_os_update_priority():
    """Verify security lockout takes precedence over app updates."""
    text = (
        "Cannot download the latest app update because my Apple ID account is locked."
    )
    intent_enum, conf, signals = classify_text_intent(text)
    assert intent_enum == SupportIntent.ACCOUNT_APPLE_ID
    assert "locked" in signals or "apple id" in signals


def test_intent_taxonomy_empty_or_ambiguous_fallback():
    """Verify empty or ambiguous utterances safely fall back to GENERAL_INQUIRY."""
    intent_enum, conf, signals = classify_text_intent("")
    assert intent_enum == SupportIntent.GENERAL_INQUIRY
    assert "empty_payload" in signals

    intent_enum_ambig, conf_ambig, signals_ambig = classify_text_intent(
        "Please help me someone right now"
    )
    assert intent_enum_ambig == SupportIntent.GENERAL_INQUIRY
