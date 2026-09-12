"""Unit tests for Phase 3: Data Preprocessing & Conversation Reconstruction."""

from pathlib import Path

import pandas as pd

from app.repositories.conversation_repository import ConversationRepository
from scripts.data.preprocess_conversations import (
    clean_tweet_text,
    extract_thread_metadata,
    reconstruct_conversations,
)


def test_clean_tweet_text_entities():
    """Verify HTML entity decoding and whitespace normalization."""
    raw = "@AppleSupport Check Settings &gt; General &gt; About   for iOS version."
    cleaned = clean_tweet_text(raw)
    assert cleaned == "Check Settings > General > About for iOS version."


def test_clean_tweet_text_leading_mentions():
    """Verify leading handles are stripped while preserving mid-sentence mentions."""
    raw = "@AppleSupport @115854 My iPhone battery is draining. I also tweeted @tim_cook."
    cleaned = clean_tweet_text(raw)
    assert cleaned == "My iPhone battery is draining. I also tweeted @tim_cook."


def test_extract_thread_metadata():
    """Verify DM, link, and resolution heuristics correctly derive operational status."""
    dm_turns = [
        {"author_role": "CUSTOMER", "text": "My phone is locked."},
        {
            "author_role": "BRAND",
            "text": "Please join us in a DM so we can verify your account.",
        },
    ]
    meta_dm = extract_thread_metadata(dm_turns)
    assert meta_dm["has_dm"] is True
    assert meta_dm["status"] == "Needs Human"

    resolved_turns = [
        {"author_role": "CUSTOMER", "text": "My screen brightness was stuck."},
        {"author_role": "BRAND", "text": "Toggle Auto-Brightness in Accessibility."},
        {"author_role": "CUSTOMER", "text": "Thank you so much, that fixed it!"},
    ]
    meta_res = extract_thread_metadata(resolved_turns)
    assert meta_res["has_resolution"] is True
    assert meta_res["status"] == "Resolved"


def test_reconstruct_conversations_graph():
    """Verify directed graph stitching into multi-turn conversation threads."""
    records = [
        {
            "tweet_id": "101",
            "author_id": "user_a",
            "inbound": True,
            "created_at": "2017-10-01 10:00:00",
            "text": "@AppleSupport My iPhone won't turn on.",
            "in_response_to_tweet_id": None,
            "response_tweet_id": "102",
        },
        {
            "tweet_id": "102",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "2017-10-01 10:05:00",
            "text": "@user_a Connect your device to a charger for at least 30 minutes.",
            "in_response_to_tweet_id": "101",
            "response_tweet_id": None,
        },
    ]
    df = pd.DataFrame(records)
    conversations = reconstruct_conversations(df, brand_handle="AppleSupport")

    assert len(conversations) == 1
    c = conversations[0]
    assert c["conversation_id"] == "conv_101"
    assert c["turn_count"] == 2
    assert c["first_inquiry"] == "My iPhone won't turn on."
    assert "Connect your device" in c["final_brand_response"]
    assert c["turns"][0]["author_role"] == "CUSTOMER"
    assert c["turns"][1]["author_role"] == "BRAND"


def test_conversation_repository_loads_sample():
    """Verify ConversationRepository loads real reconstructed AppleSupport conversations."""
    sample_file = Path("data/processed/applesupport_sample.jsonl")
    assert sample_file.exists(), "applesupport_sample.jsonl must exist"

    repo = ConversationRepository(data_path=sample_file)
    conversations = repo.list_conversations()
    assert len(conversations) > 0

    first_conv = conversations[0]
    assert "conversation_id" in first_conv
    assert "turns" in first_conv
    assert "turn_count" in first_conv
    assert first_conv["brand"] == "AppleSupport"


def test_conversation_repository_pagination_ellipsis():
    """Verify windowed pagination creates 1 2 3 4 5 ... 10 sequence and correct slice."""
    sample_file = Path("data/processed/applesupport_sample.jsonl")
    repo = ConversationRepository(data_path=sample_file)

    # Page 1 with page_size=10
    p1 = repo.paginate(page=1, page_size=10)
    assert len(p1["items"]) == 10
    assert p1["total_items"] == 100
    assert p1["total_pages"] == 10
    assert p1["current_page"] == 1
    assert p1["start_item"] == 1
    assert p1["end_item"] == 10
    assert p1["has_prev"] is False
    assert p1["has_next"] is True
    assert p1["pages_display"] == [1, 2, 3, 4, 5, "...", 10]

    # Page 6 in the middle
    p6 = repo.paginate(page=6, page_size=10)
    assert p6["current_page"] == 6
    assert p6["pages_display"] == [1, "...", 5, 6, 7, "...", 10]

    # Page 10 at the end
    p10 = repo.paginate(page=10, page_size=10)
    assert p10["current_page"] == 10
    assert p10["has_next"] is False
    assert p10["has_prev"] is True
    assert p10["pages_display"] == [1, "..."] + list(range(6, 11))


def test_conversation_repository_filters_and_sorting():
    """Verify operational filters and sorting keys work deterministically."""
    sample_file = Path("data/processed/applesupport_sample.jsonl")
    repo = ConversationRepository(data_path=sample_file)

    # Filter by decision
    auto_handled = repo.list_conversations(decision_filter="auto_handle")
    assert len(auto_handled) > 0
    assert all(c["decision"] == "AUTO_HANDLE" for c in auto_handled)

    # Filter by turns
    deep_threads = repo.list_conversations(turn_filter="deep")
    assert all(int(c["turn_count"]) >= 5 for c in deep_threads)

    # Sorting by turns descending
    sorted_turns = repo.list_conversations(sort_by="turns_desc")
    assert int(sorted_turns[0]["turn_count"]) >= int(sorted_turns[-1]["turn_count"])

    # Sorting by confidence descending
    sorted_conf = repo.list_conversations(sort_by="confidence_desc")
    assert float(sorted_conf[0]["confidence"] or 0) >= float(sorted_conf[-1]["confidence"] or 0)
