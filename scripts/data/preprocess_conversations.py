"""Data Preprocessing & Conversation Reconstruction Pipeline for AppleSupport.

Stitches flat tweet records into ordered, multi-turn conversation threads,
cleans text, extracts operational metadata, and outputs structured JSONL records.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts.data.loader import (  # noqa: E402
    get_dataset_path,
    get_repo_root,
    iter_chunks,
)

# Pre-compiled regex patterns for deterministic text cleaning
LEADING_MENTIONS_RE = re.compile(r"^(@\w+\s*)+", re.UNICODE)
MULTIPLE_SPACES_RE = re.compile(r"[ \t]+", re.UNICODE)
DM_KEYWORD_RE = re.compile(
    r"\b(dm|direct message|join us in a dm|send us a dm)\b", re.IGNORECASE
)
RESOLUTION_KEYWORD_RE = re.compile(
    r"\b(thank you|thanks|fixed|working now|worked|solved|appreciate your help|all set|awesome)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
APPLE_KB_RE = re.compile(
    r"(apple\.co|support\.apple\.com|appleid\.apple\.com)", re.IGNORECASE
)


def clean_tweet_text(text: str) -> str:
    """Normalize raw tweet text while preserving essential technical information.

    1. Unescapes HTML entities (&amp; -> &, &gt; -> >)
    2. Removes conversational leading @handles (@AppleSupport, @115854)
    3. Preserves internal domain terminology and mid-sentence references
    4. Normalizes whitespace and unprintable characters
    """
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities
    cleaned = html.unescape(text)

    # 2. Strip leading mention headers
    cleaned = LEADING_MENTIONS_RE.sub("", cleaned)

    # 3. Normalize multiple whitespace characters
    cleaned = MULTIPLE_SPACES_RE.sub(" ", cleaned)

    return cleaned.strip()


def extract_thread_metadata(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze a sequence of turns to derive operational and resolution tags."""
    has_dm = False
    has_kb_link = False
    has_resolution = False

    # Check brand and customer turns
    for turn in turns:
        text = turn.get("text", "")
        role = turn.get("author_role", "")

        if role == "BRAND":
            if DM_KEYWORD_RE.search(text):
                has_dm = True
            if APPLE_KB_RE.search(text):
                has_kb_link = True
        elif role == "CUSTOMER":
            if RESOLUTION_KEYWORD_RE.search(text):
                has_resolution = True

    # Determine default operational status for Inbox triage
    if has_dm:
        status = "Needs Human"
    elif has_resolution:
        status = "Resolved"
    else:
        status = "AI Ready"

    return {
        "has_dm": has_dm,
        "has_kb_link": has_kb_link,
        "has_resolution": has_resolution,
        "status": status,
    }


def reconstruct_conversations(
    df: pd.DataFrame,
    brand_handle: str = "AppleSupport",
) -> List[Dict[str, Any]]:
    """Reconstruct flat tweet dataframe into ordered conversation threads."""
    # Build fast in-memory lookup table
    tweet_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        t_id = str(row["tweet_id"])
        author = str(row["author_id"])
        is_inbound = bool(row["inbound"])
        in_resp_to = (
            str(row["in_response_to_tweet_id"])
            if pd.notna(row["in_response_to_tweet_id"])
            else None
        )

        tweet_dict[t_id] = {
            "tweet_id": t_id,
            "author_id": author,
            "author_role": (
                "BRAND" if author.lower() == brand_handle.lower() else "CUSTOMER"
            ),
            "inbound": is_inbound,
            "created_at": str(row["created_at"]),
            "text": str(row["text"]),
            "clean_text": clean_tweet_text(str(row["text"])),
            "in_response_to": in_resp_to,
        }

    # Identify candidate root customer inquiries
    # A root tweet is an inbound customer tweet where in_response_to is None or not in the dataset
    root_tweet_ids = [
        t_id
        for t_id, t_data in tweet_dict.items()
        if t_data["author_role"] == "CUSTOMER"
        and (
            t_data["in_response_to"] is None
            or t_data["in_response_to"] not in tweet_dict
        )
    ]

    # Map children tweets (parent_id -> list of child_ids)
    children_map: Dict[str, List[str]] = {}
    for t_id, t_data in tweet_dict.items():
        parent_id = t_data["in_response_to"]
        if parent_id:
            children_map.setdefault(parent_id, []).append(t_id)

    conversations: List[Dict[str, Any]] = []

    for root_id in root_tweet_ids:
        root_data = tweet_dict[root_id]

        # Traverse child turns in chronological sequence
        thread_tweet_ids = [root_id]
        current_id = root_id
        visited = {root_id}

        while current_id in children_map:
            children = children_map[current_id]
            # Pick first unvisited child
            next_child = None
            for child_id in children:
                if child_id not in visited:
                    next_child = child_id
                    break
            if next_child is None:
                break
            visited.add(next_child)
            thread_tweet_ids.append(next_child)
            current_id = next_child

        # We are only interested in threads with at least one brand response
        roles_in_thread = [
            tweet_dict[tid]["author_role"]
            for tid in thread_tweet_ids
            if tid in tweet_dict
        ]
        if "BRAND" not in roles_in_thread:
            continue

        # Assemble turn records
        turns = []
        for idx, tid in enumerate(thread_tweet_ids, start=1):
            t = tweet_dict[tid]
            turns.append(
                {
                    "turn_id": idx,
                    "tweet_id": tid,
                    "author_id": t["author_id"],
                    "author_role": t["author_role"],
                    "text": t["clean_text"],
                    "raw_text": t["text"],
                    "created_at": t["created_at"],
                }
            )

        # Operational metadata
        meta = extract_thread_metadata(turns)
        first_inquiry = turns[0]["text"]
        brand_turns = [t["text"] for t in turns if t["author_role"] == "BRAND"]
        final_brand_response = brand_turns[-1] if brand_turns else ""
        latest_message = turns[-1]["text"]

        conversations.append(
            {
                "conversation_id": f"conv_{root_id}",
                "ticket_id": f"TICK-{root_id}",
                "customer_id": root_data["author_id"],
                "brand": brand_handle,
                "root_tweet_id": root_id,
                "first_inquiry": first_inquiry,
                "latest_message": latest_message,
                "final_brand_response": final_brand_response,
                "intent": "Unclassified",
                "confidence": 0.90 if meta["status"] == "AI Ready" else 0.80,
                "decision": (
                    "AUTO_HANDLE"
                    if meta["status"] in ("Resolved", "AI Ready")
                    else "HUMAN_ESCALATION"
                ),
                "turn_count": len(turns),
                "turns": turns,
                "has_dm": meta["has_dm"],
                "has_kb_link": meta["has_kb_link"],
                "has_resolution": meta["has_resolution"],
                "status": meta["status"],
                "created_at": turns[0]["created_at"],
                "timestamp": turns[0]["created_at"],
            }
        )

    return conversations


def run_pipeline(
    max_source_rows: Optional[int] = None,
    output_sample_size: int = 100,
) -> Dict[str, Any]:
    """Execute the end-to-end conversation preprocessing pipeline."""
    repo_root = get_repo_root()
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = get_dataset_path()
    print(f"Reading from dataset: {dataset_path}")

    # Stream and filter AppleSupport tweets
    mention_pattern = "@AppleSupport"
    frames = []
    total_loaded = 0

    chunksize = 100000
    for chunk in iter_chunks(chunksize=chunksize):
        outbound_mask = chunk["author_id"] == "AppleSupport"
        inbound_mask = chunk["inbound"].fillna(False).astype(bool) & chunk[
            "text"
        ].fillna("").str.contains(mention_pattern, case=False, regex=False)
        matched = chunk[outbound_mask | inbound_mask].copy()

        if not matched.empty:
            frames.append(matched)
            total_loaded += len(matched)
            if max_source_rows and total_loaded >= max_source_rows:
                break

    if not frames:
        print("No AppleSupport tweets matched.")
        return {}

    df = pd.concat(frames, ignore_index=True)
    if max_source_rows:
        df = df.iloc[:max_source_rows]
    print(
        f"Loaded {len(df)} AppleSupport-related tweets. Reconstructing conversation threads..."
    )

    conversations = reconstruct_conversations(df, brand_handle="AppleSupport")
    print(
        f"Reconstructed {len(conversations)} valid multi-turn customer conversations."
    )

    # Save full processed dataset
    full_output_file = processed_dir / "applesupport_conversations.jsonl"
    with open(full_output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Saved full conversations to: {full_output_file}")

    # Save lightweight sample for fast development, CI, and test assertions
    sample_output_file = processed_dir / "applesupport_sample.jsonl"
    sample_conversations = conversations[:output_sample_size]
    with open(sample_output_file, "w", encoding="utf-8") as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(
        f"Saved {len(sample_conversations)} sample conversations to: {sample_output_file}"
    )

    summary = {
        "brand": "AppleSupport",
        "total_source_tweets": len(df),
        "total_reconstructed_conversations": len(conversations),
        "sample_size": len(sample_conversations),
        "multi_turn_conversations": sum(
            1 for c in conversations if c["turn_count"] > 2
        ),
        "dm_escalated_count": sum(1 for c in conversations if c["has_dm"]),
        "kb_linked_count": sum(1 for c in conversations if c["has_kb_link"]),
        "resolution_confirmed_count": sum(
            1 for c in conversations if c["has_resolution"]
        ),
    }

    stats_file = repo_root / "experiments" / "preprocessing_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    from scripts.data.loader import iter_chunks

    # Process with max 10000 rows when run as quick standalone or full run
    run_pipeline(max_source_rows=15000)
