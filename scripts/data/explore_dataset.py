"""Comprehensive dataset exploration script for Customer Support on Twitter.

Processes the twcs.csv dataset in streaming chunks to compute global statistics,
brand volumes, conversation topology, and resolution indicators without exhausting memory.
Saves reproducible outputs to experiments/dataset_stats.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.loader import get_dataset_path, iter_chunks  # noqa: E402


def analyze_dataset(chunksize: int = 150000) -> Dict[str, Any]:
    """Stream twcs.csv in chunks and aggregate full dataset metrics."""
    dataset_path = get_dataset_path()
    file_size_bytes = os.path.getsize(dataset_path)

    total_rows = 0
    inbound_count = 0
    outbound_count = 0

    null_counts: Counter = Counter()
    column_names = []

    brand_outbound_counts: Counter = Counter()
    unique_customer_ids = set()

    min_date = None
    max_date = None

    # Topology counters
    root_inbound_inquiries = 0  # inbound & in_response_to is null
    reply_inbound_turns = 0  # inbound & in_response_to is not null
    brand_replies = 0  # outbound & in_response_to is not null
    brand_proactive = 0  # outbound & in_response_to is null
    multi_reply_parent_count = 0

    # Heuristic resolution & escalation patterns
    dm_escalation_count = 0
    url_reference_count = 0
    gratitude_count = 0

    dm_regex = re.compile(r"\b(dm|direct message|pm|private message)\b", re.IGNORECASE)
    url_regex = re.compile(r"https?://\S+", re.IGNORECASE)
    gratitude_regex = re.compile(
        r"\b(thank|thanks|thx|resolved|fixed|worked|awesome|solved)\b", re.IGNORECASE
    )

    print(f"Analyzing dataset from: {dataset_path}")
    print(f"File size: {file_size_bytes / (1024 * 1024):.2f} MB")

    chunk_iter = iter_chunks(chunksize=chunksize)
    for chunk in tqdm(chunk_iter, desc="Processing chunks"):
        if not column_names:
            column_names = chunk.columns.tolist()

        chunk_len = len(chunk)
        total_rows += chunk_len

        # Inbound / Outbound
        inbound_mask = chunk["inbound"].fillna(False).astype(bool)
        n_inbound = int(inbound_mask.sum())
        n_outbound = chunk_len - n_inbound
        inbound_count += n_inbound
        outbound_count += n_outbound

        # Null tracking
        for col in column_names:
            null_counts[col] += int(chunk[col].isna().sum())

        # Dates
        dates = pd.to_datetime(
            chunk["created_at"],
            format="%a %b %d %H:%M:%S %z %Y",
            errors="coerce",
        ).dropna()
        if not dates.empty:
            c_min = dates.min()
            c_max = dates.max()
            min_date = c_min if min_date is None else min(min_date, c_min)
            max_date = c_max if max_date is None else max(max_date, c_max)

        # Inbound (customer) authors
        inbound_df = chunk[inbound_mask]
        unique_customer_ids.update(inbound_df["author_id"].dropna().unique().tolist())

        # Topology metrics
        root_inbound_inquiries += int(
            inbound_df["in_response_to_tweet_id"].isna().sum()
        )
        reply_inbound_turns += int(inbound_df["in_response_to_tweet_id"].notna().sum())

        # Customer gratitude check on inbound
        inbound_texts = inbound_df["text"].dropna().astype(str)
        gratitude_count += int(
            inbound_texts.apply(lambda t: bool(gratitude_regex.search(t))).sum()
        )

        # Outbound (brand) analysis
        outbound_df = chunk[~inbound_mask]
        brand_replies += int(outbound_df["in_response_to_tweet_id"].notna().sum())
        brand_proactive += int(outbound_df["in_response_to_tweet_id"].isna().sum())

        for brand, cnt in outbound_df["author_id"].value_counts().items():
            brand_outbound_counts[str(brand)] += int(cnt)

        # Multi-reply checks (response_tweet_id containing comma)
        multi_reply_parent_count += int(
            chunk["response_tweet_id"].dropna().str.contains(",").sum()
        )

        # Brand responses text cues
        outbound_texts = outbound_df["text"].dropna().astype(str)
        dm_escalation_count += int(
            outbound_texts.apply(lambda t: bool(dm_regex.search(t))).sum()
        )
        url_reference_count += int(
            outbound_texts.apply(lambda t: bool(url_regex.search(t))).sum()
        )

    # Compile Top Brands
    top_brands = []
    for brand, count in brand_outbound_counts.most_common(25):
        top_brands.append(
            {
                "brand": brand,
                "outbound_replies": count,
            }
        )

    results = {
        "dataset_metadata": {
            "source_path": str(dataset_path),
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "analyzed_at": datetime.now().isoformat(),
            "total_rows": total_rows,
            "columns": column_names,
        },
        "volume_breakdown": {
            "total_records": total_rows,
            "inbound_records": inbound_count,
            "inbound_percentage": (
                round((inbound_count / total_rows) * 100, 2) if total_rows else 0
            ),
            "outbound_records": outbound_count,
            "outbound_percentage": (
                round((outbound_count / total_rows) * 100, 2) if total_rows else 0
            ),
            "unique_customers": len(unique_customer_ids),
            "unique_brands": len(brand_outbound_counts),
        },
        "temporal_range": {
            "min_date": str(min_date) if min_date else None,
            "max_date": str(max_date) if max_date else None,
            "span_days": (
                (max_date - min_date).days if (min_date and max_date) else None
            ),
        },
        "schema_missingness": {
            col: {
                "null_count": null_counts[col],
                "null_percentage": (
                    round((null_counts[col] / total_rows) * 100, 2) if total_rows else 0
                ),
            }
            for col in column_names
        },
        "conversation_topology": {
            "root_customer_inquiries": root_inbound_inquiries,
            "root_customer_percentage": (
                round((root_inbound_inquiries / inbound_count) * 100, 2)
                if inbound_count
                else 0
            ),
            "follow_up_customer_turns": reply_inbound_turns,
            "brand_in_reply_turns": brand_replies,
            "brand_standalone_turns": brand_proactive,
            "multi_reply_parents": multi_reply_parent_count,
        },
        "behavioral_and_resolution_signals": {
            "brand_dm_referrals": dm_escalation_count,
            "brand_dm_percentage": (
                round((dm_escalation_count / outbound_count) * 100, 2)
                if outbound_count
                else 0
            ),
            "brand_link_referrals": url_reference_count,
            "brand_link_percentage": (
                round((url_reference_count / outbound_count) * 100, 2)
                if outbound_count
                else 0
            ),
            "customer_gratitude_signals": gratitude_count,
        },
        "top_25_brands_by_outbound": top_brands,
    }

    # Save to experiments/dataset_stats.json
    experiments_dir = REPO_ROOT / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    out_file = experiments_dir / "dataset_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nExploration complete. Results saved to {out_file}")
    return results


if __name__ == "__main__":
    analyze_dataset()
