"""Brand-specific profiling script for AppleSupport.

Extracts AppleSupport-specific inbound inquiries, outbound replies,
and conversational metrics to inform Phase 2 documentation and Phase 3 pipeline.
"""

import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.loader import iter_chunks  # noqa: E402


def profile_apple_support(chunksize: int = 150000):
    brand_handle = "AppleSupport"
    outbound_count = 0
    inbound_mention_count = 0
    unique_customers = set()

    apple_mention_pattern = re.compile(r"@AppleSupport\b", re.IGNORECASE)

    for chunk in tqdm(iter_chunks(chunksize=chunksize), desc="Profiling AppleSupport"):
        # Outbound from AppleSupport
        apple_outbound = chunk[chunk["author_id"] == brand_handle]
        outbound_count += len(apple_outbound)

        # Inbound directed to AppleSupport
        inbound_chunk = chunk[chunk["inbound"].fillna(False).astype(bool)]
        mention_mask = (
            inbound_chunk["text"].fillna("").str.contains(apple_mention_pattern)
        )
        apple_inbound = inbound_chunk[mention_mask]
        inbound_mention_count += len(apple_inbound)
        unique_customers.update(apple_inbound["author_id"].dropna().unique().tolist())

    results = {
        "brand": brand_handle,
        "outbound_replies": outbound_count,
        "inbound_mentions": inbound_mention_count,
        "total_brand_interactions": outbound_count + inbound_mention_count,
        "unique_customers": len(unique_customers),
    }

    out_file = REPO_ROOT / "experiments" / "brand_profile_applesupport.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAppleSupport Profile:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    profile_apple_support()
