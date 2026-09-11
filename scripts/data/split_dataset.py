"""Split preprocessed conversation dataset into Train, Validation, and Test splits.

Ensures strict leak-free isolation by completely excluding all conversation IDs
present in the curated Golden Evaluation Set (data/golden/golden_set.jsonl).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "applesupport_conversations.jsonl"
GOLDEN_PATH = BASE_DIR / "data" / "golden" / "golden_set.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "splits"


def run_split():
    # 1. Load Golden Set IDs to isolate
    golden_ids = set()
    if GOLDEN_PATH.exists():
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    golden_ids.add(rec.get("conversation_id"))

    print(f"Loaded {len(golden_ids)} isolated Golden Set IDs.")

    # 2. Ingest conversation records excluding golden set
    eligible_records: List[Dict[str, Any]] = []
    skipped_golden = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("conversation_id")
            if cid in golden_ids:
                skipped_golden += 1
                continue
            eligible_records.append(rec)

    print(
        f"Eligible records for training/testing: {len(eligible_records)} (excluded {skipped_golden} golden)"
    )

    # 3. Stratified Split: Train (70%), Val (15%), Test (15%)
    labels = [r.get("intent_code", "GENERAL_INQUIRY") for r in eligible_records]

    # First split: Train (70%) vs Temp (30%)
    train_recs, temp_recs, train_labels, temp_labels = train_test_split(
        eligible_records,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    # Second split: Val (15%) vs Test (15%)
    val_recs, test_recs, _, _ = train_test_split(
        temp_recs,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (OUTPUT_DIR / "train.jsonl", train_recs),
        "val": (OUTPUT_DIR / "val.jsonl", val_recs),
        "test": (OUTPUT_DIR / "test.jsonl", test_recs),
    }

    split_counts = {}
    for split_name, (split_path, recs) in splits.items():
        with open(split_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        split_counts[split_name] = len(recs)
        print(f"Saved {split_name} split: {len(recs)} records to {split_path}")

    # Summary
    summary = {
        "total_source_conversations": len(eligible_records) + skipped_golden,
        "golden_set_isolated": skipped_golden,
        "train_samples": split_counts["train"],
        "val_samples": split_counts["val"],
        "test_samples": split_counts["test"],
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "random_seed": 42,
        "stratified_by": "intent_code",
    }

    with open(OUTPUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Data splitting completed successfully.")


if __name__ == "__main__":
    run_split()
