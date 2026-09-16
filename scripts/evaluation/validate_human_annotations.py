"""Validate human annotation integrity, schema compliance, and completeness.

Ensures:
1. Exactly 200 expected golden set sample IDs.
2. No duplicate example IDs.
3. Every human_intent is one of the 7 MECE intent codes.
4. Every human_routing is either AUTO_HANDLE or HUMAN_ESCALATION.
5. annotated flag is True only when all required labels are present.
6. Catches accidental copying / cloning of heuristic labels.
7. Reports clear diagnostic errors.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_human_annotations")

VALID_INTENTS: Set[str] = {
    "OPERATING_SYSTEM_UPDATES",
    "BATTERY_POWER_HARDWARE",
    "ACCOUNT_APPLE_ID",
    "CONNECTIVITY_NETWORKING",
    "AUDIO_ACCESSORIES",
    "SUBSCRIPTIONS_BILLING",
    "GENERAL_INQUIRY",
}

VALID_ROUTINGS: Set[str] = {
    "AUTO_HANDLE",
    "HUMAN_ESCALATION",
}

DEFAULT_ANNOTATION_PATH = BASE_DIR / "data" / "golden" / "human_annotation_template.jsonl"
DEFAULT_GOLDEN_PATH = BASE_DIR / "data" / "golden" / "golden_set.jsonl"


def load_expected_ids(golden_path: Path) -> List[str]:
    """Load the canonical 200 sample IDs from the golden set."""
    ids = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                ids.append(record["sample_id"])
    return ids


def validate_annotations(
    annotation_path: Path, golden_path: Path, allow_pending: bool = False
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate annotation records against schema, constraints, and heuristic independence.

    Returns:
        (is_valid, error_messages, summary_stats)
    """
    errors: List[str] = []
    stats: Dict[str, Any] = {
        "total_records": 0,
        "annotated_count": 0,
        "pending_count": 0,
        "duplicate_ids": [],
        "invalid_intents": [],
        "invalid_routings": [],
        "heuristic_matches": 0,
    }

    if not annotation_path.exists():
        return False, [f"Annotation file not found: {annotation_path}"], stats

    expected_ids = load_expected_ids(golden_path) if golden_path.exists() else []
    expected_ids_set = set(expected_ids)

    # Load golden heuristic labels for leakage detection
    heuristic_labels: Dict[str, Dict[str, str]] = {}
    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    heuristic_labels[rec["sample_id"]] = {
                        "intent": rec.get("gold_intent_code", ""),
                        "routing": rec.get("gold_routing", ""),
                    }

    records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_num}: Malformed JSON - {exc}")

    stats["total_records"] = len(records)

    # 1. Total count check
    if len(records) != 200:
        errors.append(f"Expected exactly 200 records, but found {len(records)}.")

    # 2. Schema, ID and field checks
    heuristic_exact_matches = 0

    for idx, rec in enumerate(records, 1):
        eid = rec.get("example_id")
        if not eid:
            errors.append(f"Record #{idx}: Missing 'example_id'.")
            continue

        # Check duplicates
        if eid in seen_ids:
            errors.append(f"Duplicate example_id found: '{eid}' at record #{idx}.")
            stats["duplicate_ids"].append(eid)
        seen_ids.add(eid)

        # Check unexpected IDs
        if expected_ids_set and eid not in expected_ids_set:
            errors.append(f"Record #{idx}: Unexpected example_id '{eid}' not in golden set.")

        intent = rec.get("human_intent")
        routing = rec.get("human_routing")
        annotated = rec.get("annotated", False)

        # Check pending vs completed status
        if intent is None or routing is None:
            stats["pending_count"] += 1
            if annotated:
                errors.append(
                    f"Record '{eid}': Marked annotated=True but has missing labels (intent: {intent}, routing: {routing})."
                )
            if not allow_pending:
                errors.append(
                    f"Record '{eid}': Missing required human labels (annotation pending)."
                )
            continue

        stats["annotated_count"] += 1

        if not annotated:
            errors.append(f"Record '{eid}': Has complete human labels but annotated flag is False.")

        # Validate intent
        if intent not in VALID_INTENTS:
            errors.append(
                f"Record '{eid}': Invalid human_intent '{intent}'. Must be one of {sorted(VALID_INTENTS)}."
            )
            stats["invalid_intents"].append((eid, intent))

        # Validate routing
        if routing not in VALID_ROUTINGS:
            errors.append(
                f"Record '{eid}': Invalid human_routing '{routing}'. Must be one of {sorted(VALID_ROUTINGS)}."
            )
            stats["invalid_routings"].append((eid, routing))

        # Check for heuristic mirroring
        if eid in heuristic_labels:
            heur = heuristic_labels[eid]
            if intent == heur["intent"] and routing == heur["routing"]:
                heuristic_exact_matches += 1

    stats["heuristic_matches"] = heuristic_exact_matches

    # 3. Check for missing expected IDs
    if expected_ids_set:
        missing_ids = expected_ids_set - seen_ids
        if missing_ids:
            errors.append(
                f"Missing {len(missing_ids)} required golden set IDs: {sorted(list(missing_ids))[:5]}..."
            )

    # 4. Check for 100% heuristic duplication (suspected clone without human evaluation)
    if stats["annotated_count"] == 200 and heuristic_exact_matches == 200:
        errors.append(
            "CRITICAL INTEGRITY WARNING: All 200 annotations are 100% identical to the heuristic rules. "
            "Suspected copy of heuristic labels without independent human annotation."
        )

    is_valid = len(errors) == 0
    return is_valid, errors, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate human annotation records against schema and integrity rules."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_ANNOTATION_PATH,
        help=f"Path to annotations file (default: {DEFAULT_ANNOTATION_PATH})",
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help=f"Path to golden set reference file (default: {DEFAULT_GOLDEN_PATH})",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow records with null labels (used to validate blank templates).",
    )
    args = parser.parse_args()

    logger.info(f"Validating annotations file: {args.path}")
    is_valid, errors, stats = validate_annotations(
        args.path, args.golden_path, allow_pending=args.allow_pending
    )

    print("\n=== HUMAN ANNOTATION VALIDATION REPORT ===")
    print(f"File:            {args.path}")
    print(f"Total Records:   {stats['total_records']} / 200")
    print(f"Annotated:       {stats['annotated_count']}")
    print(f"Pending/Null:    {stats['pending_count']}")

    if is_valid:
        print("\nSTATUS: [VALID] All structural and schema checks PASSED.")
        if stats["pending_count"] > 0:
            print("NOTE: Template has pending human annotations as expected.")
        sys.exit(0)
    else:
        print(f"\nSTATUS: [INVALID] Found {len(errors)} validation issue(s):")
        for err in errors[:25]:
            print(f"  [ERROR] {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
