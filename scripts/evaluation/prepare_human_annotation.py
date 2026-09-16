"""Prepare human annotation template for the 200 Golden Set conversations.

Extracts customer conversations and thread context into an annotation-ready
schema with blank fields for independent human labeling of intent and routing.

IMPORTANT:
- Never copies heuristic labels into the human label fields.
- Provides all required fields for human evaluators to work independently.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_human_annotation")

DEFAULT_INPUT = BASE_DIR / "data" / "golden" / "golden_set.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "golden" / "human_annotation_template.jsonl"


def format_thread_for_display(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns into a clean, human-readable dialogue string."""
    formatted_turns = []
    for turn in turns:
        role = turn.get("author_role", "UNKNOWN")
        text = turn.get("text", turn.get("raw_text", ""))
        formatted_turns.append(f"[{role}]: {text}")
    return "\n".join(formatted_turns)


def prepare_annotation_records(golden_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform golden set samples into clean, unlabelled human annotation templates."""
    annotation_records = []

    for sample in golden_samples:
        example_id = sample.get("sample_id")
        conv_id = sample.get("conversation_id", "")
        customer_msg = sample.get("customer_message", "")
        turns = sample.get("conversation_turns", [])

        # Build comprehensive conversation display
        full_conversation = format_thread_for_display(turns) if turns else customer_msg

        record = {
            "example_id": example_id,
            "conversation_id": conv_id,
            "customer_message": customer_msg,
            "conversation": full_conversation,
            "turn_count": sample.get("turn_count", len(turns)),
            "human_intent": None,
            "human_routing": None,
            "human_notes": "",
            "annotator_id": "",
            "annotated": False,
        }
        annotation_records.append(record)

    return annotation_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare human annotation template for the 200 Golden Set cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to input golden_set.jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to output template file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found at: {args.input}")
        sys.exit(1)

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} golden set samples from {args.input}")

    records = prepare_annotation_records(samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"Generated human annotation template with {len(records)} unlabelled records at: {args.output}"
    )
    print(f"\nHuman Annotation Template Ready: {args.output}")
    print(f"Total unlabelled cases: {len(records)}")
    print("All human_intent and human_routing fields initialized to null (no heuristic leakage).")


if __name__ == "__main__":
    main()
