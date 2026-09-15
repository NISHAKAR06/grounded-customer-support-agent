"""Prepare human response evaluation template for 100 stratified agent responses.

Samples 100 cases from the 200 Golden Set across:
- All 7 intents
- Routine canonical cases
- Multi-turn extended cases
- Safety-sensitive and edge cases

Generates and persists the actual system response for each case using the
deterministic grounded precedent generator, leaving human rating dimensions
null for manual evaluation.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.agent.agent_orchestrator import AgentOrchestrator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_human_response_evaluation")

DEFAULT_INPUT = BASE_DIR / "data" / "golden" / "golden_set.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "evaluation" / "human_response_ratings_template.jsonl"

TARGET_QUOTAS = {
    "OPERATING_SYSTEM_UPDATES": 20,
    "BATTERY_POWER_HARDWARE": 18,
    "ACCOUNT_APPLE_ID": 15,
    "CONNECTIVITY_NETWORKING": 13,
    "AUDIO_ACCESSORIES": 10,
    "SUBSCRIPTIONS_BILLING": 10,
    "GENERAL_INQUIRY": 14,
}


def sample_stratified_subset(samples: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """Select 100 samples strictly stratified across intent and complexity."""
    rng = random.Random(seed)

    # Group by intent
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        intent = s.get("gold_intent_code", "GENERAL_INQUIRY")
        by_intent.setdefault(intent, []).append(s)

    selected: List[Dict[str, Any]] = []

    for intent, quota in TARGET_QUOTAS.items():
        pool = by_intent.get(intent, [])
        if len(pool) <= quota:
            selected.extend(pool)
            continue

        # Sub-group by complexity within intent
        by_complexity: Dict[str, List[Dict[str, Any]]] = {}
        for item in pool:
            comp = item.get("complexity", "CANONICAL")
            by_complexity.setdefault(comp, []).append(item)

        intent_selected: List[Dict[str, Any]] = []
        # Sample proportionally from each complexity bucket
        complexities = sorted(by_complexity.keys())
        while len(intent_selected) < quota:
            progress = False
            for comp in complexities:
                comp_items = by_complexity[comp]
                if comp_items:
                    chosen = rng.choice(comp_items)
                    comp_items.remove(chosen)
                    intent_selected.append(chosen)
                    progress = True
                    if len(intent_selected) == quota:
                        break
            if not progress:
                break
        selected.extend(intent_selected)

    # Sort back by sample_id
    selected.sort(key=lambda x: x.get("sample_id", ""))
    return selected


def generate_response_rating_records(
    selected_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run agent orchestrator to produce grounded responses and construct rating templates."""
    orchestrator = AgentOrchestrator()
    rating_records = []

    logger.info(f"Generating grounded responses for {len(selected_samples)} selected cases...")

    for idx, sample in enumerate(selected_samples, 1):
        sample_id = sample.get("sample_id")
        query = sample.get("customer_message", "")

        # Run pipeline with deterministic grounded precedent provider
        run_res = orchestrator.run(
            customer_message=query,
            brand="AppleSupport",
            provider="grounded_precedent",
        )

        draft_reply = run_res.generation.draft_reply
        evidence_cases = run_res.retrieval.evidence
        evidence_summary = ""
        if evidence_cases:
            top_case = evidence_cases[0]
            evidence_summary = (
                f"[Precedent #{top_case.case_id} (Sim: {top_case.similarity:.2f})]: "
                f"{top_case.brand_response[:200]}"
            )

        rating_record = {
            "example_id": sample_id,
            "conversation_id": sample.get("conversation_id", ""),
            "customer_message": query,
            "generated_response": draft_reply,
            "retrieved_evidence_summary": evidence_summary,
            "human_groundedness": None,
            "human_relevance": None,
            "human_brand_voice": None,
            "human_safety": None,
            "human_notes": "",
            "annotator_id": "",
            "rated": False,
        }
        rating_records.append(rating_record)

        if idx % 25 == 0 or idx == len(selected_samples):
            logger.info(f"Processed {idx}/{len(selected_samples)} responses...")

    return rating_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 100-sample human response evaluation template."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to golden set input file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to output ratings template (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified selection (default: 42)",
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

    logger.info(f"Loaded {len(samples)} golden set samples.")
    selected = sample_stratified_subset(samples, seed=args.seed)
    logger.info(f"Selected {len(selected)} samples across stratified intents and complexities.")

    records = generate_response_rating_records(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} response rating templates to: {args.output}")
    print(f"\nHuman Response Evaluation Template Ready: {args.output}")
    print(f"Total evaluated responses: {len(records)}")
    print("All rating dimensions initialized to null with rated=False.")


if __name__ == "__main__":
    main()
