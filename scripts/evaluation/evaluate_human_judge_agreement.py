"""Evaluate agreement between human raters and the automated LLM Judge.

Measures correlation, mean absolute error, exact agreement, and within-1-point
agreement across the four evaluation dimensions:
1. Groundedness & Faithfulness
2. Answer Relevance & Helpfulness
3. Brand Voice & Tone
4. Safety & Policy Compliance

IMPORTANT:
- If human ratings are pending or incomplete, cleanly reports:
  'HUMAN EVALUATION INCOMPLETE — metrics not computed.'
- Never manufactures human scores or agreement statistics.
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scipy.stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.agent.agent_orchestrator import AgentOrchestrator  # noqa: E402
from app.services.evaluation.judge_service import LLMJudgeService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_human_judge_agreement")

DEFAULT_RATINGS_PATH = BASE_DIR / "data" / "evaluation" / "human_response_ratings_template.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "experiments" / "human_judge_agreement.json"

DIMENSION_MAPPING = [
    ("groundedness", "human_groundedness", "groundedness"),
    ("answer_relevance", "human_relevance", "answer_relevance"),
    ("brand_tone", "human_brand_voice", "brand_tone"),
    ("safety_compliance", "human_safety", "safety_compliance"),
]


def load_and_verify_human_ratings(
    path: Path,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """Load human response ratings and determine if complete and rated."""
    if not path.exists():
        return False, [], {"error": f"File not found: {path}"}

    records: List[Dict[str, Any]] = []
    rated_count = 0
    incomplete_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

            is_rated = rec.get("rated", False)
            has_scores = all(rec.get(h_key) is not None for _, h_key, _ in DIMENSION_MAPPING)

            if is_rated and has_scores:
                rated_count += 1
            else:
                incomplete_count += 1

    stats = {
        "total_records": len(records),
        "rated_count": rated_count,
        "incomplete_count": incomplete_count,
    }

    is_complete = len(records) > 0 and incomplete_count == 0
    return is_complete, records, stats


def compute_dimension_metrics(
    human_scores: List[float], judge_scores: List[float]
) -> Dict[str, Any]:
    """Compute agreement metrics between human and judge rating arrays."""
    n = len(human_scores)
    if n == 0:
        return {}

    mean_human = sum(human_scores) / n
    mean_judge = sum(judge_scores) / n

    differences = [abs(h - j) for h, j in zip(human_scores, judge_scores)]
    mae = sum(differences) / n

    exact_matches = sum(1 for d in differences if d == 0)
    exact_agreement = exact_matches / n

    within_one = sum(1 for d in differences if d <= 1)
    within_one_agreement = within_one / n

    # Spearman rank correlation
    spearman_corr, p_value = scipy.stats.spearmanr(human_scores, judge_scores)
    if math.isnan(spearman_corr):
        spearman_corr = 0.0
        p_value = 1.0

    return {
        "mean_human_score": round(mean_human, 2),
        "mean_judge_score": round(mean_judge, 2),
        "mean_absolute_error": round(mae, 2),
        "exact_agreement_rate": round(exact_agreement, 4),
        "within_one_point_agreement_rate": round(within_one_agreement, 4),
        "spearman_correlation": round(float(spearman_corr), 4),
        "spearman_p_value": round(float(p_value), 6),
    }


def evaluate_human_judge_agreement(
    ratings_path: Path = DEFAULT_RATINGS_PATH,
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
) -> Optional[Dict[str, Any]]:
    """Execute agreement comparison between human ratings and LLM Judge scores."""
    is_complete, records, stats = load_and_verify_human_ratings(ratings_path)

    if not is_complete:
        print("\n=== HUMAN-VS-LLM JUDGE EVALUATION ===")
        print("Status: HUMAN EVALUATION INCOMPLETE — metrics not computed.")
        print(f"File:                 {ratings_path}")
        print(f"Total Records:        {stats.get('total_records', 0)}")
        print(f"Fully Rated:          {stats.get('rated_count', 0)}")
        print(f"Pending/Unrated:      {stats.get('incomplete_count', 0)}")
        print("\nTo compute genuine agreement metrics:")
        print("1. Open data/evaluation/human_response_ratings_template.jsonl")
        print(
            "2. Supply human ratings (1-5) across Groundedness, Relevance, Brand Voice, and Safety."
        )
        print("3. Set 'rated': true for completed records.")
        print("4. Re-run this evaluation script.\n")

        if output_path:
            pending_artifact = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "PENDING_HUMAN_ANNOTATION",
                "sample_count": stats.get("total_records", 0),
                "rated_count": stats.get("rated_count", 0),
                "methodology": "Double evaluation: 100 generated responses rated independently by human evaluator on 1-5 scale against deterministic LLM Judge rubric.",
                "dimensions": [
                    "groundedness",
                    "answer_relevance",
                    "brand_tone",
                    "safety_compliance",
                ],
                "metrics": None,
                "limitations": "Human evaluation is currently pending manual annotation. No synthetic scores have been manufactured.",
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pending_artifact, f, indent=2)

        return None

    # Proceed with full evaluation when human ratings exist
    logger.info(f"Computing agreement metrics across {len(records)} human-rated responses...")
    orchestrator = AgentOrchestrator()
    judge = LLMJudgeService()

    paired_scores: Dict[str, Dict[str, List[float]]] = {
        dim: {"human": [], "judge": []} for dim, _, _ in DIMENSION_MAPPING
    }

    all_human_flat: List[float] = []
    all_judge_flat: List[float] = []

    for rec in records:
        query = rec["customer_message"]
        draft_reply = rec.get("generated_response", "")

        # Retrieve evidence and judge response
        run_res = orchestrator.run(
            customer_message=query,
            brand="AppleSupport",
            provider="grounded_precedent",
        )
        judge_eval = judge.evaluate_response(
            customer_message=query,
            draft_reply=draft_reply or run_res.generation.draft_reply,
            evidence=run_res.retrieval.evidence,
            routing_decision=run_res.routing.decision.value,
        )

        for dim, human_key, judge_key in DIMENSION_MAPPING:
            h_val = float(rec[human_key])
            j_val = float(judge_eval[judge_key])
            paired_scores[dim]["human"].append(h_val)
            paired_scores[dim]["judge"].append(j_val)
            all_human_flat.append(h_val)
            all_judge_flat.append(j_val)

    dimension_results = {}
    for dim, _, _ in DIMENSION_MAPPING:
        dimension_results[dim] = compute_dimension_metrics(
            paired_scores[dim]["human"], paired_scores[dim]["judge"]
        )

    overall_spearman, overall_p = scipy.stats.spearmanr(all_human_flat, all_judge_flat)
    if math.isnan(overall_spearman):
        overall_spearman = 0.0
        overall_p = 1.0

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "sample_count": len(records),
        "methodology": "Independent human ratings vs deterministic LLM Judge rubric across 100 responses.",
        "dimension_metrics": dimension_results,
        "overall_correlation": {
            "spearman_correlation": round(float(overall_spearman), 4),
            "p_value": round(float(overall_p), 6),
        },
        "mean_overall_human_score": round(sum(all_human_flat) / len(all_human_flat), 2),
        "mean_overall_judge_score": round(sum(all_judge_flat) / len(all_judge_flat), 2),
        "overall_mean_absolute_error": round(
            sum(abs(h - j) for h, j in zip(all_human_flat, all_judge_flat)) / len(all_human_flat),
            2,
        ),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Human-vs-Judge agreement benchmarks saved to: {output_path}")

    print("\n=== HUMAN-VS-LLM JUDGE AGREEMENT REPORT ===")
    print(f"Evaluated Samples:               {len(records)} responses")
    print(f"Overall Human Mean Score:        {results['mean_overall_human_score']:.2f} / 5.00")
    print(f"Overall LLM Judge Mean Score:    {results['mean_overall_judge_score']:.2f} / 5.00")
    print(f"Overall Mean Absolute Error:     {results['overall_mean_absolute_error']:.2f} points")
    print(
        f"Overall Spearman Correlation:    {results['overall_correlation']['spearman_correlation']:.4f}"
    )
    print("\nPer-Dimension Breakdown:")
    for dim, metrics in results["dimension_metrics"].items():
        print(f"  [{dim.upper()}]:")
        print(f"    - Human Mean:                {metrics['mean_human_score']:.2f}")
        print(f"    - Judge Mean:                {metrics['mean_judge_score']:.2f}")
        print(f"    - MAE:                       {metrics['mean_absolute_error']:.2f}")
        print(f"    - Exact Agreement:           {metrics['exact_agreement_rate'] * 100:.1f}%")
        print(
            f"    - Within-1-Point Agreement:  {metrics['within_one_point_agreement_rate'] * 100:.1f}%"
        )
        print(f"    - Spearman rho:              {metrics['spearman_correlation']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agreement between human ratings and LLM Judge."
    )
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=DEFAULT_RATINGS_PATH,
        help=f"Path to human ratings file (default: {DEFAULT_RATINGS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to output benchmark JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()
    evaluate_human_judge_agreement(ratings_path=args.ratings_path, output_path=args.output)


if __name__ == "__main__":
    main()
