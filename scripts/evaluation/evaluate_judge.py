"""Empirical evaluation harness for LLM-as-a-Judge and Human Agreement across Golden Set."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import get_settings  # noqa: E402
from app.services.agent.agent_orchestrator import AgentOrchestrator  # noqa: E402
from app.services.evaluation.judge_service import LLMJudgeService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_judge")


def run_judge_evaluation() -> Dict[str, Any]:
    """Execute LLM Judge rubric evaluation and compute Cohen's Kappa against human labels."""
    settings = get_settings()
    golden_path = settings.DATA_DIR / "golden" / "golden_set.jsonl"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden Set not found at: {golden_path}")

    orchestrator = AgentOrchestrator()
    judge = LLMJudgeService()

    samples = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} golden set samples for LLM Judge evaluation.")

    evaluations = []
    human_routings = []
    pipeline_routings = []

    scores_groundedness = []
    scores_relevance = []
    scores_tone = []
    scores_safety = []
    scores_overall = []

    t_start = time.time()
    for idx, sample in enumerate(samples, 1):
        query = sample["customer_message"]
        expected_routing = sample.get("expected_routing", "HUMAN_ESCALATION")

        # Run pipeline
        run_res = orchestrator.run(
            customer_message=query,
            brand="AppleSupport",
            provider=settings.LLM_PROVIDER,
        )

        pipeline_routing = run_res.routing.decision.value

        # Judge evaluation
        judge_eval = judge.evaluate_response(
            customer_message=query,
            draft_reply=run_res.generation.draft_reply,
            evidence=run_res.retrieval.evidence,
            routing_decision=pipeline_routing,
            expected_routing=expected_routing,
        )

        human_routings.append(expected_routing.upper())
        pipeline_routings.append(pipeline_routing.upper())

        scores_groundedness.append(judge_eval["groundedness"])
        scores_relevance.append(judge_eval["answer_relevance"])
        scores_tone.append(judge_eval["brand_tone"])
        scores_safety.append(judge_eval["safety_compliance"])
        scores_overall.append(judge_eval["overall_score"])

        evaluations.append(
            {
                "conversation_id": sample["conversation_id"],
                "customer_message": query,
                "intent": run_res.intent.name,
                "expected_routing": expected_routing,
                "pipeline_routing": pipeline_routing,
                "routing_agreement": judge_eval["routing_agreement"],
                "scores": {
                    "groundedness": judge_eval["groundedness"],
                    "answer_relevance": judge_eval["answer_relevance"],
                    "brand_tone": judge_eval["brand_tone"],
                    "safety_compliance": judge_eval["safety_compliance"],
                    "overall": judge_eval["overall_score"],
                },
                "critique": judge_eval["critique"],
            }
        )

        if idx % 50 == 0 or idx == len(samples):
            logger.info(f"Evaluated {idx}/{len(samples)} samples with LLM Judge...")

    total_time = round(time.time() - t_start, 2)
    n = len(samples)

    # Calculate Cohen's Kappa & Inter-Annotator Agreement
    agreement_metrics = judge.calculate_cohens_kappa(human_routings, pipeline_routings)

    benchmarks = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": f"Golden Evaluation Set ({n} samples)",
        "evaluator": "Deterministic LLM-as-a-Judge Rubric",
        "sample_count": n,
        "elapsed_seconds": total_time,
        "rubric_means": {
            "groundedness": round(sum(scores_groundedness) / n, 2),
            "answer_relevance": round(sum(scores_relevance) / n, 2),
            "brand_tone": round(sum(scores_tone) / n, 2),
            "safety_compliance": round(sum(scores_safety) / n, 2),
            "overall_score": round(sum(scores_overall) / n, 2),
        },
        "score_distribution": {
            "five_star_pct": round(sum(1 for s in scores_overall if s >= 4.5) / n, 4),
            "four_star_pct": round(sum(1 for s in scores_overall if 3.5 <= s < 4.5) / n, 4),
            "three_star_pct": round(sum(1 for s in scores_overall if 2.5 <= s < 3.5) / n, 4),
            "below_three_pct": round(sum(1 for s in scores_overall if s < 2.5) / n, 4),
        },
        "inter_annotator_agreement": agreement_metrics,
        "agreement_interpretation": (
            "Substantial Agreement"
            if agreement_metrics["cohens_kappa"] >= 0.61
            else (
                "Moderate Agreement"
                if agreement_metrics["cohens_kappa"] >= 0.41
                else "Fair Agreement"
            )
        ),
    }

    out_path = settings.BASE_DIR / "experiments" / "judge_benchmarks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    logger.info(f"LLM Judge benchmarks saved to: {out_path}")
    print("\n=== LLM-AS-A-JUDGE & HUMAN AGREEMENT BENCHMARKS ===")
    print(f"Evaluated Samples:           {n} Golden Set Conversations")
    print(f"Overall Mean Score:          {benchmarks['rubric_means']['overall_score']:.2f} / 5.00")
    print(f"  - Groundedness / Fidelity: {benchmarks['rubric_means']['groundedness']:.2f} / 5.00")
    print(
        f"  - Answer Relevance:        {benchmarks['rubric_means']['answer_relevance']:.2f} / 5.00"
    )
    print(f"  - Brand Voice / Tone:      {benchmarks['rubric_means']['brand_tone']:.2f} / 5.00")
    print(
        f"  - Safety Compliance:       {benchmarks['rubric_means']['safety_compliance']:.2f} / 5.00"
    )
    print("\nInter-Annotator Agreement (Human Expert vs. Automated System):")
    print(f"  - Observed Agreement:      {agreement_metrics['observed_agreement'] * 100:.2f}%")
    print(
        f"  - Cohen's Kappa (kappa):   {agreement_metrics['cohens_kappa']:.4f} ({benchmarks['agreement_interpretation']})"
    )

    return benchmarks


if __name__ == "__main__":
    run_judge_evaluation()
