"""Empirical evaluation harness for Grounded Generation & Validation Pipeline across Golden Set."""

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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate_generation")


def evaluate_grounded_generation() -> Dict[str, Any]:
    """Run end-to-end grounded generation and validation evaluation over the 200 Golden Set cases."""
    settings = get_settings()
    golden_path = settings.DATA_DIR / "golden" / "golden_set.jsonl"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden Set not found at: {golden_path}")

    orchestrator = AgentOrchestrator()
    samples = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(
        f"Loaded {len(samples)} golden set samples for grounded generation evaluation."
    )

    results = []
    barrier_counts = {
        "non_empty_check": 0,
        "grounding_evidence_present": 0,
        "unsupported_claim_check": 0,
        "url_whitelist_check": 0,
        "pii_security_check": 0,
        "hazardous_safety_check": 0,
    }
    routing_counts = {
        "AUTO_HANDLE": 0,
        "HUMAN_ESCALATION": 0,
    }
    grounding_scores = []
    latencies = {
        "intent": [],
        "retrieval": [],
        "generation": [],
        "validation": [],
        "total": [],
    }

    t_start = time.time()
    for idx, sample in enumerate(samples, 1):
        query = sample["customer_message"]
        expected_routing = sample.get("expected_routing")

        run_res = orchestrator.run(
            customer_message=query,
            brand="AppleSupport",
            provider=settings.LLM_PROVIDER,
        )

        validation = run_res.validation
        for check_name, passed in validation.checks.items():
            if check_name in barrier_counts and passed:
                barrier_counts[check_name] += 1

        routing_str = run_res.routing.decision.value
        if routing_str in routing_counts:
            routing_counts[routing_str] += 1

        grounding_scores.append(validation.grounding_score)
        latencies["intent"].append(run_res.latency_ms.intent_ms)
        latencies["retrieval"].append(run_res.latency_ms.retrieval_ms)
        latencies["generation"].append(run_res.latency_ms.generation_ms)
        latencies["validation"].append(run_res.latency_ms.validation_ms)
        latencies["total"].append(run_res.latency_ms.total_ms)

        results.append(
            {
                "conversation_id": sample["conversation_id"],
                "customer_message": query,
                "intent": run_res.intent.name,
                "draft_reply": run_res.generation.draft_reply,
                "all_passed": validation.all_passed,
                "checks": validation.checks,
                "warnings": validation.warnings,
                "grounding_score": validation.grounding_score,
                "routing_decision": routing_str,
                "expected_routing": expected_routing,
                "reasons": run_res.routing.reasons,
            }
        )

        if idx % 50 == 0 or idx == len(samples):
            logger.info(f"Evaluated {idx}/{len(samples)} samples...")

    total_time = round(time.time() - t_start, 2)
    total_samples = len(samples)
    all_passed_count = sum(1 for r in results if r["all_passed"])

    benchmarks = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": total_samples,
        "elapsed_seconds": total_time,
        "validation_pass_rate": round(all_passed_count / total_samples, 4),
        "barrier_pass_rates": {
            k: round(v / total_samples, 4) for k, v in barrier_counts.items()
        },
        "routing_distribution": {
            "auto_handle_count": routing_counts["AUTO_HANDLE"],
            "auto_handle_pct": round(routing_counts["AUTO_HANDLE"] / total_samples, 4),
            "human_escalation_count": routing_counts["HUMAN_ESCALATION"],
            "human_escalation_pct": round(
                routing_counts["HUMAN_ESCALATION"] / total_samples, 4
            ),
        },
        "mean_grounding_fidelity_score": round(
            sum(grounding_scores) / len(grounding_scores), 4
        ),
        "mean_latencies_ms": {
            k: round(sum(v) / len(v), 2) for k, v in latencies.items()
        },
    }

    out_path = settings.BASE_DIR / "experiments" / "generation_benchmarks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    logger.info(f"Generation benchmarks saved to: {out_path}")
    print("\n=== GROUNDED GENERATION & VALIDATION BENCHMARKS SUMMARY ===")
    print(f"Total Evaluated:              {total_samples} Golden Samples")
    print(
        f"Validation Pass Rate:         {benchmarks['validation_pass_rate'] * 100:.2f}%"
    )
    print(
        f"Mean Grounding Overlap Score: {benchmarks['mean_grounding_fidelity_score']:.4f}"
    )
    print(
        f"Auto-Handle Rate:             {benchmarks['routing_distribution']['auto_handle_pct'] * 100:.1f}% ({benchmarks['routing_distribution']['auto_handle_count']})"
    )
    print(
        f"Human Escalation Rate:        {benchmarks['routing_distribution']['human_escalation_pct'] * 100:.1f}% ({benchmarks['routing_distribution']['human_escalation_count']})"
    )
    print("\nBarrier Integrity Rates:")
    for barrier, rate in benchmarks["barrier_pass_rates"].items():
        print(f"  - {barrier:<28}: {rate * 100:.1f}%")
    print("\nMean Latency Breakdown:")
    for stage, ms in benchmarks["mean_latencies_ms"].items():
        print(f"  - {stage.capitalize():<12}: {ms:.1f} ms")

    return benchmarks


if __name__ == "__main__":
    evaluate_grounded_generation()
