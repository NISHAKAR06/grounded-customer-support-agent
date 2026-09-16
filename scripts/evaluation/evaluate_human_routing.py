"""Evaluate agreement between human routing decisions and the automated system routing.

Computes:
1. Observed Agreement rate
2. 2x2 Confusion Matrix
3. Precision, Recall, and F1 for HUMAN_ESCALATION and AUTO_HANDLE
4. Cohen's Kappa coefficient (Human vs. System Routing Agreement)

IMPORTANT:
- Distinguishes 'Human vs System Routing Agreement' from 'Heuristic Policy Concordance'.
- If human annotations are incomplete, cleanly reports:
  'HUMAN EVALUATION INCOMPLETE — routing evaluation pending.'
- Never manufactures fake routing labels or metrics.
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.agent.agent_orchestrator import AgentOrchestrator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_human_routing")

DEFAULT_ANNOTATION_PATH = BASE_DIR / "data" / "golden" / "human_annotation_template.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "experiments" / "human_routing_benchmarks.json"

ROUTING_CLASSES = ["AUTO_HANDLE", "HUMAN_ESCALATION"]


def load_and_verify_human_routings(
    path: Path,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """Load human annotation records and verify completeness of routing decisions."""
    if not path.exists():
        return False, [], {"error": f"File not found: {path}"}

    records: List[Dict[str, Any]] = []
    annotated_count = 0
    incomplete_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

            is_annotated = rec.get("annotated", False)
            routing = rec.get("human_routing")

            if is_annotated and routing in ROUTING_CLASSES:
                annotated_count += 1
            else:
                incomplete_count += 1

    stats = {
        "total_records": len(records),
        "annotated_count": annotated_count,
        "incomplete_count": incomplete_count,
    }

    is_complete = len(records) == 200 and incomplete_count == 0
    return is_complete, records, stats


def calculate_routing_metrics(human_labels: List[str], system_labels: List[str]) -> Dict[str, Any]:
    """Calculate confusion matrix, precision/recall/F1, and Cohen's Kappa."""
    n = len(human_labels)
    if n == 0:
        return {}

    # Build confusion matrix: matrix[human_class][system_class]
    cm = {h: {s: 0 for s in ROUTING_CLASSES} for h in ROUTING_CLASSES}
    for h, s in zip(human_labels, system_labels):
        if h in cm and s in cm[h]:
            cm[h][s] += 1

    # Observed agreement
    observed_matches = sum(cm[cls][cls] for cls in ROUTING_CLASSES)
    p_o = observed_matches / n

    # Expected agreement by chance
    p_e = 0.0
    for cls in ROUTING_CLASSES:
        row_sum = sum(cm[cls][s] for s in ROUTING_CLASSES)  # Total times human chose cls
        col_sum = sum(cm[h][cls] for h in ROUTING_CLASSES)  # Total times system chose cls
        p_e += (row_sum * col_sum) / (n * n)

    if math.isclose(1.0, p_e):
        kappa = 1.0 if math.isclose(p_o, 1.0) else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    # Per-class precision, recall, f1
    per_class = {}
    for cls in ROUTING_CLASSES:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in ROUTING_CLASSES if other != cls)
        fn = sum(cm[cls][other] for other in ROUTING_CLASSES if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return {
        "total_evaluated": n,
        "observed_agreement": round(p_o, 4),
        "expected_agreement": round(p_e, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": (
            "Almost Perfect"
            if kappa >= 0.81
            else ("Substantial" if kappa >= 0.61 else ("Moderate" if kappa >= 0.41 else "Fair"))
        ),
        "confusion_matrix": cm,
        "per_class_metrics": per_class,
    }


def evaluate_human_routing(
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
) -> Optional[Dict[str, Any]]:
    """Execute evaluation comparing human routing labels against system routing."""
    is_complete, records, stats = load_and_verify_human_routings(annotation_path)

    if not is_complete:
        print("\n=== HUMAN VS SYSTEM ROUTING AGREEMENT ===")
        print("Status: HUMAN EVALUATION INCOMPLETE — routing evaluation pending.")
        print(f"File:                 {annotation_path}")
        print(f"Total Records:        {stats.get('total_records', 0)} / 200")
        print(f"Fully Annotated:      {stats.get('annotated_count', 0)}")
        print(f"Pending/Unannotated:  {stats.get('incomplete_count', 0)}")
        print("\nTo evaluate human-vs-system routing agreement:")
        print("1. Open data/golden/human_annotation_template.jsonl")
        print("2. Supply human_routing ('AUTO_HANDLE' or 'HUMAN_ESCALATION') for each example.")
        print("3. Set 'annotated': true for completed records.")
        print("4. Re-run this evaluation script.\n")

        if output_path:
            pending_artifact = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "PENDING_HUMAN_ANNOTATION",
                "sample_count": stats.get("total_records", 0),
                "annotated_count": stats.get("annotated_count", 0),
                "evaluation_name": "Human vs System Routing Agreement",
                "reference_comparison": "Contrasts against existing 'Heuristic Policy Concordance' (89.5% agreement, kappa 0.8118)",
                "classes": ROUTING_CLASSES,
                "metrics": None,
                "limitations": "Human routing evaluation is pending manual annotation. Existing published kappa represents heuristic policy concordance, not human agreement.",
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pending_artifact, f, indent=2)

        return None

    # Run system pipeline on the 200 records
    logger.info(
        f"Evaluating system routing decisions across {len(records)} human-annotated cases..."
    )
    orchestrator = AgentOrchestrator()

    human_routings: List[str] = []
    system_routings: List[str] = []

    for rec in records:
        query = rec["customer_message"]
        h_routing = rec["human_routing"].upper()

        run_res = orchestrator.run(
            customer_message=query,
            brand="AppleSupport",
            provider="grounded_precedent",
        )
        s_routing = run_res.routing.decision.value.upper()

        human_routings.append(h_routing)
        system_routings.append(s_routing)

    metrics = calculate_routing_metrics(human_routings, system_routings)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "evaluation_name": "Human vs System Routing Agreement",
        "sample_count": len(records),
        "methodology": "Independent human routing labels vs automated policy decisions across 200 Golden Set cases.",
        "metrics": metrics,
        "note": "This metric represents true human-vs-system agreement. Distinct from heuristic policy concordance.",
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Human routing benchmarks saved to: {output_path}")

    print("\n=== HUMAN VS SYSTEM ROUTING AGREEMENT REPORT ===")
    print(f"Evaluated Samples:               {len(records)} Golden Set cases")
    print(f"Observed Agreement:              {metrics['observed_agreement'] * 100:.2f}%")
    print(
        f"Cohen's Kappa (kappa):           {metrics['cohens_kappa']:.4f} ({metrics['interpretation']})"
    )
    print("\nPer-Class Breakdown:")
    for cls, m in metrics["per_class_metrics"].items():
        print(
            f"  [{cls}]: Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f} (Support: {m['support']})"
        )
    print("\nConfusion Matrix (Rows: Human Label, Columns: System Decision):")
    print(f"  {'':<20} {'AUTO_HANDLE':<15} {'HUMAN_ESCALATION':<15}")
    for h_cls in ROUTING_CLASSES:
        row = [str(metrics["confusion_matrix"][h_cls][s_cls]) for s_cls in ROUTING_CLASSES]
        print(f"  {h_cls:<20} {row[0]:<15} {row[1]:<15}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate human routing decisions against system routing."
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=DEFAULT_ANNOTATION_PATH,
        help=f"Path to human annotations file (default: {DEFAULT_ANNOTATION_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to output benchmark JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()
    evaluate_human_routing(annotation_path=args.annotation_path, output_path=args.output)


if __name__ == "__main__":
    main()
