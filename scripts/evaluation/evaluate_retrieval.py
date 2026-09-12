"""Empirical retrieval evaluation harness for historical case retrieval.

Evaluates dense MiniLM + FAISS vector store on:
1. Held-Out Test Split (data/splits/test.jsonl - 484 real dialogues)
2. Golden Evaluation Set (data/golden/golden_set.jsonl - 200 real dialogues)

Zero-Leakage Assurance:
The index contains ONLY cases from data/splits/train.jsonl.
Evaluations are run strictly against held-out test and golden set queries.

Metrics:
- Intent Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Mean Top-1 Cosine Similarity
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.retrieval.retriever import Retriever  # noqa: E402

TEST_SPLIT_PATH = BASE_DIR / "data" / "splits" / "test.jsonl"
GOLDEN_SET_PATH = BASE_DIR / "data" / "golden" / "golden_set.jsonl"
OUTPUT_PATH = BASE_DIR / "experiments" / "retrieval_benchmarks.json"


def load_queries_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load query conversations from jsonl."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            customer_text = (
                item.get("first_inquiry")
                or item.get("customer_text")
                or item.get("customer_message")
                or ""
            ).strip()
            intent = (
                item.get("intent_code")
                or item.get("gold_intent_code")
                or item.get("intent")
                or "GENERAL_INQUIRY"
            )
            if customer_text:
                records.append(
                    {
                        "case_id": item.get("conversation_id") or item.get("sample_id", ""),
                        "query": customer_text,
                        "ground_truth_intent": intent,
                    }
                )
    return records


def evaluate_retrieval_on_dataset(
    retriever: Retriever,
    queries: List[Dict[str, Any]],
    dataset_name: str,
    top_k_max: int = 5,
    use_intent_filter: bool = False,
) -> Dict[str, Any]:
    """Compute Recall@1, 3, 5, MRR, and Mean Top-1 Similarity for a dataset."""
    mode_label = "Intent-Conditioned" if use_intent_filter else "Unconditioned"
    logger.info(
        "Evaluating [%s] retrieval on '%s' (%d queries)...",
        mode_label,
        dataset_name,
        len(queries),
    )

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    top1_similarities = []

    per_intent_stats: Dict[str, Dict[str, int]] = {}

    for i, item in enumerate(queries):
        query = item["query"]
        target_intent = item["ground_truth_intent"]

        if target_intent not in per_intent_stats:
            per_intent_stats[target_intent] = {
                "total": 0,
                "hits_at_1": 0,
                "hits_at_3": 0,
                "hits_at_5": 0,
            }
        per_intent_stats[target_intent]["total"] += 1

        filter_arg = target_intent if use_intent_filter else None
        retrieved = retriever.retrieve(query, top_k=top_k_max, intent_filter=filter_arg)

        if not retrieved:
            reciprocal_ranks.append(0.0)
            continue

        top1_sim = retrieved[0].similarity
        top1_similarities.append(top1_sim)

        # Check hits
        first_match_rank = 0
        for rank, case in enumerate(retrieved, start=1):
            case_intent = case.metadata.get("intent", "")
            if case_intent == target_intent:
                first_match_rank = rank
                break

        if first_match_rank == 1:
            hits_at_1 += 1
            per_intent_stats[target_intent]["hits_at_1"] += 1
        if 1 <= first_match_rank <= 3:
            hits_at_3 += 1
            per_intent_stats[target_intent]["hits_at_3"] += 1
        if 1 <= first_match_rank <= 5:
            hits_at_5 += 1
            per_intent_stats[target_intent]["hits_at_5"] += 1

        rr = (1.0 / first_match_rank) if first_match_rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        if (i + 1) % 100 == 0:
            logger.info(
                "Processed %d/%d queries for %s [%s]",
                i + 1,
                len(queries),
                dataset_name,
                mode_label,
            )

    n = len(queries)
    recall_1 = round(hits_at_1 / n, 4) if n else 0.0
    recall_3 = round(hits_at_3 / n, 4) if n else 0.0
    recall_5 = round(hits_at_5 / n, 4) if n else 0.0
    mrr = round(sum(reciprocal_ranks) / n, 4) if n else 0.0
    mean_top1_sim = (
        round(sum(top1_similarities) / len(top1_similarities), 4) if top1_similarities else 0.0
    )

    intent_breakdown = {}
    for intent, stats in per_intent_stats.items():
        tot = stats["total"]
        intent_breakdown[intent] = {
            "total_queries": tot,
            "recall_at_1": round(stats["hits_at_1"] / tot, 4) if tot else 0.0,
            "recall_at_3": round(stats["hits_at_3"] / tot, 4) if tot else 0.0,
            "recall_at_5": round(stats["hits_at_5"] / tot, 4) if tot else 0.0,
        }

    return {
        "dataset": dataset_name,
        "mode": mode_label,
        "query_count": n,
        "recall_at_1": recall_1,
        "recall_at_3": recall_3,
        "recall_at_5": recall_5,
        "mrr": mrr,
        "mean_top1_similarity": mean_top1_sim,
        "intent_breakdown": intent_breakdown,
    }


def main() -> None:
    retriever = Retriever()

    test_queries = load_queries_from_jsonl(TEST_SPLIT_PATH)
    golden_queries = load_queries_from_jsonl(GOLDEN_SET_PATH)

    test_metrics_raw = evaluate_retrieval_on_dataset(
        retriever,
        test_queries,
        "Held-Out Test Split (484 samples)",
        use_intent_filter=False,
    )
    test_metrics_conditioned = evaluate_retrieval_on_dataset(
        retriever,
        test_queries,
        "Held-Out Test Split (484 samples)",
        use_intent_filter=True,
    )

    golden_metrics_raw = evaluate_retrieval_on_dataset(
        retriever,
        golden_queries,
        "Golden Evaluation Set (200 samples)",
        use_intent_filter=False,
    )
    golden_metrics_conditioned = evaluate_retrieval_on_dataset(
        retriever,
        golden_queries,
        "Golden Evaluation Set (200 samples)",
        use_intent_filter=True,
    )

    results = {
        "model": retriever.model_name,
        "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
        "source_index_size": 2245,
        "benchmarks": {
            "test_split_unconditioned": test_metrics_raw,
            "test_split_intent_conditioned": test_metrics_conditioned,
            "golden_set_unconditioned": golden_metrics_raw,
            "golden_set_intent_conditioned": golden_metrics_conditioned,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Retrieval benchmarks saved to %s", OUTPUT_PATH)
    print("\n=== RETRIEVAL BENCHMARKS SUMMARY ===")
    for key, bm in results["benchmarks"].items():
        print(f"\n--- {bm['dataset']} [{bm['mode']}] ---")
        print(f"Recall@1: {bm['recall_at_1'] * 100:.2f}%")
        print(f"Recall@3: {bm['recall_at_3'] * 100:.2f}%")
        print(f"Recall@5: {bm['recall_at_5'] * 100:.2f}%")
        print(f"MRR:      {bm['mrr']:.4f}")
        print(f"Avg Sim:  {bm['mean_top1_similarity']:.4f}")


if __name__ == "__main__":
    main()
