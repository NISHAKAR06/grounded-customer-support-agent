"""Evaluate Intent Classification baselines against independent human intent annotations.

Evaluates:
1. Baseline 1 (Majority Class Classifier)
2. Baseline 2 (TF-IDF + Logistic Regression)
3. Rule-based Domain Taxonomy Classifier

Against:
- Independently annotated human intent ground truth (200 cases)

IMPORTANT:
- Replaces the 100.0% heuristic self-evaluation artifact with empirical human-evaluated accuracy.
- If human annotations are incomplete, cleanly reports:
  'HUMAN EVALUATION INCOMPLETE — intent evaluation pending.'
- Never manufactures human labels or metrics.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.intent.intent_classifier import IntentClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_human_intent")

DEFAULT_ANNOTATION_PATH = BASE_DIR / "data" / "golden" / "human_annotation_template.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "experiments" / "human_intent_benchmarks.json"
MODELS_DIR = BASE_DIR / "models" / "intent_classifier"

CLASSES = [
    "OPERATING_SYSTEM_UPDATES",
    "BATTERY_POWER_HARDWARE",
    "ACCOUNT_APPLE_ID",
    "CONNECTIVITY_NETWORKING",
    "AUDIO_ACCESSORIES",
    "SUBSCRIPTIONS_BILLING",
    "GENERAL_INQUIRY",
]


def load_and_verify_human_intents(
    path: Path,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """Load human annotation records and verify completeness of intent labels."""
    if not path.exists():
        return False, [], [], {"error": f"File not found: {path}"}

    texts: List[str] = []
    labels: List[str] = []
    annotated_count = 0
    incomplete_count = 0
    total_records = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            rec = json.loads(line)

            is_annotated = rec.get("annotated", False)
            intent = rec.get("human_intent")
            text = rec.get("customer_message", "")

            if is_annotated and intent in CLASSES:
                annotated_count += 1
                texts.append(text)
                labels.append(intent)
            else:
                incomplete_count += 1

    stats = {
        "total_records": total_records,
        "annotated_count": annotated_count,
        "incomplete_count": incomplete_count,
    }

    is_complete = total_records == 200 and incomplete_count == 0
    return is_complete, texts, labels, stats


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculate accuracy, macro F1, weighted F1, per-class metrics, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )

    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classes": CLASSES,
    }


def evaluate_human_intent(
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
) -> Optional[Dict[str, Any]]:
    """Execute evaluation comparing models against independent human intent ground truth."""
    is_complete, texts, labels, stats = load_and_verify_human_intents(annotation_path)

    if not is_complete:
        print("\n=== HUMAN INTENT CLASSIFICATION EVALUATION ===")
        print("Status: HUMAN EVALUATION INCOMPLETE — intent evaluation pending.")
        print(f"File:                 {annotation_path}")
        print(f"Total Records:        {stats.get('total_records', 0)} / 200")
        print(f"Fully Annotated:      {stats.get('annotated_count', 0)}")
        print(f"Pending/Unannotated:  {stats.get('incomplete_count', 0)}")
        print("\nTo evaluate intent models against human ground truth:")
        print("1. Open data/golden/human_annotation_template.jsonl")
        print("2. Supply human_intent (one of the 7 valid intent codes) for each example.")
        print("3. Set 'annotated': true for completed records.")
        print("4. Re-run this evaluation script.\n")

        if output_path:
            pending_artifact = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "PENDING_HUMAN_ANNOTATION",
                "sample_count": stats.get("total_records", 0),
                "annotated_count": stats.get("annotated_count", 0),
                "evaluation_name": "Human Intent Classification Benchmarks",
                "reference_comparison": "Replaces rule-based 100% heuristic self-evaluation artifact with independent human labels.",
                "classes": CLASSES,
                "metrics": None,
                "limitations": "Human intent evaluation is pending manual annotation. Existing published 100% rule-taxonomy score on the golden set is a heuristic artifact.",
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pending_artifact, f, indent=2)

        return None

    # Load baseline model artifacts
    logger.info(f"Loading trained models from {MODELS_DIR}...")
    majority_model = joblib.load(MODELS_DIR / "majority_baseline.joblib")
    tfidf_vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_clf = joblib.load(MODELS_DIR / "tfidf_logreg_model.joblib")
    rule_classifier = IntentClassifier()

    logger.info(f"Generating model predictions across {len(texts)} human-labelled queries...")
    maj_raw = majority_model.predict(texts)
    maj_preds = maj_raw.tolist() if hasattr(maj_raw, "tolist") else list(maj_raw)
    X_vec = tfidf_vectorizer.transform(texts)
    lr_raw = tfidf_clf.predict(X_vec)
    lr_preds = lr_raw.tolist() if hasattr(lr_raw, "tolist") else list(lr_raw)
    rule_preds = [
        c.code.value if hasattr(c.code, "value") else str(c.code)
        for c in [rule_classifier.classify(t) for t in texts]
    ]

    maj_metrics = compute_classification_metrics(labels, maj_preds)
    lr_metrics = compute_classification_metrics(labels, lr_preds)
    rule_metrics = compute_classification_metrics(labels, rule_preds)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "evaluation_name": "Human Intent Classification Benchmarks",
        "dataset": f"Independent Human-Annotated Golden Set ({len(texts)} samples)",
        "classes": CLASSES,
        "models": {
            "majority_baseline": maj_metrics,
            "tfidf_logistic_regression": lr_metrics,
            "rule_based_taxonomy": rule_metrics,
        },
        "note": "Evaluated against independent human annotations. The rule-based taxonomy accuracy here represents real generalization without circular heuristic evaluation.",
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Human intent benchmarks saved to: {output_path}")

    print("\n=== HUMAN INTENT CLASSIFICATION BENCHMARKS ===")
    print(f"Evaluated Samples:               {len(texts)} Human-Annotated Queries")
    print("\nOverall Summary:")
    print(f"  {'Model':<30} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print(f"  {'-'*66}")
    print(
        f"  {'1. Majority Baseline':<30} {maj_metrics['accuracy']*100:>6.2f}%      {maj_metrics['macro_f1']:>8.4f}     {maj_metrics['weighted_f1']:>8.4f}"
    )
    print(
        f"  {'2. TF-IDF + LogReg':<30} {lr_metrics['accuracy']*100:>6.2f}%      {lr_metrics['macro_f1']:>8.4f}     {lr_metrics['weighted_f1']:>8.4f}"
    )
    print(
        f"  {'3. Rule-Based Taxonomy':<30} {rule_metrics['accuracy']*100:>6.2f}%      {rule_metrics['macro_f1']:>8.4f}     {rule_metrics['weighted_f1']:>8.4f}"
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate intent models against human annotations."
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
    evaluate_human_intent(annotation_path=args.annotation_path, output_path=args.output)


if __name__ == "__main__":
    main()
