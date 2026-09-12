"""Comprehensive evaluation harness for Intent Classification baselines.

Evaluates:
1. Baseline 1 (Majority Class Classifier)
2. Baseline 2 (TF-IDF + Logistic Regression)
3. Rule-based Domain Taxonomy Classifier

Across both:
a) Held-out Test Split (data/splits/test.jsonl)
b) 200-sample Hand-verified Golden Set (data/golden/golden_set.jsonl)

Generates empirical metrics and outputs experiments/baseline_benchmarks.json.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

TEST_SPLIT_PATH = BASE_DIR / "data" / "splits" / "test.jsonl"
GOLDEN_SET_PATH = BASE_DIR / "data" / "golden" / "golden_set.jsonl"
MODELS_DIR = BASE_DIR / "models" / "intent_classifier"
OUTPUT_PATH = BASE_DIR / "experiments" / "baseline_benchmarks.json"

CLASSES = [
    "OPERATING_SYSTEM_UPDATES",
    "BATTERY_POWER_HARDWARE",
    "ACCOUNT_APPLE_ID",
    "CONNECTIVITY_NETWORKING",
    "AUDIO_ACCESSORIES",
    "SUBSCRIPTIONS_BILLING",
    "GENERAL_INQUIRY",
]


def load_dataset(path: Path, is_golden: bool = False) -> Tuple[List[str], List[str]]:
    """Load texts and true labels from a dataset file."""
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            inquiry = r.get("customer_message") if is_golden else r.get("first_inquiry")
            intent = r.get("gold_intent_code") if is_golden else r.get("intent_code")
            if inquiry and intent:
                texts.append(inquiry.strip())
                labels.append(intent.strip())
    return texts, labels


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculate accuracy, macro F1, per-class metrics, and confusion matrix."""
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


def run_evaluation():
    print("--- Starting Intent Model Benchmark Evaluation ---")

    # Load datasets
    print(f"Loading test split: {TEST_SPLIT_PATH}")
    test_texts, test_labels = load_dataset(TEST_SPLIT_PATH, is_golden=False)
    print(f"  Test samples: {len(test_texts)}")

    print(f"Loading golden set: {GOLDEN_SET_PATH}")
    golden_texts, golden_labels = load_dataset(GOLDEN_SET_PATH, is_golden=True)
    print(f"  Golden samples: {len(golden_texts)}")

    # Load models
    print("\nLoading models...")
    majority_model = joblib.load(MODELS_DIR / "majority_baseline.joblib")
    tfidf_vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_clf = joblib.load(MODELS_DIR / "tfidf_logreg_model.joblib")
    rule_classifier = IntentClassifier()

    # Generate predictions
    print("\nRunning inference...")
    # 1. Majority Model
    maj_pred_test = majority_model.predict(test_texts)
    maj_pred_golden = majority_model.predict(golden_texts)

    # 2. TF-IDF + LogReg Model
    X_test = tfidf_vectorizer.transform(test_texts)
    X_golden = tfidf_vectorizer.transform(golden_texts)
    lr_pred_test = tfidf_clf.predict(X_test).tolist()
    lr_pred_golden = tfidf_clf.predict(X_golden).tolist()

    rule_pred_test = [
        c.code.value if hasattr(c.code, "value") else str(c.code)
        for c in [rule_classifier.classify(t) for t in test_texts]
    ]
    rule_pred_golden = [
        c.code.value if hasattr(c.code, "value") else str(c.code)
        for c in [rule_classifier.classify(t) for t in golden_texts]
    ]

    # Evaluate
    results = {
        "metadata": {
            "test_split_samples": len(test_texts),
            "golden_set_samples": len(golden_texts),
            "num_classes": len(CLASSES),
            "classes": CLASSES,
        },
        "held_out_test_split": {
            "majority_baseline": evaluate_predictions(test_labels, maj_pred_test),
            "tfidf_logreg_baseline": evaluate_predictions(test_labels, lr_pred_test),
            "rule_based_taxonomy": evaluate_predictions(test_labels, rule_pred_test),
        },
        "golden_evaluation_set": {
            "majority_baseline": evaluate_predictions(golden_labels, maj_pred_golden),
            "tfidf_logreg_baseline": evaluate_predictions(golden_labels, lr_pred_golden),
            "rule_based_taxonomy": evaluate_predictions(golden_labels, rule_pred_golden),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print("\n[Held-out Test Split - 484 samples]")
    print(
        f"  Majority Baseline:        Acc: {results['held_out_test_split']['majority_baseline']['accuracy']*100:.2f}% | Macro-F1: {results['held_out_test_split']['majority_baseline']['macro_f1']*100:.2f}%"
    )
    print(
        f"  TF-IDF + LogReg:          Acc: {results['held_out_test_split']['tfidf_logreg_baseline']['accuracy']*100:.2f}% | Macro-F1: {results['held_out_test_split']['tfidf_logreg_baseline']['macro_f1']*100:.2f}%"
    )
    print(
        f"  Rule-Based Taxonomy:      Acc: {results['held_out_test_split']['rule_based_taxonomy']['accuracy']*100:.2f}% | Macro-F1: {results['held_out_test_split']['rule_based_taxonomy']['macro_f1']*100:.2f}%"
    )

    print("\n[Hand-Verified Golden Set - 200 samples]")
    print(
        f"  Majority Baseline:        Acc: {results['golden_evaluation_set']['majority_baseline']['accuracy']*100:.2f}% | Macro-F1: {results['golden_evaluation_set']['majority_baseline']['macro_f1']*100:.2f}%"
    )
    print(
        f"  TF-IDF + LogReg:          Acc: {results['golden_evaluation_set']['tfidf_logreg_baseline']['accuracy']*100:.2f}% | Macro-F1: {results['golden_evaluation_set']['tfidf_logreg_baseline']['macro_f1']*100:.2f}%"
    )
    print(
        f"  Rule-Based Taxonomy:      Acc: {results['golden_evaluation_set']['rule_based_taxonomy']['accuracy']*100:.2f}% | Macro-F1: {results['golden_evaluation_set']['rule_based_taxonomy']['macro_f1']*100:.2f}%"
    )

    print(f"\nEmpirical results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
