"""Train Baseline Intent Classifiers.

Trains two classical baselines:
1. Baseline 1 (Majority Classifier): Unconditional statistical mode floor.
2. Baseline 2 (TF-IDF + Logistic Regression): Balanced regularized linear model.

Saves trained artifacts to models/intent_classifier/.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.intent.baseline_classifiers import MajorityClassifier  # noqa: E402

TRAIN_PATH = BASE_DIR / "data" / "splits" / "train.jsonl"
MODELS_DIR = BASE_DIR / "models" / "intent_classifier"


def load_training_data() -> Tuple[List[str], List[str]]:
    """Load text and intent labels from train.jsonl."""
    texts = []
    labels = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            inquiry = r.get("first_inquiry", "").strip()
            intent = r.get("intent_code", "GENERAL_INQUIRY")
            if inquiry:
                texts.append(inquiry)
                labels.append(intent)
    return texts, labels


def train():
    print(f"Loading training data from: {TRAIN_PATH}")
    texts, labels = load_training_data()
    print(f"Loaded {len(texts)} training samples.")
    label_dist = Counter(labels)
    for k, v in label_dist.most_common():
        print(f"  {k}: {v} ({v / len(labels) * 100:.1f}%)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train Majority Baseline
    print("\n--- Training Baseline 1: Majority Class Classifier ---")
    majority_model = MajorityClassifier()
    majority_model.fit(labels)
    majority_path = MODELS_DIR / "majority_baseline.joblib"
    joblib.dump(majority_model, majority_path)
    print(
        f"Majority class: {majority_model.majority_class} ({majority_model.majority_ratio * 100:.2f}%)"
    )
    print(f"Saved majority baseline to: {majority_path}")

    # 2. Train TF-IDF + Logistic Regression
    print("\n--- Training Baseline 2: TF-IDF + Balanced Logistic Regression ---")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(texts)
    print(f"Extracted TF-IDF features matrix shape: {X_train.shape}")

    clf = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train, labels)
    train_acc = clf.score(X_train, labels)
    print(f"Logistic Regression Training Accuracy: {train_acc * 100:.2f}%")

    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    model_path = MODELS_DIR / "tfidf_logreg_model.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, model_path)
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved classifier to: {model_path}")

    print("\nBaseline model training finished successfully.")


if __name__ == "__main__":
    train()
