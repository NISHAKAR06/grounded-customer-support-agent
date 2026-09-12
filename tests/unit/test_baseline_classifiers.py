"""Unit tests for Phase 6 Baseline Classifiers and Evaluation Harness."""

import json

import joblib
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.evaluation.evaluation_service import EvaluationService
from app.services.intent.baseline_classifiers import MajorityClassifier

client = TestClient(app)
settings = get_settings()

MODELS_DIR = settings.BASE_DIR / "models" / "intent_classifier"
SPLITS_DIR = settings.BASE_DIR / "data" / "splits"
GOLDEN_PATH = settings.BASE_DIR / "data" / "golden" / "golden_set.jsonl"
BENCHMARKS_PATH = settings.BASE_DIR / "experiments" / "baseline_benchmarks.json"


def test_majority_baseline_loads_and_predicts():
    """Verify MajorityClassifier loads and makes valid predictions."""
    model_path = MODELS_DIR / "majority_baseline.joblib"
    assert model_path.exists(), f"Missing model artifact at {model_path}"

    model = joblib.load(model_path)
    assert isinstance(model, MajorityClassifier)
    assert model.majority_class == "GENERAL_INQUIRY"

    preds = model.predict(["My screen is broken", "Battery dies immediately"])
    assert len(preds) == 2
    assert preds[0] == "GENERAL_INQUIRY"


def test_tfidf_logreg_baseline_loads_and_predicts():
    """Verify TF-IDF vectorizer and Logistic Regression model load and predict."""
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    clf_path = MODELS_DIR / "tfidf_logreg_model.joblib"
    assert vec_path.exists(), f"Missing vectorizer artifact at {vec_path}"
    assert clf_path.exists(), f"Missing classifier artifact at {clf_path}"

    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)

    X = vectorizer.transform(
        [
            "My iPhone battery is draining so fast and running hot",
            "I need to update to iOS 11 software update",
        ]
    )
    preds = clf.predict(X)
    assert len(preds) == 2
    assert preds[0] in [
        "BATTERY_POWER_HARDWARE",
        "OPERATING_SYSTEM_UPDATES",
        "GENERAL_INQUIRY",
    ]


def test_train_val_test_splits_exist_and_leak_free():
    """Verify splits exist and have 0% overlap with the golden evaluation set."""
    golden_ids = set()
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_ids.add(json.loads(line).get("conversation_id"))

    assert len(golden_ids) == 200

    split_ids = set()
    for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = SPLITS_DIR / name
        assert path.exists(), f"Missing split file: {path}"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cid = json.loads(line).get("conversation_id")
                    assert (
                        cid not in golden_ids
                    ), f"Data leakage detected: {cid} in golden set and {name}"
                    assert cid not in split_ids, f"Duplicate conversation {cid} across splits"
                    split_ids.add(cid)

    assert len(split_ids) == 3223


def test_baseline_benchmark_results_structure():
    """Verify baseline benchmark results artifact is complete and valid."""
    assert BENCHMARKS_PATH.exists(), f"Missing benchmark results at {BENCHMARKS_PATH}"
    with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "held_out_test_split" in data
    assert "golden_evaluation_set" in data

    for partition in ["held_out_test_split", "golden_evaluation_set"]:
        assert "majority_baseline" in data[partition]
        assert "tfidf_logreg_baseline" in data[partition]
        assert "rule_based_taxonomy" in data[partition]

        maj = data[partition]["majority_baseline"]
        lr = data[partition]["tfidf_logreg_baseline"]

        assert 0.0 < maj["accuracy"] <= 1.0
        assert 0.0 < lr["accuracy"] <= 1.0
        assert 0.0 < lr["macro_f1"] <= 1.0


def test_evaluation_service_returns_completed_benchmarks():
    """Verify EvaluationService parses the real benchmarks without fallback."""
    service = EvaluationService()
    summary = service.get_benchmark_summary()

    assert summary["status"] == "completed"
    assert summary["golden_set_count"] == 200
    assert summary["test_set_count"] == 484

    models = summary["models"]
    assert "majority_baseline" in models
    assert "tfidf_logreg_baseline" in models
    assert "final_system" in models

    assert models["tfidf_logreg_baseline"]["accuracy"] > 0.70


def test_evaluation_page_renders_live_benchmarks():
    """Verify /evaluation HTTP route returns status 200 and renders benchmark tables."""
    res = client.get("/evaluation")
    assert res.status_code == 200
    html = res.text
    assert "Held-Out Test Split Benchmarks" in html
    assert "Golden Evaluation Set Benchmarks" in html
    assert "TF-IDF + Logistic Regression" in html
    assert "Empirical Benchmark Execution Complete" in html
