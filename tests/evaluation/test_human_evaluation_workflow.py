"""Unit tests for the independent human evaluation pipeline and validation tooling.

Verifies:
1. Human annotation schema validation.
2. Missing annotation detection.
3. Duplicate sample ID detection.
4. Invalid intent classification detection.
5. Invalid routing decision detection.
6. Human and LLM Judge ID matching.
7. Graceful incomplete human evaluation handling.
8. Agreement metric calculation math (MAE, exact, within-1-point, Spearman).
9. Routing confusion matrix and Cohen's Kappa calculations.
10. Intent baseline classification metric calculations.

All test inputs use explicitly synthetic fixtures from tests/fixtures/synthetic_human_evaluation_fixtures.py.
"""

import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_human_intent import compute_classification_metrics
from scripts.evaluation.evaluate_human_judge_agreement import (
    compute_dimension_metrics,
    load_and_verify_human_ratings,
)
from scripts.evaluation.evaluate_human_routing import (
    calculate_routing_metrics,
    load_and_verify_human_routings,
)
from scripts.evaluation.validate_human_annotations import validate_annotations
from tests.fixtures.synthetic_human_evaluation_fixtures import (
    get_synthetic_blank_annotations,
    get_synthetic_invalid_annotations,
    get_synthetic_valid_annotations,
)


@pytest.fixture
def temp_eval_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test files."""
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def test_human_annotation_schema_validation_on_blank_template(temp_eval_dir: Path):
    """Test that blank template passes validation when allow_pending=True."""
    blank_file = temp_eval_dir / "blank_template.jsonl"
    records = get_synthetic_blank_annotations(count=200)
    with open(blank_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, errors, stats = validate_annotations(
        annotation_path=blank_file,
        golden_path=Path("non_existent_file.jsonl"),
        allow_pending=True,
    )
    assert is_valid is True
    assert len(errors) == 0
    assert stats["total_records"] == 200
    assert stats["pending_count"] == 200
    assert stats["annotated_count"] == 0


def test_missing_annotation_detection(temp_eval_dir: Path):
    """Test that missing labels are flagged as errors when allow_pending=False."""
    incomplete_file = temp_eval_dir / "incomplete.jsonl"
    records = get_synthetic_blank_annotations(count=200)
    with open(incomplete_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, errors, stats = validate_annotations(
        annotation_path=incomplete_file,
        golden_path=Path("non_existent_file.jsonl"),
        allow_pending=False,
    )
    assert is_valid is False
    assert len(errors) > 0
    assert any("Missing required human labels" in err for err in errors)


def test_duplicate_id_detection(temp_eval_dir: Path):
    """Test that duplicate sample IDs are detected loudly."""
    dup_file = temp_eval_dir / "duplicates.jsonl"
    records = get_synthetic_valid_annotations(count=5)
    # Inject a duplicate
    records.append(records[0])
    with open(dup_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, errors, stats = validate_annotations(
        annotation_path=dup_file,
        golden_path=Path("non_existent.jsonl"),
        allow_pending=True,
    )
    assert is_valid is False
    assert any("Duplicate example_id found" in err for err in errors)
    assert len(stats["duplicate_ids"]) > 0


def test_invalid_intent_detection(temp_eval_dir: Path):
    """Test that invalid intent taxonomy labels are rejected."""
    bad_file = temp_eval_dir / "bad_intent.jsonl"
    records = get_synthetic_invalid_annotations()
    with open(bad_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, errors, stats = validate_annotations(
        annotation_path=bad_file,
        golden_path=Path("non_existent.jsonl"),
        allow_pending=True,
    )
    assert is_valid is False
    assert any("Invalid human_intent" in err for err in errors)


def test_invalid_routing_detection(temp_eval_dir: Path):
    """Test that routing values outside AUTO_HANDLE and HUMAN_ESCALATION are rejected."""
    bad_file = temp_eval_dir / "bad_routing.jsonl"
    records = get_synthetic_invalid_annotations()
    with open(bad_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, errors, stats = validate_annotations(
        annotation_path=bad_file,
        golden_path=Path("non_existent.jsonl"),
        allow_pending=True,
    )
    assert is_valid is False
    assert any("Invalid human_routing" in err for err in errors)


def test_incomplete_response_evaluation_detection(temp_eval_dir: Path):
    """Test that response evaluation cleanly flags incomplete or unrated templates."""
    unrated_file = temp_eval_dir / "unrated.jsonl"
    records = [
        {
            "example_id": "test_001",
            "customer_message": "test",
            "human_groundedness": None,
            "human_relevance": None,
            "human_brand_voice": None,
            "human_safety": None,
            "rated": False,
        }
    ]
    with open(unrated_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_complete, loaded_records, stats = load_and_verify_human_ratings(unrated_file)
    assert is_complete is False
    assert stats["incomplete_count"] == 1
    assert stats["rated_count"] == 0


def test_agreement_metric_calculations():
    """Test mathematical correctness of MAE, exact agreement, within-1-point, and Spearman."""
    human_scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    judge_scores = [5.0, 4.0, 2.0, 1.0, 1.0]

    metrics = compute_dimension_metrics(human_scores, judge_scores)
    assert metrics["mean_human_score"] == 3.0
    assert metrics["mean_judge_score"] == 2.6
    # Absolute differences: [0, 0, 1, 1, 0] -> sum = 2 -> MAE = 2/5 = 0.4
    assert metrics["mean_absolute_error"] == 0.4
    # Exact matches: 3 out of 5 = 0.60
    assert metrics["exact_agreement_rate"] == 0.60
    # Within 1 point: 5 out of 5 = 1.00
    assert metrics["within_one_point_agreement_rate"] == 1.00
    # Spearman rank correlation should be strongly positive
    assert metrics["spearman_correlation"] > 0.90


def test_human_routing_confusion_matrix_and_kappa():
    """Test computation of observed agreement, 2x2 confusion matrix, and Cohen's Kappa."""
    human_labels = [
        "AUTO_HANDLE",
        "AUTO_HANDLE",
        "HUMAN_ESCALATION",
        "HUMAN_ESCALATION",
    ]
    system_labels = [
        "AUTO_HANDLE",
        "HUMAN_ESCALATION",
        "HUMAN_ESCALATION",
        "HUMAN_ESCALATION",
    ]

    metrics = calculate_routing_metrics(human_labels, system_labels)
    assert metrics["total_evaluated"] == 4
    # 3 matches out of 4 -> 0.75 observed agreement
    assert metrics["observed_agreement"] == 0.75
    # Confusion matrix checks
    cm = metrics["confusion_matrix"]
    assert cm["AUTO_HANDLE"]["AUTO_HANDLE"] == 1
    assert cm["AUTO_HANDLE"]["HUMAN_ESCALATION"] == 1
    assert cm["HUMAN_ESCALATION"]["AUTO_HANDLE"] == 0
    assert cm["HUMAN_ESCALATION"]["HUMAN_ESCALATION"] == 2
    # Check per-class metrics exist
    assert "precision" in metrics["per_class_metrics"]["HUMAN_ESCALATION"]
    assert "recall" in metrics["per_class_metrics"]["HUMAN_ESCALATION"]
    assert "f1" in metrics["per_class_metrics"]["HUMAN_ESCALATION"]


def test_human_intent_classification_metrics():
    """Test intent classification metric computation against multiclass labels."""
    y_true = [
        "OPERATING_SYSTEM_UPDATES",
        "BATTERY_POWER_HARDWARE",
        "ACCOUNT_APPLE_ID",
    ]
    y_pred = [
        "OPERATING_SYSTEM_UPDATES",
        "BATTERY_POWER_HARDWARE",
        "CONNECTIVITY_NETWORKING",
    ]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.6667, abs=1e-3)
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["classes"]) == 7


def test_load_and_verify_human_routings_incomplete(temp_eval_dir: Path):
    """Test routing verification flags incomplete annotations."""
    incomplete_file = temp_eval_dir / "incomplete_routings.jsonl"
    records = get_synthetic_blank_annotations(count=10)
    with open(incomplete_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_complete, _, stats = load_and_verify_human_routings(incomplete_file)
    assert is_complete is False
    assert stats["incomplete_count"] == 10
    assert stats["annotated_count"] == 0
