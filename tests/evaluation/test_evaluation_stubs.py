"""Tests for evaluation metrics and benchmark structure."""

from app.services.evaluation.evaluation_service import EvaluationService


def test_benchmark_summary_structure():
    """Verify evaluation benchmark summary adheres to zero-mock contract."""
    service = EvaluationService()
    summary = service.get_benchmark_summary()

    assert "status" in summary
    assert "golden_set_count" in summary
    assert "models" in summary
    assert isinstance(summary["models"], dict)
    assert "retrieval" in summary
    assert "llm_judge" in summary
