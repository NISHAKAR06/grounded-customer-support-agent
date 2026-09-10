"""Evaluation and metrics API routes."""

from fastapi import APIRouter

from app.services.evaluation.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])
_eval_service = EvaluationService()


@router.get("/metrics")
def get_evaluation_metrics():
    """Return benchmark metrics across baselines, final model, and LLM judge."""
    return _eval_service.get_benchmark_summary()
