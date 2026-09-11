"""Evaluation and metrics API routes."""

from typing import Optional

from fastapi import APIRouter, Query

from app.repositories.golden_set_repository import GoldenSetRepository
from app.services.evaluation.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])
_eval_service = EvaluationService()
_golden_repo = GoldenSetRepository()


@router.get("/metrics")
def get_evaluation_metrics():
    """Return benchmark metrics across baselines, final model, and LLM judge."""
    return _eval_service.get_benchmark_summary()


@router.get("/golden-set")
def get_golden_set(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    intent_code: Optional[str] = Query("all"),
    routing: Optional[str] = Query("all"),
):
    """Return paginated samples from the 200-sample hand-verified Golden Evaluation Set."""
    items, total = _golden_repo.paginate(
        limit=limit,
        offset=offset,
        intent_code=intent_code,
        routing=routing,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "samples": items,
        "summary": _golden_repo.get_summary(),
    }


@router.get("/golden-set/summary")
def get_golden_set_summary():
    """Return distribution breakdown and stratification statistics for the Golden Set."""
    return _golden_repo.get_summary()
