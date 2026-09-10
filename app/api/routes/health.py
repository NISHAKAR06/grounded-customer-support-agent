"""Health and readiness probe routes."""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.response_models import HealthResponse
from app.repositories.model_repository import ModelRepository

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return system readiness, active LLM provider, and model checkpoint statuses."""
    settings = get_settings()
    model_repo = ModelRepository()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        llm_provider=settings.LLM_PROVIDER,
        models=model_repo.get_model_status(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
