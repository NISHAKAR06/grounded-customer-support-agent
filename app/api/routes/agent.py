"""Agent execution and simulation API routes."""

import asyncio
import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.logging import logger
from app.models.request_models import SimulateRequest
from app.models.response_models import AgentRunResult
from app.services.agent.agent_orchestrator import AgentOrchestrator
from app.services.generation.provider_factory import LLMProviderFactory

router = APIRouter(prefix="/agent", tags=["Agent"])

# Global orchestrator dependency injection
_orchestrator = AgentOrchestrator()


def get_orchestrator() -> AgentOrchestrator:
    return _orchestrator


@router.get("/providers", response_model=List[Dict[str, Any]])
def get_llm_providers() -> List[Dict[str, Any]]:
    """Return all supported LLM providers and their configuration status."""
    return LLMProviderFactory.list_available_providers()


@router.post("/run", response_model=AgentRunResult)
def run_agent(
    payload: SimulateRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> AgentRunResult:
    """Execute the end-to-end grounded support pipeline for an incoming customer message."""
    try:
        result = orchestrator.run(
            customer_message=payload.customer_message,
            conversation_id=payload.conversation_id,
            brand=payload.brand,
            provider=payload.provider,
        )
        return result
    except Exception as ex:
        logger.error(f"Agent execution failed: {ex}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Agent pipeline failure: {str(ex)}"
        )


@router.get("/events/{run_id}")
async def stream_agent_events(run_id: str):
    """Server-Sent Events (SSE) endpoint to monitor live execution status."""

    async def event_generator():
        # Yields execution status updates
        yield {
            "event": "connected",
            "data": json.dumps({"run_id": run_id, "status": "listening"}),
        }
        await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())
