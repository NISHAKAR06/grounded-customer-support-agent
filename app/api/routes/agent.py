"""Agent execution and simulation API routes."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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


@router.get("/stream")
async def stream_agent_execution(
    request: Request,
    customer_message: str = Query(..., description="Customer support inquiry text"),
    conversation_id: Optional[str] = Query(None, description="Optional thread/conversation ID"),
    brand: Optional[str] = Query("AppleSupport", description="Target brand"),
    provider: Optional[str] = Query(None, description="LLM provider name"),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Server-Sent Events (SSE) streaming endpoint for live real-time pipeline execution trace."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sync_event_callback(event_name: str, event_payload: dict):
        loop.call_soon_threadsafe(queue.put_nowait, (event_name, event_payload))

    async def run_pipeline():
        try:
            result = await loop.run_in_executor(
                None,
                lambda: orchestrator.run(
                    customer_message=customer_message,
                    conversation_id=conversation_id,
                    brand=brand,
                    provider=provider,
                    event_callback=sync_event_callback,
                ),
            )
            # Serialize result cleanly for frontend consumption
            await queue.put(("RESULT", json.loads(result.model_dump_json())))
        except Exception as ex:
            logger.error(f"Live pipeline streaming failed: {ex}", exc_info=True)
            await queue.put(("ERROR", {"message": str(ex)}))
        finally:
            await queue.put(("DONE", {}))

    asyncio.create_task(run_pipeline())

    async def event_generator():
        while True:
            # Detect client disconnection
            if await request.is_disconnected():
                break
            event_name, data = await queue.get()
            if event_name == "DONE":
                break
            yield {
                "event": event_name,
                "data": json.dumps(data),
            }

    return EventSourceResponse(event_generator())


@router.get("/events/{run_id}")
async def stream_agent_events(run_id: str):
    """Server-Sent Events (SSE) endpoint to monitor live execution status."""

    async def event_generator():
        yield {
            "event": "connected",
            "data": json.dumps({"run_id": run_id, "status": "listening"}),
        }
        await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())

