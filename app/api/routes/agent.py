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


def _persist_to_inbox(
    result: AgentRunResult, message: str, customer_handle: Optional[str], brand: str
) -> None:
    """Save processed inquiry as an operational ticket in the Support Inbox database."""
    try:
        import time
        import uuid

        from app.repositories.conversation_repository import ConversationRepository

        repo = ConversationRepository()
        cid = (
            result.metadata.conversation_id or f"conv_sim_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        )
        ticket_id = f"TICK-SIM-{cid[-6:].upper()}"
        h = (customer_handle or "@Customer").strip()
        if not h.startswith("@"):
            h = f"@{h}"
        decision_str = result.routing.decision.value
        status_str = "AI Ready" if decision_str == "AUTO_HANDLE" else "Needs Human"
        now_str = time.strftime("%a %b %d %H:%M:%S +0000 %Y", time.gmtime())

        rec = {
            "conversation_id": cid,
            "ticket_id": ticket_id,
            "customer_id": h.replace("@", ""),
            "brand": brand or "AppleSupport",
            "root_tweet_id": str(int(time.time())),
            "first_inquiry": message,
            "latest_message": message,
            "final_brand_response": result.generation.draft_reply,
            "intent": result.intent.name,
            "intent_code": result.intent.code,
            "confidence": result.intent.confidence,
            "decision": decision_str,
            "turn_count": 2,
            "turns": [
                {
                    "turn_id": 1,
                    "author_id": h.replace("@", ""),
                    "author_role": "CUSTOMER",
                    "text": message,
                    "created_at": now_str,
                },
                {
                    "turn_id": 2,
                    "author_id": brand or "AppleSupport",
                    "author_role": "BRAND",
                    "text": result.generation.draft_reply,
                    "created_at": now_str,
                },
            ],
            "has_dm": "DM" in result.generation.draft_reply,
            "has_kb_link": "http" in result.generation.draft_reply,
            "has_resolution": decision_str == "AUTO_HANDLE",
            "status": status_str,
            "created_at": now_str,
            "timestamp": now_str,
            "intent_confidence": result.intent.confidence,
            "intent_signals": result.intent.signals,
        }
        repo.save_conversation(rec)
    except Exception as e:
        logger.warning(f"Failed to persist ticket to inbox DB: {e}")


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
            customer_handle=payload.customer_handle,
            provider=payload.provider,
        )
        _persist_to_inbox(result, payload.customer_message, payload.customer_handle, payload.brand)
        return result
    except Exception as ex:
        logger.error(f"Agent execution failed: {ex}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent pipeline failure: {str(ex)}")


@router.get("/stream")
async def stream_agent_execution(
    request: Request,
    customer_message: str = Query(..., description="Customer support inquiry text"),
    conversation_id: Optional[str] = Query(None, description="Optional thread/conversation ID"),
    brand: Optional[str] = Query("AppleSupport", description="Target brand"),
    customer_handle: Optional[str] = Query(None, description="Customer handle or username"),
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
                    customer_handle=customer_handle,
                    provider=provider,
                    event_callback=sync_event_callback,
                ),
            )
            _persist_to_inbox(result, customer_message, customer_handle, brand)
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
