"""FastAPI application entrypoint for Grounded Customer Support Agent."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes.agent import router as agent_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.health import router as health_router
from app.api.routes.taxonomy import router as taxonomy_router
from app.core.config import get_settings
from app.core.logging import setup_logger
from app.repositories.conversation_repository import ConversationRepository
from app.services.evaluation.evaluation_service import EvaluationService
from scripts.data.classify_intents import load_intent_taxonomy

settings = get_settings()
logger = setup_logger(log_level=settings.LOG_LEVEL)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Grounded Customer Support Agent with verifiable historical grounding and safe escalation.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Static files and template engine
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# Root health probe
app.include_router(health_router)

# Include API route groups (supporting both /api and /api/v1)
for prefix in ("/api", "/api/v1"):
    app.include_router(health_router, prefix=prefix)
    app.include_router(agent_router, prefix=prefix)
    app.include_router(conversations_router, prefix=prefix)
    app.include_router(evaluation_router, prefix=prefix)
    app.include_router(taxonomy_router, prefix=prefix)

# Repositories & Services for Page Contexts
_conversation_repo = ConversationRepository()
_evaluation_service = EvaluationService()


# ---------------------------------------------------------------------------
# UI Page Routes (Rendered using Jinja2 templates)
# ---------------------------------------------------------------------------


@app.get("/", response_class=RedirectResponse)
def root():
    """Redirect root to the primary Simulate workspace."""
    return RedirectResponse(url="/simulate", status_code=302)


@app.get("/simulate", response_class=HTMLResponse)
def page_simulate(request: Request, msg: str = ""):
    """Render the primary Simulate Incoming Message workspace."""
    return templates.TemplateResponse(
        request=request,
        name="simulate.html",
        context={
            "active_page": "simulate",
            "initial_message": msg,
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        },
    )


@app.get("/inbox", response_class=HTMLResponse)
def page_inbox(
    request: Request,
    filter: str = "all",
    decision: str = "all",
    turns: str = "all",
    intent: str = "all",
    sort: str = "newest",
    search: str = "",
    page: int = 1,
    page_size: int = 10,
):
    """Render the Support Inbox workspace with filtering, sorting, and pagination."""
    pagination_data = _conversation_repo.paginate(
        status_filter=filter,
        decision_filter=decision,
        turn_filter=turns,
        intent_filter=intent,
        sort_by=sort,
        search_query=search,
        page=page,
        page_size=page_size,
    )
    return templates.TemplateResponse(
        request=request,
        name="inbox.html",
        context={
            "active_page": "inbox",
            "conversations": pagination_data["items"],
            "pagination": pagination_data,
            "active_filter": filter,
            "active_decision": decision,
            "active_turns": turns,
            "active_intent": intent,
            "active_sort": sort,
            "search_query": search,
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        },
    )


@app.get("/evaluation", response_class=HTMLResponse)
def page_evaluation(request: Request):
    """Render the model benchmarks and evaluation summary."""
    benchmarks = _evaluation_service.get_benchmark_summary()
    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context={
            "active_page": "evaluation",
            "benchmarks": benchmarks,
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        },
    )


@app.get("/failures", response_class=HTMLResponse)
def page_failures(request: Request):
    """Render the top 5 failure modes and headline limitation analysis."""
    return templates.TemplateResponse(
        request=request,
        name="failures.html",
        context={
            "active_page": "failures",
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        },
    )


@app.get("/decisions", response_class=HTMLResponse)
def page_decisions(request: Request):
    """Render the engineering decision log."""
    return templates.TemplateResponse(
        request=request,
        name="decisions.html",
        context={
            "active_page": "decisions",
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        },
    )


@app.get("/methodology", response_class=HTMLResponse)
def page_methodology(request: Request):
    """Render grounding methodology, intent taxonomy, and escalation policy specification."""
    import json
    from pathlib import Path

    taxonomy_schema = None
    try:
        taxonomy_schema = load_intent_taxonomy()
    except Exception:
        pass

    distribution_data = None
    dist_file = Path("experiments/intent_distribution.json")
    if dist_file.exists():
        try:
            with open(dist_file, "r", encoding="utf-8") as f:
                distribution_data = json.load(f)
        except Exception:
            pass

    return templates.TemplateResponse(
        request=request,
        name="methodology.html",
        context={
            "active_page": "methodology",
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "taxonomy": taxonomy_schema,
            "distribution": distribution_data,
        },
    )
