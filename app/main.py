"""FastAPI application entrypoint for Grounded Customer Support Agent."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes.agent import router as agent_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.logging import setup_logger
from app.repositories.conversation_repository import ConversationRepository
from app.services.evaluation.evaluation_service import EvaluationService

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

# Include API route groups
app.include_router(health_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")

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
def page_simulate(request: Request):
    """Render the primary Simulate Incoming Message workspace."""
    return templates.TemplateResponse(
        request=request,
        name="simulate.html",
        context={
            "active_page": "simulate",
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        },
    )


@app.get("/inbox", response_class=HTMLResponse)
def page_inbox(request: Request, filter: str = "all"):
    """Render the Support Inbox workspace."""
    conversations = _conversation_repo.list_conversations(status_filter=filter)
    return templates.TemplateResponse(
        request=request,
        name="inbox.html",
        context={
            "active_page": "inbox",
            "conversations": conversations,
            "active_filter": filter,
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
    """Render grounding methodology and escalation policy specification."""
    return templates.TemplateResponse(
        request=request,
        name="methodology.html",
        context={
            "active_page": "methodology",
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        },
    )
