"""Integration tests for service wiring and orchestration."""

from app.models.domain_models import RoutingDecision
from app.services.agent.agent_orchestrator import AgentOrchestrator
from app.services.generation.llm_service import BaseLLMProvider, LLMService
from tests.fixtures.sample_fixtures import get_sample_historical_case


class FailingProvider(BaseLLMProvider):
    def provider_name(self) -> str:
        return "failing_primary"

    def generate(self, prompt: str) -> str:
        raise RuntimeError("Simulated API failure")


class MockFallbackProvider(BaseLLMProvider):
    def provider_name(self) -> str:
        return "working_fallback"

    def generate(self, prompt: str) -> str:
        return "Fallback reply grounded in test context."


class FixtureRetriever:
    def retrieve(self, query: str, top_k: int = 3):
        return [get_sample_historical_case()]


def test_orchestrator_wiring_with_fixture_retriever():
    """Verify AgentOrchestrator coordinates all stages when evidence is present."""
    orchestrator = AgentOrchestrator(retriever=FixtureRetriever())
    result = orchestrator.run(customer_message="I need to change my shipping address.")

    assert result.run_id.startswith("run_")
    assert result.intent.name != ""
    assert result.intent.confidence > 0.0
    assert len(result.retrieval.evidence) > 0
    assert result.generation.draft_reply != ""
    assert result.validation.all_passed is True
    assert result.routing.decision == RoutingDecision.AUTO_HANDLE
    assert result.latency_ms.total_ms > 0


def test_orchestrator_wiring_without_index_zero_mock_escalation():
    """Verify AgentOrchestrator safely escalates to human when no vector index exists."""
    orchestrator = (
        AgentOrchestrator()
    )  # Default retriever has no index prior to Phase 7
    result = orchestrator.run(customer_message="I need to change my shipping address.")

    assert result.run_id.startswith("run_")
    assert len(result.retrieval.evidence) == 0
    assert result.routing.decision == RoutingDecision.HUMAN_ESCALATION
    assert any(
        "No historical resolved support cases found" in r
        for r in result.routing.reasons
    )


def test_llm_service_circuit_breaker_fallback():
    """Verify LLMService catches primary failure and automatically uses fallback."""
    service = LLMService(
        primary_provider=FailingProvider(),
        fallback_provider=MockFallbackProvider(),
    )
    reply, provider = service.generate_reply("Test prompt")
    assert reply == "Fallback reply grounded in test context."
    assert provider == "working_fallback"
