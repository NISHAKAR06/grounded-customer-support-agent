"""Unit and integration tests for multi-provider LLM system.

Tests all 6 providers: mock, ollama, openai, groq, gemini, claude,
as well as the provider factory, fallback circuit breaker, and API endpoints.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import LLMProviderException
from app.main import app
from app.services.generation.claude_provider import ClaudeProvider
from app.services.generation.gemini_provider import GeminiProvider
from app.services.generation.groq_provider import GroqProvider
from app.services.generation.llm_service import BaseLLMProvider, LLMService
from app.services.generation.mock_provider import MockProvider
from app.services.generation.ollama_provider import OllamaProvider
from app.services.generation.openai_provider import OpenAIProvider
from app.services.generation.provider_factory import LLMProviderFactory

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. MockProvider Tests
# ---------------------------------------------------------------------------


def test_mock_provider_grounded_in_resolution():
    """Verify MockProvider extracts historical resolution from prompt."""
    provider = MockProvider()
    prompt = (
        "GROUNDING RULES:\n1. Rely on evidence\n\n"
        "CUSTOMER INTENT: OPERATING_SYSTEM_UPDATES\n"
        "CUSTOMER MESSAGE: 'iOS 11 battery drain'\n\n"
        "HISTORICAL RESOLVED CASES:\n"
        "[Historical Case #1] (ID: case_001)\n"
        "Customer: My battery is draining fast\n"
        "Brand Resolution: Please try updating your iPhone to the latest iOS build.\n\n"
        "DRAFT REPLY:"
    )
    reply = provider.generate(prompt)
    assert "Please try updating your iPhone" in reply
    assert "mock" in provider.provider_name()


def test_mock_provider_fallback_templates():
    """Verify MockProvider generates appropriate response when no evidence is present."""
    provider = MockProvider()
    prompt = "CUSTOMER INTENT: BATTERY_POWER_HARDWARE\nCUSTOMER MESSAGE: 'battery swollen'\nDRAFT REPLY:"
    reply = provider.generate(prompt)
    assert "hardware" in reply.lower() or "battery" in reply.lower()


# ---------------------------------------------------------------------------
# 2. OllamaProvider Tests
# ---------------------------------------------------------------------------


def test_ollama_provider_name_and_config():
    """Verify OllamaProvider configuration and naming."""
    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3.2")
    assert provider.provider_name() == "ollama (llama3.2)"


@patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused"))
def test_ollama_provider_connection_error(mock_post):
    """Verify OllamaProvider handles offline daemon gracefully."""
    provider = OllamaProvider(base_url="http://localhost:11434")
    with pytest.raises(LLMProviderException) as exc_info:
        provider.generate("Hello world")
    assert "Ollama connection refused" in str(exc_info.value)


@patch("httpx.Client.post")
def test_ollama_provider_successful_generation(mock_post):
    """Verify OllamaProvider parses successful HTTP 200 response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Hello from Ollama Llama 3.2!"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3.2")
    reply = provider.generate("Test prompt")
    assert reply == "Hello from Ollama Llama 3.2!"


# ---------------------------------------------------------------------------
# 3. OpenAIProvider Tests
# ---------------------------------------------------------------------------


def test_openai_provider_missing_key():
    """Verify OpenAIProvider raises exception when API key is missing."""
    provider = OpenAIProvider(api_key="")
    with pytest.raises(LLMProviderException) as exc_info:
        provider.generate("Test prompt")
    assert "OpenAI API key is missing" in str(exc_info.value)


@patch("httpx.Client.post")
def test_openai_provider_successful_generation(mock_post):
    """Verify OpenAIProvider parses chat completion response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello from GPT-4o mini!"}}]
    }
    mock_post.return_value = mock_resp

    provider = OpenAIProvider(api_key="sk-real-test-key", model_name="gpt-4o-mini")
    reply = provider.generate("Test prompt")
    assert reply == "Hello from GPT-4o mini!"
    assert provider.provider_name() == "openai (gpt-4o-mini)"


# ---------------------------------------------------------------------------
# 4. GroqProvider Tests
# ---------------------------------------------------------------------------


def test_groq_provider_missing_key():
    """Verify GroqProvider raises exception when API key is missing."""
    provider = GroqProvider(api_key="")
    with pytest.raises(LLMProviderException) as exc_info:
        provider.generate("Test prompt")
    assert "Groq API key is missing" in str(exc_info.value)


@patch("httpx.Client.post")
def test_groq_provider_successful_generation(mock_post):
    """Verify GroqProvider parses ultra-fast inference response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Ultra-fast response from Groq!"}}]
    }
    mock_post.return_value = mock_resp

    provider = GroqProvider(api_key="gsk-test-key", model_name="llama-3.1-8b-instant")
    reply = provider.generate("Test prompt")
    assert reply == "Ultra-fast response from Groq!"
    assert provider.provider_name() == "groq (llama-3.1-8b-instant)"


# ---------------------------------------------------------------------------
# 5. GeminiProvider Tests
# ---------------------------------------------------------------------------


def test_gemini_provider_missing_key():
    """Verify GeminiProvider raises exception when API key is missing."""
    provider = GeminiProvider(api_key="")
    with pytest.raises(LLMProviderException) as exc_info:
        provider.generate("Test prompt")
    assert "Gemini API key is missing" in str(exc_info.value)


@patch("httpx.Client.post")
def test_gemini_provider_successful_generation(mock_post):
    """Verify GeminiProvider parses candidate content."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "Hello from Gemini 1.5 Flash!"}]}}
        ]
    }
    mock_post.return_value = mock_resp

    provider = GeminiProvider(api_key="AIzaSyTestKey", model_name="gemini-1.5-flash")
    reply = provider.generate("Test prompt")
    assert reply == "Hello from Gemini 1.5 Flash!"
    assert provider.provider_name() == "gemini (gemini-1.5-flash)"


# ---------------------------------------------------------------------------
# 6. ClaudeProvider Tests
# ---------------------------------------------------------------------------


def test_claude_provider_missing_key():
    """Verify ClaudeProvider raises exception when API key is missing."""
    provider = ClaudeProvider(api_key="")
    with pytest.raises(LLMProviderException) as exc_info:
        provider.generate("Test prompt")
    assert "Claude API key is missing" in str(exc_info.value) or "Anthropic" in str(
        exc_info.value
    )


@patch("httpx.Client.post")
def test_claude_provider_successful_generation(mock_post):
    """Verify ClaudeProvider parses Anthropic Messages response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "Hello from Claude 3.5 Haiku!"}]
    }
    mock_post.return_value = mock_resp

    provider = ClaudeProvider(
        api_key="sk-ant-test-key", model_name="claude-3-5-haiku-20241022"
    )
    reply = provider.generate("Test prompt")
    assert reply == "Hello from Claude 3.5 Haiku!"
    assert provider.provider_name() == "claude (claude-3-5-haiku-20241022)"


# ---------------------------------------------------------------------------
# 7. LLMProviderFactory Tests
# ---------------------------------------------------------------------------


def test_factory_creates_all_six_providers():
    """Verify factory instantiates all 6 supported providers."""
    for name in ["mock", "ollama", "openai", "groq", "gemini", "claude"]:
        provider = LLMProviderFactory.create_provider(name)
        assert isinstance(provider, BaseLLMProvider)


def test_factory_normalizes_aliases():
    """Verify factory handles common aliases."""
    local_p = LLMProviderFactory.create_provider("local")
    assert isinstance(local_p, MockProvider)

    anthropic_p = LLMProviderFactory.create_provider("anthropic")
    assert isinstance(anthropic_p, ClaudeProvider)


def test_factory_unsupported_raises_error():
    """Verify factory raises exception on invalid provider identifier."""
    with pytest.raises(LLMProviderException) as exc_info:
        LLMProviderFactory.create_provider("unknown_super_llm")
    assert "Unsupported LLM provider" in str(exc_info.value)


def test_factory_list_available_providers():
    """Verify metadata list contains all 6 providers."""
    providers = LLMProviderFactory.list_available_providers()
    assert len(providers) == 6
    ids = [p["id"] for p in providers]
    for expected in ["mock", "ollama", "openai", "groq", "gemini", "claude"]:
        assert expected in ids


# ---------------------------------------------------------------------------
# 8. LLMService Fallback & Dynamic Routing
# ---------------------------------------------------------------------------


def test_llm_service_dynamic_provider_override():
    """Verify LLMService uses the provider specified at runtime."""
    service = LLMService(
        primary_provider=MockProvider(model_name="primary-mock"),
        fallback_provider=MockProvider(model_name="fallback-mock"),
    )
    reply, provider_used = service.generate_reply("Test prompt", provider_name="mock")
    assert "mock" in provider_used


def test_llm_service_falls_back_when_selected_provider_fails():
    """Verify LLMService transparently falls back to safe mock when target provider fails."""
    service = LLMService(
        primary_provider=MockProvider(model_name="primary-mock"),
        fallback_provider=MockProvider(model_name="safe-fallback"),
    )
    # Target openai with no API key; will fail and trigger fallback
    reply, provider_used = service.generate_reply("Test prompt", provider_name="openai")
    assert reply != ""
    assert "safe-fallback" in provider_used or "mock" in provider_used


# ---------------------------------------------------------------------------
# 9. API Endpoints for Providers
# ---------------------------------------------------------------------------


def test_api_get_providers():
    """Verify GET /api/agent/providers and /api/v1/agent/providers return 6 providers."""
    for path in ["/api/agent/providers", "/api/v1/agent/providers"]:
        res = client.get(path)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 6
        provider_ids = [p["id"] for p in data]
        assert "mock" in provider_ids
        assert "groq" in provider_ids
        assert "claude" in provider_ids


def test_api_run_agent_with_provider_selection():
    """Verify POST /api/agent/run accepts provider override."""
    payload = {
        "customer_message": "My iPhone battery dies so fast after the update.",
        "provider": "mock",
    }
    res = client.post("/api/agent/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "mock" in data["generation"]["provider"].lower()
    assert data["generation"]["draft_reply"] != ""
