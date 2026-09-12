"""Unit and integration tests for multi-provider LLM system.

Tests all 5 real inference providers: groq, ollama, openai, gemini, claude,
as well as the provider factory, fallback circuit breaker, and API endpoints.
Zero-mock policy: No offline mock simulation engines in production runtime.
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
from app.services.generation.ollama_provider import OllamaProvider
from app.services.generation.openai_provider import OpenAIProvider
from app.services.generation.provider_factory import LLMProviderFactory

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Fixture: Minimal In-Memory Provider for LLMService Testing
# ---------------------------------------------------------------------------


class DummyTestProvider(BaseLLMProvider):
    """In-memory test stub for verifying LLMService circuit breaking."""

    def __init__(self, name: str = "dummy", should_fail: bool = False):
        self._name = name
        self._should_fail = should_fail

    def provider_name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> str:
        if self._should_fail:
            raise LLMProviderException(f"Simulated failure on {self._name}")
        return f"Response from {self._name}: {prompt[:20]}"


# ---------------------------------------------------------------------------
# 1. OllamaProvider Tests
# ---------------------------------------------------------------------------


def test_ollama_provider_name_and_config():
    """Verify OllamaProvider configuration and naming."""
    provider = OllamaProvider(base_url="http://localhost:11434", model_name="qwen2.5-coder:7b")
    assert "ollama" in provider.provider_name()


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
    mock_resp.json.return_value = {"response": "Hello from Ollama!"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(base_url="http://localhost:11434", model_name="qwen2.5-coder:7b")
    reply = provider.generate("Test prompt")
    assert reply == "Hello from Ollama!"


# ---------------------------------------------------------------------------
# 2. OpenAIProvider Tests
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
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Hello from GPT-4o mini!"}}]}
    mock_post.return_value = mock_resp

    provider = OpenAIProvider(api_key="sk-real-test-key", model_name="gpt-4o-mini")
    reply = provider.generate("Test prompt")
    assert reply == "Hello from GPT-4o mini!"
    assert provider.provider_name() == "openai (gpt-4o-mini)"


# ---------------------------------------------------------------------------
# 3. GroqProvider Tests
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

    provider = GroqProvider(api_key="gsk-test-key", model_name="groq/compound-mini")
    reply = provider.generate("Test prompt")
    assert reply == "Ultra-fast response from Groq!"
    assert "groq" in provider.provider_name()


# ---------------------------------------------------------------------------
# 4. GeminiProvider Tests
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
        "candidates": [{"content": {"parts": [{"text": "Hello from Gemini 1.5 Flash!"}]}}]
    }
    mock_post.return_value = mock_resp

    provider = GeminiProvider(api_key="AIzaSyTestKey", model_name="gemini-1.5-flash")
    reply = provider.generate("Test prompt")
    assert reply == "Hello from Gemini 1.5 Flash!"
    assert provider.provider_name() == "gemini (gemini-1.5-flash)"


# ---------------------------------------------------------------------------
# 5. ClaudeProvider Tests
# ---------------------------------------------------------------------------


def test_claude_provider_missing_key():
    """Verify ClaudeProvider raises exception when API key is missing."""
    provider = ClaudeProvider(api_key="")
    with pytest.raises(LLMProviderException) as exc_info:
        provider.generate("Test prompt")
    assert "Claude API key is missing" in str(exc_info.value) or "Anthropic" in str(exc_info.value)


@patch("httpx.Client.post")
def test_claude_provider_successful_generation(mock_post):
    """Verify ClaudeProvider parses Anthropic Messages response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "Hello from Claude 3.5 Haiku!"}]
    }
    mock_post.return_value = mock_resp

    provider = ClaudeProvider(api_key="sk-ant-test-key", model_name="claude-3-5-haiku-20241022")
    reply = provider.generate("Test prompt")
    assert reply == "Hello from Claude 3.5 Haiku!"
    assert provider.provider_name() == "claude (claude-3-5-haiku-20241022)"


# ---------------------------------------------------------------------------
# 6. LLMProviderFactory Tests
# ---------------------------------------------------------------------------


def test_factory_creates_all_five_providers():
    """Verify factory instantiates all 5 supported real inference providers."""
    for name in ["ollama", "openai", "groq", "gemini", "claude"]:
        provider = LLMProviderFactory.create_provider(name)
        assert isinstance(provider, BaseLLMProvider)


def test_factory_normalizes_aliases():
    """Verify factory handles common aliases."""
    local_p = LLMProviderFactory.create_provider("local")
    assert isinstance(local_p, OllamaProvider)

    anthropic_p = LLMProviderFactory.create_provider("anthropic")
    assert isinstance(anthropic_p, ClaudeProvider)


def test_factory_unsupported_raises_error():
    """Verify factory raises exception on invalid provider identifier or deprecated mock."""
    with pytest.raises(LLMProviderException) as exc_info:
        LLMProviderFactory.create_provider("unknown_super_llm")
    assert "Unsupported LLM provider" in str(exc_info.value)


def test_factory_list_available_providers():
    """Verify metadata list contains all 5 real providers without mock."""
    providers = LLMProviderFactory.list_available_providers()
    assert len(providers) == 5
    ids = [p["id"] for p in providers]
    assert "mock" not in ids
    for expected in ["ollama", "openai", "groq", "gemini", "claude"]:
        assert expected in ids


def test_factory_fallback_creates_real_provider():
    """Verify fallback provider creates a real alternative engine."""
    fallback = LLMProviderFactory.create_fallback_provider()
    assert isinstance(fallback, (OllamaProvider, GroqProvider, OpenAIProvider))
    assert "mock" not in fallback.provider_name()


# ---------------------------------------------------------------------------
# 7. LLMService Fallback & Dynamic Routing
# ---------------------------------------------------------------------------


def test_llm_service_dynamic_provider_override():
    """Verify LLMService uses the provider specified at runtime."""
    primary = DummyTestProvider(name="primary-p")
    fallback = DummyTestProvider(name="fallback-p")
    service = LLMService(primary_provider=primary, fallback_provider=fallback)

    with patch.object(LLMProviderFactory, "create_provider", return_value=primary):
        reply, provider_used = service.generate_reply("Test prompt", provider_name="groq")
        assert "primary-p" in provider_used


def test_llm_service_falls_back_when_primary_fails():
    """Verify LLMService transparently falls back to secondary real provider when primary fails."""
    primary = DummyTestProvider(name="primary-failing", should_fail=True)
    fallback = DummyTestProvider(name="fallback-live", should_fail=False)
    service = LLMService(primary_provider=primary, fallback_provider=fallback)

    reply, provider_used = service.generate_reply("Test prompt")
    assert "fallback-live" in provider_used
    assert "Response from fallback-live" in reply


# ---------------------------------------------------------------------------
# 8. API Endpoints for Providers
# ---------------------------------------------------------------------------


def test_api_get_providers():
    """Verify GET /api/agent/providers returns 5 real providers and excludes mock."""
    for path in ["/api/agent/providers", "/api/v1/agent/providers"]:
        res = client.get(path)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 5
        provider_ids = [p["id"] for p in data]
        assert "mock" not in provider_ids
        assert "groq" in provider_ids
        assert "ollama" in provider_ids
