"""Factory and registry for all supported LLM generation providers."""

from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.services.generation.claude_provider import ClaudeProvider
from app.services.generation.gemini_provider import GeminiProvider
from app.services.generation.groq_provider import GroqProvider
from app.services.generation.llm_service import BaseLLMProvider
from app.services.generation.mock_provider import MockProvider
from app.services.generation.ollama_provider import OllamaProvider
from app.services.generation.openai_provider import OpenAIProvider


class LLMProviderFactory:
    """Factory for dynamically creating and inspecting LLM providers.

    Supported Providers:
        - mock: Zero-cost offline deterministic simulation
        - ollama: Local self-hosted LLM (Llama, Mistral, Phi)
        - openai: OpenAI Chat Completions (GPT-4o mini, GPT-4o)
        - groq: Groq ultra-low latency cloud inference (Llama-3.1, Mixtral)
        - gemini: Google Gemini (1.5 Flash, 2.0 Flash)
        - claude: Anthropic Claude (3.5 Haiku, 3.5 Sonnet)
    """

    SUPPORTED_PROVIDERS = ("mock", "ollama", "openai", "groq", "gemini", "claude")

    @classmethod
    def normalize_provider_name(cls, provider_name: Optional[str]) -> str:
        """Normalize provider aliases and casing."""
        if not provider_name:
            settings = get_settings()
            provider_name = settings.LLM_PROVIDER

        clean = provider_name.strip().lower()
        alias_map = {
            "local": "mock",
            "offline": "mock",
            "anthropic": "claude",
            "google": "gemini",
        }
        return alias_map.get(clean, clean)

    @classmethod
    def create_provider(
        cls, provider_name: Optional[str] = None, **kwargs
    ) -> BaseLLMProvider:
        """Instantiate an LLM provider by identifier name."""
        name = cls.normalize_provider_name(provider_name)

        if name == "mock":
            return MockProvider(**kwargs)
        elif name == "ollama":
            return OllamaProvider(**kwargs)
        elif name == "openai":
            return OpenAIProvider(**kwargs)
        elif name == "groq":
            return GroqProvider(**kwargs)
        elif name == "gemini":
            return GeminiProvider(**kwargs)
        elif name == "claude":
            return ClaudeProvider(**kwargs)
        else:
            raise LLMProviderException(
                f"Unsupported LLM provider '{provider_name}'. "
                f"Supported providers: {list(cls.SUPPORTED_PROVIDERS)}"
            )

    @classmethod
    def list_available_providers(cls) -> List[Dict[str, Any]]:
        """Return metadata, status, and model info for all supported providers."""
        settings = get_settings()
        active_provider = cls.normalize_provider_name(settings.LLM_PROVIDER)

        def is_valid_key(key: Optional[str], placeholder: str) -> bool:
            return bool(key and not key.startswith(placeholder) and len(key) > 5)

        return [
            {
                "id": "mock",
                "name": "Mock Provider",
                "badge": "Offline Simulation",
                "description": "Deterministic, zero-latency grounded simulation without API keys",
                "model": settings.MOCK_MODEL_NAME,
                "configured": True,
                "is_active": active_provider == "mock",
                "type": "local_mock",
            },
            {
                "id": "ollama",
                "name": "Ollama",
                "badge": "Local Self-Hosted",
                "description": f"Local inference server at {settings.OLLAMA_BASE_URL}",
                "model": settings.OLLAMA_MODEL_NAME,
                "configured": bool(settings.OLLAMA_BASE_URL),
                "is_active": active_provider == "ollama",
                "type": "local_daemon",
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "badge": "Cloud API",
                "description": "OpenAI official Chat Completions API",
                "model": settings.OPENAI_MODEL_NAME,
                "configured": is_valid_key(
                    settings.OPENAI_API_KEY, "your_openai_api_key"
                ),
                "is_active": active_provider == "openai",
                "type": "cloud_api",
            },
            {
                "id": "groq",
                "name": "Groq",
                "badge": "Ultra-Fast LPU",
                "description": "Groq cloud high-speed inference engine",
                "model": settings.GROQ_MODEL_NAME,
                "configured": is_valid_key(settings.GROQ_API_KEY, "your_groq_api_key"),
                "is_active": active_provider == "groq",
                "type": "cloud_api",
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "badge": "Cloud Multimodal",
                "description": "Google AI Studio / Vertex Gemini API",
                "model": settings.GEMINI_MODEL_NAME,
                "configured": is_valid_key(
                    settings.GEMINI_API_KEY, "your_gemini_api_key"
                ),
                "is_active": active_provider == "gemini",
                "type": "cloud_api",
            },
            {
                "id": "claude",
                "name": "Anthropic Claude",
                "badge": "Cloud Reasoning",
                "description": "Anthropic Claude Messages API",
                "model": settings.CLAUDE_MODEL_NAME,
                "configured": is_valid_key(
                    settings.ANTHROPIC_API_KEY or settings.CLAUDE_API_KEY,
                    "your_anthropic_api_key",
                ),
                "is_active": active_provider == "claude",
                "type": "cloud_api",
            },
        ]
