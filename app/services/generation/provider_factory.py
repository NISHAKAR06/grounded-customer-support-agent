"""Factory and registry for all supported LLM generation providers."""

from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.services.generation.claude_provider import ClaudeProvider
from app.services.generation.gemini_provider import GeminiProvider
from app.services.generation.groq_provider import GroqProvider
from app.services.generation.llm_service import BaseLLMProvider
from app.services.generation.ollama_provider import OllamaProvider
from app.services.generation.openai_provider import OpenAIProvider


class LLMProviderFactory:
    """Factory for dynamically creating and inspecting LLM providers.

    Supported Real Inference Providers:
        - groq: Groq ultra-low latency cloud inference (Llama-3.1, Compound-Mini)
        - ollama: Local self-hosted LLM (Llama, Qwen, Mistral)
        - openai: OpenAI Chat Completions (GPT-4o mini, GPT-4o)
        - gemini: Google Gemini (1.5 Flash, 2.0 Flash)
        - claude: Anthropic Claude (3.5 Haiku, 3.5 Sonnet)
    """

    SUPPORTED_PROVIDERS = ("groq", "ollama", "openai", "gemini", "claude")

    @classmethod
    def normalize_provider_name(cls, provider_name: Optional[str]) -> str:
        """Normalize provider aliases and casing."""
        if not provider_name:
            settings = get_settings()
            provider_name = settings.LLM_PROVIDER

        clean = provider_name.strip().lower()
        alias_map = {
            "local": "ollama",
            "anthropic": "claude",
            "google": "gemini",
        }
        return alias_map.get(clean, clean)

    @classmethod
    def create_provider(cls, provider_name: Optional[str] = None, **kwargs) -> BaseLLMProvider:
        """Instantiate an LLM provider by identifier name."""
        name = cls.normalize_provider_name(provider_name)

        if name == "groq":
            return GroqProvider(**kwargs)
        elif name == "ollama":
            return OllamaProvider(**kwargs)
        elif name == "openai":
            return OpenAIProvider(**kwargs)
        elif name == "gemini":
            return GeminiProvider(**kwargs)
        elif name == "claude":
            return ClaudeProvider(**kwargs)
        else:
            raise LLMProviderException(
                f"Unsupported LLM provider '{provider_name}'. "
                f"Supported real inference providers: {list(cls.SUPPORTED_PROVIDERS)}"
            )

    @classmethod
    def create_fallback_provider(cls) -> BaseLLMProvider:
        """Create a real secondary inference provider when primary engine fails.

        Zero-Mock Policy: Engages alternative live models (Groq <-> Ollama),
        never offline mock simulations.
        """
        settings = get_settings()
        primary = cls.normalize_provider_name(settings.LLM_PROVIDER)

        # If Groq is primary, fallback to local Ollama
        if primary == "groq":
            return OllamaProvider()
        # If Ollama is primary, fallback to cloud Groq if configured
        elif primary == "ollama":
            if settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("your_"):
                return GroqProvider()
            return OpenAIProvider()
        elif settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("your_"):
            return GroqProvider()
        else:
            return OllamaProvider()

    @classmethod
    def list_available_providers(cls, **kwargs) -> List[Dict[str, Any]]:
        """Return catalog of supported real providers and configuration status."""
        settings = get_settings()
        active_provider = cls.normalize_provider_name(settings.LLM_PROVIDER)

        def is_valid_key(key: Optional[str], placeholder: str) -> bool:
            return bool(key and not key.startswith(placeholder) and len(key) > 5)

        return [
            {
                "id": "groq",
                "name": "Groq",
                "badge": "Ultra-Fast LPU",
                "description": "Groq cloud high-speed inference engine (Llama-3.1 / Compound-Mini)",
                "model": settings.GROQ_MODEL_NAME,
                "configured": is_valid_key(settings.GROQ_API_KEY, "your_groq_api_key"),
                "is_active": active_provider == "groq",
                "type": "cloud_api",
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
                "configured": is_valid_key(settings.OPENAI_API_KEY, "your_openai_api_key"),
                "is_active": active_provider == "openai",
                "type": "cloud_api",
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "badge": "Cloud Multimodal",
                "description": "Google AI Studio / Vertex Gemini API",
                "model": settings.GEMINI_MODEL_NAME,
                "configured": is_valid_key(settings.GEMINI_API_KEY, "your_gemini_api_key"),
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
