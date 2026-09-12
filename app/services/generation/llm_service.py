"""LLM Provider abstraction and service coordinator."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from app.core.logging import logger


class BaseLLMProvider(ABC):
    """Abstract interface for language model generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response string from the given prompt."""
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """Return identifier name of the provider."""
        pass


class LLMService:
    """Coordinates LLM generation with multi-provider support, circuit breaking, and fallback."""

    def __init__(
        self,
        primary_provider: Optional[BaseLLMProvider] = None,
        fallback_provider: Optional[BaseLLMProvider] = None,
    ):
        from app.services.generation.provider_factory import LLMProviderFactory

        self.primary = primary_provider or LLMProviderFactory.create_provider()
        self.fallback = fallback_provider or LLMProviderFactory.create_fallback_provider()

    def generate_reply(self, prompt: str, provider_name: Optional[str] = None) -> Tuple[str, str]:
        """Generate reply using the designated or active provider, falling back on failure.

        Args:
            prompt: Text prompt with grounded constraints and historical evidence.
            provider_name: Optional provider override ('groq', 'ollama', 'openai', 'gemini', 'claude').

        Returns:
            Tuple of (generated_text, active_provider_name)
        """
        target_provider = self.primary

        if provider_name:
            from app.services.generation.provider_factory import LLMProviderFactory

            try:
                target_provider = LLMProviderFactory.create_provider(provider_name)
            except Exception as ex:
                logger.warning(
                    f"Failed to initialize requested provider '{provider_name}': {ex}. "
                    f"Falling back to default primary ({self.primary.provider_name()})."
                )
                target_provider = self.primary

        try:
            reply = target_provider.generate(prompt)
            return reply, target_provider.provider_name()
        except Exception as ex:
            # Automatic circuit breaker fallback
            logger.warning(
                f"LLM generation failed on provider '{target_provider.provider_name()}': {ex}. "
                f"Engaging fallback provider '{self.fallback.provider_name()}'."
            )
            reply = self.fallback.generate(prompt)
            return reply, self.fallback.provider_name()
