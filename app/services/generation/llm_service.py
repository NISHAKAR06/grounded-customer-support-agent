"""LLM Provider abstraction and service coordinator."""

from abc import ABC, abstractmethod
from typing import Tuple


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
    """Coordinates LLM generation with transparent circuit breaking and fallback."""

    def __init__(
        self, primary_provider: BaseLLMProvider, fallback_provider: BaseLLMProvider
    ):
        self.primary = primary_provider
        self.fallback = fallback_provider

    def generate_reply(self, prompt: str) -> Tuple[str, str]:
        """Generate reply, attempting primary provider first and failing gracefully to fallback.

        Returns: (generated_text, active_provider_name)
        """
        try:
            reply = self.primary.generate(prompt)
            return reply, self.primary.provider_name()
        except Exception:
            # Automatic circuit breaker fallback
            reply = self.fallback.generate(prompt)
            return reply, self.fallback.provider_name()
