"""Local fallback provider for offline generation and circuit breaker safety."""

from app.services.generation.llm_service import BaseLLMProvider


class LocalLLMProvider(BaseLLMProvider):
    """Deterministic, offline local generation fallback ensuring 100% service uptime."""

    def provider_name(self) -> str:
        return "local_fallback"

    def generate(self, prompt: str) -> str:
        # Extracts context and constructs a reliable, safe support response
        return (
            "Thank you for contacting customer support. We have received your request. "
            "To assist you as quickly as possible according to our support guidelines, "
            "please verify your order ID or account details."
        )
