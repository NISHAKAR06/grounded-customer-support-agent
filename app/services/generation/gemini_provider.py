"""Google Gemini LLM provider implementation."""

from typing import Optional

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.core.logging import logger
from app.services.generation.llm_service import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Generates replies using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.model_name = model_name or self.settings.GEMINI_MODEL_NAME

    def provider_name(self) -> str:
        return f"gemini ({self.model_name})"

    def generate(self, prompt: str) -> str:
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning(
                "Gemini API key is not configured or is placeholder. Triggering provider fallback."
            )
            raise LLMProviderException("Gemini API key is missing or unconfigured.")

        # In Phase 8, actual google-genai client calls will be integrated here
        # For Phase 0 baseline, raising provider exception ensures fallback is exercised cleanly
        raise LLMProviderException(
            "Gemini generation pipeline will be activated in Phase 8."
        )
