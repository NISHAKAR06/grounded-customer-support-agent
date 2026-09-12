"""Google Gemini LLM provider implementation."""

from typing import Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.core.logging import logger
from app.services.generation.llm_service import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Generates grounded replies using Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.model_name = model_name or self.settings.GEMINI_MODEL_NAME
        self.timeout = timeout

    def provider_name(self) -> str:
        return f"gemini ({self.model_name})"

    def generate(self, prompt: str) -> str:
        """Call Gemini REST API synchronously."""
        if not self.api_key or self.api_key.startswith("your_gemini_api_key"):
            logger.warning(
                "Gemini API key is not configured or is placeholder. Triggering provider fallback."
            )
            raise LLMProviderException(
                "Gemini API key is missing or unconfigured. Set GEMINI_API_KEY in .env."
            )

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are a grounded customer support assistant. "
                                "Adhere strictly to the historical evidence and resolution guidelines provided. "
                                "Draft a helpful, professional reply.\n\n"
                                f"{prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload)
                if response.status_code != 200:
                    error_msg = response.text
                    logger.error(f"Gemini error (HTTP {response.status_code}): {error_msg}")
                    raise LLMProviderException(
                        f"Gemini API error ({response.status_code}): {error_msg}"
                    )

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise LLMProviderException("Gemini returned zero response candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise LLMProviderException("Gemini returned empty candidate parts.")

                reply = parts[0].get("text", "").strip()
                return reply
        except httpx.TimeoutException as ex:
            raise LLMProviderException(
                f"Gemini API request timed out after {self.timeout}s."
            ) from ex
        except LLMProviderException:
            raise
        except Exception as ex:
            raise LLMProviderException(f"Gemini execution failed: {str(ex)}") from ex
