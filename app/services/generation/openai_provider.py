"""OpenAI LLM provider implementation."""

from typing import Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.core.logging import logger
from app.services.generation.llm_service import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Generates grounded replies using OpenAI chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.OPENAI_API_KEY
        self.model_name = model_name or self.settings.OPENAI_MODEL_NAME
        self.base_url = (base_url or self.settings.OPENAI_BASE_URL).rstrip("/")
        self.timeout = timeout

    def provider_name(self) -> str:
        return f"openai ({self.model_name})"

    def generate(self, prompt: str) -> str:
        """Call OpenAI chat completions endpoint."""
        if not self.api_key or self.api_key.startswith("your_openai_api_key"):
            raise LLMProviderException(
                "OpenAI API key is missing or unconfigured. Set OPENAI_API_KEY in .env."
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional customer support representative. "
                        "Draft responses strictly grounded in the historical evidence provided. "
                        "Do not invent facts, unverified warranties, or unauthorized refunds."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    error_msg = response.text
                    logger.error(
                        f"OpenAI error (HTTP {response.status_code}): {error_msg}"
                    )
                    raise LLMProviderException(
                        f"OpenAI API error ({response.status_code}): {error_msg}"
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMProviderException(
                        "OpenAI returned zero completion choices."
                    )

                reply = choices[0].get("message", {}).get("content", "").strip()
                return reply
        except httpx.TimeoutException as ex:
            raise LLMProviderException(
                f"OpenAI API request timed out after {self.timeout}s."
            ) from ex
        except LLMProviderException:
            raise
        except Exception as ex:
            raise LLMProviderException(f"OpenAI execution failed: {str(ex)}") from ex
