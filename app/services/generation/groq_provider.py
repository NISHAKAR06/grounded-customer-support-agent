"""Groq LLM provider implementation for ultra-low latency inference."""

from typing import Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.core.logging import logger
from app.services.generation.llm_service import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Generates grounded replies using Groq LPU cloud inference API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 20.0,
    ):
        self.settings = get_settings()
        self.api_key = self.settings.GROQ_API_KEY if api_key is None else api_key
        self.model_name = model_name or self.settings.GROQ_MODEL_NAME
        self.base_url = (base_url or self.settings.GROQ_BASE_URL).rstrip("/")
        self.timeout = timeout

    def provider_name(self) -> str:
        return f"groq ({self.model_name})"

    def generate(self, prompt: str) -> str:
        """Call Groq chat completions endpoint."""
        if not self.api_key or self.api_key.startswith("your_groq_api_key"):
            raise LLMProviderException(
                "Groq API key is missing or unconfigured. Set GROQ_API_KEY in .env."
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
                        "You are a professional customer support agent. "
                        "Follow grounded historical evidence strictly without hallucination."
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
                    if response.status_code == 404 and "model_not_found" in error_msg:
                        logger.warning(
                            f"Groq model '{self.model_name}' not found. "
                            f"Auto-adapting to available model 'groq/compound-mini'."
                        )
                        self.model_name = "groq/compound-mini"
                        payload["model"] = "groq/compound-mini"
                        response = client.post(endpoint, headers=headers, json=payload)

                    if response.status_code != 200:
                        error_msg = response.text
                        logger.error(f"Groq error (HTTP {response.status_code}): {error_msg}")
                        raise LLMProviderException(
                            f"Groq API error ({response.status_code}): {error_msg}"
                        )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMProviderException("Groq returned zero completion choices.")

                reply = choices[0].get("message", {}).get("content", "").strip()
                return reply
        except httpx.TimeoutException as ex:
            raise LLMProviderException(f"Groq API request timed out after {self.timeout}s.") from ex
        except LLMProviderException:
            raise
        except Exception as ex:
            raise LLMProviderException(f"Groq execution failed: {str(ex)}") from ex
