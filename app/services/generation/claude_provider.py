"""Anthropic Claude LLM provider implementation."""

from typing import Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.core.logging import logger
from app.services.generation.llm_service import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """Generates grounded replies using Anthropic Claude Messages API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.settings = get_settings()
        self.api_key = (
            api_key or self.settings.ANTHROPIC_API_KEY or self.settings.CLAUDE_API_KEY
        )
        self.model_name = model_name or self.settings.CLAUDE_MODEL_NAME
        self.timeout = timeout

    def provider_name(self) -> str:
        return f"claude ({self.model_name})"

    def generate(self, prompt: str) -> str:
        """Call Anthropic Messages API."""
        if not self.api_key or self.api_key.startswith("your_anthropic_api_key"):
            raise LLMProviderException(
                "Anthropic/Claude API key is missing or unconfigured. Set ANTHROPIC_API_KEY in .env."
            )

        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 512,
            "temperature": 0.2,
            "system": (
                "You are an enterprise customer support agent. "
                "Adhere strictly to the provided historical resolution guidelines and evidence. "
                "Never promise refunds or services not verified by the grounding evidence."
            ),
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    error_msg = response.text
                    logger.error(
                        f"Claude error (HTTP {response.status_code}): {error_msg}"
                    )
                    raise LLMProviderException(
                        f"Claude API error ({response.status_code}): {error_msg}"
                    )

                data = response.json()
                content_blocks = data.get("content", [])
                if not content_blocks:
                    raise LLMProviderException(
                        "Claude returned zero message content blocks."
                    )

                text_blocks = [
                    b.get("text", "") for b in content_blocks if b.get("type") == "text"
                ]
                reply = "\n".join(text_blocks).strip()
                return reply
        except httpx.TimeoutException as ex:
            raise LLMProviderException(
                f"Claude API request timed out after {self.timeout}s."
            ) from ex
        except LLMProviderException:
            raise
        except Exception as ex:
            raise LLMProviderException(f"Claude execution failed: {str(ex)}") from ex
