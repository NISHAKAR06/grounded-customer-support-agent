"""Ollama local LLM provider implementation."""

from typing import Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderException
from app.core.logging import logger
from app.services.generation.llm_service import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Generates replies via local Ollama instance (e.g. Llama-3.2, Mistral, Phi-3)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 45.0,
    ):
        self.settings = get_settings()
        self.base_url = (base_url or self.settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or self.settings.OLLAMA_MODEL_NAME
        self.timeout = timeout

    def provider_name(self) -> str:
        return f"ollama ({self.model_name})"

    def generate(self, prompt: str) -> str:
        """Call Ollama /api/generate endpoint synchronously."""
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload)
                if response.status_code != 200:
                    error_detail = response.text
                    # If model not found, try to auto-adapt to any installed local model
                    if response.status_code == 404 and "not found" in error_detail.lower():
                        try:
                            tags_resp = client.get(f"{self.base_url}/api/tags", timeout=5.0)
                            if tags_resp.status_code == 200:
                                models = [
                                    m.get("name")
                                    for m in tags_resp.json().get("models", [])
                                    if m.get("name")
                                ]
                                if models:
                                    fallback_model = models[0]
                                    logger.warning(
                                        f"Ollama model '{self.model_name}' not found. "
                                        f"Auto-adapting to installed local model '{fallback_model}'."
                                    )
                                    self.model_name = fallback_model
                                    payload["model"] = fallback_model
                                    response = client.post(endpoint, json=payload)
                        except Exception as tag_err:
                            logger.debug(f"Failed to query Ollama tags for fallback: {tag_err}")

                    if response.status_code != 200:
                        error_detail = response.text
                        logger.error(f"Ollama returned HTTP {response.status_code}: {error_detail}")
                        raise LLMProviderException(
                            f"Ollama error (HTTP {response.status_code}): {error_detail}"
                        )

                data = response.json()
                reply = data.get("response", "").strip()
                if not reply:
                    raise LLMProviderException("Ollama returned an empty response string.")
                return reply
        except httpx.ConnectError as ex:
            logger.warning(f"Cannot connect to Ollama at {self.base_url}. Service may be offline.")
            raise LLMProviderException(
                f"Ollama connection refused at {self.base_url}. Ensure 'ollama serve' is running."
            ) from ex
        except httpx.TimeoutException as ex:
            logger.warning(f"Ollama request timed out after {self.timeout}s.")
            raise LLMProviderException(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from ex
        except LLMProviderException:
            raise
        except Exception as ex:
            logger.error(f"Unexpected error during Ollama generation: {ex}")
            raise LLMProviderException(f"Ollama execution failed: {str(ex)}") from ex
