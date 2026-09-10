"""Custom domain exceptions for Grounded Customer Support Agent."""

from typing import Any, Dict, Optional


class AppException(Exception):
    """Base exception for all domain-specific errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelNotLoadedException(AppException):
    """Raised when an intent classifier or embedding model is required but not loaded."""

    pass


class IntentClassificationException(AppException):
    """Raised when intent classification encounters an unrecoverable failure."""

    pass


class RetrievalException(AppException):
    """Raised when vector similarity search or evidence retrieval fails."""

    pass


class LLMProviderException(AppException):
    """Raised when an LLM provider (e.g. Gemini) encounters an API or connectivity failure."""

    pass


class ResponseValidationException(AppException):
    """Raised when response validation detects a fatal hallucination or policy failure."""

    pass


class EscalationPolicyException(AppException):
    """Raised when escalation policy evaluation fails."""

    pass
