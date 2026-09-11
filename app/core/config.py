"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System settings for Grounded Customer Support Agent."""

    # Application Information
    APP_NAME: str = "Grounded Customer Support Agent"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # Target Brand Selection (Phase 2)
    TARGET_BRAND: str = "AppleSupport"
    TARGET_BRAND_NAME: str = "Apple Support"
    TARGET_BRAND_HANDLE: str = "@AppleSupport"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"

    # LLM Settings
    LLM_PROVIDER: str = "mock"  # 'mock', 'ollama', 'openai', 'groq', 'gemini', 'claude'

    # Mock Provider
    MOCK_MODEL_NAME: str = "mock-grounded-v1"

    # Ollama Provider (Local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "llama3.2"

    # OpenAI Provider
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # Groq Provider
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Gemini Provider
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

    # Claude (Anthropic) Provider
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    CLAUDE_MODEL_NAME: str = "claude-3-5-haiku-20241022"

    # Legacy / Local Model Fallback Path
    LOCAL_MODEL_PATH: str = "models/local_llm/default"
    LOCAL_MODEL_DEVICE: str = "cpu"

    # Intent Classification
    INTENT_MODEL_PATH: str = "models/intent_classifier"
    INTENT_CONFIDENCE_THRESHOLD: float = 0.85

    # Historical Retrieval & Vector Store
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "models/embedding_model/faiss.index"
    RETRIEVAL_TOP_K: int = 3
    RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.55

    # Escalation Policy Thresholds
    ESCALATION_MIN_CONFIDENCE: float = 0.80
    ESCALATION_MIN_RETRIEVAL_SCORE: float = 0.55

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
