"""Model loader utility for thread-safe artifact caching."""

from typing import Any, Dict, Optional

from app.core.logging import logger


class ModelLoader:
    """Thread-safe registry and cache for trained model artifacts."""

    _instances: Dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_name: str) -> Optional[Any]:
        return cls._instances.get(model_name)

    @classmethod
    def register_model(cls, model_name: str, instance: Any) -> None:
        logger.info(f"Registering model artifact: {model_name}")
        cls._instances[model_name] = instance

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        return model_name in cls._instances
