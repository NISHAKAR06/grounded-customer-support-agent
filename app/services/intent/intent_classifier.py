"""Intent classification base classes and abstractions."""

from abc import ABC, abstractmethod

from app.models.domain_models import IntentPrediction


class BaseIntentClassifier(ABC):
    """Abstract base class for all intent classification models."""

    @abstractmethod
    def classify(self, text: str) -> IntentPrediction:
        """Classify customer text into an intent with confidence and attribution signals."""
        pass


class IntentClassifier(BaseIntentClassifier):
    """Production intent classifier coordinator that delegates to the active model artifact."""

    def __init__(self, model: BaseIntentClassifier = None):
        self._model = model

    def set_model(self, model: BaseIntentClassifier) -> None:
        self._model = model

    def classify(self, text: str) -> IntentPrediction:
        if self._model is None:
            # Safe architectural fallback stub prior to Phase 6 training
            return IntentPrediction(
                name="General Inquiry / Support",
                confidence=0.90,
                signals=["inquiry", "support request"],
                alternatives=[{"Other": 0.10}],
            )
        return self._model.classify(text)
