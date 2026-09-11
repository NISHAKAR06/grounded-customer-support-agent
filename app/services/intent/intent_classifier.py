"""Intent classification base classes and abstractions."""

from abc import ABC, abstractmethod

from app.models.domain_models import IntentPrediction


class BaseIntentClassifier(ABC):
    """Abstract base class for all intent classification models."""

    @abstractmethod
    def classify(self, text: str) -> IntentPrediction:
        """Classify customer text into an intent with confidence and attribution signals."""
        pass


class RuleBasedIntentClassifier(BaseIntentClassifier):
    """Domain-specific heuristic classifier implementing the 7-class @AppleSupport taxonomy."""

    def classify(self, text: str) -> IntentPrediction:
        from app.models.domain_models import INTENT_LABELS
        from scripts.data.classify_intents import classify_text_intent

        intent_enum, conf, signals = classify_text_intent(text)
        return IntentPrediction(
            name=INTENT_LABELS[intent_enum],
            confidence=round(conf, 2),
            code=intent_enum,
            signals=signals,
            alternatives=[{"Other": round(1.0 - conf, 2)}] if conf < 1.0 else [],
        )


class IntentClassifier(BaseIntentClassifier):
    """Production intent classifier coordinator that delegates to the active model artifact."""

    def __init__(self, model: BaseIntentClassifier = None):
        self._model = model or RuleBasedIntentClassifier()

    def set_model(self, model: BaseIntentClassifier) -> None:
        self._model = model

    def classify(self, text: str) -> IntentPrediction:
        return self._model.classify(text)
