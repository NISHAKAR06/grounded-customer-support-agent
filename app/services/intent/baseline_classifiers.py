"""Baseline Intent Classifier implementations."""

from collections import Counter
from typing import List

import numpy as np


class MajorityClassifier:
    """Trivial baseline classifier that unconditionally predicts the statistical training mode."""

    def __init__(self, majority_class: str = "GENERAL_INQUIRY"):
        self.majority_class: str = majority_class
        self.majority_ratio: float = 0.0
        self.class_priors: dict = {}

    def fit(self, y: List[str]):
        counts = Counter(y)
        total = len(y)
        self.majority_class = counts.most_common(1)[0][0]
        self.majority_ratio = counts[self.majority_class] / total
        self.class_priors = {k: v / total for k, v in counts.items()}
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_class] * len(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        classes = sorted(self.class_priors.keys())
        priors = [self.class_priors.get(c, 0.0) for c in classes]
        return np.tile(priors, (len(texts), 1))
