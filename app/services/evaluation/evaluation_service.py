"""Evaluation harness service for baseline benchmarks and metrics aggregation."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import get_settings


class EvaluationService:
    """Manages benchmark metrics across models, golden set runs, and judge rubrics.

    Zero-mock policy: Reads exclusively from evaluation/benchmark_results.json when
    reproducible benchmarks have executed. Never fabricates evaluation numbers.
    """

    def __init__(self, results_path: Optional[Path] = None):
        settings = get_settings()
        self.results_path = results_path or (
            settings.BASE_DIR / "evaluation" / "benchmark_results.json"
        )

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Return benchmark comparison metrics across systems."""
        if self.results_path and self.results_path.exists():
            try:
                with open(self.results_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Explicit unmeasured/pending state prior to Phase 6 & 11 execution
        return {
            "status": "pending_model_training",
            "message": "Model training and evaluation benchmarks scheduled in Phase 6 & Phase 11.",
            "golden_set_count": 0,
            "models": {},
            "retrieval": {},
            "llm_judge": {},
        }
