"""Golden Set Repository for programmatic access to verified benchmark samples."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings


class GoldenSetRepository:
    """Thread-safe access to the 200-sample hand-verified Golden Evaluation Set."""

    def __init__(self, data_path: Optional[Path] = None):
        self.settings = get_settings()
        self.data_path = data_path or (self.settings.DATA_DIR / "golden" / "golden_set.jsonl")
        self.summary_path = self.settings.DATA_DIR / "golden" / "golden_set_summary.json"
        self._samples: Optional[List[Dict[str, Any]]] = None

    def _ensure_loaded(self) -> List[Dict[str, Any]]:
        """Load and cache golden records in memory."""
        if self._samples is None:
            records = []
            if self.data_path.exists():
                with open(self.data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            self._samples = records
        return self._samples

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all curated golden set samples."""
        return self._ensure_loaded()

    def count(self) -> int:
        """Return total sample count."""
        return len(self._ensure_loaded())

    def get_by_sample_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Find a golden sample by its sample_id (e.g. 'gold_001')."""
        for sample in self._ensure_loaded():
            if sample.get("sample_id") == sample_id:
                return sample
        return None

    def filter(
        self,
        intent_code: Optional[str] = None,
        routing: Optional[str] = None,
        complexity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter golden samples by intent, routing decision, or complexity."""
        samples = self._ensure_loaded()
        if intent_code and intent_code != "all":
            samples = [s for s in samples if s.get("gold_intent_code") == intent_code]
        if routing and routing != "all":
            samples = [s for s in samples if s.get("gold_routing") == routing]
        if complexity and complexity != "all":
            samples = [s for s in samples if s.get("complexity") == complexity]
        return samples

    def paginate(
        self,
        limit: int = 50,
        offset: int = 0,
        intent_code: Optional[str] = None,
        routing: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Paginate filtered golden records, returning (page_items, total_count)."""
        filtered = self.filter(intent_code=intent_code, routing=routing)
        total = len(filtered)
        paged = filtered[offset : offset + limit]
        return paged, total

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics loaded from summary artifact or computed on the fly."""
        if self.summary_path.exists():
            with open(self.summary_path, "r", encoding="utf-8") as f:
                return json.load(f)

        samples = self._ensure_loaded()
        intent_dist: Dict[str, int] = {}
        routing_dist = {"AUTO_HANDLE": 0, "HUMAN_ESCALATION": 0}

        for s in samples:
            ic = s.get("gold_intent_code", "UNKNOWN")
            intent_dist[ic] = intent_dist.get(ic, 0) + 1
            rt = s.get("gold_routing", "HUMAN_ESCALATION")
            routing_dist[rt] = routing_dist.get(rt, 0) + 1

        return {
            "total_samples": len(samples),
            "intent_distribution": intent_dist,
            "routing_distribution": routing_dist,
        }
