"""Historical conversation retriever."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.models.domain_models import HistoricalCase, ResolutionStatus

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant historical resolved support cases from FAISS vector store.

    Strict Zero-Mock Policy:
    Queries real MiniLM + FAISS vector store built over historical AppleSupport
    conversations. Never fabricates fake historical cases.
    """

    _model: Optional[SentenceTransformer] = None

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.index_path = Path(index_path or self.settings.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or (self.index_path.parent / "case_metadata.json"))
        self.model_name = model_name or self.settings.EMBEDDING_MODEL_NAME

        self._index: Optional[faiss.Index] = None
        self._metadata: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def _get_embedding_model(cls, model_name: str) -> SentenceTransformer:
        """Cached singleton embedding model."""
        if cls._model is None:
            logger.info("Loading retrieval embedding model: %s", model_name)
            cls._model = SentenceTransformer(model_name)
        return cls._model

    def _ensure_loaded(self) -> bool:
        """Lazy load FAISS index and paired case metadata."""
        if self._index is not None and self._metadata is not None:
            return True

        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning(
                "FAISS index (%s) or metadata (%s) not found on disk.",
                self.index_path,
                self.metadata_path,
            )
            return False

        try:
            logger.info("Reading FAISS index from %s", self.index_path)
            self._index = faiss.read_index(str(self.index_path))

            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            logger.info(
                "Loaded FAISS index with %d vectors and %d metadata records.",
                self._index.ntotal,
                len(self._metadata),
            )
            return True
        except Exception as e:
            logger.error("Failed to load FAISS index or metadata: %s", e)
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        intent_filter: Optional[str] = None,
    ) -> List[HistoricalCase]:
        """Retrieve top-k historical resolved cases matching the query with cosine similarity."""
        query = (query or "").strip()
        if not query:
            return []

        if not self._ensure_loaded():
            return []

        try:
            model = self._get_embedding_model(self.model_name)
            embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(embedding)

            fetch_k = top_k * 10 if intent_filter else top_k
            fetch_k = min(fetch_k, self._index.ntotal)
            scores, indices = self._index.search(embedding, fetch_k)

            matched_results: List[HistoricalCase] = []
            fallback_results: List[HistoricalCase] = []

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._metadata):
                    continue

                case_data = self._metadata[idx]
                similarity = float(np.clip(score, 0.0, 1.0))

                case_obj = HistoricalCase(
                    case_id=case_data["case_id"],
                    similarity=round(similarity, 4),
                    customer_text=case_data["customer_text"],
                    brand_response=case_data["brand_response"],
                    resolution_status=ResolutionStatus(
                        case_data.get("resolution_status", "RESOLVED")
                    ),
                    metadata={
                        "ticket_id": case_data.get("ticket_id", ""),
                        "intent": case_data.get("intent", ""),
                        "has_resolution": case_data.get("has_resolution", False),
                        "has_dm": case_data.get("has_dm", False),
                        "has_kb_link": case_data.get("has_kb_link", False),
                    },
                )

                if intent_filter and case_data.get("intent") == intent_filter:
                    matched_results.append(case_obj)
                else:
                    fallback_results.append(case_obj)

            # Prioritize intent-matched cases, backfill with remaining highest similarity
            combined = matched_results + fallback_results
            return combined[:top_k]
        except Exception as e:
            logger.error("Error during vector retrieval: %s", e)
            return []
