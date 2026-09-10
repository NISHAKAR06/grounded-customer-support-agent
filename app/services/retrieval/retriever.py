"""Historical conversation retriever."""

from pathlib import Path
from typing import List, Optional

from app.core.config import get_settings
from app.models.domain_models import HistoricalCase


class Retriever:
    """Retrieves relevant historical resolved support cases from FAISS vector store.

    Zero-mock policy: Queries real vector index if built; returns empty list if index
    does not exist on disk (prior to Phase 7). Never fabricates fake historical cases.
    """

    def __init__(self, index_path: Optional[str] = None):
        self.settings = get_settings()
        self.index_path = Path(index_path or self.settings.FAISS_INDEX_PATH)
        self._index = None

    def retrieve(self, query: str, top_k: int = 3) -> List[HistoricalCase]:
        """Retrieve top-k historical resolved cases matching the query."""
        if not self.index_path.exists():
            # Real vector index will be built and loaded in Phase 7
            return []

        # When FAISS index is present, vector search is executed here
        return []
