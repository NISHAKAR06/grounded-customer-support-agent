"""Evidence ranker for filtering and ordering historical cases."""

from typing import List

from app.models.domain_models import HistoricalCase, ResolutionStatus


class EvidenceRanker:
    """Ranks and filters candidate historical support cases."""

    def __init__(self, min_similarity: float = 0.65):
        self.min_similarity = min_similarity

    def rank_and_filter(self, candidates: List[HistoricalCase]) -> List[HistoricalCase]:
        """Filter out unresolved cases and those below similarity threshold, sorted by similarity."""
        filtered = [
            c
            for c in candidates
            if c.resolution_status == ResolutionStatus.RESOLVED
            and c.similarity >= self.min_similarity
        ]
        return sorted(filtered, key=lambda x: x.similarity, reverse=True)
