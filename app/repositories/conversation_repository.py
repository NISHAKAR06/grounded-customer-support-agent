"""Conversation repository accessing reconstructed customer support records."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings


class ConversationRepository:
    """Provides structured data access to real reconstructed conversations for the support inbox.

    Zero-mock policy: Reads exclusively from data/processed/conversations.jsonl if present.
    Returns an empty list when data has not yet been processed (prior to Phase 3).
    """

    def __init__(self, data_path: Optional[Path] = None):
        settings = get_settings()
        self.data_path = self._resolve_data_path(data_path, settings)
        self._conversations: List[Dict[str, Any]] = []
        self._load_conversations()

    def _resolve_data_path(self, data_path: Optional[Path], settings: Any) -> Optional[Path]:
        if data_path and data_path.is_file():
            return data_path

        candidates = [
            settings.DATA_DIR / "processed" / "applesupport_conversations.jsonl",
            settings.DATA_DIR / "processed" / "applesupport_sample.jsonl",
            settings.DATA_DIR / "processed" / "conversations.jsonl",
        ]
        for p in candidates:
            if p.is_file():
                return p
        return settings.DATA_DIR / "processed" / "applesupport_conversations.jsonl"

    def _load_conversations(self) -> None:
        """Load real reconstructed conversations from disk if available."""
        if self.data_path and self.data_path.exists():
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._conversations.append(json.loads(line))
            except Exception:
                self._conversations = []
        else:
            self._conversations = []

    def list_conversations(
        self,
        status_filter: str = "all",
        decision_filter: str = "all",
        turn_filter: str = "all",
        intent_filter: str = "all",
        sort_by: str = "newest",
        search_query: str = "",
    ) -> List[Dict[str, Any]]:
        """Filter and sort conversations."""
        results = list(self._conversations)

        # 1. Filter by operational status
        if status_filter == "auto_ready":
            results = [c for c in results if c.get("status") == "AI Ready"]
        elif status_filter == "needs_human":
            results = [c for c in results if c.get("status") == "Needs Human"]
        elif status_filter == "resolved":
            results = [c for c in results if c.get("status") == "Resolved"]

        # 2. Filter by routing decision
        if decision_filter == "auto_handle":
            results = [c for c in results if c.get("decision") == "AUTO_HANDLE"]
        elif decision_filter == "human_escalation":
            results = [c for c in results if c.get("decision") == "HUMAN_ESCALATION"]

        # 3. Filter by conversation depth (turn count)
        if turn_filter == "short":
            results = [c for c in results if int(c.get("turn_count") or 0) <= 2]
        elif turn_filter == "medium":
            results = [c for c in results if 3 <= int(c.get("turn_count") or 0) <= 4]
        elif turn_filter == "deep":
            results = [c for c in results if int(c.get("turn_count") or 0) >= 5]

        # 4. Filter by classified intent
        if intent_filter and intent_filter != "all":
            results = [
                c
                for c in results
                if str(c.get("intent_code", "")).lower() == intent_filter.lower()
                or str(c.get("intent", "")).lower() == intent_filter.lower()
            ]

        # 5. Filter by search query (ticket ID, customer ID, or message text)
        if search_query:
            q = search_query.strip().lower()
            results = [
                c
                for c in results
                if q in str(c.get("conversation_id", "")).lower()
                or q in str(c.get("ticket_id", "")).lower()
                or q in str(c.get("customer_id", "")).lower()
                or q in str(c.get("latest_message", "")).lower()
                or q in str(c.get("first_inquiry", "")).lower()
            ]

        # 6. Sort conversations
        if sort_by == "oldest":
            results.sort(key=lambda x: str(x.get("created_at", "")))
        elif sort_by == "confidence_desc":
            results.sort(key=lambda x: float(x.get("confidence") or 0.0), reverse=True)
        elif sort_by == "confidence_asc":
            results.sort(key=lambda x: float(x.get("confidence") or 0.0))
        elif sort_by == "turns_desc":
            results.sort(key=lambda x: int(x.get("turn_count") or 0), reverse=True)
        elif sort_by == "turns_asc":
            results.sort(key=lambda x: int(x.get("turn_count") or 0))
        elif sort_by == "ticket_asc":
            results.sort(key=lambda x: str(x.get("ticket_id", "")))
        else:  # newest default
            results.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)

        return results

    @staticmethod
    def _build_pagination_display(current_page: int, total_pages: int) -> List[Any]:
        """
        Build windowed pagination with ellipsis markers.
        Produces page sequences like: [1, 2, 3, 4, 5, '...', 10]
        """
        if total_pages <= 7:
            return list(range(1, total_pages + 1))

        if current_page <= 4:
            return [1, 2, 3, 4, 5, "...", total_pages]

        if current_page >= total_pages - 3:
            return [1, "..."] + list(range(total_pages - 4, total_pages + 1))

        return [
            1,
            "...",
            current_page - 1,
            current_page,
            current_page + 1,
            "...",
            total_pages,
        ]

    def paginate(
        self,
        status_filter: str = "all",
        decision_filter: str = "all",
        turn_filter: str = "all",
        intent_filter: str = "all",
        sort_by: str = "newest",
        search_query: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        """Return paginated slice with full pagination window metadata."""
        import math

        items = self.list_conversations(
            status_filter=status_filter,
            decision_filter=decision_filter,
            turn_filter=turn_filter,
            intent_filter=intent_filter,
            sort_by=sort_by,
            search_query=search_query,
        )
        total_items = len(items)
        page_size = max(5, min(page_size, 100))
        total_pages = max(1, math.ceil(total_items / page_size)) if total_items > 0 else 1
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        sliced_items = items[start_idx:end_idx]

        pages_display = self._build_pagination_display(page, total_pages)

        return {
            "items": sliced_items,
            "total_items": total_items,
            "total_pages": total_pages,
            "current_page": page,
            "page_size": page_size,
            "start_item": start_idx + 1 if total_items > 0 else 0,
            "end_item": end_idx,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "prev_page": page - 1,
            "next_page": page + 1,
            "pages_display": pages_display,
            "pages_range": [p for p in pages_display if isinstance(p, int)],
        }

    def get_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        for c in self._conversations:
            if c.get("conversation_id") == conversation_id:
                return c
        return None
