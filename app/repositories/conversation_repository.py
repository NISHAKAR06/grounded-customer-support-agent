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
        self.data_path = data_path or (
            settings.DATA_DIR / "processed" / "conversations.jsonl"
        )
        self._conversations: List[Dict[str, Any]] = []
        self._load_conversations()

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

    def list_conversations(self, status_filter: str = "all") -> List[Dict[str, Any]]:
        """Filter conversations by status."""
        if status_filter == "auto_ready":
            return [c for c in self._conversations if c.get("status") == "AI Ready"]
        elif status_filter == "needs_human":
            return [c for c in self._conversations if c.get("status") == "Needs Human"]
        elif status_filter == "resolved":
            return [c for c in self._conversations if c.get("status") == "Resolved"]
        return self._conversations

    def get_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        for c in self._conversations:
            if c.get("conversation_id") == conversation_id:
                return c
        return None
