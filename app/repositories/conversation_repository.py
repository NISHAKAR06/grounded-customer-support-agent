"""Conversation repository accessing customer support records backed by SQLite and PostgreSQL."""

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core.config import get_settings

logger = logging.getLogger("conversation_repository")


class ConversationRepository:
    """Provides structured data access to customer support records for the support inbox.

    Supports PostgreSQL (production, e.g. Neon) via DATABASE_URL and SQLite (local/fallback),
    with automatic table creation, indexing, and auto-seeding from sample datasets.
    """

    _pg_initialized: bool = False

    def __init__(
        self,
        data_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        db_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.settings = settings

        # Determine database mode
        self.db_url = db_url or getattr(settings, "DATABASE_URL", None)
        self._is_postgres = False

        if db_path and str(db_path) == ":memory:":
            # Explicit in-memory SQLite requested (unit tests)
            self._is_postgres = False
            self.db_path = Path(":memory:")
        elif data_path and data_path.is_file() and not db_url:
            # Explicit dataset file in test without db_url -> isolated in-memory SQLite
            self._is_postgres = False
            self.db_path = Path(":memory:")
        elif self.db_url and (
            self.db_url.startswith("postgres://") or self.db_url.startswith("postgresql://")
        ):
            # Production PostgreSQL
            self._is_postgres = True
            # Normalize Neon pooler to direct connection for connection stability if needed
            self.db_url = self.db_url.replace("-pooler.", ".")
            self.db_path = None
        elif db_path:
            self.db_path = db_path
        else:
            self.db_path = settings.DATA_DIR / "inbox.db"

        self.data_path = self._resolve_data_path(data_path, settings)

        # SQLite memory connection caching
        self._is_memory = not self._is_postgres and str(self.db_path) == ":memory:"
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self._is_memory:
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row

        if self._is_postgres:
            if not ConversationRepository._pg_initialized:
                self._init_db()
                self._ensure_seeded()
                ConversationRepository._pg_initialized = True
        else:
            self._init_db()
            self._ensure_seeded()

    def _get_connection(self):
        if self._is_postgres:
            import psycopg2

            conn = psycopg2.connect(self.db_url, connect_timeout=15)
            conn.autocommit = True
            return conn

        if self._is_memory and self._mem_conn is not None:
            return self._mem_conn

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _resolve_data_path(self, data_path: Optional[Path], settings: Any) -> Optional[Path]:
        if data_path and data_path.is_file():
            return data_path

        candidates = [
            settings.DATA_DIR / "processed" / "applesupport_sample.jsonl",
            settings.DATA_DIR / "processed" / "applesupport_conversations.jsonl",
            settings.DATA_DIR / "golden" / "golden_set.jsonl",
            settings.DATA_DIR / "processed" / "conversations.jsonl",
        ]
        for p in candidates:
            if p.is_file():
                return p
        return settings.DATA_DIR / "processed" / "applesupport_sample.jsonl"

    def _init_db(self) -> None:
        """Create database tables and indexes if they do not exist."""
        if self._is_postgres:
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                        CREATE TABLE IF NOT EXISTS conversations (
                            conversation_id VARCHAR(128) PRIMARY KEY,
                            ticket_id VARCHAR(64),
                            customer_id VARCHAR(64),
                            brand VARCHAR(64),
                            root_tweet_id VARCHAR(64),
                            first_inquiry TEXT,
                            latest_message TEXT,
                            final_brand_response TEXT,
                            intent VARCHAR(128),
                            intent_code VARCHAR(64),
                            confidence DOUBLE PRECISION,
                            decision VARCHAR(64),
                            status VARCHAR(64),
                            turn_count INTEGER,
                            has_dm BOOLEAN,
                            has_kb_link BOOLEAN,
                            has_resolution BOOLEAN,
                            created_at VARCHAR(128),
                            timestamp VARCHAR(128),
                            intent_confidence DOUBLE PRECISION,
                            intent_signals_json TEXT,
                            turns_json TEXT
                        );
                        CREATE INDEX IF NOT EXISTS idx_pg_status ON conversations(status);
                        CREATE INDEX IF NOT EXISTS idx_pg_decision ON conversations(decision);
                        CREATE INDEX IF NOT EXISTS idx_pg_intent_code ON conversations(intent_code);
                        CREATE INDEX IF NOT EXISTS idx_pg_turn_count ON conversations(turn_count);
                        CREATE INDEX IF NOT EXISTS idx_pg_created_at ON conversations(created_at);
                        """
                        )
                finally:
                    conn.close()
            except Exception as ex:
                logger.warning(f"Failed to initialize PostgreSQL schema: {ex}")
        else:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        ticket_id TEXT,
                        customer_id TEXT,
                        brand TEXT,
                        root_tweet_id TEXT,
                        first_inquiry TEXT,
                        latest_message TEXT,
                        final_brand_response TEXT,
                        intent TEXT,
                        intent_code TEXT,
                        confidence REAL,
                        decision TEXT,
                        status TEXT,
                        turn_count INTEGER,
                        has_dm INTEGER,
                        has_kb_link INTEGER,
                        has_resolution INTEGER,
                        created_at TEXT,
                        timestamp TEXT,
                        intent_confidence REAL,
                        intent_signals_json TEXT,
                        turns_json TEXT
                    )
                    """
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON conversations(status)")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_decision ON conversations(decision)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_intent_code ON conversations(intent_code)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_turn_count ON conversations(turn_count)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at)"
                    )
            finally:
                if not self._is_memory:
                    conn.close()

    def _ensure_seeded(self) -> None:
        """Seed database if currently empty."""
        try:
            if self._is_postgres:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM conversations")
                        count = cur.fetchone()[0]
                        if count == 0 and self.data_path and self.data_path.exists():
                            self.seed_from_file(self.data_path)
                finally:
                    conn.close()
            else:
                conn = self._get_connection()
                try:
                    cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                    count = cursor.fetchone()[0]
                    if count == 0 and self.data_path and self.data_path.exists():
                        self.seed_from_file(self.data_path)
                finally:
                    if not self._is_memory:
                        conn.close()
        except Exception as ex:
            logger.warning(f"Database seeding check failed: {ex}")

    def seed_from_file(self, file_path: Path) -> int:
        """Load conversation records from JSONL file into the database."""
        if not file_path or not file_path.exists():
            return 0

        records = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
        except Exception as ex:
            logger.warning(f"Failed to read file {file_path}: {ex}")
            return 0

        if not records:
            return 0

        if self._is_postgres:
            from psycopg2.extras import execute_batch

            insert_sql = """
            INSERT INTO conversations (
                conversation_id, ticket_id, customer_id, brand, root_tweet_id,
                first_inquiry, latest_message, final_brand_response, intent,
                intent_code, confidence, decision, status, turn_count,
                has_dm, has_kb_link, has_resolution, created_at, timestamp,
                intent_confidence, intent_signals_json, turns_json
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s
            ) ON CONFLICT (conversation_id) DO NOTHING;
            """
            params = []
            for c in records:
                turns = c.get("turns") or []
                signals = c.get("intent_signals") or []
                params.append(
                    (
                        str(c.get("conversation_id") or ""),
                        str(c.get("ticket_id") or ""),
                        str(c.get("customer_id") or ""),
                        str(c.get("brand") or "AppleSupport"),
                        str(c.get("root_tweet_id") or ""),
                        str(c.get("first_inquiry") or ""),
                        str(c.get("latest_message") or ""),
                        str(c.get("final_brand_response") or ""),
                        str(c.get("intent") or "General Inquiry"),
                        str(c.get("intent_code") or "general_inquiry"),
                        float(c.get("confidence") or 0.0),
                        str(c.get("decision") or "AUTO_HANDLE"),
                        str(c.get("status") or "AI Ready"),
                        int(c.get("turn_count") or len(turns)),
                        bool(c.get("has_dm", False)),
                        bool(c.get("has_kb_link", False)),
                        bool(c.get("has_resolution", False)),
                        str(c.get("created_at") or ""),
                        str(c.get("timestamp") or ""),
                        float(c.get("intent_confidence") or c.get("confidence") or 0.0),
                        json.dumps(signals, ensure_ascii=False),
                        json.dumps(turns, ensure_ascii=False),
                    )
                )
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        execute_batch(cur, insert_sql, params, page_size=50)
                    return len(params)
                finally:
                    conn.close()
            except Exception as ex:
                logger.warning(f"Batch seeding PostgreSQL failed: {ex}")
                return 0
        else:
            seeded = 0
            for r in records:
                try:
                    self.save_conversation(r)
                    seeded += 1
                except Exception as ex:
                    logger.debug(f"Failed to seed record: {ex}")
            return seeded

    def save_conversation(self, conv: Dict[str, Any]) -> None:
        """Insert or replace a conversation record in the database."""
        turns = conv.get("turns") or []
        turns_json = json.dumps(turns, ensure_ascii=False)
        intent_signals = conv.get("intent_signals") or []
        signals_json = json.dumps(intent_signals, ensure_ascii=False)

        val_tuple = (
            conv.get("conversation_id"),
            conv.get("ticket_id"),
            conv.get("customer_id"),
            conv.get("brand", "AppleSupport"),
            conv.get("root_tweet_id"),
            conv.get("first_inquiry"),
            conv.get("latest_message"),
            conv.get("final_brand_response"),
            conv.get("intent"),
            conv.get("intent_code"),
            float(conv.get("confidence") or 0.0),
            conv.get("decision"),
            conv.get("status", "Needs Human"),
            int(conv.get("turn_count") or len(turns)),
            bool(conv.get("has_dm")),
            bool(conv.get("has_kb_link")),
            bool(conv.get("has_resolution")),
            str(conv.get("created_at") or ""),
            str(conv.get("timestamp") or ""),
            float(conv.get("intent_confidence") or conv.get("confidence") or 0.0),
            signals_json,
            turns_json,
        )

        if self._is_postgres:
            pg_query = """
            INSERT INTO conversations (
                conversation_id, ticket_id, customer_id, brand, root_tweet_id,
                first_inquiry, latest_message, final_brand_response, intent,
                intent_code, confidence, decision, status, turn_count,
                has_dm, has_kb_link, has_resolution, created_at, timestamp,
                intent_confidence, intent_signals_json, turns_json
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s
            ) ON CONFLICT (conversation_id) DO UPDATE SET
                ticket_id = EXCLUDED.ticket_id,
                customer_id = EXCLUDED.customer_id,
                brand = EXCLUDED.brand,
                root_tweet_id = EXCLUDED.root_tweet_id,
                first_inquiry = EXCLUDED.first_inquiry,
                latest_message = EXCLUDED.latest_message,
                final_brand_response = EXCLUDED.final_brand_response,
                intent = EXCLUDED.intent,
                intent_code = EXCLUDED.intent_code,
                confidence = EXCLUDED.confidence,
                decision = EXCLUDED.decision,
                status = EXCLUDED.status,
                turn_count = EXCLUDED.turn_count,
                has_dm = EXCLUDED.has_dm,
                has_kb_link = EXCLUDED.has_kb_link,
                has_resolution = EXCLUDED.has_resolution,
                created_at = EXCLUDED.created_at,
                timestamp = EXCLUDED.timestamp,
                intent_confidence = EXCLUDED.intent_confidence,
                intent_signals_json = EXCLUDED.intent_signals_json,
                turns_json = EXCLUDED.turns_json;
            """
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(pg_query, val_tuple)
            finally:
                conn.close()
        else:
            query = """
            INSERT OR REPLACE INTO conversations (
                conversation_id, ticket_id, customer_id, brand, root_tweet_id,
                first_inquiry, latest_message, final_brand_response, intent,
                intent_code, confidence, decision, status, turn_count,
                has_dm, has_kb_link, has_resolution, created_at, timestamp,
                intent_confidence, intent_signals_json, turns_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        query,
                        (
                            val_tuple[0],
                            val_tuple[1],
                            val_tuple[2],
                            val_tuple[3],
                            val_tuple[4],
                            val_tuple[5],
                            val_tuple[6],
                            val_tuple[7],
                            val_tuple[8],
                            val_tuple[9],
                            val_tuple[10],
                            val_tuple[11],
                            val_tuple[12],
                            val_tuple[13],
                            1 if val_tuple[14] else 0,
                            1 if val_tuple[15] else 0,
                            1 if val_tuple[16] else 0,
                            val_tuple[17],
                            val_tuple[18],
                            val_tuple[19],
                            val_tuple[20],
                            val_tuple[21],
                        ),
                    )
            finally:
                if not self._is_memory:
                    conn.close()

    @staticmethod
    def _row_to_dict(row: Union[sqlite3.Row, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert a database row into a standardized conversation dictionary."""
        turns = []
        raw_turns = row["turns_json"]
        if raw_turns:
            try:
                turns = json.loads(raw_turns)
            except Exception:
                turns = []

        signals = []
        raw_signals = row["intent_signals_json"]
        if raw_signals:
            try:
                signals = json.loads(raw_signals)
            except Exception:
                signals = []

        return {
            "conversation_id": row["conversation_id"],
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "brand": row["brand"],
            "root_tweet_id": row["root_tweet_id"],
            "first_inquiry": row["first_inquiry"],
            "latest_message": row["latest_message"],
            "final_brand_response": row["final_brand_response"],
            "intent": row["intent"],
            "intent_code": row["intent_code"],
            "confidence": row["confidence"],
            "decision": row["decision"],
            "status": row["status"],
            "turn_count": row["turn_count"],
            "has_dm": bool(row["has_dm"]),
            "has_kb_link": bool(row["has_kb_link"]),
            "has_resolution": bool(row["has_resolution"]),
            "created_at": row["created_at"],
            "timestamp": row["timestamp"],
            "intent_confidence": row["intent_confidence"],
            "intent_signals": signals,
            "turns": turns,
        }

    def list_conversations(
        self,
        status_filter: str = "all",
        decision_filter: str = "all",
        turn_filter: str = "all",
        intent_filter: str = "all",
        sort_by: str = "newest",
        search_query: str = "",
    ) -> List[Dict[str, Any]]:
        """Filter and sort conversations from SQLite or PostgreSQL."""
        where_clauses = []
        params = []

        # 1. Filter by status
        if status_filter == "auto_ready":
            where_clauses.append("status = 'AI Ready'")
        elif status_filter == "needs_human":
            where_clauses.append("status = 'Needs Human'")
        elif status_filter == "resolved":
            where_clauses.append("status = 'Resolved'")

        # 2. Filter by decision
        if decision_filter == "auto_handle":
            where_clauses.append("decision = 'AUTO_HANDLE'")
        elif decision_filter == "human_escalation":
            where_clauses.append("decision = 'HUMAN_ESCALATION'")

        # 3. Filter by turns
        if turn_filter == "short":
            where_clauses.append("turn_count <= 2")
        elif turn_filter == "medium":
            where_clauses.append("turn_count BETWEEN 3 AND 4")
        elif turn_filter == "deep":
            where_clauses.append("turn_count >= 5")

        # 4. Filter by intent
        if intent_filter and intent_filter != "all":
            where_clauses.append("(LOWER(intent_code) = ? OR LOWER(intent) = ?)")
            params.extend([intent_filter.lower(), intent_filter.lower()])

        # 5. Search query
        if search_query:
            q = f"%{search_query.strip().lower()}%"
            where_clauses.append(
                """(
                LOWER(conversation_id) LIKE ? OR
                LOWER(ticket_id) LIKE ? OR
                LOWER(customer_id) LIKE ? OR
                LOWER(latest_message) LIKE ? OR
                LOWER(first_inquiry) LIKE ?
            )"""
            )
            params.extend([q, q, q, q, q])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        base_query = f"SELECT * FROM conversations {where_sql}"

        if self._is_postgres:
            from psycopg2.extras import RealDictCursor

            pg_query = base_query.replace("?", "%s")
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if params:
                        cur.execute(pg_query, params)
                    else:
                        cur.execute(pg_query)
                    rows = cur.fetchall()
                    results = [self._row_to_dict(r) for r in rows]
            finally:
                conn.close()
        else:
            conn = self._get_connection()
            try:
                cursor = conn.execute(base_query, params)
                rows = cursor.fetchall()
                results = [self._row_to_dict(r) for r in rows]
            finally:
                if not self._is_memory:
                    conn.close()

        # Deterministic sorting
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
        """Build windowed pagination with ellipsis markers."""
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
        """Retrieve a single conversation record by ID."""
        if self._is_postgres:
            from psycopg2.extras import RealDictCursor

            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM conversations WHERE conversation_id = %s", (conversation_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_dict(row)
                    return None
            finally:
                conn.close()
        else:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_dict(row)
                return None
            finally:
                if not self._is_memory:
                    conn.close()
