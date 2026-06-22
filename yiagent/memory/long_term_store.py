"""
PostgreSQL + pgvector LongTermStore — permanent memory with hybrid search.

Implements the architecture plan's three-tier keyword search (tsvector →
pg_trgm → ILIKE) plus vector cosine search, all fused via RRF
(Reciprocal Rank Fusion) with k=60.

Design:
  - memory_chunks table: vector + FTS + trigram indexes
  - messages table: archival conversation storage
  - conversations table: session metadata
  - Pool-based asyncpg access with connection management
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.common.utils import compute_hash

# Lazy import — only required when PG is actually connected
asyncpg: Any = None  # type: ignore


# ---------------------------------------------------------------------------
# Schemas (DDL)
# ---------------------------------------------------------------------------

PG_SCHEMA_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Conversations (session metadata)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255),
    channel_type VARCHAR(64) NOT NULL DEFAULT '',
    title VARCHAR(255) NOT NULL DEFAULT '',
    context_start_seq INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    msg_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Messages (archival)
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role VARCHAR(16) NOT NULL CHECK (role IN ('user','assistant','system')),
    content JSONB NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extras JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (conversation_id, seq)
);

-- Core long-term memory store
CREATE TABLE IF NOT EXISTS memory_chunks (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(255),
    scope VARCHAR(32) NOT NULL DEFAULT 'shared',
    source VARCHAR(32) NOT NULL DEFAULT 'memory',
    path VARCHAR(1024) NOT NULL,
    start_line INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    embedding vector(1536),
    content_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON memory_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
    ON memory_chunks USING GIN (text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
    ON memory_chunks USING GIN (to_tsvector('simple', text));
CREATE INDEX IF NOT EXISTS idx_chunks_scope ON memory_chunks(scope);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON memory_chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON memory_chunks(user_id);

-- File metadata for incremental sync
CREATE TABLE IF NOT EXISTS memory_files (
    path VARCHAR(1024) PRIMARY KEY,
    source VARCHAR(32) NOT NULL DEFAULT 'memory',
    content_hash VARCHAR(64) NOT NULL,
    mtime INTEGER NOT NULL,
    size BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Dream diary (nightly distillation output)
CREATE TABLE IF NOT EXISTS dream_diaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    user_id VARCHAR(255),
    memory_before TEXT,
    memory_after TEXT,
    dream_text TEXT,
    dedup_key VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, dedup_key)
);

-- Evolution logs
CREATE TABLE IF NOT EXISTS evolution_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    backup_id VARCHAR(128),
    summary TEXT NOT NULL,
    changed_files JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: str = "memory"
    user_id: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass
class MemoryChunk:
    id: str
    user_id: Optional[str]
    scope: str
    source: str
    path: str
    start_line: int
    end_line: int
    text: str
    embedding: Optional[List[float]] = None
    content_hash: str = ""
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SyncReport:
    files_scanned: int = 0
    files_changed: int = 0
    chunks_updated: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LongTermStore
# ---------------------------------------------------------------------------

class LongTermStore:
    """PostgreSQL + pgvector backed long-term memory with RRF hybrid search."""

    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or conf().get("pg_dsn", "")
        self._pool: Any = None  # asyncpg.Pool (lazy)
        self._pool_lock = asyncio.Lock()
        self._initialized = False

    async def _get_pool(self):
        global asyncpg
        if asyncpg is None:
            import asyncpg as _asyncpg
            asyncpg = _asyncpg
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is not None:
                return self._pool
            cfg = conf()
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=cfg.get("pg_min_connections", 5),
                max_size=cfg.get("pg_max_connections", 50),
            )
            return self._pool

    async def initialize(self) -> None:
        """Run DDL. Idempotent."""
        if self._initialized:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Execute DDL block-by-block for safety
            for stmt in PG_SCHEMA_DDL.split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        await conn.execute(stmt)
                    except Exception as e:
                        logger.warning(f"[LongTermStore] DDL skip: {e}")
        self._initialized = True
        logger.info("[LongTermStore] Schema initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    # Messages (conversation archival)
    # ------------------------------------------------------------------

    async def append_message(
        self, session_id: str, message: Dict[str, Any], channel_type: str = ""
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Upsert conversation
                await conn.execute(
                    """
                    INSERT INTO conversations (session_id, channel_type, last_active, msg_count)
                    VALUES ($1, $2, NOW(), 1)
                    ON CONFLICT (session_id) DO UPDATE
                        SET last_active = NOW(),
                            msg_count = conversations.msg_count + 1,
                            channel_type = COALESCE(NULLIF($2, ''), conversations.channel_type)
                    """,
                    session_id, channel_type,
                )

                # Get conversation UUID
                conv_id = await conn.fetchval(
                    "SELECT id FROM conversations WHERE session_id = $1", session_id
                )

                seq = message.get("seq", 0)
                role = message.get("role", "")
                content = message.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                extras = message.get("extras", {})

                await conn.execute(
                    """
                    INSERT INTO messages (conversation_id, seq, role, content, extras)
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                    ON CONFLICT (conversation_id, seq) DO NOTHING
                    """,
                    conv_id, seq, role, content, json.dumps(extras, ensure_ascii=False),
                )

    async def load_messages(
        self,
        session_id: str,
        max_turns: int = 30,
        context_start_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load messages from PostgreSQL (cold path)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            conv_id = await conn.fetchval(
                "SELECT id FROM conversations WHERE session_id = $1", session_id
            )
            if not conv_id:
                return []

            query = """
                SELECT seq, role, content, extras
                FROM messages
                WHERE conversation_id = $1
            """
            params: list = [conv_id]

            if context_start_seq:
                query += " AND seq >= $2"
                params.append(context_start_seq)

            query += " ORDER BY seq ASC"

            rows = await conn.fetch(query, *params)

            messages = []
            for row in rows:
                content = row["content"]
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                msg = {"role": row["role"], "content": content, "seq": row["seq"]}
                extras = row.get("extras")
                if extras:
                    if isinstance(extras, str):
                        try:
                            extras = json.loads(extras)
                        except json.JSONDecodeError:
                            pass
                    msg["extras"] = extras
                messages.append(msg)

            # Visible-turn trim
            return ConversationStore._trim_by_visible_turns(messages, max_turns)

    async def delete_session(self, session_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conversations WHERE session_id = $1", session_id
            )

    # ------------------------------------------------------------------
    # Memory chunks CRUD
    # ------------------------------------------------------------------

    async def upsert_chunks_batch(self, chunks: List[MemoryChunk]) -> None:
        """Insert or update chunks in batch. Uses ON CONFLICT for idempotency."""
        if not chunks:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO memory_chunks
                        (id, user_id, scope, source, path, start_line, end_line,
                         text, embedding, content_hash, metadata, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        scope = EXCLUDED.scope,
                        source = EXCLUDED.source,
                        path = EXCLUDED.path,
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        content_hash = EXCLUDED.content_hash,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    [
                        (
                            c.id, c.user_id, c.scope, c.source, c.path,
                            c.start_line, c.end_line, c.text,
                            _encode_vector(c.embedding),
                            c.content_hash,
                            json.dumps(c.metadata or {}, ensure_ascii=False),
                        )
                        for c in chunks
                    ],
                )

    async def delete_by_path(self, path: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_chunks WHERE path = $1", path)

    async def get_file_hash(self, path: str) -> Optional[str]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT content_hash FROM memory_files WHERE path = $1", path
            )
            return row["content_hash"] if row else None

    async def update_file_metadata(
        self, path: str, source: str, content_hash: str, mtime: int, size: int
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_files (path, source, content_hash, mtime, size, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (path) DO UPDATE SET
                    source = EXCLUDED.source,
                    content_hash = EXCLUDED.content_hash,
                    mtime = EXCLUDED.mtime,
                    size = EXCLUDED.size,
                    updated_at = NOW()
                """,
                path, source, content_hash, mtime, size,
            )

    # ------------------------------------------------------------------
    # Vector search (cosine similarity via pgvector)
    # ------------------------------------------------------------------

    async def search_vector(
        self,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[SearchResult]:
        """Vector cosine similarity search via pgvector."""
        if scopes is None:
            scopes = ["shared"]
            if user_id:
                scopes.append("user")

        vec_str = _encode_vector_str(query_embedding)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            scope_placeholders = ", ".join(f"${i+2}" for i in range(len(scopes)))
            query = f"""
                SELECT id, path, start_line, end_line, text, source, user_id,
                       1.0 - (embedding <=> $1) AS score
                FROM memory_chunks
                WHERE embedding IS NOT NULL
                  AND scope IN ({scope_placeholders})
            """
            params: list = [vec_str] + scopes

            if user_id:
                query += f" AND (scope = 'shared' OR user_id = ${len(params) + 1})"
                params.append(user_id)

            query += f" ORDER BY embedding <=> $1 LIMIT ${len(params) + 1}"
            params.append(limit)

            try:
                rows = await conn.fetch(query, *params)
            except Exception as e:
                logger.error(f"[LongTermStore] Vector search failed: {e}")
                return []

            return [
                SearchResult(
                    path=row["path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    score=float(row["score"]),
                    snippet=_trunc_text(row["text"], 500),
                    source=row["source"],
                    user_id=row["user_id"],
                    chunk_id=row["id"],
                )
                for row in rows if float(row["score"]) > 0
            ]

    # ------------------------------------------------------------------
    # Keyword search (3-tier: tsvector → pg_trgm → ILIKE)
    # ------------------------------------------------------------------

    async def search_keyword(
        self,
        query: str,
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[SearchResult]:
        """Three-tier keyword search with graceful degradation."""
        if scopes is None:
            scopes = ["shared"]
            if user_id:
                scopes.append("user")

        # Tier 1: tsvector (ASCII/English)
        results = await self._search_tsvector(query, user_id, scopes, limit)
        if results:
            return results

        # Tier 2: pg_trgm (CJK + mixed language)
        results = await self._search_trigram(query, user_id, scopes, limit)
        if results:
            return results

        # Tier 3: ILIKE (fallback)
        return await self._search_ilike(query, user_id, scopes, limit)

    async def _search_tsvector(
        self, query: str, user_id: Optional[str], scopes: List[str], limit: int
    ) -> List[SearchResult]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                params: list = [query]
                scope_ph = ", ".join(f"${i+2}" for i in range(len(scopes)))
                sql = f"""
                    SELECT id, path, start_line, end_line, text, source, user_id,
                           ts_rank(to_tsvector('simple', text), plainto_tsquery('simple', $1)) AS rank
                    FROM memory_chunks
                    WHERE to_tsvector('simple', text) @@ plainto_tsquery('simple', $1)
                      AND scope IN ({scope_ph})
                """
                params += scopes
                if user_id:
                    sql += f" AND (scope = 'shared' OR user_id = ${len(params) + 1})"
                    params.append(user_id)
                sql += f" ORDER BY rank DESC LIMIT ${len(params) + 1}"
                params.append(limit)

                rows = await conn.fetch(sql, *params)
                return [
                    SearchResult(
                        path=row["path"], start_line=row["start_line"],
                        end_line=row["end_line"],
                        score=_rank_to_score(float(row["rank"])),
                        snippet=_trunc_text(row["text"], 500),
                        source=row["source"], user_id=row["user_id"],
                        chunk_id=row["id"],
                    )
                    for row in rows
                ]
            except Exception as e:
                logger.warning(f"[LongTermStore] tsvector search failed: {e}")
                return []

    async def _search_trigram(
        self, query: str, user_id: Optional[str], scopes: List[str], limit: int
    ) -> List[SearchResult]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                params: list = [query]
                scope_ph = ", ".join(f"${i+2}" for i in range(len(scopes)))
                sql = f"""
                    SELECT id, path, start_line, end_line, text, source, user_id,
                           similarity(text, $1) AS sim
                    FROM memory_chunks
                    WHERE similarity(text, $1) > 0.05
                      AND scope IN ({scope_ph})
                """
                params += scopes
                if user_id:
                    sql += f" AND (scope = 'shared' OR user_id = ${len(params) + 1})"
                    params.append(user_id)
                sql += f" ORDER BY sim DESC LIMIT ${len(params) + 1}"
                params.append(limit)

                rows = await conn.fetch(sql, *params)
                return [
                    SearchResult(
                        path=row["path"], start_line=row["start_line"],
                        end_line=row["end_line"],
                        score=0.3 + 0.69 * float(row["sim"]),
                        snippet=_trunc_text(row["text"], 500),
                        source=row["source"], user_id=row["user_id"],
                        chunk_id=row["id"],
                    )
                    for row in rows
                ]
            except Exception as e:
                logger.warning(f"[LongTermStore] trigram search failed: {e}")
                return []

    async def _search_ilike(
        self, query: str, user_id: Optional[str], scopes: List[str], limit: int
    ) -> List[SearchResult]:
        """ILIKE fallback — last resort for very short tokens."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                pattern = f"%{query}%"
                params: list = [pattern]
                scope_ph = ", ".join(f"${i+2}" for i in range(len(scopes)))
                sql = f"""
                    SELECT id, path, start_line, end_line, text, source, user_id
                    FROM memory_chunks
                    WHERE text ILIKE $1
                      AND scope IN ({scope_ph})
                """
                params += scopes
                if user_id:
                    sql += f" AND (scope = 'shared' OR user_id = ${len(params) + 1})"
                    params.append(user_id)
                sql += f" LIMIT ${len(params) + 1}"
                params.append(limit)

                rows = await conn.fetch(sql, *params)
                return [
                    SearchResult(
                        path=row["path"], start_line=row["start_line"],
                        end_line=row["end_line"],
                        score=0.3,
                        snippet=_trunc_text(row["text"], 500),
                        source=row["source"], user_id=row["user_id"],
                        chunk_id=row["id"],
                    )
                    for row in rows
                ]
            except Exception as e:
                logger.warning(f"[LongTermStore] ILIKE fallback failed: {e}")
                return []

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def fuse_rrf(
        ranked_lists: List[List[SearchResult]],
        k: int = 60,
        temporal_half_life_days: float = 30.0,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion across multiple search channels.

        RRF_Score(d) = Σ 1/(k + r_m(d)) for m in channels
        Then multiplied by temporal_decay(path).

        Args:
            ranked_lists: One list per search channel, pre-ranked.
            k: RRF constant (60 per the classic paper).
            temporal_half_life_days: Half-life for temporal decay.

        Returns:
            Fused and re-ranked results.
        """
        # Build score map keyed by chunk_id
        score_map: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}

        for channel_results in ranked_lists:
            for rank, result in enumerate(channel_results, start=1):
                chunk_key = result.chunk_id or f"{result.path}:{result.start_line}:{result.end_line}"
                rrf_contrib = 1.0 / (k + rank)
                if chunk_key in score_map:
                    score_map[chunk_key] += rrf_contrib
                else:
                    score_map[chunk_key] = rrf_contrib
                    result_map[chunk_key] = result

        # Apply temporal decay
        for chunk_key, result in result_map.items():
            decay = _compute_temporal_decay(result.path, temporal_half_life_days)
            score_map[chunk_key] *= decay
            result.score = score_map[chunk_key]

        # Sort descending
        fused = sorted(result_map.values(), key=lambda r: r.score, reverse=True)
        return fused

    async def get_stats(self) -> Dict[str, int]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            chunks = await conn.fetchval("SELECT COUNT(*) FROM memory_chunks")
            files = await conn.fetchval("SELECT COUNT(*) FROM memory_files")
            embedded = await conn.fetchval(
                "SELECT COUNT(*) FROM memory_chunks WHERE embedding IS NOT NULL"
            )
            return {"chunks": int(chunks or 0), "files": int(files or 0), "embedded": int(embedded or 0)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_vector(vec: Optional[List[float]]) -> Optional[str]:
    """Encode a float list to pgvector-compatible string."""
    if vec is None:
        return None
    return "[" + ",".join(str(v) for v in vec) + "]"


def _encode_vector_str(vec: List[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"


def _trunc_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _rank_to_score(rank: float) -> float:
    """Convert ts_rank to [0,1) score."""
    if rank <= 0:
        return 0.0
    return 0.3 + 0.69 * (rank / (1.0 + rank))


_CJK_RANGES = (
    r'　-ヿ' r'㐀-鿿' r'가-힯' r'豈-﫿' r'\U00020000-\U0002fa1f'
)
_RE_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.md$")


def _compute_temporal_decay(path: str, half_life_days: float = 30.0) -> float:
    """Exponential temporal decay for dated memory files."""
    match = _RE_DATE.search(path)
    if not match:
        return 1.0  # evergreen
    try:
        file_date = datetime(
            int(match.group(1)), int(match.group(2)), int(match.group(3)),
            tzinfo=timezone.utc,
        )
        age_days = (datetime.now(timezone.utc) - file_date).days
        if age_days <= 0:
            return 1.0
        decay_lambda = math.log(2) / half_life_days
        return math.exp(-decay_lambda * age_days)
    except (ValueError, OverflowError):
        return 1.0


# Import here to avoid circular dependency
from yiagent.memory.conversation_store import ConversationStore

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store_instance: Optional[LongTermStore] = None
_store_lock = asyncio.Lock()


def get_long_term_store() -> LongTermStore:
    """Return the process-wide LongTermStore singleton (sync init, async init later)."""
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    _store_instance = LongTermStore()
    return _store_instance
