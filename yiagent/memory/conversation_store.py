"""
Redis-backed ConversationStore — short-term memory for active sessions.

Design:
  - Hot path: Redis List for context window, Redis Hash for session metadata
  - Cold path: PostgreSQL messages table (async persistence)
  - Distributed lock: SET NX EX per session to prevent concurrent runs

Key patterns borrowed from CowAgent:
  - Visible-turn counting (only real user text, not tool_result)
  - context_start_seq for boundary-based context trimming
  - Message extras for TTS/attachment metadata
"""

from __future__ import annotations

import asyncio
import json
import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.common.redis_pool import (
    get_redis, redis_get_json, redis_set_json,
    redis_hget_json, redis_hset_json, redis_hgetall_json,
)
from yiagent.common.utils import compute_hash


# ---------------------------------------------------------------------------
# Redis key helpers
# ---------------------------------------------------------------------------

def _ctx_key(session_id: str) -> str:
    return f"session:{session_id}:context"

def _meta_key(session_id: str) -> str:
    return f"session:{session_id}:meta"

def _lock_key(session_id: str) -> str:
    return f"session:{session_id}:run_lock"

def _embed_cache_key(provider: str, model: str, query_hash: str) -> str:
    return f"embed_cache:{provider}:{model}:{query_hash}"

# Per-session singleflight lock — a short-lived asyncio.Lock that coalesces
# concurrent load_context() calls when Redis is empty, so they don't all
# stampede to PostgreSQL.
def _singleflight_key(session_id: str) -> str:
    return f"session:{session_id}:load_lock"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RedisUnavailableError(Exception):
    """Redis connection failed — session can proceed in degraded mode."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class SessionMeta:
    session_id: str
    user_id: str = ""
    channel_type: str = ""
    receiver: str = ""
    created_at: float = 0.0
    last_active: float = 0.0
    msg_count: int = 0
    context_start_seq: int = 0
    base_seq: int = 0  # offset added to list-index seq after ltrim
    evo_done_seq: int = 0
    evo_last_active: float = 0.0
    evo_turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "channel_type": self.channel_type,
            "receiver": self.receiver,
            "created_at": str(self.created_at),
            "last_active": str(self.last_active),
            "msg_count": str(self.msg_count),
            "context_start_seq": str(self.context_start_seq),
            "base_seq": str(self.base_seq),
            "evo_done_seq": str(self.evo_done_seq),
            "evo_last_active": str(self.evo_last_active),
            "evo_turns": str(self.evo_turns),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionMeta":
        return cls(
            session_id=d.get("session_id", ""),
            user_id=d.get("user_id", ""),
            channel_type=d.get("channel_type", ""),
            receiver=d.get("receiver", ""),
            created_at=float(d.get("created_at", 0)),
            last_active=float(d.get("last_active", 0)),
            msg_count=int(d.get("msg_count", 0)),
            context_start_seq=int(d.get("context_start_seq", 0)),
            base_seq=int(d.get("base_seq", 0)),
            evo_done_seq=int(d.get("evo_done_seq", 0)),
            evo_last_active=float(d.get("evo_last_active", 0)),
            evo_turns=int(d.get("evo_turns", 0)),
        )


# ---------------------------------------------------------------------------
# Background-task tracker — keeps strong references to create_task fire-and-
# forget tasks so their exceptions are logged instead of silently lost.
# ---------------------------------------------------------------------------

class _BgTracker:
    """Tracks in-flight background tasks so failures are never silent."""

    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()

    def spawn(self, coro) -> None:
        """Create a tracked background task.

        Safe to call both inside and outside an event loop: outside a loop
        the coroutine is simply logged and discarded (no-op) rather than
        raising RuntimeError.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — cannot spawn, log a warning
            logger.warning("[ConvStore] _BgTracker.spawn called outside event loop; task dropped")
            return
        task = loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._log_failure)

    @staticmethod
    def _log_failure(task: asyncio.Task) -> None:
        try:
            exc = task.exception()
            if exc is not None:
                logger.error(f"[ConvStore] Background task failed: {exc}")
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass

    async def drain(self) -> None:
        """Wait for all tracked tasks to complete (for graceful shutdown)."""
        if not self._tasks:
            return
        for task in list(self._tasks):
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass


_bg_tracker = _BgTracker()


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------

class ConversationStore:
    """Redis-backed short-term memory for conversation sessions."""

    def __init__(self):
        self._cfg = conf()
        self._context_ttl = self._cfg["redis_context_ttl"]
        self._session_ttl = self._cfg["redis_session_ttl"]
        self._lock_ttl = self._cfg["redis_lock_ttl"]
        # Singleflight locks per session (weak-ref: cleaned up when no-one waits)
        self._sf_locks: Dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Distributed lock
    # ------------------------------------------------------------------

    async def acquire_lock(self, session_id: str) -> bool:
        """Acquire distributed lock. Returns False if held by another process.
        Raises RedisError on connection failure (caller can decide to proceed without lock)."""
        try:
            r = await get_redis()
            acquired = await r.set(_lock_key(session_id), "1", nx=True, ex=self._lock_ttl)
            return bool(acquired)
        except Exception as e:
            logger.warning(f"[ConvStore] Redis unavailable for lock on {session_id}: {e}")
            raise RedisUnavailableError(f"Redis unavailable: {e}") from e

    async def release_lock(self, session_id: str) -> None:
        try:
            r = await get_redis()
            await r.delete(_lock_key(session_id))
        except Exception as e:
            logger.warning(f"[ConvStore] Lock release failed for {session_id}: {e}")

    async def extend_lock(self, session_id: str) -> None:
        try:
            r = await get_redis()
            await r.expire(_lock_key(session_id), self._lock_ttl)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    async def get_meta(self, session_id: str) -> Optional[SessionMeta]:
        try:
            r = await get_redis()
            raw = await redis_hgetall_json(r, _meta_key(session_id))
            if not raw:
                return None
            raw["session_id"] = session_id
            return SessionMeta.from_dict(raw)
        except Exception as e:
            logger.warning(f"[ConvStore] get_meta failed for {session_id}: {e}")
            return None

    async def upsert_meta(self, session_id: str, **fields) -> None:
        """Create or update session metadata fields.

        Uses HSETNX for created_at so concurrent callers cannot race on
        the initial timestamp — the first writer wins atomically.
        """
        try:
            r = await get_redis()
            now = str(time.time())
            meta_key = _meta_key(session_id)
            # Atomically set created_at on first write (no TOCTOU gap)
            await r.hsetnx(meta_key, "created_at", now)
            fields.setdefault("last_active", now)
            await redis_hset_json(r, meta_key, fields)
            await r.expire(meta_key, self._session_ttl)
        except Exception as e:
            logger.warning(f"[ConvStore] upsert_meta failed for {session_id}: {e}")

    async def touch_session(self, session_id: str) -> None:
        try:
            r = await get_redis()
            meta_key = _meta_key(session_id)
            ctx_key = _ctx_key(session_id)
            now = str(time.time())
            await r.hset(meta_key, "last_active", now)
            await r.expire(meta_key, self._session_ttl)
            await r.expire(ctx_key, self._context_ttl)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context window (Redis List)
    # ------------------------------------------------------------------

    async def load_context(
        self,
        session_id: str,
        max_turns: int = 30,
        context_start_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load recent conversation context from Redis.

        Returns messages as [{role, content, extras}, ...] in chronological order.

        Cold-path singleflight: concurrent callers for the same empty cache
        coalesce into ONE PostgreSQL query, preventing cache stampede.
        """
        try:
            r = await get_redis()
            raw_list = await r.lrange(_ctx_key(session_id), 0, -1)
        except Exception as e:
            logger.warning(f"[ConvStore] Redis load failed for {session_id}: {e}, falling back to PG")
            raw_list = []

        if raw_list:
            # Load base_seq for seq correction
            meta = await self.get_meta(session_id)
            base_seq = meta.base_seq if meta else 0

            messages = []
            for i, item in enumerate(raw_list):
                try:
                    msg = json.loads(item)
                    # Correct seq to account for ltrim offset
                    if base_seq:
                        msg["seq"] = base_seq + i
                    messages.append(msg)
                except json.JSONDecodeError:
                    continue

            if context_start_seq:
                messages = [m for m in messages if m.get("seq", 0) >= context_start_seq]

            messages = self._trim_by_visible_turns(messages, max_turns)
            return messages

        # Cold path with singleflight
        return await self._load_from_pg_singleflight(session_id, max_turns, context_start_seq)

    async def _load_from_pg_singleflight(
        self, session_id: str, max_turns: int, context_start_seq: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Singleflight wrapper around PG cold-path load.

        All concurrent callers acquire the same per-session lock exactly
        once, then check Redis inside the critical section.  The first
        caller sees an empty cache and does the PG query; every other
        caller wakes up after the first has populated Redis and returns
        the cached result.  This guarantees exactly one PG query per
        cache miss regardless of concurrency.
        """
        sf_key = _singleflight_key(session_id)
        lock = self._sf_locks.get(sf_key)
        if lock is None:
            lock = asyncio.Lock()
            self._sf_locks[sf_key] = lock

        try:
            async with lock:
                # Double-check Redis: someone who held the lock before us
                # may have already populated the cache.
                try:
                    r = await get_redis()
                    raw_list = await r.lrange(_ctx_key(session_id), 0, -1)
                    if raw_list:
                        messages = []
                        for item in raw_list:
                            try:
                                messages.append(json.loads(item))
                            except json.JSONDecodeError:
                                continue
                        return self._trim_by_visible_turns(messages, max_turns)
                except Exception:
                    pass
                return await self._load_from_pg(session_id, max_turns, context_start_seq)
        finally:
            # Safe cleanup: only remove if no-one is currently waiting.
            try:
                if not lock.locked():
                    self._sf_locks.pop(sf_key, None)
            except Exception:
                pass

    async def append_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        channel_type: str = "",
    ) -> int:
        """
        Append a message to the Redis context list.
        Returns the new message count.
        """
        try:
            r = await get_redis()
            ctx_key = _ctx_key(session_id)

            # Load base_seq for consistent seq numbering across ltrim boundaries
            meta = await self.get_meta(session_id)
            base_seq = meta.base_seq if meta else 0
            current_len = await r.llen(ctx_key)
            # seq = base_offset + position_in_list
            message["seq"] = base_seq + current_len
            message["ts"] = time.time()

            raw = json.dumps(message, ensure_ascii=False)
            await r.rpush(ctx_key, raw)

            # Cap list size (keep max ~200 messages in hot store).
            # When trimming from the head, advance base_seq so seq numbers
            # stay monotonic rather than resetting to list-index.
            max_msgs = 200
            overflow = max(0, current_len + 1 - max_msgs)
            if overflow:
                await r.ltrim(ctx_key, overflow, -1)
                base_seq += overflow
                await redis_hset_json(r, _meta_key(session_id), {"base_seq": str(base_seq)})

            await r.expire(ctx_key, self._context_ttl)

            await self.upsert_meta(session_id, msg_count=str(base_seq + current_len + 1))
            if channel_type:
                await self.upsert_meta(session_id, channel_type=channel_type)

            # Tracked background persist — failure is logged, never silently lost
            _bg_tracker.spawn(self._persist_to_pg(session_id, [message], channel_type))

            return base_seq + current_len + 1
        except Exception as e:
            logger.error(f"[ConvStore] append_message failed for {session_id}: {e}")
            return 0

    async def append_messages_batch(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        channel_type: str = "",
    ) -> Tuple[int, int]:
        """Append multiple messages in sequence.

        Returns (success_count, total_count).  Each message is appended
        independently — partial failures are logged but do not roll back
        already-persisted messages.  If you need atomicity, wrap the
        caller in a transaction at the application level.
        """
        total = len(messages)
        success = 0
        for msg in messages:
            c = await self.append_message(session_id, msg, channel_type)
            if c > 0:
                success += 1
            else:
                logger.warning(
                    f"[ConvStore] Batch: message seq={msg.get('seq')} "
                    f"failed for {session_id} ({success}/{total} ok so far)"
                )
        if success < total:
            logger.warning(
                f"[ConvStore] Batch partial success: {success}/{total} messages "
                f"persisted for {session_id}"
            )
        return (success, total)

    async def clear_context(self, session_id: str) -> int:
        try:
            r = await get_redis()
            meta = await self.get_meta(session_id)
            base_seq = meta.base_seq if meta else 0
            current_len = await r.llen(_ctx_key(session_id))
            new_boundary = base_seq + current_len
            await self.upsert_meta(session_id, context_start_seq=str(new_boundary))
            return new_boundary
        except Exception as e:
            logger.warning(f"[ConvStore] clear_context failed for {session_id}: {e}")
            return 0

    async def clear_session(self, session_id: str) -> None:
        try:
            r = await get_redis()
            await r.delete(_ctx_key(session_id), _meta_key(session_id), _lock_key(session_id))
            _bg_tracker.spawn(self._delete_from_pg(session_id))
        except Exception as e:
            logger.warning(f"[ConvStore] clear_session failed for {session_id}: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        try:
            r = await get_redis()
            session_keys = await r.keys("session:*:meta")
            return {"active_sessions": len(session_keys)}
        except Exception:
            return {"active_sessions": "unknown"}

    async def list_session_ids(self) -> List[str]:
        try:
            r = await get_redis()
            keys = await r.keys("session:*:meta")
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    async def get_embed_cache(self, provider: str, model: str, query: str) -> Optional[List[float]]:
        import hashlib
        qh = hashlib.md5(query.encode()).hexdigest()
        try:
            r = await get_redis()
            return await redis_get_json(r, _embed_cache_key(provider, model, qh))
        except Exception:
            return None

    async def set_embed_cache(self, provider: str, model: str, query: str, embedding: List[float]) -> None:
        import hashlib
        qh = hashlib.md5(query.encode()).hexdigest()
        try:
            r = await get_redis()
            await redis_set_json(r, _embed_cache_key(provider, model, qh), embedding, ex=3600)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_by_visible_turns(messages: List[Dict], max_turns: int) -> List[Dict]:
        if not messages:
            return messages

        visible_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    has_text = any(
                        isinstance(b, dict) and b.get("type") == "text"
                        for b in content
                    )
                    has_tool = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in content
                    )
                    if has_text and not has_tool:
                        visible_indices.append(i)
                elif isinstance(content, str) and content.strip():
                    visible_indices.append(i)

        if len(visible_indices) <= max_turns:
            return messages

        cutoff = visible_indices[-(max_turns)]
        return messages[cutoff:]

    async def _load_from_pg(
        self, session_id: str, max_turns: int, context_start_seq: Optional[int],
    ) -> List[Dict[str, Any]]:
        from yiagent.memory.long_term_store import get_long_term_store
        try:
            store = get_long_term_store()
            return await store.load_messages(session_id, max_turns, context_start_seq)
        except Exception as e:
            logger.warning(f"[ConvStore] PG fallback failed for {session_id}: {e}")
            return []

    async def _persist_to_pg(
        self, session_id: str, messages: List[Dict[str, Any]], channel_type: str,
    ) -> None:
        from yiagent.memory.long_term_store import get_long_term_store
        try:
            store = get_long_term_store()
            for msg in messages:
                await store.append_message(session_id, msg, channel_type)
        except Exception as e:
            logger.warning(f"[ConvStore] PG persist failed for {session_id}: {e}")

    async def _delete_from_pg(self, session_id: str) -> None:
        try:
            from yiagent.memory.long_term_store import get_long_term_store
            store = get_long_term_store()
            await store.delete_session(session_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store_instance: Optional[ConversationStore] = None
_store_lock = asyncio.Lock()


async def get_conversation_store() -> ConversationStore:
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    async with _store_lock:
        if _store_instance is not None:
            return _store_instance
        _store_instance = ConversationStore()
        return _store_instance
