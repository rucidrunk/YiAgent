"""
Redis connection pool — single process-wide pool shared by all components.

Provides both raw Redis and a high-level async interface for the
ConversationStore and embed cache.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

from yiagent.common.config import conf
from yiagent.common.log import logger

_pool: Optional[aioredis.ConnectionPool] = None
_pool_lock = asyncio.Lock()
# Track whether our warm-up ping passed — a ping failure must never leave
# _pool set, so later callers cannot accidentally reuse a broken pool.
_pool_ready = False


class _JSONEncoder(json.JSONEncoder):
    """ISO-8601 aware encoder that never uses default=str fallback."""

    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable. "
            f"Serialize it explicitly before storing."
        )


async def get_redis() -> Redis:
    """Return a Redis client from the process-wide pool. Idempotent.

    The pool is created lazily on first call and verified with a ping
    BEFORE ``_pool`` is published — a ping failure leaves no trace so
    retry-able errors (DNS propagation, port not open yet) are not
    turned into permanent corruption.
    """
    global _pool, _pool_ready
    if _pool_ready and _pool is not None:
        return Redis(connection_pool=_pool)

    async with _pool_lock:
        if _pool_ready and _pool is not None:
            return Redis(connection_pool=_pool)

        cfg = conf()
        url = cfg["redis_url"]
        max_conn = cfg["redis_max_connections"]
        timeout = cfg["redis_socket_timeout"]

        # Build the pool but don't publish it yet
        candidate = aioredis.ConnectionPool.from_url(
            url,
            max_connections=max_conn,
            socket_timeout=timeout,
            decode_responses=True,
        )

        # Warm-up probe — MUST pass before we publish
        try:
            r = Redis(connection_pool=candidate)
            await r.ping()
            logger.info(f"[Redis] Connected to {url}  pool={max_conn}")
        except Exception:
            # Best-effort cleanup; never let disconnect() shadow the root cause
            try:
                await candidate.disconnect()
            except Exception:
                pass
            logger.error(f"[Redis] Connection failed to {url}")
            raise

        # Publish only after successful ping
        _pool = candidate
        _pool_ready = True
        return Redis(connection_pool=_pool)


async def close_redis() -> None:
    """Disconnect the process-wide pool. Thread-safe via _pool_lock."""
    global _pool, _pool_ready
    async with _pool_lock:
        if _pool is not None:
            try:
                await _pool.disconnect()
            except Exception as e:
                logger.warning(f"[Redis] Error during pool disconnect: {e}")
            _pool = None
            _pool_ready = False
            logger.info("[Redis] Pool disconnected")


# ---------------------------------------------------------------------------
# High-level Redis helpers used by ConversationStore
# ---------------------------------------------------------------------------

async def redis_set_json(r: Redis, key: str, value: Any, ex: Optional[int] = None) -> None:
    """Set a key to a JSON-encoded value. Uses strict encoder — no silent type coercion."""
    await r.set(key, json.dumps(value, ensure_ascii=False, cls=_JSONEncoder), ex=ex)


async def redis_get_json(r: Redis, key: str) -> Optional[Any]:
    raw = await r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


async def redis_hset_json(r: Redis, name: str, mapping: Dict[str, Any]) -> None:
    """HSET with JSON-encoded values. Uses strict encoder."""
    encoded = {k: json.dumps(v, ensure_ascii=False, cls=_JSONEncoder)
               for k, v in mapping.items()}
    if encoded:
        await r.hset(name, mapping=encoded)


async def redis_hget_json(r: Redis, name: str, key: str) -> Optional[Any]:
    raw = await r.hget(name, key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


async def redis_hgetall_json(r: Redis, name: str) -> Dict[str, Any]:
    raw = await r.hgetall(name)
    result = {}
    for k, v in raw.items():
        try:
            result[k] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            result[k] = v
    return result
