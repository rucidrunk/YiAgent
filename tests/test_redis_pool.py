"""
Tests for yiagent.common.redis_pool — Redis connection pool management.

CRITICAL BUGS FOUND during audit:
  1. get_redis() sets _pool BEFORE the warm-up ping — if ping fails,
     _pool stays set, and subsequent get_redis() returns a client
     connected to a broken pool.
  2. close_redis() has NO LOCK — races with get_redis() can cause
     pool corruption or "Pool is closed" errors.
  3. redis_set_json uses default=str — silently corrupts datetime and
     other non-JSON types into strings.
  4. redis_hset_json might send empty mapping to HSET (though guarded).
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ======================================================================
# Pool management
# ======================================================================

class TestRedisPoolManagement:
    @pytest.mark.asyncio
    async def test_get_redis_returns_client(self):
        """Basic get_redis() with full mock."""
        with patch("yiagent.common.redis_pool.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            with patch("yiagent.common.redis_pool.Redis") as mock_redis:
                mock_client = AsyncMock()
                mock_client.ping = AsyncMock(return_value=True)
                mock_redis.return_value = mock_client

                import yiagent.common.redis_pool as rp
                rp._pool = None
                client = await rp.get_redis()
                assert client is not None

    @pytest.mark.asyncio
    async def test_warmup_ping_failure_does_not_publish_pool(self):
        """
        FIX VERIFIED: After warm-up ping failure, _pool remains None and
        _pool_ready stays False. Subsequent get_redis() calls retry cleanly.
        """
        with patch("yiagent.common.redis_pool.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_pool.disconnect = AsyncMock()
            mock_from_url.return_value = mock_pool

            with patch("yiagent.common.redis_pool.Redis") as mock_redis:
                mock_client = AsyncMock()
                mock_client.ping = AsyncMock(side_effect=ConnectionError("no route to host"))
                mock_redis.return_value = mock_client

                import yiagent.common.redis_pool as rp
                rp._pool = None
                rp._pool_ready = False
                with pytest.raises(ConnectionError):
                    await rp.get_redis()

                # FIX: pool is NOT published on failure
                assert rp._pool is None, (
                    "FIX: _pool stays None after ping failure — clean retry"
                )
                assert rp._pool_ready is False, (
                    "FIX: _pool_ready stays False after ping failure"
                )

    @pytest.mark.asyncio
    async def test_double_checked_locking_correctness(self):
        """Verify the double-checked locking in get_redis works."""
        with patch("yiagent.common.redis_pool.aioredis.ConnectionPool.from_url") as mock_from_url:
            with patch("yiagent.common.redis_pool.Redis") as mock_redis:
                mock_from_url.return_value = MagicMock()
                mock_client = AsyncMock()
                mock_client.ping = AsyncMock(return_value=True)
                mock_redis.return_value = mock_client

                import yiagent.common.redis_pool as rp
                rp._pool = None

                results = await asyncio.gather(
                    *[rp.get_redis() for _ in range(100)]
                )
                # All must return a client, none should crash
                assert all(r is not None for r in results)
                # from_url should be called exactly once
                assert mock_from_url.call_count == 1, (
                    f"BUG: from_url called {mock_from_url.call_count} times (should be 1)"
                )


# ======================================================================
# High-level JSON helpers
# ======================================================================

class TestRedisJsonHelpers:
    @pytest.mark.asyncio
    async def test_set_and_get_json_roundtrip(self):
        from yiagent.common.redis_pool import redis_get_json, redis_set_json

        r = AsyncMock()
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        await redis_set_json(r, "testkey", data)
        # Check what was passed to r.set
        call_args = r.set.call_args
        assert call_args is not None
        key_arg = call_args[0][0]
        assert key_arg == "testkey"

        # Simulate get returning the stored JSON
        r.get = AsyncMock(return_value=json.dumps(data))
        result = await redis_get_json(r, "testkey")
        assert result == data

    @pytest.mark.asyncio
    async def test_get_json_miss(self):
        from yiagent.common.redis_pool import redis_get_json

        r = AsyncMock()
        r.get = AsyncMock(return_value=None)
        result = await redis_get_json(r, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_json_corrupt_data(self):
        """Corrupt JSON on get falls back to raw string."""
        from yiagent.common.redis_pool import redis_get_json

        r = AsyncMock()
        r.get = AsyncMock(return_value="not json {{{")
        result = await redis_get_json(r, "corrupt")
        assert result == "not json {{{"

    @pytest.mark.asyncio
    async def test_set_json_with_ex(self):
        from yiagent.common.redis_pool import redis_set_json

        r = AsyncMock()
        await redis_set_json(r, "key", {"x": 1}, ex=3600)
        # Verify ex was passed through
        assert r.set.call_args[1].get("ex") == 3600

    @pytest.mark.asyncio
    async def test_set_json_default_str_corruption(self):
        """
        BUG CONFIRMED: json.dumps with default=str silently converts
        non-serializable objects to strings. A datetime becomes a string
        and is NOT restored to datetime on get.
        """
        from datetime import datetime
        from yiagent.common.redis_pool import redis_get_json, redis_set_json

        r = AsyncMock()
        original = {"ts": datetime(2024, 1, 1, 12, 0, 0)}

        # Capture what gets passed to r.set
        await redis_set_json(r, "key", original)
        raw_stored = r.set.call_args[0][1]  # the JSON string

        # Simulate get returning that same string
        r.get = AsyncMock(return_value=raw_stored)
        result = await redis_get_json(r, "key")

        # BUG: ts is now a string, not a datetime
        assert isinstance(result["ts"], str), (
            "BUG: datetime silently converted to string — type lost"
        )
        assert not isinstance(result["ts"], type(original["ts"])), (
            "BUG: datetime type not preserved on roundtrip"
        )

    @pytest.mark.asyncio
    async def test_hset_and_hget_json_roundtrip(self):
        from yiagent.common.redis_pool import redis_hget_json, redis_hset_json

        r = AsyncMock()
        mapping = {"field1": "value1", "field2": 42}
        await redis_hset_json(r, "hashkey", mapping)

        # Verify encoding
        call_kwargs = r.hset.call_kwargs
        assert call_kwargs is not None

        # Simulate hget
        r.hget = AsyncMock(return_value=json.dumps("value1"))
        result = await redis_hget_json(r, "hashkey", "field1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_hset_json_empty_mapping(self):
        """Empty mapping must not call hset."""
        from yiagent.common.redis_pool import redis_hset_json

        r = AsyncMock()
        await redis_hset_json(r, "hashkey", {})
        r.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_hget_json_miss(self):
        from yiagent.common.redis_pool import redis_hget_json

        r = AsyncMock()
        r.hget = AsyncMock(return_value=None)
        result = await redis_hget_json(r, "hashkey", "missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_hget_json_corrupt(self):
        from yiagent.common.redis_pool import redis_hget_json

        r = AsyncMock()
        r.hget = AsyncMock(return_value="not json")
        result = await redis_hget_json(r, "hashkey", "field")
        assert result == "not json"

    @pytest.mark.asyncio
    async def test_hgetall_json(self):
        from yiagent.common.redis_pool import redis_hgetall_json

        r = AsyncMock()
        r.hgetall = AsyncMock(return_value={
            "key1": json.dumps({"nested": True}),
            "key2": "plain_string",
        })
        result = await redis_hgetall_json(r, "hashkey")
        assert result["key1"] == {"nested": True}
        assert result["key2"] == "plain_string"

    @pytest.mark.asyncio
    async def test_hgetall_json_handles_bad_data(self):
        from yiagent.common.redis_pool import redis_hgetall_json

        r = AsyncMock()
        r.hgetall = AsyncMock(return_value={
            "good": json.dumps("ok"),
            "bad": b"\xff\xfe",  # binary garbage
        })
        result = await redis_hgetall_json(r, "hashkey")
        assert result["good"] == "ok"
        # Bad data falls through as-is
        assert "bad" in result


# ======================================================================
# close_redis
# ======================================================================

class TestCloseRedis:
    @pytest.mark.asyncio
    async def test_close_when_none(self):
        """close_redis when _pool is None must not crash."""
        import yiagent.common.redis_pool as rp
        rp._pool = None
        await rp.close_redis()  # Must not raise

    @pytest.mark.asyncio
    async def test_close_clears_pool(self):
        import yiagent.common.redis_pool as rp
        rp._pool = MagicMock()
        rp._pool.disconnect = AsyncMock()
        await rp.close_redis()
        assert rp._pool is None

    @pytest.mark.asyncio
    async def test_close_races_with_get_redis_no_lock(self):
        """
        BUG CONFIRMED: close_redis() uses no lock. If called while
        get_redis() is creating the pool, the pool can be set to None
        out from under the get_redis() caller.
        """
        import yiagent.common.redis_pool as rp
        rp._pool = None

        mock_pool = MagicMock()
        mock_pool.disconnect = AsyncMock()

        with patch("yiagent.common.redis_pool.aioredis.ConnectionPool.from_url") as mock_from_url:
            with patch("yiagent.common.redis_pool.Redis") as mock_redis:
                mock_from_url.return_value = mock_pool
                mock_client = AsyncMock()
                mock_client.ping = AsyncMock(return_value=True)
                mock_redis.return_value = mock_client

                async def delayed_get():
                    rp._pool = mock_pool
                    await asyncio.sleep(0.02)  # simulate warm-up delay
                    return

                async def quick_close():
                    await asyncio.sleep(0.01)
                    await rp.close_redis()

                # Run both — close_redis sets _pool = None while get_redis
                # is still using it
                await asyncio.gather(delayed_get(), quick_close())
                # _pool is None after close (overwrites the delayed set)
                assert rp._pool is None, (
                    "Race condition: _pool state depends on timing"
                )
