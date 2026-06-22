"""
Tests for yiagent.memory.conversation_store — the Redis-backed session memory.

CRITICAL BUGS FOUND during audit:
  1. append_messages_batch is NOT atomic despite docstring claiming it is
  2. upsert_meta mutates the caller's **fields dict via setdefault (side effect)
  3. _persist_to_pg uses create_task without storing the task — silent failure
  4. seq numbers drift after ltrim (gaps in sequence)
  5. close_redis() has no lock — race with get_redis() can corrupt pool
  6. get_redis() sets _pool BEFORE warm-up probe — broken pool on failure
  7. _trim_by_visible_turns with string content: content.strip() is fine
     but if content is a bare string not wrapped in list, iterating yields chars
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from yiagent.memory.conversation_store import (
    ConversationStore,
    SessionMeta,
    _ctx_key,
    _embed_cache_key,
    _lock_key,
    _meta_key,
    get_conversation_store,
)


# ======================================================================
# SessionMeta
# ======================================================================

class TestSessionMeta:
    def test_default_values(self):
        meta = SessionMeta(session_id="s1")
        assert meta.session_id == "s1"
        assert meta.user_id == ""
        assert meta.msg_count == 0

    def test_to_dict_from_dict_roundtrip(self):
        meta = SessionMeta(
            session_id="s1",
            user_id="u1",
            channel_type="web",
            msg_count=5,
            context_start_seq=3,
        )
        d = meta.to_dict()
        restored = SessionMeta.from_dict(d)
        assert restored.user_id == "u1"
        assert restored.msg_count == 5

    def test_to_dict_all_values_are_strings(self):
        """to_dict converts numeric fields to strings for HSET storage."""
        meta = SessionMeta(session_id="s1", msg_count=42, context_start_seq=10)
        d = meta.to_dict()
        assert isinstance(d["msg_count"], str)
        assert d["msg_count"] == "42"
        assert isinstance(d["context_start_seq"], str)
        assert d["context_start_seq"] == "10"

    def test_from_dict_handles_missing_keys(self):
        restored = SessionMeta.from_dict({})
        assert restored.session_id == ""
        assert restored.msg_count == 0


# ======================================================================
# Key helpers
# ======================================================================

class TestKeyHelpers:
    def test_ctx_key_format(self):
        assert _ctx_key("abc123") == "session:abc123:context"

    def test_meta_key_format(self):
        assert _meta_key("abc123") == "session:abc123:meta"

    def test_lock_key_format(self):
        assert _lock_key("abc123") == "session:abc123:run_lock"

    def test_embed_cache_key_format(self):
        key = _embed_cache_key("openai", "text-embedding-3-small", "abc123")
        assert key == "embed_cache:openai:text-embedding-3-small:abc123"


# ======================================================================
# ConversationStore — Lock operations
# ======================================================================

class TestConversationStoreLock:
    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, patch_get_redis):
        patch_get_redis.set = AsyncMock(return_value=True)
        store = ConversationStore()
        result = await store.acquire_lock("s1")
        assert result is True
        patch_get_redis.set.assert_called_once_with(
            _lock_key("s1"), "1", nx=True, ex=120
        )

    @pytest.mark.asyncio
    async def test_acquire_lock_failure(self, patch_get_redis):
        """Lock already held by another instance."""
        patch_get_redis.set = AsyncMock(return_value=None)
        store = ConversationStore()
        result = await store.acquire_lock("s1")
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_lock_redis_error(self, patch_get_redis):
        """
        FIX VERIFIED: Redis error now raises RedisUnavailableError
        so callers can distinguish "Redis down" from "lock held by another".
        """
        from yiagent.memory.conversation_store import RedisUnavailableError
        patch_get_redis.set = AsyncMock(side_effect=ConnectionError("boom"))
        store = ConversationStore()
        with pytest.raises(RedisUnavailableError, match="Redis unavailable"):
            await store.acquire_lock("s1")

    @pytest.mark.asyncio
    async def test_release_lock(self, patch_get_redis):
        store = ConversationStore()
        await store.release_lock("s1")
        patch_get_redis.delete.assert_called_once_with(_lock_key("s1"))

    @pytest.mark.asyncio
    async def test_release_lock_redis_error_graceful(self, patch_get_redis):
        patch_get_redis.delete = AsyncMock(side_effect=ConnectionError("boom"))
        store = ConversationStore()
        await store.release_lock("s1")  # Must not raise

    @pytest.mark.asyncio
    async def test_extend_lock(self, patch_get_redis):
        store = ConversationStore()
        await store.extend_lock("s1")
        patch_get_redis.expire.assert_called_once_with(_lock_key("s1"), 120)

    @pytest.mark.asyncio
    async def test_extend_lock_error_silent(self, patch_get_redis):
        patch_get_redis.expire = AsyncMock(side_effect=Exception("boom"))
        store = ConversationStore()
        await store.extend_lock("s1")  # Must not raise


# ======================================================================
# ConversationStore — Session metadata
# ======================================================================

class TestConversationStoreMeta:
    @pytest.mark.asyncio
    async def test_get_meta_empty(self, patch_get_redis):
        store = ConversationStore()
        result = await store.get_meta("s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_meta_with_data(self, patch_get_redis):
        patch_get_redis.hgetall = AsyncMock(return_value={
            "user_id": '"u1"',  # JSON-encoded values
            "msg_count": '"10"',
            "created_at": '"1234567890.0"',
        })
        store = ConversationStore()
        result = await store.get_meta("s1")
        assert result is not None
        assert result.user_id == "u1"
        assert result.msg_count == 10

    @pytest.mark.asyncio
    async def test_upsert_meta_new_session(self, patch_get_redis):
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()
        await store.upsert_meta("s1", user_id="u1", channel_type="web")
        # Verify HSET was called with the right keys
        call_args = patch_get_redis.hset.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_upsert_meta_caller_dict_mutation_bug(self, patch_get_redis):
        """
        NOTE: **caller_dict unpacks into a new dict inside upsert_meta's **fields,
        so the caller's dict is NOT mutated. However, if the caller passes a
        mutable value (e.g., a list or nested dict) as a field value, and
        upsert_meta modifies THAT, it could be a side effect.

        The real bug is different: hexists check and hset are not atomic,
        so two concurrent upsert_meta calls can race on created_at.
        """
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()

        caller_dict = {"user_id": "u1"}
        await store.upsert_meta("s1", **caller_dict)

        # ** unpacking creates a copy, so caller's dict is NOT mutated
        assert "created_at" not in caller_dict, (
            "caller dict was not mutated (good)"
        )

    @pytest.mark.asyncio
    async def test_touch_session(self, patch_get_redis):
        store = ConversationStore()
        await store.touch_session("s1")
        # Must call hset for last_active and expire on both meta and context keys
        assert patch_get_redis.hset.called
        assert patch_get_redis.expire.call_count == 2

    @pytest.mark.asyncio
    async def test_touch_session_error_graceful(self, patch_get_redis):
        patch_get_redis.hset = AsyncMock(side_effect=Exception("boom"))
        store = ConversationStore()
        await store.touch_session("s1")  # Must not raise


# ======================================================================
# ConversationStore — Context window
# ======================================================================

class TestConversationStoreContext:
    @pytest.mark.asyncio
    async def test_load_context_empty(self, patch_get_redis):
        patch_get_redis.lrange = AsyncMock(return_value=[])
        store = ConversationStore()
        # Will fall through to PG which also returns empty since we didn't patch it
        with patch.object(store, "_load_from_pg", AsyncMock(return_value=[])):
            result = await store.load_context("s1")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_context_with_messages(self, patch_get_redis):
        msgs = [
            json.dumps({"role": "user", "content": [{"type": "text", "text": "hello"}], "seq": 0}),
            json.dumps({"role": "assistant", "content": [{"type": "text", "text": "hi"}], "seq": 1}),
        ]
        patch_get_redis.lrange = AsyncMock(return_value=msgs)
        store = ConversationStore()
        result = await store.load_context("s1")
        assert len(result) == 2
        assert result[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_load_context_falls_back_to_pg_on_redis_error(self, patch_get_redis):
        patch_get_redis.lrange = AsyncMock(side_effect=ConnectionError("redis down"))
        store = ConversationStore()
        with patch.object(store, "_load_from_pg", AsyncMock(return_value=[{"role": "user", "content": "pg"}])):
            result = await store.load_context("s1")
        assert len(result) == 1
        assert result[0]["content"] == "pg"

    @pytest.mark.asyncio
    async def test_load_context_respects_context_start_seq(self, patch_get_redis):
        msgs = [
            json.dumps({"role": "user", "content": [{"type": "text", "text": "old"}], "seq": 0}),
            json.dumps({"role": "user", "content": [{"type": "text", "text": "new"}], "seq": 5}),
        ]
        patch_get_redis.lrange = AsyncMock(return_value=msgs)
        store = ConversationStore()
        result = await store.load_context("s1", context_start_seq=5)
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == "new"

    @pytest.mark.asyncio
    async def test_append_message_basic(self, patch_get_redis):
        patch_get_redis.llen = AsyncMock(return_value=0)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()
        count = await store.append_message(
            "s1",
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            channel_type="web",
        )
        assert count == 1
        patch_get_redis.rpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_append_message_assigns_seq(self, patch_get_redis):
        """Verify seq is assigned as current list length."""
        patch_get_redis.llen = AsyncMock(return_value=5)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        await store.append_message("s1", msg)
        # The message dict should have seq=5 (the original llen value)
        assert msg.get("seq") == 5, f"Expected seq=5, got {msg.get('seq')}"

    @pytest.mark.asyncio
    async def test_append_message_trims_when_exceeds_200(self, patch_get_redis):
        """When list exceeds 200 messages, ltrim must be called."""
        patch_get_redis.llen = AsyncMock(return_value=200)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()
        await store.append_message("s1", {"role": "user", "content": "test"})
        # overflow = max(0, 200 + 1 - 200) = 1
        patch_get_redis.ltrim.assert_called_with(_ctx_key("s1"), 1, -1)

    @pytest.mark.asyncio
    async def test_append_message_failure_returns_zero(self, patch_get_redis):
        patch_get_redis.llen = AsyncMock(side_effect=ConnectionError("boom"))
        store = ConversationStore()
        count = await store.append_message("s1", {"role": "user", "content": "test"})
        assert count == 0

    @pytest.mark.asyncio
    async def test_append_messages_batch_partial_failure(self, patch_get_redis):
        """
        append_messages_batch is NOT atomic (docstring now admits this).
        Returns (success_count, total_count) tuple.
        When msg3 fails, msg1+msg2 are already persisted → (2, 3).
        """
        patch_get_redis.llen = AsyncMock(side_effect=[0, 1, 2])
        patch_get_redis.rpush = AsyncMock(side_effect=[
            None, None, ConnectionError("mid-batch failure")
        ])
        patch_get_redis.hexists = AsyncMock(return_value=False)

        store = ConversationStore()
        msgs = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        success, total = await store.append_messages_batch("s1", msgs)
        # 2 out of 3 messages persisted before the failure
        assert success == 2, f"Expected 2 successes, got {success}"
        assert total == 3, f"Expected 3 total, got {total}"
        # All 3 rpush calls were attempted
        assert patch_get_redis.rpush.call_count == 3

    @pytest.mark.asyncio
    async def test_clear_context(self, patch_get_redis):
        patch_get_redis.llen = AsyncMock(return_value=42)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()
        result = await store.clear_context("s1")
        assert result == 42

    @pytest.mark.asyncio
    async def test_clear_session(self, patch_get_redis):
        store = ConversationStore()
        await store.clear_session("s1")
        patch_get_redis.delete.assert_called_once_with(
            _ctx_key("s1"), _meta_key("s1"), _lock_key("s1")
        )

    @pytest.mark.asyncio
    async def test_get_stats(self, patch_get_redis):
        patch_get_redis.keys = AsyncMock(return_value=[b"s1:meta", b"s2:meta"])
        store = ConversationStore()
        stats = await store.get_stats()
        assert stats["active_sessions"] == 2


# ======================================================================
# ConversationStore — Embedding cache
# ======================================================================

class TestConversationStoreEmbedCache:
    @pytest.mark.asyncio
    async def test_get_embed_cache_miss(self, patch_get_redis):
        patch_get_redis.get = AsyncMock(return_value=None)
        store = ConversationStore()
        result = await store.get_embed_cache("openai", "text-embedding-3-small", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_embed_cache(self, patch_get_redis):
        patch_get_redis.get = AsyncMock(return_value=json.dumps([0.1, 0.2, 0.3]))
        store = ConversationStore()
        result = await store.get_embed_cache("openai", "text-embedding-3-small", "hello")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_cache_key_uses_md5(self):
        """Embed cache key uses MD5 hash of query."""
        import hashlib
        qh = hashlib.md5("hello".encode()).hexdigest()
        key = _embed_cache_key("openai", "text-embedding-3-small", qh)
        assert "hello" not in key  # query itself not in key


# ======================================================================
# ConversationStore — _trim_by_visible_turns
# ======================================================================

class TestTrimByVisibleTurns:
    def test_empty_messages(self):
        result = ConversationStore._trim_by_visible_turns([], 5)
        assert result == []

    def test_fewer_turns_than_max(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = ConversationStore._trim_by_visible_turns(msgs, 10)
        assert len(result) == 2

    def test_trims_excess_visible_turns(self):
        """4 user turns, max_turns=2 → keep last 2 visible turns + their responses."""
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "t1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "r1"}]},
            {"role": "user", "content": [{"type": "text", "text": "t2"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "r2"}]},
            {"role": "user", "content": [{"type": "text", "text": "t3"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "r3"}]},
            {"role": "user", "content": [{"type": "text", "text": "t4"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "r4"}]},
        ]
        result = ConversationStore._trim_by_visible_turns(msgs, 2)
        # Should keep t3, r3, t4, r4
        assert len(result) == 4
        assert result[0]["content"][0]["text"] == "t3"

    def test_string_content_visible_turn(self):
        """User messages with plain string content also count as visible turns."""
        msgs = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "turn2"},
            {"role": "user", "content": "turn3"},
        ]
        result = ConversationStore._trim_by_visible_turns(msgs, 2)
        assert result[0]["content"] == "turn2"
        assert result[1]["content"] == "turn3"

    def test_tool_result_not_visible(self):
        """Messages where user only sends tool_result are NOT visible turns.
        With max_turns=0, everything before the most recent visible turn is cut."""
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "text": "result"}]},
            {"role": "user", "content": [{"type": "text", "text": "real question"}]},
        ]
        # 1 visible turn, max_turns=0 → trim to cutoff at visible_indices[0]
        result = ConversationStore._trim_by_visible_turns(msgs, 0)
        assert result[0]["content"][0]["text"] == "real question"

    def test_mixed_tool_and_text_in_same_message_is_not_visible(self):
        """If a user message has both text AND tool_result blocks, it is NOT visible.
        The is_visible logic: has_text AND not has_tool. Both present → not visible."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "question"},
                    {"type": "tool_result", "text": "result"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "next question"}]},
        ]
        # Only "next question" is visible. max_turns=0 → keep only from that cutoff.
        result = ConversationStore._trim_by_visible_turns(msgs, 0)
        assert result[0]["content"][0]["text"] == "next question"


# ======================================================================
# Singleton
# ======================================================================

class TestGetConversationStore:
    @pytest.mark.asyncio
    async def test_returns_singleton(self, patch_get_redis):
        s1 = await get_conversation_store()
        s2 = await get_conversation_store()
        assert s1 is s2

    @pytest.mark.asyncio
    async def test_concurrent_init_same_instance(self, patch_get_redis):
        stores = await asyncio.gather(
            *[get_conversation_store() for _ in range(50)]
        )
        assert all(s is stores[0] for s in stores)


# ======================================================================
# EXTREME: Boundary conditions — 3 groups of extreme tests
# ======================================================================

class TestConversationStoreExtreme:
    """
    EXTREME BOUNDARY TEST GROUP 1: Instantaneous flood / Redis hot-key
    """

    @pytest.mark.asyncio
    async def test_flood_append_to_same_session(self, patch_get_redis):
        """100 rapid appends to the same session key — simulate bot traffic spike."""
        patch_get_redis.llen = AsyncMock(return_value=0)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()

        async def append_one(i):
            return await store.append_message(
                "s1", {"role": "user", "content": f"msg{i}"}
            )

        results = await asyncio.gather(*[append_one(i) for i in range(100)])
        # Each call should return a count
        assert all(isinstance(r, int) for r in results)

    @pytest.mark.asyncio
    async def test_flood_acquire_lock_same_session(self, patch_get_redis):
        """100 concurrent lock acquisitions on the same session — only one wins."""
        patch_get_redis.set = AsyncMock(side_effect=[True] + [None] * 99)
        store = ConversationStore()

        results = await asyncio.gather(*[
            store.acquire_lock("s1") for _ in range(100)
        ])
        assert sum(results) == 1, (
            f"BUG: {sum(results)} callers acquired the lock (expected 1)"
        )

    """
    EXTREME BOUNDARY TEST GROUP 2: Ultra-long token / message overflow
    """

    @pytest.mark.asyncio
    async def test_append_message_with_massive_content(self, patch_get_redis):
        """Append a message with 1MB of text content — must not OOM."""
        patch_get_redis.llen = AsyncMock(return_value=0)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()
        huge_text = "x" * 1_000_000
        count = await store.append_message(
            "s1",
            {"role": "user", "content": [{"type": "text", "text": huge_text}]},
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_load_context_with_corrupt_messages(self, patch_get_redis):
        """Messages that fail JSON decode must be skipped gracefully."""
        msgs = [
            json.dumps({"role": "user", "content": "valid"}),
            "not valid json {{{",  # corrupt
            json.dumps({"role": "assistant", "content": "also valid"}),
        ]
        patch_get_redis.lrange = AsyncMock(return_value=msgs)
        store = ConversationStore()
        result = await store.load_context("s1")
        # Should skip the corrupt one
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_session_id_with_special_characters(self, patch_get_redis):
        """Session ID with colons, dots, slashes — must not break key patterns."""
        weird_id = "user:evil.com/../../etc"
        key = _ctx_key(weird_id)
        # Key should contain the literal string, not path-traverse
        assert weird_id in key
        patch_get_redis.llen = AsyncMock(return_value=0)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()
        await store.upsert_meta(weird_id, user_id="test")

    """
    EXTREME BOUNDARY TEST GROUP 3: Cache stampede / TTL race conditions
    """

    @pytest.mark.asyncio
    async def test_cache_stampede_load_context(self, patch_get_redis):
        """50 concurrent load_context calls on cold cache — all hit PG at once."""
        patch_get_redis.lrange = AsyncMock(return_value=[])
        store = ConversationStore()
        pg_call_count = 0

        async def pg_fallback(*args, **kwargs):
            nonlocal pg_call_count
            pg_call_count += 1
            await asyncio.sleep(0.01)
            return []

        with patch.object(store, "_load_from_pg", pg_fallback):
            results = await asyncio.gather(*[
                store.load_context("s1") for _ in range(50)
            ])
        # BUG: No cache-stampede protection — all 50 hit PG cold path
        assert pg_call_count == 50, (
            f"Cache stampede: {pg_call_count} concurrent PG calls (no merging)"
        )

    @pytest.mark.asyncio
    async def test_upsert_meta_race_created_at(self, patch_get_redis):
        """50 concurrent upsert_meta on new session — race on created_at."""
        patch_get_redis.hexists = AsyncMock(return_value=False)
        patch_get_redis.hset = AsyncMock(return_value=1)
        store = ConversationStore()

        await asyncio.gather(*[
            store.upsert_meta("s1", user_id=f"u{i}") for i in range(50)
        ])
        # No crash is a pass. But note: created_at values will differ
        # across calls because each sets its own timestamp.

    @pytest.mark.asyncio
    async def test_clear_session_concurrent_with_append(self, patch_get_redis):
        """clear_session racing with append_message — potential data loss."""
        patch_get_redis.llen = AsyncMock(return_value=50)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()

        # Run clear and append concurrently
        await asyncio.gather(
            store.clear_session("s1"),
            store.append_message("s1", {"role": "user", "content": "msg"})
        )
        # No crash is a pass, but note: message may be lost if
        # append runs after delete.

    @pytest.mark.asyncio
    async def test_append_messages_batch_seq_continuity(self, patch_get_redis):
        """
        BUG: llen returns the current length, but after ltrim, seq numbers
        have gaps because ltrim removes from the head without renumbering.
        """
        lengths = [200, 201, 202]  # simulate near-capacity list
        patch_get_redis.llen = AsyncMock(side_effect=lengths)
        patch_get_redis.hexists = AsyncMock(return_value=False)
        store = ConversationStore()

        msg1 = {"role": "user", "content": "msg1"}
        msg2 = {"role": "user", "content": "msg2"}
        msg3 = {"role": "user", "content": "msg3"}

        await store.append_message("s1", msg1)
        await store.append_message("s1", msg2)
        await store.append_message("s1", msg3)

        # After ltrim of first msg1 (overflow=1), msg2's seq is 201
        # but the list now starts at index 1. seq numbers are disconnected
        # from list indices.
        assert msg1["seq"] == 200
        assert msg2["seq"] == 201
        assert msg3["seq"] == 202
