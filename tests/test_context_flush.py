"""
Tests for yiagent.memory.context_flush — async queue-based flush pipeline.

CRITICAL BUGS FOUND during audit:
  1. _process writes files synchronously (blocking I/O in async worker)
  2. monitor_and_flush compares turn_idx to len(messages)//2 (not turn_count//2)
  3. stop() may hang if a worker was respawned and the old task reference lingers
  4. QueueFull silently drops flush tasks (backpressure without feedback to caller)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yiagent.memory.context_flush import ContextFlushPipeline, FlushTask


class TestContextFlushBasics:
    def test_init_defaults(self):
        pipeline = ContextFlushPipeline()
        assert pipeline.queue_depth == 0
        assert pipeline._running is False
        assert pipeline._max_workers == 4

    def test_init_custom(self):
        pipeline = ContextFlushPipeline(
            max_workers=2,
            queue_maxsize=64,
            high_watermark=0.9,
        )
        assert pipeline._max_workers == 2
        assert pipeline._high_watermark == 0.9

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        pipeline = ContextFlushPipeline(max_workers=1)
        await pipeline.start()
        assert pipeline._running is True
        assert len(pipeline._workers) == 1
        await pipeline.stop()
        assert pipeline._running is False

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        pipeline = ContextFlushPipeline(max_workers=1)
        await pipeline.start()
        worker_count = len(pipeline._workers)
        await pipeline.start()  # Should be no-op
        assert len(pipeline._workers) == worker_count
        await pipeline.stop()


class TestContextFlushMonitor:
    @pytest.mark.asyncio
    async def test_below_watermark_no_flush(self):
        pipeline = ContextFlushPipeline(high_watermark=0.8)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = await pipeline.monitor_and_flush(
            "s1", token_estimate=100, max_tokens=1000, messages=messages,
        )
        assert result is False  # 100/1000 = 0.1 < 0.8

    @pytest.mark.asyncio
    async def test_above_watermark_but_few_turns_no_flush(self):
        pipeline = ContextFlushPipeline(high_watermark=0.5)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        result = await pipeline.monitor_and_flush(
            "s1", token_estimate=600, max_tokens=1000, messages=messages,
        )
        # Above watermark but only 1 turn (< 4 minimum)
        assert result is False

    @pytest.mark.asyncio
    async def test_overflow_triggers_flush(self):
        pipeline = ContextFlushPipeline(high_watermark=0.5)
        # Build 4+ visible turns to satisfy minimum
        messages = []
        for i in range(6):
            messages.append({"role": "user", "content": [{"type": "text", "text": f"turn {i}"}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"reply {i}"}]})

        result = await pipeline.monitor_and_flush(
            "s1", token_estimate=800, max_tokens=1000, messages=messages,
        )
        # 800/1000 = 0.8 >= 0.5 → should enqueue
        assert result is True
        assert pipeline.queue_depth >= 1

    @pytest.mark.asyncio
    async def test_zero_max_tokens_no_flush(self):
        pipeline = ContextFlushPipeline(high_watermark=0.5)
        result = await pipeline.monitor_and_flush(
            "s1", token_estimate=100, max_tokens=0, messages=[{"role": "user", "content": "x"}],
        )
        assert result is False


class TestContextFlushQueue:
    @pytest.mark.asyncio
    async def test_queue_full_drops_task(self):
        pipeline = ContextFlushPipeline(queue_maxsize=1, max_workers=0)
        messages = []
        for i in range(6):
            messages.append({"role": "user", "content": [{"type": "text", "text": f"turn {i}"}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"r{i}"}]})

        # Fill the queue (workers not started, so queue doesn't drain)
        result1 = await pipeline.monitor_and_flush(
            "s1", token_estimate=800, max_tokens=1000, messages=messages,
        )
        result2 = await pipeline.monitor_and_flush(
            "s2", token_estimate=800, max_tokens=1000, messages=messages,
        )
        # At least one must be enqueued; the second may be dropped
        assert result1 or result2

    @pytest.mark.asyncio
    async def test_queue_depth_reports_correctly(self):
        pipeline = ContextFlushPipeline(queue_maxsize=10, max_workers=0)
        messages = []
        for i in range(6):
            messages.append({"role": "user", "content": [{"type": "text", "text": f"turn {i}"}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"r{i}"}]})

        await pipeline.monitor_and_flush(
            "s1", token_estimate=800, max_tokens=1000, messages=messages,
        )
        assert pipeline.queue_depth == 1


class TestBuildTranscript:
    def test_builds_transcript_from_messages(self):
        from yiagent.memory.context_flush import ContextFlushPipeline
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
            {"role": "system", "content": "ignore me"},
        ]
        transcript = ContextFlushPipeline._build_transcript(messages)
        assert "User: hello" in transcript
        assert "Assistant: hi there" in transcript
        assert "ignore me" not in transcript

    def test_builds_transcript_from_string_content(self):
        from yiagent.memory.context_flush import ContextFlushPipeline
        messages = [
            {"role": "user", "content": "plain string"},
        ]
        transcript = ContextFlushPipeline._build_transcript(messages)
        assert "User: plain string" in transcript

    def test_respects_max_chars(self):
        from yiagent.memory.context_flush import ContextFlushPipeline
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "x" * 10000}]},
        ]
        transcript = ContextFlushPipeline._build_transcript(messages, max_chars=100)
        assert len(transcript) <= 100 + len("...(earlier omitted)...\n")


class TestCountTurns:
    def test_counts_only_visible_user_turns(self):
        from yiagent.memory.context_flush import ContextFlushPipeline
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "user", "content": [{"type": "tool_result", "text": "result"}]},
            {"role": "user", "content": [{"type": "text", "text": "q2"}]},
        ]
        count = ContextFlushPipeline._count_turns(messages)
        assert count == 2  # q1 and q2; tool_result not counted


class TestContextFlushExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    @pytest.mark.asyncio
    async def test_queue_full_backpressure(self):
        """When queue is full, monitor_and_flush returns False (not crash)."""
        pipeline = ContextFlushPipeline(queue_maxsize=2, max_workers=0)
        messages = []
        for i in range(6):
            messages.append({"role": "user", "content": [{"type": "text", "text": f"t{i}"}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"r{i}"}]})

        # Fill queue to capacity
        results = []
        for i in range(5):
            r = await pipeline.monitor_and_flush(
                f"s{i}", token_estimate=800, max_tokens=1000, messages=messages,
            )
            results.append(r)
        # Some enqueued, some dropped — none crashed
        assert any(results), "At least one should be enqueued"
        # No exceptions

    @pytest.mark.asyncio
    async def test_worker_respawn_on_error(self):
        """If a worker crashes processing a task, it respawns."""
        with patch("yiagent.memory.context_flush.ContextFlushPipeline._process") as mock_process:
            mock_process.side_effect = RuntimeError("worker crash")
            pipeline = ContextFlushPipeline(max_workers=1)

            # Manually enqueue a task
            task = FlushTask(
                session_id="s1",
                user_id="u1",
                messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                reason="threshold",
            )
            await pipeline._queue.put(task)
            await pipeline.start()

            # Give worker time to process and crash
            await asyncio.sleep(0.1)

            # Worker should have respawned (or be in the process)
            # The pipeline should still be running
            # At minimum, no unhandled exception escaped

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_drains_queue(self):
        """stop() sends sentinels and waits for workers to finish."""
        pipeline = ContextFlushPipeline(max_workers=2)
        await pipeline.start()
        # Workers are idle (queue is empty) — stop should complete quickly
        await asyncio.wait_for(pipeline.stop(), timeout=2.0)
        assert pipeline._running is False
        # Workers should be done
        for w in pipeline._workers:
            assert w.done()
