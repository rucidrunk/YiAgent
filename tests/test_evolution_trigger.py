"""
Tests for yiagent.agent.evolution.trigger — idle-based evolution scanner.

CRITICAL AREAS:
  1. note_user_turn / mark_run_active — activity tracking
  2. start/stop lifecycle — idempotent, cancellation
  3. _scan_once — session filtering, concurrency handling
  4. _context_pressure — token budget overflow detection
"""
from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules["asyncpg"] = MagicMock()

from yiagent.agent.evolution.trigger import (
    _context_pressure,
    _FALLBACK_BUDGET,
    _CONTEXT_RATIO,
    _SCAN_INTERVAL,
    note_user_turn,
    mark_run_active,
    start_evolution_trigger,
    stop_evolution_trigger,
    _trigger_task,
)


class TestNoteUserTurn:
    def test_records_activity(self):
        agent = MagicMock()
        agent._evo_last_active = 0
        agent._evo_turns = 0

        before = time.time()
        note_user_turn(agent, channel_type="web", receiver="ch1")
        after = time.time()

        assert before <= agent._evo_last_active <= after
        assert agent._evo_turns == 1
        assert agent._evo_channel_type == "web"
        assert agent._evo_receiver == "ch1"

    def test_increments_turns(self):
        agent = MagicMock()
        agent._evo_turns = 5
        note_user_turn(agent)
        assert agent._evo_turns == 6

    def test_no_crash_on_missing_attrs(self):
        agent = object()  # no _evo_last_active
        note_user_turn(agent)  # must not crash


class TestMarkRunActive:
    def test_sets_active_flag(self):
        agent = MagicMock()
        agent._evo_run_active = False
        mark_run_active(agent, True)
        assert agent._evo_run_active is True

    def test_clears_active_flag(self):
        agent = MagicMock()
        agent._evo_run_active = True
        mark_run_active(agent, False)
        assert agent._evo_run_active is False

    def test_updates_last_active_when_active(self):
        agent = MagicMock()
        agent._evo_last_active = 0
        before = time.time()
        mark_run_active(agent, True)
        after = time.time()
        assert before <= agent._evo_last_active <= after


class TestStartStopTrigger:
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """start → stop must clean up _trigger_task."""
        import yiagent.agent.evolution.trigger as trig
        trig._trigger_task = None
        try:
            mock_bridge = MagicMock()
            mock_bridge.agents = {}
            await trig.start_evolution_trigger(mock_bridge, interval=3600)
            assert trig._trigger_task is not None
            await trig.stop_evolution_trigger()
            assert trig._trigger_task is None
        finally:
            trig._trigger_task = None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Second start() when already running is a no-op."""
        import yiagent.agent.evolution.trigger as trig
        trig._trigger_task = None
        try:
            mock_bridge = MagicMock()
            mock_bridge.agents = {}
            await trig.start_evolution_trigger(mock_bridge, interval=3600)
            task1 = trig._trigger_task
            await trig.start_evolution_trigger(mock_bridge, interval=3600)
            assert trig._trigger_task is task1  # same task
            await trig.stop_evolution_trigger()
        finally:
            trig._trigger_task = None

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """stop() when not running is a no-op."""
        import yiagent.agent.evolution.trigger as trig
        trig._trigger_task = None
        await trig.stop_evolution_trigger()  # must not crash


class TestContextPressure:
    def test_no_messages_returns_false(self):
        agent = MagicMock()
        agent.messages = []
        assert _context_pressure(agent) is False

    def test_below_threshold_returns_false(self):
        agent = MagicMock()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent._estimate_message_tokens = lambda m: 100
        agent.max_context_tokens = 10000
        # 100 / 10000 = 0.01 < 0.8
        assert _context_pressure(agent) is False

    def test_above_threshold_returns_true(self):
        agent = MagicMock()
        agent.messages = [{"role": "user", "content": "big message"}]
        agent._estimate_message_tokens = lambda m: 9000
        agent.max_context_tokens = 10000
        # 9000 / 10000 = 0.9 > 0.8
        assert _context_pressure(agent) is True

    def test_fallback_budget_used_when_none(self):
        agent = MagicMock()
        agent.messages = [{"role": "user", "content": "huge"}]
        agent._estimate_message_tokens = lambda m: _FALLBACK_BUDGET + 1
        agent.max_context_tokens = None
        assert _context_pressure(agent) is True

    def test_exception_returns_false(self):
        agent = MagicMock()
        agent.messages = None  # will cause AttributeError
        assert _context_pressure(agent) is False


class TestEvolutionTriggerExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    @pytest.mark.asyncio
    async def test_many_rapid_note_user_turns(self):
        """100 rapid note_user_turn calls must not crash."""
        agent = MagicMock()
        agent._evo_turns = 0
        for _ in range(100):
            note_user_turn(agent)
        assert agent._evo_turns == 100

    def test_none_agent_does_not_crash(self):
        """None agent must be handled gracefully."""
        note_user_turn(None)
        mark_run_active(None, True)

    def test_default_scan_interval(self):
        assert _SCAN_INTERVAL == 60
