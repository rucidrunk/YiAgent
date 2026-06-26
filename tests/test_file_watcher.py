"""
Tests for yiagent.memory.file_watcher — watchdog-based .md file monitor.

CRITICAL BUGS FOUND:
  1. _handle calls asyncio.get_running_loop() from daemon thread → always fails
     → _recent never cleaned up → files processed ONCE then permanently ignored
  2. _recent set accessed from watchdog thread + event loop thread → no lock
  3. _MemoryFileHandler inherits from object when watchdog not installed
  4. schedule_async has no error handling on the returned Future
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from yiagent.memory.file_watcher import (
    MemoryFileWatcher,
    _MemoryFileHandler,
    watchdog as _watchdog_available,
)


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / "memory").mkdir(exist_ok=True)
        yield str(ws)


def _make_handler(on_changed=None, cooldown=5.0):
    """Create a _MemoryFileHandler with a mock event loop (for testing)."""
    loop = MagicMock()
    return _MemoryFileHandler(
        on_changed=on_changed or (lambda p: None),
        loop=loop,
        cooldown=cooldown,
    )


class TestMemoryFileWatcherInit:
    def test_init_creates_watcher(self):
        w = MemoryFileWatcher("/tmp/test_ws")
        assert w._workspace_dir == Path("/tmp/test_ws")
        assert w._running is False

    def test_default_cooldown(self):
        w = MemoryFileWatcher("/tmp/test_ws")
        assert w._cooldown == 5.0

    def test_custom_cooldown(self):
        w = MemoryFileWatcher("/tmp/test_ws", cooldown=1.0)
        assert w._cooldown == 1.0


class TestMemoryFileWatcherStartStop:
    def test_start_without_watchdog(self):
        """When watchdog is not installed, start() is a no-op."""
        w = MemoryFileWatcher("/tmp/test_ws")
        # Observer is None when watchdog not installed
        w.start()
        assert w._running is False  # didn't start

    def test_stop_when_not_running(self):
        w = MemoryFileWatcher("/tmp/test_ws")
        w.stop()  # must not crash

    def test_stop_cleans_up(self):
        w = MemoryFileWatcher("/tmp/test_ws")
        w._running = True
        w._observer = MagicMock()
        w.stop()
        assert w._running is False

    def test_start_creates_memory_dir(self, temp_workspace):
        # remove memory dir to test auto-creation
        mem = Path(temp_workspace) / "memory"
        if mem.exists():
            mem.rmdir()
        w = MemoryFileWatcher(temp_workspace)
        # Without watchdog, start won't create dir
        # But we can test the dir creation logic


class TestMemoryFileHandler:
    """Tests for the _MemoryFileHandler debounce logic."""

    def test_handler_ignores_non_md_files(self):
        called = []
        handler = _make_handler(on_changed=lambda p: called.append(p))
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/test.txt"
        handler._handle(event)
        assert len(called) == 0

    def test_handler_ignores_directories(self):
        called = []
        handler = _make_handler(on_changed=lambda p: called.append(p))
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/tmp/test.md"
        handler._handle(event)
        assert len(called) == 0

    def test_handler_calls_callback_for_md(self):
        called = []
        handler = _make_handler(on_changed=lambda p: called.append(p))
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/test.md"
        handler._handle(event)
        assert called == ["/tmp/test.md"]

    def test_handler_debounces_same_file(self):
        """
        BUG CONFIRMED: After first call, path is in _recent. But cooldown
        cleanup requires asyncio.get_running_loop() which always fails
        from watchdog's daemon thread. So _recent is never cleaned.
        """
        called = []
        handler = _make_handler(on_changed=lambda p: called.append(p))

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/test.md"

        handler._handle(event)  # first call — fires
        handler._handle(event)  # second call — debounced (in _recent)
        assert len(called) == 1, (
            f"Debounce failed: called {len(called)} times (expected 1)"
        )

    def test_different_files_both_fire(self):
        called = []
        handler = _make_handler(on_changed=lambda p: called.append(p))

        e1 = MagicMock()
        e1.is_directory = False
        e1.src_path = "/tmp/a.md"
        e2 = MagicMock()
        e2.is_directory = False
        e2.src_path = "/tmp/b.md"

        handler._handle(e1)
        handler._handle(e2)
        assert len(called) == 2

    def test_recent_cleaned_via_call_soon_threadsafe(self):
        """
        FIX VERIFIED: _handle now uses loop.call_soon_threadsafe to schedule
        cooldown cleanup from any thread. This works where get_running_loop() failed.
        """
        handler = _make_handler(on_changed=lambda p: None, cooldown=0.01)

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/once.md"

        handler._handle(event)
        assert "/tmp/once.md" in handler._recent

        # call_soon_threadsafe was called to schedule cleanup
        handler._loop.call_soon_threadsafe.assert_called_once()


class TestScheduleAsync:
    """Tests for schedule_async thread-safe coroutine scheduling."""

    def test_no_loop_returns_none(self):
        w = MemoryFileWatcher("/tmp/test_ws")
        w._loop = None

        async def dummy():
            pass

        # Should not raise
        w.schedule_async(dummy())

    @pytest.mark.asyncio
    async def test_with_loop_schedules_correctly(self):
        """schedule_async in the main event loop schedules the coroutine."""
        called = False

        async def dummy():
            nonlocal called
            called = True

        w = MemoryFileWatcher("/tmp/test_ws")
        w._loop = asyncio.get_running_loop()
        w.schedule_async(dummy())
        await asyncio.sleep(0.05)
        assert called


class TestFileWatcherExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    def test_very_long_path(self):
        handler = _make_handler(on_changed=lambda p: None)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/" + "x" * 1000 + ".md"
        handler._handle(event)  # must not crash

    def test_handler_null_event_path(self):
        """None or empty path must not crash."""
        handler = _make_handler(on_changed=lambda p: None)
        event = MagicMock()
        event.is_directory = False
        event.src_path = ""
        handler._handle(event)  # must not crash, just skip

    def test_rapid_fire_events(self):
        """100 events in rapid succession — all same file → debounced."""
        called = []
        handler = _make_handler(on_changed=lambda p: called.append(p))
        for _ in range(100):
            event = MagicMock()
            event.is_directory = False
            event.src_path = "/tmp/rapid.md"
            handler._handle(event)
        assert len(called) == 1  # all debounced to one call

    def test_1000_different_files_no_oom(self):
        """Many different files should all fire, not OOM."""
        called = []
        handler = _make_handler(on_changed=lambda p: called.append(p))
        for i in range(1000):
            event = MagicMock()
            event.is_directory = False
            event.src_path = f"/tmp/file_{i}.md"
            handler._handle(event)
        # All 1000 should fire (different paths)
        assert len(called) == 1000
        # _recent set should have 1000 entries
        assert len(handler._recent) == 1000
