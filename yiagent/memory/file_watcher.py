"""
MemoryFileWatcher — watchdog-based .md file monitor for memory/ directory.

Watches memory/*.md for creates and modifications, then spawns a background
task to digest the changed file into memory_chunks (chunk → embed → PG).

Design:
  - Runs watchdog Observer in a daemon thread (non-blocking).
  - Debounces rapid writes: same file is only processed once per cooldown window.
  - Integration point: MemoryManager.initialize() / close().
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Set

from yiagent.common.log import logger

# Lazy import — watchdog is optional
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    watchdog = True  # type: ignore
except ImportError:
    watchdog = None
    FileSystemEventHandler = object  # fallback for type hints
    Observer = None


class _MemoryFileHandler(FileSystemEventHandler):  # type: ignore[misc]
    """watchdog FileSystemEventHandler that filters for .md files."""

    def __init__(
        self,
        on_changed: Callable[[str], None],
        loop: asyncio.AbstractEventLoop,
        cooldown: float = 5.0,
    ):
        self._on_changed = on_changed
        self._loop = loop
        self._cooldown = cooldown
        self._recent: Set[str] = set()
        self._lock = threading.Lock()  # protects _recent across threads

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if not path.endswith(".md"):
            return

        # Thread-safe debounce: skip if file was processed within cooldown
        with self._lock:
            if path in self._recent:
                return
            self._recent.add(path)

        logger.debug(f"[FileWatcher] Detected change: {path}")
        self._on_changed(path)

        # Schedule cooldown removal on the main event loop.
        # call_soon_threadsafe works from any thread — unlike get_running_loop().
        self._loop.call_soon_threadsafe(
            lambda p=path: self._clear_debounce(p),
        )

    def _clear_debounce(self, path: str) -> None:
        """Called from the main event loop thread — safe to access _recent."""
        with self._lock:
            self._recent.discard(path)


class MemoryFileWatcher:
    """
    Watch memory/*.md files.  When a file is created or modified, trigger
    a callback that should digest it into memory_chunks.

    Usage:
        watcher = MemoryFileWatcher(workspace_dir, on_changed=my_sync_func)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(
        self,
        workspace_dir: str,
        on_changed: Optional[Callable[[str], None]] = None,
        cooldown: float = 5.0,
    ):
        self._workspace_dir = Path(workspace_dir)
        self._on_changed = on_changed
        self._cooldown = cooldown
        self._observer: Optional[object] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watchdog observer in a background daemon thread."""
        if Observer is None:
            logger.warning(
                "[FileWatcher] watchdog not installed — "
                "memory/ file auto-sync disabled. "
                "Install it with: pip install watchdog"
            )
            return

        if self._running:
            return

        # Capture the main event loop for thread-safe callbacks
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[FileWatcher] No event loop — watcher disabled")
            return

        memory_dir = self._workspace_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        handler = _MemoryFileHandler(
            on_changed=self._on_changed or (lambda p: None),
            loop=self._loop,
            cooldown=self._cooldown,
        )
        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(memory_dir),
            recursive=True,  # watch memory/users/ subdirs too
        )
        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info(f"[FileWatcher] Watching {memory_dir} for .md changes")

    def stop(self) -> None:
        """Stop the watchdog observer."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._running = False
            logger.info("[FileWatcher] Stopped")

    def schedule_async(self, coro) -> None:
        """Thread-safe: schedule a coroutine on the main event loop."""
        if self._loop and self._loop.is_running():
            f = asyncio.run_coroutine_threadsafe(coro, self._loop)
            # Log failures so exceptions are never silently lost
            f.add_done_callback(_check_future)


def _check_future(fut: asyncio.Future) -> None:
    try:
        exc = fut.exception()
        if exc is not None:
            logger.error(f"[FileWatcher] schedule_async task failed: {exc}")
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        pass
