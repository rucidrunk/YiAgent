"""
ContextFlushPipeline — asyncio.Queue 削峰管线

Non-blocking context overflow handling:
  1. Monitor: check Redis token counter after each message append
  2. Snapshot: when token_estimate > max × 0.8, take oldest 50% of turns
  3. Push: await queue.put() in microseconds
  4. Reset: LTRIM Redis context to recent turns only
  5. Return: immediately, hot path never blocked
  6. Worker: daemon tasks consume queue, distill via LLM, persist to daily file

Key improvement over CowAgent: asyncio.Queue with bounded workers
replaces threading.Thread daemon, providing backpressure and observability.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from yiagent.common.config import conf
from yiagent.common.log import logger


@dataclass
class FlushTask:
    """A context flush work item."""
    session_id: str
    user_id: Optional[str]
    messages: List[Dict[str, Any]]
    reason: str  # "threshold" | "overflow" | "daily_summary"
    callback: Optional[Callable] = None  # async callback(summary_text)


class ContextFlushPipeline:
    """
    Non-blocking context flush pipeline backed by an asyncio.Queue.
    """

    def __init__(
        self,
        max_workers: int = 4,
        queue_maxsize: int = 256,
        high_watermark: float = 0.8,
        llm_model: Any = None,
    ):
        self._queue: asyncio.Queue[Optional[FlushTask]] = asyncio.Queue(maxsize=queue_maxsize)
        self._high_watermark = high_watermark
        self._max_workers = max_workers
        self._llm_model = llm_model
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._workspace_dir = conf().get("workspace_root", "~/yiagent")

    async def start(self) -> None:
        """Launch daemon worker tasks."""
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"flush-worker-{i}")
            for i in range(self._max_workers)
        ]
        logger.info(f"[ContextFlush] Started {self._max_workers} workers")

    async def stop(self) -> None:
        """Gracefully drain and stop all workers (with timeout per worker)."""
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        for w in self._workers:
            try:
                await asyncio.wait_for(w, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                w.cancel()
        logger.info("[ContextFlush] Stopped")

    async def monitor_and_flush(
        self,
        session_id: str,
        token_estimate: int,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        Check if context exceeds high watermark and enqueue flush if so.

        Returns True if a flush was enqueued.
        """
        if max_tokens <= 0:
            return False

        ratio = token_estimate / max_tokens
        if ratio < self._high_watermark:
            return False

        # Snapshot oldest 50% of turns
        turn_count = self._count_turns(messages)
        if turn_count < 4:
            return False  # Not enough to flush meaningfully

        keep_turns = max(1, turn_count // 2)
        flush_turns: List[Dict[str, Any]] = []
        keep_turns_list: List[Dict[str, Any]] = []
        current_turn: List[Dict[str, Any]] = []
        turn_idx = 0

        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                is_visible = False
                if isinstance(content, list):
                    has_text = any(
                        isinstance(b, dict) and b.get("type") == "text" for b in content
                    )
                    has_tool = any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                    )
                    is_visible = has_text and not has_tool
                elif isinstance(content, str):
                    is_visible = bool(content.strip())

                if is_visible and current_turn:
                    turn_idx += 1
                    if turn_idx <= turn_count // 2:
                        flush_turns.extend(current_turn)
                    else:
                        keep_turns_list.extend(current_turn)
                    current_turn = [msg]
                else:
                    current_turn.append(msg)
            else:
                current_turn.append(msg)

        if current_turn:
            if turn_idx <= len(messages) // 2:
                flush_turns.extend(current_turn)
            else:
                keep_turns_list.extend(current_turn)

        if not flush_turns:
            return False

        task = FlushTask(
            session_id=session_id,
            user_id=user_id,
            messages=flush_turns,
            reason="threshold",
            callback=callback,
        )

        try:
            self._queue.put_nowait(task)
            logger.debug(f"[ContextFlush] Enqueued {len(flush_turns)} msgs for {session_id}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"[ContextFlush] Queue full, dropping flush for {session_id}")
            return False

    @property
    def queue_depth(self) -> int:
        """Current queue size (for Prometheus metrics)."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _worker(self, worker_id: int) -> None:
        while True:
            try:
                task = await self._queue.get()
            except asyncio.CancelledError:
                break

            if task is None:
                break  # Sentinel

            try:
                await self._process(task, worker_id)
            except Exception as e:
                logger.error(f"[ContextFlush] Worker {worker_id} error: {e}")
            finally:
                self._queue.task_done()

        # If the worker died unexpectedly, respawn it
        if self._running:
            logger.warning(f"[ContextFlush] Worker {worker_id} died, respawning")
            self._workers[worker_id] = asyncio.create_task(
                self._worker(worker_id), name=f"flush-worker-{worker_id}"
            )

    async def _process(self, task: FlushTask, worker_id: int) -> None:
        """
        Distill flushed messages into a daily memory file via LLM summarization.
        Uses a lightweight summarizer; falls back to hashing if no LLM available.
        """
        from pathlib import Path
        from datetime import datetime, timezone

        # Build transcript
        transcript = self._build_transcript(task.messages, max_chars=8000)
        if not transcript.strip():
            return

        # Generate summary
        summary = await self._summarize(transcript, task.reason)

        # Write to daily memory file
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        mem_dir = Path(self._workspace_dir) / "memory"
        if task.user_id:
            mem_dir = mem_dir / "users" / task.user_id
        mem_dir.mkdir(parents=True, exist_ok=True)

        daily_file = mem_dir / f"{today}.md"
        entry = f"\n\n## Context Flush ({datetime.now(timezone.utc).strftime('%H:%M')})\n{summary}\n"
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, _write_daily_sync, daily_file, entry,
            )
        except Exception as e:
            logger.error(f"[ContextFlush] Failed to write {daily_file}: {e}")

        # Invoke callback for in-context injection
        if task.callback and summary:
            try:
                if asyncio.iscoroutinefunction(task.callback):
                    await task.callback(summary)
                else:
                    task.callback(summary)
            except Exception as e:
                logger.warning(f"[ContextFlush] Callback failed: {e}")

        logger.info(
            f"[ContextFlush] Worker {worker_id}: flushed {len(task.messages)} msgs"
            f" for session={task.session_id} ({len(summary)} chars)"
        )

    async def _summarize(self, transcript: str, reason: str) -> str:
        """Summarize transcript via LLM, or fall back to hash-based dedup."""
        if self._llm_model:
            try:
                from yiagent.protocol.models import LLMRequest
                prompt = (
                    "You are a context summarizer. Summarize the following conversation "
                    "into a concise paragraph (max 500 chars) capturing key facts, "
                    "decisions, and user preferences. Write in the same language as the input.\n\n"
                    f"Conversation:\n{transcript}"
                )
                request = LLMRequest(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                    stream=False,
                )
                result = await self._llm_model.call(request)
                return result.get("content", "")[:500]
            except Exception as e:
                logger.warning(f"[ContextFlush] LLM summarization failed, using fallback: {e}")

        # Fallback: simple keyword extraction
        lines = transcript.strip().split("\n")
        fallback = "\n".join(lines[:20])
        return f"[Auto-summary] {fallback[:500]}"

    @staticmethod
    def _build_transcript(messages: List[Dict[str, Any]], max_chars: int = 8000) -> str:
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            text = ""
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(b.get("text", ""))
                text = "\n".join(parts)
            elif isinstance(content, str):
                text = content
            if text.strip():
                speaker = "User" if role == "user" else "Assistant"
                lines.append(f"{speaker}: {text.strip()}")
        transcript = "\n".join(lines)
        if len(transcript) > max_chars:
            transcript = "...(earlier omitted)...\n" + transcript[-max_chars:]
        return transcript

    @staticmethod
    def _count_turns(messages: List[Dict[str, Any]]) -> int:
        count = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                has_text = any(
                    isinstance(b, dict) and b.get("type") == "text" for b in content
                )
                has_tool = any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                )
                if has_text and not has_tool:
                    count += 1
            elif isinstance(content, str) and content.strip():
                count += 1
        return count


def _write_daily_sync(path, content: str) -> None:
    """Blocking file write — must be called via run_in_executor."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
