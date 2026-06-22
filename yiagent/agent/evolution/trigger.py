"""
Idle-based evolution trigger — background asyncio task.

Scans live agent sessions every scan_interval seconds. When a session
has been idle >= idle_seconds AND has enough signal (min_turns OR
context pressure), launches an isolated evolution pass.

"Enough signal" = EITHER >= min_turns since last evo, OR live context
exceeds 80% of the agent's token budget.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from yiagent.common.log import logger
from yiagent.agent.evolution.config import get_evolution_config
from yiagent.agent.evolution.executor import run_evolution_for_session

_CONTEXT_RATIO = 0.8
_FALLBACK_BUDGET = 50000
_SCAN_INTERVAL = 60

_trigger_task: asyncio.Task = None


def note_user_turn(agent: Any, channel_type: str = "", receiver: str = "") -> None:
    """Record activity for a session's agent. Called once per user turn."""
    try:
        agent._evo_last_active = time.time()
        agent._evo_turns = int(getattr(agent, "_evo_turns", 0)) + 1
        if channel_type:
            agent._evo_channel_type = channel_type
        if receiver:
            agent._evo_receiver = receiver
    except Exception:
        pass


def mark_run_active(agent: Any, active: bool) -> None:
    """Flag agent as mid-run so idle scans skip it."""
    try:
        agent._evo_run_active = bool(active)
        if active:
            agent._evo_last_active = time.time()
    except Exception:
        pass


async def start_evolution_trigger(
    agent_bridge: Any,
    interval: int = _SCAN_INTERVAL,
) -> None:
    """Launch the idle-scan background task. Idempotent."""
    global _trigger_task
    if _trigger_task is not None and not _trigger_task.done():
        return

    _trigger_task = asyncio.create_task(
        _scan_loop(agent_bridge, interval),
        name="evolution-trigger",
    )
    logger.info("[Evolution] Idle trigger started")


async def stop_evolution_trigger() -> None:
    global _trigger_task
    if _trigger_task and not _trigger_task.done():
        _trigger_task.cancel()
        try:
            await _trigger_task
        except asyncio.CancelledError:
            pass
        _trigger_task = None
        logger.info("[Evolution] Idle trigger stopped")


async def _scan_loop(agent_bridge: Any, interval: int) -> None:
    while True:
        try:
            await asyncio.sleep(interval)
            cfg = get_evolution_config()
            if not cfg.enabled:
                continue
            await _scan_once(agent_bridge, cfg)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[Evolution] Scan loop error: {e}")


async def _scan_once(agent_bridge: Any, cfg) -> None:
    now = time.time()
    sessions: Dict[str, Any] = dict(getattr(agent_bridge, "agents", {}))
    for session_id, agent in sessions.items():
        try:
            if getattr(agent, "_evo_run_active", False):
                continue

            last_active = getattr(agent, "_evo_last_active", 0)
            turns = int(getattr(agent, "_evo_turns", 0))
            enough_signal = turns >= cfg.min_turns or _context_pressure(agent)
            if not enough_signal:
                continue

            idle = now - last_active if last_active > 0 else -1
            if last_active <= 0 or idle < cfg.idle_seconds:
                continue

            ch = getattr(agent, "_evo_channel_type", "") or ""
            rcv = getattr(agent, "_evo_receiver", "") or ""
            agent._evo_turns = 0

            await run_evolution_for_session(
                agent_bridge=agent_bridge,
                session_id=session_id,
                channel_type=ch,
                receiver=rcv,
                idle_minutes=(now - last_active) / 60 if last_active > 0 else 0,
            )
        except Exception as e:
            logger.warning(f"[Evolution] Scan failed for {session_id}: {e}")


def _context_pressure(agent: Any) -> bool:
    """True if live context exceeds 80% of token budget."""
    try:
        msgs = list(getattr(agent, "messages", []))
        if not msgs:
            return False
        est = sum(agent._estimate_message_tokens(m) for m in msgs)
        budget = getattr(agent, "max_context_tokens", None) or _FALLBACK_BUDGET
        return est / budget > _CONTEXT_RATIO
    except Exception:
        return False
