"""
Self-evolution executor — runs an isolated review agent on an idle session.

Flow:
  1. Build transcript from session messages
  2. Snapshot MEMORY.md + daily file + skills (for undo) → backup_id
  3. Run isolated agent (restricted tools, evolution prompt)
  4. [SILENT] → no changes → return
  5. Otherwise → log, inject [EVOLUTION] note, notify user

Conservative by design: most runs return [SILENT].
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

from yiagent.common.log import logger
from yiagent.agent.evolution.config import get_evolution_config
from yiagent.agent.evolution.backup import create_backup
from yiagent.agent.evolution.prompts import (
    EVOLUTION_MARKER,
    EVOLUTION_SYSTEM_PROMPT,
    SILENT_TOKEN,
    build_review_user_message,
)
from yiagent.agent.evolution.record import append_session_evolution

_ALLOWED_TOOLS: Set[str] = {"read", "write", "edit", "ls", "bash", "memory_search", "memory_get"}
_WATCH_SUBDIRS = ("MEMORY.md", "AGENT.md", "skills", "output")
_MAX_CONCURRENT = 2

_running_count = 0
_running_lock = asyncio.Lock()


async def run_evolution_for_session(
    agent_bridge,
    session_id: str,
    channel_type: str = "",
    receiver: str = "",
    user_id: Optional[str] = None,
    idle_minutes: float = 0.0,
) -> bool:
    """Run one evolution pass. Returns True if changes were made."""
    cfg = get_evolution_config()
    if not cfg.enabled:
        return False

    global _running_count
    async with _running_lock:
        if _running_count >= _MAX_CONCURRENT:
            logger.info(f"[Evolution] Busy ({_running_count}/{_MAX_CONCURRENT}), skipping {session_id}")
            return False
        _running_count += 1

    try:
        agent = agent_bridge.agents.get(session_id)
        if not agent:
            return False

        # Build transcript from new messages since last pass
        all_msgs = list(getattr(agent, "messages", []))
        total = len(all_msgs)
        done = int(getattr(agent, "_evo_done_msg_count", 0))
        if done > total:
            done = 0
        new_msgs = all_msgs[done:]
        transcript = _build_transcript(new_msgs)
        if not transcript.strip():
            agent._evo_done_msg_count = total
            return False

        logger.info(
            f"[Evolution] Reviewing {session_id} "
            f"(idle={idle_minutes:.1f}m, {len(new_msgs)}/{total} msgs)"
        )

        # Resolve workspace and backup targets
        from yiagent.memory.config import get_memory_config
        mem_cfg = get_memory_config()
        ws = mem_cfg.get_workspace()

        mem_file = ws / "MEMORY.md"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if user_id:
            daily_file = ws / "memory" / "users" / user_id / f"{today}.md"
        else:
            daily_file = ws / "memory" / f"{today}.md"
        agent_file = ws / "AGENT.md"

        backup_files = [p for p in (mem_file, daily_file, agent_file) if p.exists()]
        backup_id = create_backup(ws, backup_files)

        # Snapshot workspace before the run
        pre_snapshot = _workspace_snapshot(ws)

        # Build isolated review agent
        all_tools = list(getattr(agent, "tools", []) or [])
        review_tools = [t for t in all_tools if getattr(t, "name", None) in _ALLOWED_TOOLS]

        review_agent = agent_bridge.create_agent(
            system_prompt="",
            tools=review_tools,
            description="Self-evolution review agent",
            max_steps=cfg.max_steps,
            workspace_dir=str(ws),
            skill_manager=getattr(agent, "skill_manager", None),
            memory_manager=getattr(agent, "memory_manager", None),
            enable_skills=True,
            runtime_info=getattr(agent, "runtime_info", None),
        )
        review_agent.model = agent.model
        review_agent.extra_system_suffix = EVOLUTION_SYSTEM_PROMPT

        user_msg = build_review_user_message(transcript)
        try:
            result = review_agent.run_stream(user_msg, clear_history=True)
        except Exception as e:
            logger.warning(f"[Evolution] Review agent failed for {session_id}: {e}")
            return False
        result = (result or "").strip()

        agent._evo_done_msg_count = total

        if not result or result.startswith(SILENT_TOKEN):
            logger.info(f"[Evolution] No change for {session_id} ([SILENT])")
            return False

        # Verify actual file changes
        if not _workspace_changed(ws, pre_snapshot):
            logger.info(f"[Evolution] Text produced but no files changed — staying silent")
            return False

        result = result.replace(SILENT_TOKEN, "").strip()
        if not result:
            return False

        logger.info(f"[Evolution] Session {session_id} evolved:\n{result}")
        append_session_evolution(ws, result, backup_id=backup_id, user_id=user_id)

        # Advance cursor past injected note
        try:
            agent._evo_done_msg_count = len(getattr(agent, "messages", []))
        except Exception:
            pass

        return True

    except Exception as e:
        logger.warning(f"[Evolution] Run failed for {session_id}: {e}")
        return False
    finally:
        async with _running_lock:
            _running_count -= 1


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_transcript(messages: List[dict], max_chars: int = 12000) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        text = _extract_text(content)
        if not text.strip():
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {text.strip()}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = "...(earlier omitted)...\n" + transcript[-max_chars:]
    return transcript


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _workspace_snapshot(ws: Path) -> dict:
    """Map relative path → (mtime, size) for watched files."""
    snap: dict = {}
    for name in _WATCH_SUBDIRS:
        root = ws / name
        if root.is_file():
            try:
                st = root.stat()
                snap[name] = (st.st_mtime, st.st_size)
            except OSError:
                pass
            continue
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.name == "skills_config.json":
                continue
            try:
                st = p.stat()
                snap[str(p.relative_to(ws))] = (st.st_mtime, st.st_size)
            except OSError:
                pass
    return snap


def _workspace_changed(ws: Path, pre: dict) -> bool:
    return _workspace_snapshot(ws) != pre
