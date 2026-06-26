"""
Memory tools — search, read, and write to the long-term memory store.

These tools give the agent direct access to the MemoryManager so it can:
  - memory_search: hybrid vector + keyword search over memory_chunks
  - memory_get:    read exact content of a memory file by path
  - memory_add:    proactively save new facts / decisions / preferences
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from yiagent.common.log import logger
from yiagent.agent.tools.base_tool import BaseTool, ToolResult


# ---------------------------------------------------------------------------
# Cached MemoryManager singleton (BUG-8 fix)
# ---------------------------------------------------------------------------

_memory_manager = None
_memory_manager_lock = asyncio.Lock()


async def _get_memory_manager():
    """Return the cached MemoryManager, creating it lazily once."""
    global _memory_manager
    if _memory_manager is not None:
        return _memory_manager
    async with _memory_manager_lock:
        if _memory_manager is not None:
            return _memory_manager
        from yiagent.memory.manager import MemoryManager
        mgr = MemoryManager()
        await mgr.initialize()
        _memory_manager = mgr
        return mgr


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = (
        "Search long-term memory for relevant facts, preferences, and past decisions. "
        "Use this before answering questions about the user's history, preferences, "
        "or past interactions. Returns the top matching memory snippets."
    )
    params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query in natural language. Be specific (e.g. 'user preference Python version' not just 'preference').",
            },
            "user_id": {
                "type": "string",
                "description": "Optional user ID to restrict search to a specific user.",
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        user_id = params.get("user_id") or None

        if not query.strip():
            return ToolResult.fail("query is required")

        try:
            mgr = await _get_memory_manager()
            if mgr is None:
                return ToolResult.fail("Memory manager not available (PostgreSQL required)")

            results = await mgr.search(query=query, user_id=user_id)
            if not results:
                return ToolResult.success("No matching memories found.")

            formatted = []
            for r in results:
                formatted.append({
                    "path": r.path,
                    "score": round(r.score, 3),
                    "snippet": r.snippet,
                })
            return ToolResult.success(formatted)
        except Exception as e:
            logger.error(f"[memory_search] Failed: {e}")
            return ToolResult.fail(str(e))


class MemoryGetTool(BaseTool):
    name = "memory_get"
    description = (
        "Read the exact content of a memory file. Use this when you already know "
        "which file to read (e.g. from a previous memory_search result, or when "
        "checking today's daily memory file)."
    )
    params = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the memory file relative to workspace, e.g. 'MEMORY.md', 'memory/2026-06-26.md', or a path returned by memory_search.",
            },
            "user_id": {
                "type": "string",
                "description": "Optional user ID when reading user-scoped memory files.",
            },
        },
        "required": ["path"],
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", "")
        user_id = params.get("user_id") or None

        if not path.strip():
            return ToolResult.fail("path is required")

        # BUG-6 fix: reject null bytes before they reach os.path.realpath()
        if "\x00" in path:
            return ToolResult.fail("Invalid path: null byte detected")

        from yiagent.common.config import conf
        ws = os.path.expanduser(conf().get("workspace_root", "~/yiagent"))
        full_path = os.path.join(ws, path)

        # Security: prevent path traversal (catch ValueError from null bytes)
        try:
            real = os.path.realpath(full_path)
        except (ValueError, OSError) as e:
            return ToolResult.fail(f"Invalid path: {e}")

        real_ws = os.path.realpath(ws)
        if not real.startswith(real_ws):
            return ToolResult.fail(f"Access denied: {path} is outside workspace")

        # BUG-7 fix: non-blocking file I/O via run_in_executor
        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, _read_file_sync, real)
            if len(content) > 20000:
                content = content[:20000] + "\n...(truncated)..."
            return ToolResult.success(content)
        except FileNotFoundError:
            return ToolResult.fail(f"File not found: {path}")
        except Exception as e:
            logger.error(f"[memory_get] Failed: {e}")
            return ToolResult.fail(str(e))


class MemoryAddTool(BaseTool):
    name = "memory_add"
    description = (
        "Save a new fact, decision, preference, or important information to long-term "
        "memory so it can be recalled later via memory_search. "
        "Use this proactively when the user says things like 'remember', 'always', "
        "'never', 'prefer', 'don't like', 'my name is', or shares important context "
        "that should persist across conversations.\n"
        "Do NOT use for transient information (current weather, temporary state)."
    )
    params = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or information to remember. Write in clear, searchable language (e.g. 'User prefers Python 3.11+ for all projects' not 'ok got it').",
            },
            "user_id": {
                "type": "string",
                "description": "The user this memory belongs to. Required for user-scoped memories.",
            },
            "scope": {
                "type": "string",
                "enum": ["shared", "user"],
                "description": "Memory scope: 'user' for personal preferences, 'shared' for project-wide knowledge. Default is 'user' when user_id is provided.",
            },
        },
        "required": ["content"],
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        content = params.get("content", "")
        user_id = params.get("user_id") or None
        scope = params.get("scope", "user" if user_id else "shared")

        if not content.strip():
            return ToolResult.fail("content is required")

        try:
            mgr = await _get_memory_manager()
            if mgr is None:
                return ToolResult.fail("Memory manager not available (PostgreSQL + embedding required)")

            await mgr.initialize()
            await mgr.add_memory(
                content=content,
                user_id=user_id,
                scope=scope,
            )

            logger.info(f"[memory_add] Saved: scope={scope} user_id={user_id} text={content[:80]}")
            return ToolResult.success({
                "status": "saved",
                "scope": scope,
                "user_id": user_id,
                "hint": "Memory has been saved and embedded. It will be searchable via memory_search.",
            })
        except Exception as e:
            logger.error(f"[memory_add] Failed: {e}")
            return ToolResult.fail(str(e))


def _read_file_sync(path: str) -> str:
    """Blocking file read — meant for run_in_executor."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
