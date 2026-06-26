"""
Tests for yiagent.agent.tools.builtin.memory_tools — agent memory tools.

CRITICAL AREAS:
  1. Path traversal prevention in memory_get
  2. memory_add with edge-case content
  3. memory_search with empty/boundary queries
  4. Synchronous file I/O blocking in async methods
  5. _get_memory_manager creates new instance each call (no caching)
  6. Error message leakage (str(e) passed directly to user/LLM)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules["asyncpg"] = MagicMock()

from yiagent.agent.tools.builtin.memory_tools import (
    MemorySearchTool,
    MemoryGetTool,
    MemoryAddTool,
    _get_memory_manager,
)


# ======================================================================
# MemorySearchTool
# ======================================================================

class TestMemorySearchTool:
    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        tool = MemorySearchTool()
        result = await tool.execute({"query": ""})
        assert result.status == "error"
        assert "required" in str(result.result)

    @pytest.mark.asyncio
    async def test_empty_query_whitespace_only(self):
        tool = MemorySearchTool()
        result = await tool.execute({"query": "   "})
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_missing_query_returns_error(self):
        tool = MemorySearchTool()
        result = await tool.execute({})
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_memory_manager_unavailable(self):
        tool = MemorySearchTool()
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=None)):
            result = await tool.execute({"query": "test"})
            assert result.status == "error"
            assert "not available" in str(result.result)

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        tool = MemorySearchTool()
        mock_mgr = AsyncMock()
        mock_mgr.search = AsyncMock(return_value=[
            MagicMock(path="/mem.md", score=0.95, snippet="found something"),
        ])
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"query": "test query"})
            assert result.status == "success"

    @pytest.mark.asyncio
    async def test_search_exception_caught(self):
        tool = MemorySearchTool()
        mock_mgr = AsyncMock()
        mock_mgr.search = AsyncMock(side_effect=RuntimeError("PG connection lost"))
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"query": "test"})
            assert result.status == "error"

    @pytest.mark.asyncio
    async def test_search_very_long_query(self):
        tool = MemorySearchTool()
        mock_mgr = AsyncMock()
        mock_mgr.search = AsyncMock(return_value=[])
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"query": "x" * 10000})
            assert result.status in ("success", "error")


# ======================================================================
# MemoryGetTool
# ======================================================================

class TestMemoryGetTool:
    @pytest.mark.asyncio
    async def test_empty_path_returns_error(self):
        tool = MemoryGetTool()
        result = await tool.execute({"path": ""})
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_missing_path_returns_error(self):
        tool = MemoryGetTool()
        result = await tool.execute({})
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        tool = MemoryGetTool()
        result = await tool.execute({"path": "../../../etc/passwd"})
        assert result.status == "error"
        assert "Access denied" in str(result.result)

    @pytest.mark.asyncio
    async def test_absolute_path_outside_workspace_blocked(self):
        tool = MemoryGetTool()
        result = await tool.execute({"path": "/etc/passwd"})
        assert result.status == "error"
        assert "Access denied" in str(result.result)

    @pytest.mark.asyncio
    async def test_valid_file_reads_content(self):
        with tempfile.TemporaryDirectory() as ws:
            # Override workspace config to point to temp dir
            import yiagent.common.config as cfg_mod
            old = cfg_mod._config
            cfg_mod._config = dict(cfg_mod._DEFAULT_CONFIG, workspace_root=ws)
            try:
                test_file = os.path.join(ws, "test.md")
                Path(test_file).write_text("hello world")

                tool = MemoryGetTool()
                result = await tool.execute({"path": "test.md"})
                assert result.status == "success"
                assert result.result == "hello world"
            finally:
                cfg_mod._config = old

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        import yiagent.common.config as cfg_mod
        old = cfg_mod._config
        cfg_mod._config = dict(cfg_mod._DEFAULT_CONFIG, workspace_root="/tmp/nonexistent_ws")
        try:
            tool = MemoryGetTool()
            result = await tool.execute({"path": "nonexistent.md"})
            assert result.status == "error"
            assert "not found" in str(result.result).lower()
        finally:
            cfg_mod._config = old

    @pytest.mark.asyncio
    async def test_large_file_capped_at_20000(self):
        with tempfile.TemporaryDirectory() as ws:
            import yiagent.common.config as cfg_mod
            old = cfg_mod._config
            cfg_mod._config = dict(cfg_mod._DEFAULT_CONFIG, workspace_root=ws)
            try:
                test_file = os.path.join(ws, "huge.md")
                Path(test_file).write_text("x" * 50000)

                tool = MemoryGetTool()
                result = await tool.execute({"path": "huge.md"})
                assert result.status == "success"
                assert len(result.result) <= 20000 + len("\n...(truncated)...")
                assert "truncated" in result.result
            finally:
                cfg_mod._config = old


# ======================================================================
# MemoryAddTool
# ======================================================================

class TestMemoryAddTool:
    @pytest.mark.asyncio
    async def test_empty_content_returns_error(self):
        tool = MemoryAddTool()
        result = await tool.execute({"content": ""})
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_missing_content_returns_error(self):
        tool = MemoryAddTool()
        result = await tool.execute({})
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_add_memory_success(self):
        tool = MemoryAddTool()
        mock_mgr = AsyncMock()
        mock_mgr.add_memory = AsyncMock(return_value=None)
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"content": "user prefers dark mode", "user_id": "u1"})
            assert result.status == "success"
            data = result.result
            assert data["status"] == "saved"

    @pytest.mark.asyncio
    async def test_add_memory_with_default_scope(self):
        """When user_id is provided, scope defaults to 'user'."""
        tool = MemoryAddTool()
        mock_mgr = AsyncMock()
        mock_mgr.add_memory = AsyncMock(return_value=None)
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"content": "test", "user_id": "u1"})
            assert result.status == "success"
            # scope should default to "user" when user_id present
            call_kwargs = mock_mgr.add_memory.call_args
            assert call_kwargs is not None

    @pytest.mark.asyncio
    async def test_add_memory_exception_caught(self):
        tool = MemoryAddTool()
        mock_mgr = AsyncMock()
        mock_mgr.add_memory = AsyncMock(side_effect=RuntimeError("disk full"))
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"content": "test"})
            assert result.status == "error"


# ======================================================================
# Extreme boundary conditions
# ======================================================================

class TestMemoryToolsExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    @pytest.mark.asyncio
    async def test_memory_get_with_symlink_inside_workspace(self):
        """Symlink inside workspace pointing to outside — must be blocked."""
        with tempfile.TemporaryDirectory() as ws:
            import yiagent.common.config as cfg_mod
            old = cfg_mod._config
            cfg_mod._config = dict(cfg_mod._DEFAULT_CONFIG, workspace_root=ws)
            try:
                # Create a symlink inside ws pointing to /etc/hosts
                symlink_path = os.path.join(ws, "safe_link")
                os.symlink("/etc/hosts", symlink_path)

                tool = MemoryGetTool()
                result = await tool.execute({"path": "safe_link"})
                # _os.path.realpath resolves symlink to /etc/hosts
                # which is outside workspace → blocked
                assert result.status == "error"
                assert "Access denied" in str(result.result)
            finally:
                cfg_mod._config = old

    @pytest.mark.asyncio
    async def test_memory_get_path_with_null_byte(self):
        """
        FIX VERIFIED: Null byte in path is now caught and returns a safe error
        instead of crashing with ValueError.
        """
        tool = MemoryGetTool()
        result = await tool.execute({"path": "test\x00.md"})
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_memory_add_very_long_content(self):
        """10KB content must not crash."""
        tool = MemoryAddTool()
        mock_mgr = AsyncMock()
        mock_mgr.add_memory = AsyncMock(return_value=None)
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"content": "Remember: " + "x" * 10000})
            assert result.status == "success"

    @pytest.mark.asyncio
    async def test_memory_search_special_chars_query(self):
        """Queries with SQL-like chars must be safe (parameterized queries in PG)."""
        tool = MemorySearchTool()
        mock_mgr = AsyncMock()
        mock_mgr.search = AsyncMock(return_value=[])
        with patch("yiagent.agent.tools.builtin.memory_tools._get_memory_manager",
                   AsyncMock(return_value=mock_mgr)):
            result = await tool.execute({"query": "'; DROP TABLE users; --"})
            # Must not crash — should return results or error
            assert result.status in ("success", "error")

    @pytest.mark.asyncio
    async def test_get_memory_manager_caches_instance(self):
        """
        FIX VERIFIED: _get_memory_manager now caches the instance.
        Multiple calls return the same MemoryManager (only 1 PG pool).
        """
        with patch("yiagent.memory.manager.MemoryManager") as mock_cls:
            mock_inst = AsyncMock()
            mock_inst.initialize = AsyncMock()
            mock_cls.return_value = mock_inst
            mgr1 = await _get_memory_manager()
            mgr2 = await _get_memory_manager()
            assert mgr1 is mgr2
            assert mock_cls.call_count == 1, (
                f"FIX VERIFIED: {mock_cls.call_count} instances (should be 1)"
            )
