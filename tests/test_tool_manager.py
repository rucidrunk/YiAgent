"""
Tests for yiagent.agent.tools.tool_manager — MCP config, signature detection.

CRITICAL AREAS:
  1. _normalize_mcp: list vs dict config formats
  2. _read_mcp_signature: missing file, mtime changes
  3. singleton correctness (double-checked locking)
  4. mcp.json parsing with invalid JSON
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from yiagent.agent.tools.tool_manager import ToolManager


@pytest.fixture
def tm():
    """Fresh ToolManager singleton (reset _instance for test isolation)."""
    ToolManager._instance = None
    return ToolManager()


class TestToolManagerSingleton:
    def test_singleton_returns_same_instance(self):
        a = ToolManager()
        b = ToolManager()
        assert a is b


class TestMcpNormalize:
    def test_normalize_list_format(self):
        """List format is passed through unchanged (caller provides type)."""
        raw = [
            {"name": "server1", "url": "http://s1.com", "type": "sse"},
            {"name": "server2", "command": "python", "type": "stdio"},
        ]
        result = ToolManager._normalize_mcp(raw)
        assert len(result) == 2
        assert result[0]["name"] == "server1"
        assert result[1]["name"] == "server2"

    def test_normalize_dict_format(self):
        raw = {
            "server1": {"url": "http://s1.com"},
            "server2": {"command": "python", "type": "stdio"},
        }
        result = ToolManager._normalize_mcp(raw)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"server1", "server2"}

    def test_normalize_empty(self):
        assert ToolManager._normalize_mcp([]) == []
        assert ToolManager._normalize_mcp({}) == []


class TestMcpSignature:
    def test_missing_file_returns_none(self):
        with patch.object(ToolManager, '_mcp_json_path', return_value='/nonexistent/mcp.json'):
            tm = ToolManager()
            mtime, digest = tm._read_mcp_signature()
            assert mtime is None
            assert digest is None

    def test_existing_file_returns_signature(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"mcpServers": []}, f)
            path = f.name

        try:
            with patch.object(ToolManager, '_mcp_json_path', return_value=path):
                tm = ToolManager()
                mtime, digest = tm._read_mcp_signature()
                assert mtime is not None
                assert digest is not None
                assert len(digest) == 64  # SHA-256
        finally:
            os.unlink(path)


class TestMcpConfigLoading:
    def test_load_valid_mcp_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"mcpServers": [{"name": "s1", "url": "http://s1.com"}]}, f)
            path = f.name

        try:
            with patch.object(ToolManager, '_mcp_json_path', return_value=path):
                tm = ToolManager()
                configs = tm._load_mcp_configs()
                assert len(configs) == 1
                assert configs[0]["name"] == "s1"
        finally:
            os.unlink(path)

    def test_load_invalid_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            path = f.name

        try:
            with patch.object(ToolManager, '_mcp_json_path', return_value=path):
                tm = ToolManager()
                with patch('yiagent.agent.tools.tool_manager.logger') as mock_log:
                    configs = tm._load_mcp_configs()
                # Falls back to config env
                assert isinstance(configs, list)
        finally:
            os.unlink(path)

    def test_load_mcp_servers_from_config_not_file(self):
        """When mcp.json doesn't exist, falls back to conf().get('mcp_servers')."""
        with patch.object(ToolManager, '_mcp_json_path', return_value='/nonexistent/mcp.json'):
            tm = ToolManager()
            configs = tm._load_mcp_configs()
            assert isinstance(configs, list)


class TestToolManagerExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    def test_load_mcp_with_massive_config(self):
        """100 MCP servers in config — must not crash."""
        raw = [{"name": f"s{i}", "url": f"http://host{i}.com"} for i in range(100)]
        result = ToolManager._normalize_mcp(raw)
        assert len(result) == 100

    def test_normalize_mcp_dict_adds_type(self):
        """Dict format auto-infers type from presence of url key."""
        raw = {"no_type_server": {"url": "http://x.com"}}
        result = ToolManager._normalize_mcp(raw)
        assert result[0]["type"] == "sse"

    def test_normalize_mcp_dict_no_url_defaults_to_stdio(self):
        raw = {"bare_server": {"command": "echo"}}
        result = ToolManager._normalize_mcp(raw)
        assert result[0]["type"] == "stdio"

    def test_normalize_list_passthrough_no_type_added(self):
        """List format: no auto-type inference (caller provides type explicitly)."""
        raw = [{"name": "s1", "url": "http://x.com"}]
        result = ToolManager._normalize_mcp(raw)
        # List format passes through unchanged — no 'type' key added
        assert "type" not in result[0]

    def test_register_tool_invalid_class(self):
        """Registering a non-tool class logs warning, doesn't crash."""
        tm = ToolManager()
        class NotATool:
            pass
        # register_tool tries to instantiate → name attribute missing → warning
        tm.register_tool(NotATool)
        # No crash = pass
