"""
Tests for yiagent.common.config — the config loading and singleton layer.

Critical areas under audit:
  - Double-checked locking correctness under concurrent access
  - Env var override precedence
  - Missing/malformed config file resilience
  - Type coercion from env vars (string vs JSON)
  - get_workspace() path expansion correctness
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from yiagent.common.config import (
    _DEFAULT_CONFIG,
    conf,
    get_pg_dsn,
    get_redis_url,
    get_workspace,
    load_config,
)


# ======================================================================
# Basic correctness
# ======================================================================

class TestConfigBasics:
    def test_load_config_returns_defaults_when_no_file_or_env(self):
        cfg = load_config()
        assert cfg["agent_name"] == "YiAgent"
        assert cfg["agent_language"] == "zh"
        assert cfg["redis_max_connections"] == 50

    def test_conf_returns_same_dict_on_repeated_calls(self):
        c1 = conf()
        c2 = conf()
        assert c1 is c2

    def test_load_config_is_idempotent(self):
        """load_config() called twice must return the same instance."""
        c1 = load_config()
        c2 = load_config()
        assert c1 is c2

    def test_get_workspace_expands_tilde(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/testuser")
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        result = get_workspace()
        assert str(result) == "/home/testuser/yiagent"

    def test_get_redis_url_returns_default(self):
        assert get_redis_url() == "redis://127.0.0.1:6379/0"

    def test_get_pg_dsn_returns_default(self):
        assert get_pg_dsn() == "postgresql://yiagent:yiagent@127.0.0.1:5432/yiagent"


# ======================================================================
# Env var overrides
# ======================================================================

class TestConfigEnvVars:
    def test_env_var_overrides_default_string(self, monkeypatch):
        monkeypatch.setenv("YIAGENT_AGENT_NAME", "TestBot")
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        assert cfg["agent_name"] == "TestBot"

    def test_env_var_overrides_json_int(self, monkeypatch):
        monkeypatch.setenv("YIAGENT_REDIS_MAX_CONNECTIONS", "100")
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        assert cfg["redis_max_connections"] == 100

    def test_env_var_overrides_json_list(self, monkeypatch):
        monkeypatch.setenv("YIAGENT_MCP_SERVERS", '["server1","server2"]')
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        assert cfg["mcp_servers"] == ["server1", "server2"]

    def test_env_var_float_json_parsing(self, monkeypatch):
        monkeypatch.setenv("YIAGENT_FLUSH_HIGH_WATERMARK", "0.95")
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        assert cfg["flush_high_watermark"] == 0.95

    def test_env_var_invalid_json_falls_back_to_string(self, monkeypatch):
        monkeypatch.setenv("YIAGENT_AGENT_NAME", '{"broken"')
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        # Falls back to raw string when JSON parse fails
        assert cfg["agent_name"] == '{"broken"'

    def test_env_var_not_in_defaults_is_included(self):
        """FIX VERIFIED: Any YIAGENT_* env var is now included, not just defaults."""
        import os
        os.environ["YIAGENT_CUSTOM_FIELD"] = "custom_value"
        try:
            import yiagent.common.config as cfg_mod
            cfg_mod._config = None
            cfg = load_config()
            assert "custom_field" in cfg
        finally:
            del os.environ["YIAGENT_CUSTOM_FIELD"]

    def test_yiagent_config_env_var_points_to_file(self, monkeypatch):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        try:
            json.dump({"agent_name": "FileBot"}, tmp)
            tmp.close()
            monkeypatch.setenv("YIAGENT_CONFIG", tmp.name)
            import yiagent.common.config as cfg_mod
            cfg_mod._config = None
            cfg = load_config()
            assert cfg["agent_name"] == "FileBot"
        finally:
            os.unlink(tmp.name)


# ======================================================================
# Config file loading
# ======================================================================

class TestConfigFile:
    def test_loads_valid_json_file(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        try:
            json.dump({"agent_name": "JSONBot", "redis_max_connections": 200}, tmp)
            tmp.close()
            import yiagent.common.config as cfg_mod
            cfg_mod._config = None
            cfg = load_config(tmp.name)
            assert cfg["agent_name"] == "JSONBot"
            assert cfg["redis_max_connections"] == 200
        finally:
            os.unlink(tmp.name)

    def test_missing_config_file_silently_ignored(self):
        """load_config with a non-existent file path must not crash."""
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config("/nonexistent/path/config.json")
        assert cfg["agent_name"] == "YiAgent"  # falls back to default

    def test_malformed_json_file_logs_warning_and_continues(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        try:
            tmp.write("not valid json {{{")
            tmp.close()
            import yiagent.common.config as cfg_mod
            cfg_mod._config = None
            cfg = load_config(tmp.name)
            # Must still return defaults
            assert cfg["agent_name"] == "YiAgent"
        finally:
            os.unlink(tmp.name)

    def test_env_var_takes_precedence_over_file(self, monkeypatch):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        try:
            json.dump({"agent_name": "FileBot"}, tmp)
            tmp.close()
            monkeypatch.setenv("YIAGENT_AGENT_NAME", "EnvBot")
            import yiagent.common.config as cfg_mod
            cfg_mod._config = None
            cfg = load_config(tmp.name)
            # Env var should win over file
            assert cfg["agent_name"] == "EnvBot"
        finally:
            os.unlink(tmp.name)


# ======================================================================
# EXTREME: Concurrent access (Cache Stampede / Thread-safety)
# ======================================================================

class TestConfigConcurrency:
    """CORE AUDIT: Simulate N threads hammering load_config() simultaneously."""

    def test_massive_concurrent_load_config(self):
        """100 threads all calling load_config() at the same time — must
        not crash, must all get the same config object."""
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        errors = []
        results = []

        def worker():
            try:
                results.append(load_config())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Got errors during concurrent load: {errors}"
        # All must return the same object
        assert all(r is results[0] for r in results), (
            "BUG: concurrent load_config returned different dict objects!"
        )

    def test_conf_is_thread_safe_snapshot(self, monkeypatch):
        """conf() called after env change should see the frozen snapshot."""
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        c1 = conf()
        monkeypatch.setenv("YIAGENT_AGENT_NAME", "ChangedAfterInit")
        c2 = conf()
        # conf() returns the frozen snapshot — env changes after first load
        # are NOT re-read. This is by design (frozen singleton).
        assert c1["agent_name"] == c2["agent_name"]


# ======================================================================
# EXTREME: Boundary value tests
# ======================================================================

class TestConfigBoundaries:
    def test_empty_env_var_value(self, monkeypatch):
        """Empty env var must be stored as empty string."""
        monkeypatch.setenv("YIAGENT_AGENT_NAME", "")
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        assert cfg["agent_name"] == ""

    def test_massive_env_var_value(self, monkeypatch):
        """Very long env var value must not crash."""
        huge = "x" * 1_000_000
        monkeypatch.setenv("YIAGENT_AGENT_NAME", huge)
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        assert cfg["agent_name"] == huge

    def test_config_with_special_characters_in_values(self, monkeypatch):
        monkeypatch.setenv("YIAGENT_AGENT_NAME", 'Bot with "quotes" and \\backslashes')
        import yiagent.common.config as cfg_mod
        cfg_mod._config = None
        cfg = load_config()
        assert "quotes" in cfg["agent_name"]
        assert "backslashes" in cfg["agent_name"]
