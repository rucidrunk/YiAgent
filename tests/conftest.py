"""Shared fixtures for the YiAgent test suite."""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Ensure yiagent is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Config isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    """Reset the config singleton before and after each test."""
    import yiagent.common.config as cfg_mod
    cfg_mod._config = None
    monkeypatch.delenv("YIAGENT_CONFIG", raising=False)
    for key in list(os.environ):
        if key.startswith("YIAGENT_"):
            monkeypatch.delenv(key)
    yield
    cfg_mod._config = None


# ---------------------------------------------------------------------------
# Redis mocks (for unit tests that don't need real Redis)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis_client():
    """Return a fully mocked async Redis client."""
    r = AsyncMock()
    r.ping = AsyncMock(return_value=True)
    r.set = AsyncMock(return_value=True)
    r.get = AsyncMock(return_value=None)
    r.delete = AsyncMock(return_value=1)
    r.expire = AsyncMock(return_value=True)
    r.lrange = AsyncMock(return_value=[])
    r.rpush = AsyncMock(return_value=1)
    r.llen = AsyncMock(return_value=0)
    r.ltrim = AsyncMock(return_value="OK")
    r.hset = AsyncMock(return_value=1)
    r.hget = AsyncMock(return_value=None)
    r.hgetall = AsyncMock(return_value={})
    r.hexists = AsyncMock(return_value=False)
    r.keys = AsyncMock(return_value=[])
    return r


@pytest.fixture
def patch_get_redis(mock_redis_client):
    """
    Patch get_redis() wherever it's imported.

    conversation_store.py imports get_redis and the helpers directly,
    so we must patch the module where they are *used*, not *defined*.
    """
    targets = [
        "yiagent.memory.conversation_store.get_redis",
        "yiagent.memory.conversation_store.redis_get_json",
        "yiagent.memory.conversation_store.redis_set_json",
        "yiagent.memory.conversation_store.redis_hget_json",
        "yiagent.memory.conversation_store.redis_hset_json",
        "yiagent.memory.conversation_store.redis_hgetall_json",
    ]
    with patch(targets[0], AsyncMock(return_value=mock_redis_client)):
        yield mock_redis_client
