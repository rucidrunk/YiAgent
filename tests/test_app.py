"""
Tests for app.py — FastAPI lifespan, health endpoint, startup/shutdown.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Mock asyncpg before it's imported by long_term_store
class _FakeAsyncPGPool:
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def execute(self, *args, **kwargs): return "OK"
    async def fetch(self, *args, **kwargs): return []
    async def fetchrow(self, *args, **kwargs): return None
    def acquire(self): return self

_pg_mock = MagicMock()
_pg_mock.create_pool = AsyncMock(return_value=_FakeAsyncPGPool())
sys.modules["asyncpg"] = _pg_mock


def _mock_app():
    """Create a mock FastAPI app with assignable state."""
    m = MagicMock()
    m.state = MagicMock()
    return m


def _ensure_lifespan(app):
    """FastAPI TestClient may not trigger lifespan — ensure state attrs exist."""
    if not hasattr(app.state, 'redis_ok'):
        app.state.redis_ok = False
        app.state.pg_ok = False
        app.state.web_adapter = MagicMock()


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        from app import app as fastapi_app
        from fastapi.testclient import TestClient
        _ensure_lifespan(fastapi_app)
        client = TestClient(fastapi_app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "redis" in data
        assert "postgres" in data


class TestAppStartup:
    @pytest.mark.asyncio
    async def test_startup_without_redis(self):
        with patch("yiagent.common.redis_pool.get_redis") as mock_get:
            mock_get.side_effect = ConnectionError("no redis")
            from app import lifespan
            try:
                async with lifespan(_mock_app()):
                    pass
            except Exception as e:
                pytest.fail(f"Startup should not crash: {e}")

    @pytest.mark.asyncio
    async def test_startup_without_pg(self):
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        with patch("yiagent.common.redis_pool.get_redis", AsyncMock(return_value=mock_client)):
            with patch("yiagent.memory.long_term_store.get_long_term_store") as mock_get_store:
                mock_store = AsyncMock()
                mock_store.initialize = AsyncMock(side_effect=ConnectionError("no pg"))
                mock_get_store.return_value = mock_store
                from app import lifespan
                try:
                    async with lifespan(_mock_app()):
                        pass
                except Exception as e:
                    pytest.fail(f"Startup should not crash: {e}")

    @pytest.mark.asyncio
    async def test_startup_both_unavailable(self):
        with patch("yiagent.common.redis_pool.get_redis") as mock_get:
            mock_get.side_effect = ConnectionError("no redis")
            with patch("yiagent.memory.long_term_store.get_long_term_store") as mock_get_store:
                mock_store = AsyncMock()
                mock_store.initialize = AsyncMock(side_effect=ConnectionError("no pg"))
                mock_get_store.return_value = mock_store
                from app import lifespan
                try:
                    async with lifespan(_mock_app()):
                        pass
                except Exception as e:
                    pytest.fail(f"Startup should not crash: {e}")

    @pytest.mark.asyncio
    async def test_startup_uses_get_redis_warmup(self):
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        with patch("yiagent.common.redis_pool.get_redis") as mock_get:
            mock_get.return_value = mock_client
            with patch("yiagent.memory.long_term_store.get_long_term_store") as mock_get_store:
                mock_store = AsyncMock()
                mock_store.initialize = AsyncMock(return_value=None)
                mock_get_store.return_value = mock_store
                from app import lifespan
                async with lifespan(_mock_app()):
                    pass
        mock_get.assert_called_once()


class TestAppShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_redis_pool(self):
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        with patch("yiagent.common.redis_pool.get_redis", AsyncMock(return_value=mock_client)):
            with patch("yiagent.common.redis_pool.close_redis") as mock_close:
                mock_close.return_value = None
                with patch("yiagent.memory.long_term_store.get_long_term_store") as mock_get_store:
                    mock_store = AsyncMock()
                    mock_store.initialize = AsyncMock(return_value=None)
                    mock_get_store.return_value = mock_store
                    from app import lifespan
                    async with lifespan(_mock_app()):
                        pass
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_close_failure(self):
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        with patch("yiagent.common.redis_pool.get_redis", AsyncMock(return_value=mock_client)):
            with patch("yiagent.common.redis_pool.close_redis") as mock_close:
                mock_close.side_effect = RuntimeError("cleanup failed")
                with patch("yiagent.memory.long_term_store.get_long_term_store") as mock_get_store:
                    mock_store = AsyncMock()
                    mock_store.initialize = AsyncMock(return_value=None)
                    mock_get_store.return_value = mock_store
                    from app import lifespan
                    try:
                        async with lifespan(_mock_app()):
                            pass
                    except RuntimeError:
                        pass
                    mock_close.assert_called_once()
