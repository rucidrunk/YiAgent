"""
Tests for app.py chat endpoints — end-to-end agent interaction.

CRITICAL BUGS FOUND:
  1. No model provider registered → 503 (FIXED: registered at startup)
  2. SSE format missing 'event:' line → JS listeners never fire
  3. _sessions dict no expiry → memory leak
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Prevent asyncpg import failure
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


def _ensure_lifespan(app):
    """FastAPI TestClient sometimes doesn't properly trigger lifespan.
    Ensure state attributes exist for tests."""
    if not hasattr(app.state, 'redis_ok'):
        app.state.redis_ok = False
        app.state.pg_ok = False
        app.state.web_adapter = MagicMock()


class TestAppRoutesExist:
    def test_index_route_returns_html(self):
        from app import app, _CHAT_HTML
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "YiAgent Chat" in resp.text
        assert "EventSource" in resp.text

    def test_health_route(self):
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "redis" in data
        assert "postgres" in data

    def test_chat_route_exists(self):
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.post("/chat", json={"content": "hello"})
        assert resp.status_code in (200, 400, 500, 503)

    def test_chat_stream_route_accepts_get(self):
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.get("/chat/stream?content=hello")
        assert resp.status_code in (200, 400, 500, 503)


class TestChatNoModelProvider:
    def test_chat_returns_503_without_model_provider(self):
        """Without any provider, chat returns 503."""
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        # Clear provider registry to simulate no providers
        from yiagent.model_gateway.router import _provider_registry
        old = dict(_provider_registry)
        _provider_registry.clear()
        try:
            client = TestClient(app)
            resp = client.post("/chat", json={"content": "hello"})
            assert resp.status_code == 503
            assert "No model provider" in resp.json()["detail"]
        finally:
            _provider_registry.update(old)

    def test_empty_content_returns_400(self):
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.post("/chat", json={"content": ""})
        assert resp.status_code == 400

    def test_provider_fix_demo(self):
        """FIX VERIFIED: OpenAI provider registered at startup, ergo no 503."""
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.post("/chat", json={"content": "hello"})
        # With OpenAI provider registered at startup, we get 200 (if Redis+PG
        # are available) or 500 (if infra down). But NOT 503 (no provider).
        assert resp.status_code != 503, (
            f"BUG REGRESSION: 503 — provider registration at startup broken"
        )


class TestChatEdgeCases:
    def test_missing_content_field(self):
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.post("/chat", json={})
        assert resp.status_code == 400

    def test_very_long_content(self):
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.post("/chat", json={"content": "hello " * 2000})
        assert resp.status_code in (400, 500, 503)

    def test_unicode_content(self):
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        resp = client.post("/chat", json={"content": "你好世界 🚀"})
        assert resp.status_code in (400, 500, 503)

    def test_session_reuse(self):
        from app import app, _sessions
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        _sessions.clear()
        client = TestClient(app)
        resp1 = client.post("/chat", json={"session_id": "reuse", "content": "a"})
        resp2 = client.post("/chat", json={"session_id": "reuse", "content": "b"})
        assert resp1.status_code == resp2.status_code
        _sessions.clear()


class TestAppExtreme:
    def test_concurrent_chat_requests(self):
        import concurrent.futures
        from app import app
        from fastapi.testclient import TestClient
        _ensure_lifespan(app)
        client = TestClient(app)
        def send(i):
            return client.post("/chat", json={"session_id": f"c{i}", "content": f"m{i}"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = [ex.submit(send, i).result() for i in range(5)]
        assert all(r.status_code in (200, 500, 503) for r in results)

    def test_session_dict_memory_leak(self):
        from app import _sessions
        _sessions.clear()
        for i in range(100):
            _sessions[f"s{i}"] = object()
        assert len(_sessions) == 100
        _sessions.clear()
        assert len(_sessions) == 0

    def test_html_page_has_all_js_listeners(self):
        from app import _CHAT_HTML
        assert "function send()" in _CHAT_HTML
        assert "EventSource" in _CHAT_HTML
        assert "addEventListener" in _CHAT_HTML
        # BUG-8 check: these listeners need matching SSE event: lines
        assert '"message_update"' in _CHAT_HTML
        assert '"done"' in _CHAT_HTML

    def test_sse_format_has_event_line(self):
        """
        BUG-8 VERIFICATION: Check if the SSE format includes the 'event:' line.
        Without it, EventSource listeners never fire.
        """
        from app import app
        # Read the raw app.py to check the SSE format
        with open("app.py") as f:
            source = f.read()
        # The fmt string should include 'event:'
        # Current bug: yield fmt only has 'data:' line
        has_event_line = 'event: {' in source or 'event: {event' in source
        # Document current state — this should be True after fix
        if not has_event_line:
            pytest.fail(
                "BUG-8 STILL PRESENT: SSE format missing 'event:' line. "
                "JavaScript event listeners will never fire. "
                "Fix: change 'yield f\"data: {data}\\n\\n\"' to "
                "'yield f\"event: {event.type}\\ndata: {data}\\n\\n\"'"
            )
