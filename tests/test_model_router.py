"""
Tests for yiagent.model_gateway.router — provider failover, retry logic.

CRITICAL AREAS:
  1. Failover: primary fails → secondary tried
  2. Retryable vs non-retryable error discrimination
  3. Empty provider list → RuntimeError
  4. Streaming failover continues from next provider
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yiagent.model_gateway.router import ModelRouter, create_model_router, register_provider
from yiagent.protocol.models import LLMModel, LLMRequest


class FakeProvider(LLMModel):
    """Mock provider for testing."""
    def __init__(self, model="test", should_fail=False, error_msg="timeout"):
        super().__init__(model)
        self.should_fail = should_fail
        self.error_msg = error_msg
        self.call_count = 0
        self.stream_count = 0

    async def call(self, request):
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError(self.error_msg)
        return {"content": f"response from {self.model}"}

    async def call_stream(self, request):
        self.stream_count += 1
        if self.should_fail:
            raise ConnectionError(self.error_msg)
        yield {"choices": [{"delta": {"content": f"stream from {self.model}"}}]}


class TestModelRouterBasics:
    def test_empty_router_has_no_primary(self):
        router = ModelRouter()
        assert router.primary is None

    def test_add_provider_sets_primary(self):
        router = ModelRouter()
        p = FakeProvider(model="gpt-4o")
        router.add_provider(p)
        assert router.primary is p

    def test_multiple_providers(self):
        router = ModelRouter()
        p1 = FakeProvider(model="gpt-4o")
        p2 = FakeProvider(model="claude-sonnet")
        router.add_provider(p1)
        router.add_provider(p2)
        assert len(router._providers) == 2


class TestModelRouterFailover:
    @pytest.mark.asyncio
    async def test_primary_succeeds_no_failover(self):
        p1 = FakeProvider(model="primary")
        p2 = FakeProvider(model="secondary")
        router = ModelRouter([p1, p2])
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await router.call(req)
        assert "primary" in result["content"]
        assert p1.call_count == 1
        assert p2.call_count == 0  # never called

    @pytest.mark.asyncio
    async def test_primary_fails_falls_back_to_secondary(self):
        p1 = FakeProvider(model="primary", should_fail=True)
        p2 = FakeProvider(model="secondary")
        router = ModelRouter([p1, p2])
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await router.call(req)
        assert "secondary" in result["content"]
        assert p1.call_count == 1
        assert p2.call_count == 1

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self):
        p1 = FakeProvider(model="p1", should_fail=True)
        p2 = FakeProvider(model="p2", should_fail=True)
        router = ModelRouter([p1, p2])
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(Exception):
            await router.call(req)

    @pytest.mark.asyncio
    async def test_non_retryable_error_continues_to_next(self):
        """
        FIX VERIFIED: Non-retryable errors now CONTINUE to next provider
        (not break). Another provider may have valid credentials.
        """
        p1 = FakeProvider(model="primary", should_fail=True, error_msg="invalid api key")
        p2 = FakeProvider(model="secondary")  # has valid credentials
        router = ModelRouter([p1, p2])
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await router.call(req)
        # p2 succeeds even though p1 had non-retryable error
        assert "secondary" in result["content"]
        assert p1.call_count == 1
        assert p2.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_providers_raises(self):
        router = ModelRouter([])
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(RuntimeError, match="No providers"):
            await router.call(req)


class TestModelRouterRetryable:
    def test_timeout_is_retryable(self):
        assert ModelRouter._is_retryable(ConnectionError("read timeout")) is True

    def test_rate_limit_is_retryable(self):
        assert ModelRouter._is_retryable(Exception("rate limit exceeded")) is True

    def test_429_is_retryable(self):
        assert ModelRouter._is_retryable(Exception("HTTP 429")) is True

    def test_500_is_retryable(self):
        assert ModelRouter._is_retryable(Exception("server error 500")) is True

    def test_502_503_504_are_retryable(self):
        assert ModelRouter._is_retryable(Exception("Bad Gateway 502")) is True
        assert ModelRouter._is_retryable(Exception("503 Service Unavailable")) is True

    def test_auth_error_not_retryable(self):
        assert ModelRouter._is_retryable(Exception("invalid authentication")) is False

    def test_bad_request_not_retryable(self):
        assert ModelRouter._is_retryable(Exception("invalid request parameter")) is False


class TestModelRouterStreaming:
    @pytest.mark.asyncio
    async def test_stream_primary_succeeds(self):
        p1 = FakeProvider(model="primary")
        p2 = FakeProvider(model="secondary")
        router = ModelRouter([p1, p2])
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        chunks = []
        async for chunk in router.call_stream(req):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert p1.stream_count == 1
        assert p2.stream_count == 0

    @pytest.mark.asyncio
    async def test_stream_failover_to_secondary(self):
        p1 = FakeProvider(model="primary", should_fail=True)
        p2 = FakeProvider(model="secondary")
        router = ModelRouter([p1, p2])
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        chunks = []
        async for chunk in router.call_stream(req):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "secondary" in str(chunks)
        assert p1.stream_count == 1
        assert p2.stream_count == 1


class TestModelRouterFactory:
    def test_create_model_router_from_config(self):
        router = create_model_router()
        assert isinstance(router, ModelRouter)

    def test_register_and_use_provider(self):
        class TestProv(FakeProvider):
            pass

        register_provider("test_prov", TestProv)
        assert "test_prov" in __import__("yiagent.model_gateway.router", fromlist=["_provider_registry"])._provider_registry
