"""
Model Gateway — provider-agnostic LLM routing with failover.

Design:
  - Routes requests to configured providers (OpenAI, Claude, DeepSeek, ...).
  - Automatic failover: if primary provider fails with a retryable error,
    the next provider in the chain is tried.
  - Each provider is a subclass of LLMModel — pluggable via register_provider().
  - Streaming and non-streaming calls are both supported.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.protocol.models import LLMRequest, LLMModel


class ModelRouter:
    """Routes LLM requests through configured providers with failover."""

    def __init__(self, providers: Optional[List[LLMModel]] = None):
        self._providers: List[LLMModel] = providers or []
        self._primary: Optional[LLMModel] = self._providers[0] if self._providers else None

    def add_provider(self, provider: LLMModel) -> None:
        self._providers.append(provider)
        if self._primary is None:
            self._primary = provider

    @property
    def primary(self) -> Optional[LLMModel]:
        return self._primary

    async def call(self, request: LLMRequest) -> Dict[str, Any]:
        """Non-streaming call with failover."""
        last_error: Optional[Exception] = None
        for i, provider in enumerate(self._providers):
            try:
                return await provider.call(request)
            except Exception as e:
                logger.warning(f"[ModelRouter] Provider {i} ({provider.model}) failed: {e}")
                last_error = e
                if not self._is_retryable(e):
                    continue  # try next provider — another may have valid credentials
        raise last_error or RuntimeError("No providers available")

    async def call_stream(self, request: LLMRequest) -> AsyncIterator[Dict[str, Any]]:
        """Streaming call. Falls back to next provider on connection error only."""
        last_error: Optional[Exception] = None
        for i, provider in enumerate(self._providers):
            try:
                async for chunk in provider.call_stream(request):
                    yield chunk
                return  # success — stop iterating providers
            except Exception as e:
                logger.warning(f"[ModelRouter] Provider {i} ({provider.model}) stream failed: {e}")
                last_error = e
                if not self._is_retryable(e):
                    continue
        raise last_error or RuntimeError("No providers available")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        msg = str(exc).lower()
        etype = type(exc).__name__.lower()
        return any(kw in msg for kw in (
            "timeout", "connection", "connect", "rate limit", "overloaded",
            "429", "500", "502", "503", "504",
        )) or any(kw in etype for kw in (
            "timeout", "connect", "read", "write", "remote",
        ))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider_registry: Dict[str, type] = {}


def register_provider(name: str, cls: type) -> None:
    _provider_registry[name] = cls


def create_model_router() -> ModelRouter:
    """Create a ModelRouter from config. Extension: reads model_providers list."""
    router = ModelRouter()
    cfg = conf()

    # Default: single provider from config
    provider_name = cfg.get("model_provider", "openai")
    model_name = cfg.get("model_name", "gpt-4o")

    cls = _provider_registry.get(provider_name)
    if cls is not None:
        try:
            router.add_provider(cls(model=model_name))
        except Exception as e:
            logger.warning(f"[ModelRouter] Failed to create {provider_name}: {e}")

    # Extra providers for failover
    for entry in cfg.get("model_providers", []):
        name = entry.get("provider", "")
        model = entry.get("model", "")
        if name and model and name != provider_name:
            cls2 = _provider_registry.get(name)
            if cls2:
                try:
                    router.add_provider(cls2(model=model))
                except Exception as e:
                    logger.warning(f"[ModelRouter] Failed to create {name}: {e}")

    return router
