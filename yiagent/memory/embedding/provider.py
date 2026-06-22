"""
Embedding provider abstraction — pluggable vendors, shared cache.

Design:
  - EmbeddingProvider: ABC with embed_query / embed_batch
  - EmbeddingCache: in-process LRU + Redis shared cache
  - Vendor implementations register via factory pattern
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Type

import httpx

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.common.redis_pool import get_redis, redis_get_json, redis_set_json
from yiagent.common.utils import compute_md5


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Pluggable embedding provider."""

    def __init__(self, model: str, dim: int = 1536):
        self.model = model
        self.dim = dim

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# In-process LRU cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """In-process LRU cache (hot path) + Redis shared cache (cold path)."""

    def __init__(self, maxsize: int = 512):
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._maxsize = maxsize

    def _key(self, provider: str, model: str, text: str) -> str:
        return f"{provider}:{model}:{compute_md5(text)}"

    async def get(self, provider: str, model: str, text: str) -> Optional[List[float]]:
        key = self._key(provider, model, text)
        # In-process hot path
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        # Redis shared cache
        try:
            r = await get_redis()
            cached = await r.hget(f"embed_cache:{provider}:{model}", key)
            if cached:
                import json
                vec = json.loads(cached)
                self._cache[key] = vec
                return vec
        except Exception:
            pass
        return None

    async def put(self, provider: str, model: str, text: str, embedding: List[float]) -> None:
        key = self._key(provider, model, text)
        self._cache[key] = embedding
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        # Async write to Redis
        try:
            r = await get_redis()
            import json
            await r.hset(
                f"embed_cache:{provider}:{model}", key,
                json.dumps(embedding, ensure_ascii=False),
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        cfg = conf()
        self.model_name = model or cfg.get("embedding_model", "text-embedding-3-small")
        self._api_key = api_key or cfg.get("openai_api_key", "")
        self._base_url = base_url or cfg.get("openai_base_url", "https://api.openai.com/v1")
        super().__init__(model=self.model_name, dim=cfg.get("pg_vector_dim", 1536))
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
        return self._client

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = await self._get_client()
        try:
            resp = await client.post(
                "/embeddings",
                json={"input": texts, "model": self.model_name},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            logger.debug(f"[Embedding] OpenAI: {len(texts)} texts → {len(embeddings)} vectors")
            return embeddings
        except Exception as e:
            logger.error(f"[Embedding] OpenAI batch failed: {e}")
            raise

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider_registry: Dict[str, Type[EmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
}


def register_embedding_provider(name: str, cls: Type[EmbeddingProvider]) -> None:
    """Register a custom embedding provider."""
    _provider_registry[name] = cls


def create_embedding_provider(provider_name: Optional[str] = None, **kwargs) -> Optional[EmbeddingProvider]:
    """Factory: create an embedding provider by name. Returns None if disabled."""
    name = provider_name or conf().get("embedding_provider", "openai")
    if not name or name == "disabled":
        return None
    cls = _provider_registry.get(name)
    if cls is None:
        logger.warning(f"[Embedding] Unknown provider '{name}', embedding disabled")
        return None
    try:
        return cls(**kwargs)
    except Exception as e:
        logger.error(f"[Embedding] Failed to create provider '{name}': {e}")
        return None
