"""
OpenAI-compatible embedding provider — covers Doubao, Zhipu, Moonshot,
and any other vendor that exposes an OpenAI-compatible /embeddings endpoint.

Configure with:
  embedding_provider: "doubao"      # or "zhipu"
  doubao_api_key: "..."
  doubao_base_url: "https://ark.cn-beijing.volces.com/api/v3"
  doubao_embedding_model: "doubao-embedding"
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.memory.embedding.provider import EmbeddingProvider


class OpenAICompatEmbeddingProvider(EmbeddingProvider):
    """Generic OpenAI-compatible embedding endpoint.

    Key names are derived from the provider prefix in config (e.g. "doubao"):
      - {prefix}_api_key
      - {prefix}_base_url
      - {prefix}_embedding_model
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai_compat",
    ):
        cfg = conf()
        self._provider = provider
        self._model = model or cfg.get(f"{provider}_embedding_model", "text-embedding-3-small")
        self._api_key = api_key or cfg.get(f"{provider}_api_key", "")
        self._base_url = base_url or cfg.get(f"{provider}_base_url", "https://api.openai.com/v1")
        self._client: Optional[httpx.AsyncClient] = None
        super().__init__(model=self._model, dim=cfg.get("pg_vector_dim", 1536))

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
                json={"model": self._model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            logger.debug(f"[Embedding] {self._provider}: {len(texts)} → {len(embeddings)}")
            return embeddings
        except Exception as e:
            logger.error(f"[Embedding] {self._provider} failed: {e}")
            raise

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
