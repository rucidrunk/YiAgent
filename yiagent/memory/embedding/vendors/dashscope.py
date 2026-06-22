"""
DashScope (Alibaba Tongyi) embedding provider.

Uses DashScope's text-embedding API via HTTP.
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.memory.embedding.provider import EmbeddingProvider


class DashScopeEmbeddingProvider(EmbeddingProvider):
    """DashScope / Alibaba Tongyi embedding."""

    DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        cfg = conf()
        self._model = model or cfg.get("dashscope_embedding_model", "text-embedding-v3")
        self._api_key = api_key or cfg.get("dashscope_api_key", "")
        self._base_url = cfg.get("dashscope_base_url", self.DASHSCOPE_URL)
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
                json={
                    "model": self._model,
                    "input": texts,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            logger.debug(f"[Embedding] DashScope: {len(texts)} → {len(embeddings)}")
            return embeddings
        except Exception as e:
            logger.error(f"[Embedding] DashScope failed: {e}")
            raise

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
