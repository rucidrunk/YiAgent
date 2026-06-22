"""
MemoryManager — unified API for the three-layer memory architecture.

Orchestrates:
  - ConversationStore (Redis short-term)
  - LongTermStore (PostgreSQL + pgvector long-term)
  - EmbeddingProvider (vector generation + cache)
  - TextChunker (document → chunks)
  - ContextFlushPipeline (asyncio.Queue 削峰)
  - RRF fusion via LongTermStore.fuse_rrf()

Two-pass incremental sync (from CowAgent):
  Pass 1: walk files, detect changes by content_hash, collect pending chunks
  Pass 2: single batched embed across ALL pending chunks, then persist

Key design decisions:
  - Hybrid search with RRF fusion (vector + tsvector + pg_trgm)
  - Temporal decay for dated memory files
  - Graceful degradation: vector failure → keyword-only
  - Non-blocking context flush
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.common.utils import compute_hash
from yiagent.memory.config import MemoryConfig, get_memory_config
from yiagent.memory.chunker import TextChunker
from yiagent.memory.embedding.provider import EmbeddingProvider, EmbeddingCache, create_embedding_provider
from yiagent.memory.long_term_store import (
    LongTermStore, SearchResult, MemoryChunk, get_long_term_store,
)
from yiagent.memory.context_flush import ContextFlushPipeline


class MemoryManager:
    """
    Unified memory API.

    Usage:
        mgr = MemoryManager()
        await mgr.initialize()

        results = await mgr.search("Python tutorial", user_id="user_1")
        await mgr.sync_files(workspace_dir, force=False)
        await mgr.flush_context(session_id, messages, reason="threshold")
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_model: Any = None,
    ):
        self.config = config or get_memory_config()
        self._llm_model = llm_model

        # Storage
        self._long_term = get_long_term_store()

        # Chunker
        self._chunker = TextChunker(
            max_tokens=self.config.chunk_max_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens,
        )

        # Embedding
        self._embedding_provider: Optional[EmbeddingProvider] = embedding_provider
        self._embedding_cache = EmbeddingCache()

        # Context flush
        self._flush_pipeline = ContextFlushPipeline(
            max_workers=conf().get("flush_max_workers", 4),
            queue_maxsize=conf().get("flush_queue_maxsize", 256),
            high_watermark=conf().get("flush_high_watermark", 0.8),
            llm_model=llm_model,
        )

        self._dirty = False
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize PG schema and flush workers. Idempotent."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._long_term.initialize()
            if self._embedding_provider is None:
                self._embedding_provider = create_embedding_provider()
            self._ensure_workspace()
            await self._flush_pipeline.start()
            self._initialized = True
            logger.info(
                f"[MemoryManager] Initialized "
                f"(embedding={'enabled' if self._embedding_provider else 'disabled'})"
            )

    async def close(self) -> None:
        await self._flush_pipeline.stop()
        await self._long_term.close()

    def _ensure_workspace(self) -> None:
        ws = self.config.get_workspace()
        (ws / "memory").mkdir(parents=True, exist_ok=True)
        # Create default MEMORY.md if missing
        mem_md = ws / "MEMORY.md"
        if not mem_md.exists():
            mem_md.write_text("# Memory Index\n\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Search (hybrid: vector + keyword, RRF fusion)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_results: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search with graceful degradation.

        1. Try vector search (if embedding provider available)
        2. Always run keyword search (3-tier: tsvector / pg_trgm / ILIKE)
        3. Fuse results via RRF
        4. Apply temporal decay + score filter
        """
        max_results = max_results or self.config.max_results
        min_score = min_score or self.config.min_score

        scopes = ["shared"]
        if user_id:
            scopes.append("user")

        ranked_lists: List[List[SearchResult]] = []

        # Vector search (best-effort)
        if self._embedding_provider:
            try:
                provider_name = self._embedding_provider.provider_name
                model = self._embedding_provider.model
                cached = await self._embedding_cache.get(provider_name, model, query)
                if cached is not None:
                    query_embedding = cached
                else:
                    query_embedding = await self._embedding_provider.embed_query(query)
                    await self._embedding_cache.put(provider_name, model, query, query_embedding)

                vec_results = await self._long_term.search_vector(
                    query_embedding, user_id, scopes, max_results * 2
                )
                ranked_lists.append(vec_results)
                logger.debug(f"[MemoryManager] Vector search: {len(vec_results)} results")
            except Exception as e:
                logger.warning(f"[MemoryManager] Vector search degraded: {e}")

        # Keyword search (always runs)
        try:
            kw_results = await self._long_term.search_keyword(
                query, user_id, scopes, max_results * 2
            )
            ranked_lists.append(kw_results)
            logger.debug(f"[MemoryManager] Keyword search: {len(kw_results)} results")
        except Exception as e:
            logger.warning(f"[MemoryManager] Keyword search failed: {e}")

        if not ranked_lists:
            return []

        # RRF fusion
        fused = LongTermStore.fuse_rrf(
            ranked_lists,
            k=conf().get("rrf_k", 60),
            temporal_half_life_days=conf().get("temporal_half_life_days", 30.0),
        )

        # Filter and limit
        filtered = [r for r in fused if r.score >= min_score]
        return filtered[:max_results]

    # ------------------------------------------------------------------
    # Memory ingestion
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        scope: str = "shared",
        source: str = "memory",
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add new memory content, chunked and embedded."""
        if not content.strip():
            return

        if not path:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            if user_id and scope == "user":
                path = f"memory/users/{user_id}/memory_{content_hash}.md"
            else:
                path = f"memory/shared/memory_{content_hash}.md"

        chunks = self._chunker.chunk_text(content)
        texts = [c.text for c in chunks]

        # Batch embed if provider available
        if self._embedding_provider:
            try:
                embeddings = await self._embedding_provider.embed_batch(texts)
            except Exception as e:
                logger.warning(f"[MemoryManager] Embedding failed, storing without vectors: {e}")
                embeddings = [None] * len(texts)
        else:
            embeddings = [None] * len(texts)

        memory_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = hashlib.md5(
                f"{path}:{chunk.start_line}:{chunk.end_line}".encode()
            ).hexdigest()
            memory_chunks.append(MemoryChunk(
                id=chunk_id,
                user_id=user_id,
                scope=scope,
                source=source,
                path=path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
                embedding=embedding,
                content_hash=compute_hash(chunk.text),
                metadata=metadata,
            ))

        await self._long_term.upsert_chunks_batch(memory_chunks)
        file_hash = compute_hash(content)
        await self._long_term.update_file_metadata(
            path=path, source=source, content_hash=file_hash,
            mtime=int(time.time()), size=len(content),
        )
        self._dirty = True

    # ------------------------------------------------------------------
    # Two-pass incremental sync
    # ------------------------------------------------------------------

    async def sync_files(self, workspace_dir: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        Two-pass incremental sync from memory files.

        Pass 1: walk all .md files, detect changes via content_hash
        Pass 2: single batched embed across ALL pending chunks
        """
        ws = Path(workspace_dir or str(self.config.get_workspace()))
        memory_dir = ws / "memory"

        files_to_scan: List[Tuple[Path, str, str, Optional[str]]] = []

        # MEMORY.md (evergreen)
        mem_md = ws / "MEMORY.md"
        if mem_md.exists():
            files_to_scan.append((mem_md, "memory", "shared", None))

        # All .md files under memory/ (exclude backups, dreams, evolution)
        if memory_dir.exists():
            for file_path in memory_dir.rglob("*.md"):
                rel_parts = file_path.relative_to(ws).parts
                if any(part.startswith(".") for part in rel_parts):
                    continue
                # Exclude bookkeeping dirs
                skip_dirs = {"dreams", ".evolution_backups", "evolution"}
                if any(s in rel_parts for s in skip_dirs):
                    continue

                # Resolve scope and user_id
                if "users" in rel_parts:
                    user_idx = rel_parts.index("users") + 1
                    uid = rel_parts[user_idx] if user_idx < len(rel_parts) else None
                    files_to_scan.append((file_path, "memory", "user", uid))
                else:
                    files_to_scan.append((file_path, "memory", "shared", None))

        # Pass 1: detect changes
        import time as _time
        pending: List[Dict[str, Any]] = []
        for file_path, source, scope, uid in files_to_scan:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            file_hash = compute_hash(content)
            rel_path = str(file_path.relative_to(ws))

            if not force:
                stored_hash = await self._long_term.get_file_hash(rel_path)
                if stored_hash == file_hash:
                    continue

            chunks = self._chunker.chunk_text(content)
            if not chunks:
                continue

            pending.append({
                "file_path": file_path,
                "rel_path": rel_path,
                "source": source,
                "scope": scope,
                "user_id": uid,
                "file_hash": file_hash,
                "chunks": chunks,
                "texts": [c.text for c in chunks],
            })

        if not pending:
            self._dirty = False
            return {"files_scanned": len(files_to_scan), "files_changed": 0, "chunks_updated": 0}

        # Pass 2: batch embed
        all_texts: List[str] = []
        for entry in pending:
            all_texts.extend(entry["texts"])

        if not self._embedding_provider:
            all_embeddings: List[Optional[List[float]]] = [None] * len(all_texts)
        else:
            try:
                all_embeddings = await self._embedding_provider.embed_batch(all_texts)
            except Exception as e:
                logger.error(
                    f"[MemoryManager] Batch embed failed for {len(all_texts)} chunks: {e}. "
                    f"Index left untouched."
                )
                return {"files_scanned": len(files_to_scan), "files_changed": len(pending),
                        "chunks_updated": 0, "error": str(e)}

        # Persist
        cursor = 0
        total_chunks = 0
        for entry in pending:
            n = len(entry["texts"])
            entry_embeddings = all_embeddings[cursor:cursor + n]
            cursor += n

            await self._long_term.delete_by_path(entry["rel_path"])
            memory_chunks = []
            for chunk, embedding in zip(entry["chunks"], entry_embeddings):
                chunk_id = hashlib.md5(
                    f"{entry['rel_path']}:{chunk.start_line}:{chunk.end_line}".encode()
                ).hexdigest()
                memory_chunks.append(MemoryChunk(
                    id=chunk_id,
                    user_id=entry["user_id"],
                    scope=entry["scope"],
                    source=entry["source"],
                    path=entry["rel_path"],
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    embedding=embedding,
                    content_hash=compute_hash(chunk.text),
                ))
            await self._long_term.upsert_chunks_batch(memory_chunks)
            stat = entry["file_path"].stat()
            await self._long_term.update_file_metadata(
                path=entry["rel_path"],
                source=entry["source"],
                content_hash=entry["file_hash"],
                mtime=int(stat.st_mtime),
                size=stat.st_size,
            )
            total_chunks += len(memory_chunks)

        self._dirty = False
        logger.info(
            f"[MemoryManager] Sync: {len(files_to_scan)} files, "
            f"{len(pending)} changed, {total_chunks} chunks updated"
        )
        return {
            "files_scanned": len(files_to_scan),
            "files_changed": len(pending),
            "chunks_updated": total_chunks,
        }

    # ------------------------------------------------------------------
    # Context flush (non-blocking, via ContextFlushPipeline)
    # ------------------------------------------------------------------

    async def monitor_and_flush(
        self,
        session_id: str,
        token_estimate: int,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> bool:
        """Delegate to ContextFlushPipeline for non-blocking flush."""
        return await self._flush_pipeline.monitor_and_flush(
            session_id=session_id,
            token_estimate=token_estimate,
            max_tokens=max_tokens,
            messages=messages,
            user_id=user_id,
            callback=callback,
        )

    def get_flush_pipeline(self) -> ContextFlushPipeline:
        return self._flush_pipeline

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> Dict[str, Any]:
        lt_stats = await self._long_term.get_stats()
        return {
            "long_term": lt_stats,
            "dirty": self._dirty,
            "embedding_enabled": self._embedding_provider is not None,
            "flush_queue_depth": self._flush_pipeline.queue_depth,
        }

    def mark_dirty(self) -> None:
        self._dirty = True
