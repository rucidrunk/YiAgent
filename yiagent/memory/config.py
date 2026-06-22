"""Memory configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from yiagent.common.config import conf
from yiagent.common.utils import expand_path


@dataclass
class MemoryConfig:
    """Configuration for memory storage and search."""

    workspace_root: str = field(default_factory=lambda: str(expand_path(conf().get("workspace_root", "~/yiagent"))))

    # Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Chunking
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Search
    max_results: int = 10
    min_score: float = 0.1
    rrf_k: int = 60
    temporal_half_life_days: float = 30.0

    # Sync
    sync_on_search: bool = True

    def get_workspace(self) -> Path:
        return Path(self.workspace_root)

    def get_memory_dir(self) -> Path:
        return self.get_workspace() / "memory"

    def get_skills_dir(self) -> Path:
        return self.get_workspace() / "skills"

    def get_db_path(self) -> Path:
        return self.get_memory_dir() / "long-term" / "index.db"


# Singleton
_mem_cfg: Optional[MemoryConfig] = None


def get_memory_config() -> MemoryConfig:
    global _mem_cfg
    if _mem_cfg is None:
        _mem_cfg = MemoryConfig()
    return _mem_cfg
