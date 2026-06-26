"""
Global configuration with environment-aware loading.

Load order: defaults → config.json → env vars (YIAGENT_ prefix)
All access is thread-safe via the process-wide singleton.
"""

from __future__ import annotations

import json

# Auto-load .env file at import time (idempotent)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_CONFIG: Dict[str, Any] = {
    # --- Workspace ---
    "workspace_root": "~/yiagent",
    "memory_dir": "memory",
    "skills_dir": "skills",
    "knowledge_dir": "knowledge",

    # --- Redis (short-term memory) ---
    "redis_url": "redis://127.0.0.1:6379/0",
    "redis_max_connections": 50,
    "redis_socket_timeout": 5.0,
    "redis_session_ttl": 86400,       # 24h
    "redis_context_ttl": 7200,         # 2h
    "redis_lock_ttl": 120,             # 2min

    # --- PostgreSQL + pgvector (long-term memory) ---
    "pg_dsn": "postgresql://yiagent:yiagent@127.0.0.1:5432/yiagent",
    "pg_min_connections": 5,
    "pg_max_connections": 50,
    "pg_search_limit": 20,
    "pg_vector_dim": 1536,

    # --- Embedding ---
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "embedding_cache_ttl": 3600,

    # --- Hybrid search ---
    "rrf_k": 60,                       # RRF constant
    "min_score": 0.005,
    "max_results": 10,
    "temporal_half_life_days": 30.0,

    # --- Chunking ---
    "chunk_max_tokens": 500,
    "chunk_overlap_tokens": 50,

    # --- Agent ---
    "agent_max_steps": 50,
    "agent_max_context_turns": 30,
    "agent_max_context_tokens": 50000,
    "agent_name": "YiAgent",
    "agent_language": "zh",

    # --- Context flush ---
    "flush_high_watermark": 0.8,
    "flush_max_workers": 4,
    "flush_queue_maxsize": 256,

    # --- Evolution ---
    "evolution_enabled": True,
    "evolution_idle_seconds": 300,     # 5 min
    "evolution_min_turns": 3,
    "evolution_max_steps": 20,
    "evolution_max_concurrent": 2,

    # --- MCP ---
    "mcp_servers": [],

    # --- Model ---
    "model_provider": "openai",
    "model_name": "gpt-4o",
    "model_temperature": 0.0,
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",

    # --- Observability ---
    "log_level": "INFO",
    "log_format": "json",
}

_config: Optional[Dict[str, Any]] = None
_config_lock = threading.Lock()


def _expand_path(path: str) -> Path:
    """Expand ~ and env vars in a path."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load config from file + env, merge over defaults. Idempotent, thread-safe."""
    global _config
    with _config_lock:
        if _config is not None:
            return _config

        merged = dict(_DEFAULT_CONFIG)

        # Layer 1: config file
        if config_path is None:
            config_path = os.environ.get("YIAGENT_CONFIG", "")
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
                merged.update(file_cfg)
            except FileNotFoundError:
                pass
            except Exception as e:
                import logging
                logging.warning(f"[Config] Failed to load {config_path}: {e}")

        # Layer 2: env vars — first, all keys from defaults
        for key in _DEFAULT_CONFIG:
            env_key = f"YIAGENT_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    merged[key] = json.loads(env_val)
                except json.JSONDecodeError:
                    merged[key] = env_val

        # Then, any other YIAGENT_* env vars not in defaults
        for env_key, env_val in os.environ.items():
            if env_key.startswith("YIAGENT_") and env_val:
                key = env_key[8:].lower()  # strip "YIAGENT_" prefix
                if key not in merged:
                    try:
                        merged[key] = json.loads(env_val)
                    except json.JSONDecodeError:
                        merged[key] = env_val

        _config = merged
        return _config


def conf() -> Dict[str, Any]:
    """Return the current config dict (thread-safe snapshot)."""
    global _config
    if _config is None:
        return load_config()
    return _config


def get_workspace() -> Path:
    return _expand_path(conf()["workspace_root"])


def get_redis_url() -> str:
    return conf()["redis_url"]


def get_pg_dsn() -> str:
    return conf()["pg_dsn"]
