"""Self-evolution configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from yiagent.common.config import conf


@dataclass
class EvolutionConfig:
    enabled: bool = True
    idle_seconds: int = 300       # scan session if idle >= 5 min
    min_turns: int = 3            # min user turns since last evolution
    max_steps: int = 20           # max tool-call steps for review agent
    max_concurrent: int = 2       # cap concurrent evolution passes
    scan_interval: int = 60       # seconds between scans


def get_evolution_config() -> EvolutionConfig:
    cfg = conf()
    return EvolutionConfig(
        enabled=cfg.get("evolution_enabled", True),
        idle_seconds=cfg.get("evolution_idle_seconds", 300),
        min_turns=cfg.get("evolution_min_turns", 3),
        max_steps=cfg.get("evolution_max_steps", 20),
        max_concurrent=cfg.get("evolution_max_concurrent", 2),
        scan_interval=60,
    )
