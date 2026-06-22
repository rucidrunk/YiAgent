"""Skill filtering helpers."""

from typing import Any, Dict, Optional

from yiagent.agent.skills.types import SkillEntry


def should_include_skill(entry: SkillEntry, config: Optional[Dict[str, Any]] = None) -> bool:
    """Check if a skill's requirements are met.

    Extension point: override to add env-var / API-key gating for specific skills.
    """
    meta = entry.metadata
    if meta is None:
        return True

    env_var = meta.primary_env
    if not env_var:
        return True

    import os
    value = os.environ.get(env_var)
    if config:
        value = value or config.get(env_var) or config.get(env_var.lower())
    return bool(value)


def get_missing_requirements(entry: SkillEntry) -> Dict[str, str]:
    """Return a dict of {env_var: hint} for unmet skill requirements."""
    missing: Dict[str, str] = {}
    meta = entry.metadata
    if meta and meta.primary_env:
        import os
        if not os.environ.get(meta.primary_env):
            missing[meta.primary_env] = f"Set {meta.primary_env} environment variable"
    return missing
