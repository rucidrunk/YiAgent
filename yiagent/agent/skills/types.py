"""Skill data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillMetadata:
    """Frontmatter metadata from a SKILL.md file."""
    name: str = ""
    description: str = ""
    primary_env: Optional[str] = None
    default_enabled: bool = True
    skill_key: str = ""


@dataclass
class Skill:
    """A loaded skill with its file path and parsed frontmatter."""
    name: str
    description: str
    source: str  # "builtin" | "custom"
    file_path: str
    base_dir: str
    metadata: Optional[SkillMetadata] = None


@dataclass
class SkillEntry:
    """A skill entry held by the SkillManager."""
    skill: Skill
    metadata: Optional[SkillMetadata] = None


@dataclass
class SkillSnapshot:
    """Snapshot of available skills for a specific run."""
    prompt: str
    skills: List[Dict[str, str]]  # [{name, primary_env}, ...]
    resolved_skills: List[Skill] = field(default_factory=list)
    version: Optional[int] = None
