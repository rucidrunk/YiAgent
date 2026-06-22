"""
SkillLoader — scans skill directories and parses SKILL.md frontmatter.

Each skill is a directory containing a SKILL.md file with YAML-like
frontmatter delimited by --- markers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

from yiagent.common.log import logger
from yiagent.agent.skills.types import Skill, SkillEntry, SkillMetadata


class SkillLoader:
    """Loads skills from builtin and custom directories."""

    def load_all_skills(
        self,
        builtin_dir: str,
        custom_dir: str,
    ) -> Dict[str, SkillEntry]:
        """Scan both directories and return {name: SkillEntry}."""
        skills: Dict[str, SkillEntry] = {}

        if os.path.isdir(builtin_dir):
            self._load_dir(builtin_dir, "builtin", skills)

        if os.path.isdir(custom_dir):
            self._load_dir(custom_dir, "custom", skills)

        return skills

    def _load_dir(self, dir_path: str, source: str, skills: Dict[str, SkillEntry]) -> None:
        for entry in os.scandir(dir_path):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                skill = self._load_skill(Path(entry.path), source)
                if skill:
                    # Custom overrides builtin with same name
                    skills[skill.name] = SkillEntry(
                        skill=skill,
                        metadata=skill.metadata,
                    )
            except Exception as e:
                logger.warning(f"[SkillLoader] Failed to load {entry.path}: {e}")

    def _load_skill(self, skill_dir: Path, source: str) -> Optional[Skill]:
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None

        content = skill_md.read_text(encoding="utf-8")
        meta = self._parse_frontmatter(content)

        return Skill(
            name=meta.name or skill_dir.name,
            description=meta.description or "",
            source=source,
            file_path=str(skill_md),
            base_dir=str(skill_dir),
            metadata=meta,
        )

    @staticmethod
    def _parse_frontmatter(content: str) -> SkillMetadata:
        """Extract YAML-like frontmatter between --- delimiters."""
        meta = SkillMetadata()

        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return meta

        front = match.group(1)
        for line in front.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip().strip("\"'")

            if key == "name":
                meta.name = val
            elif key == "description":
                meta.description = val
            elif key == "primary_env":
                meta.primary_env = val
            elif key == "default_enabled":
                meta.default_enabled = val.lower() != "false"
            elif key == "skill_key":
                meta.skill_key = val

        return meta
