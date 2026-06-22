"""
SkillManager — lifecycle management for agent skills.

Orchestrates loading from disk, enable/disable persistence via
skills_config.json, filtering by requirements, and formatting for
system prompt injection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from yiagent.common.log import logger
from yiagent.agent.skills.types import SkillEntry, SkillSnapshot
from yiagent.agent.skills.loader import SkillLoader
from yiagent.agent.skills.formatter import format_skill_entries_for_prompt
from yiagent.agent.skills.config import should_include_skill, get_missing_requirements

SKILLS_CONFIG_FILE = "skills_config.json"


class SkillManager:
    """Manages skill lifecycle: load, filter, format, enable/disable."""

    def __init__(
        self,
        builtin_dir: Optional[str] = None,
        custom_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.builtin_dir = builtin_dir or os.path.join(project_root, "skills")
        self.custom_dir = custom_dir or ""
        self.config = config or {}
        if self.custom_dir:
            self._config_path = os.path.join(self.custom_dir, SKILLS_CONFIG_FILE)
        else:
            self._config_path = ""

        self.skills: Dict[str, SkillEntry] = {}
        self.skills_config: Dict[str, dict] = {}
        self.loader = SkillLoader()

        self.refresh_skills()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def refresh_skills(self) -> None:
        """Reload all skills from disk and sync the config file."""
        self.skills = self.loader.load_all_skills(
            builtin_dir=self.builtin_dir,
            custom_dir=self.custom_dir,
        )
        self._sync_config()
        logger.debug(f"[SkillManager] Loaded {len(self.skills)} skills")

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config_file(self) -> Dict[str, dict]:
        if not self._config_path or not os.path.exists(self._config_path):
            return {}
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"[SkillManager] Failed to load {SKILLS_CONFIG_FILE}: {e}")
            return {}

    def _save_config_file(self) -> None:
        if not self._config_path:
            return
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self.skills_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[SkillManager] Failed to save {SKILLS_CONFIG_FILE}: {e}")

    def _sync_config(self) -> None:
        saved = self._load_config_file()
        merged: Dict[str, dict] = {}

        for name, entry in self.skills.items():
            skill = entry.skill
            prev = saved.get(name, {})
            enabled = prev.get("enabled", True) if name in saved else (
                entry.metadata.default_enabled if entry.metadata else True
            )
            merged[name] = {
                "name": name,
                "description": skill.description,
                "source": prev.get("source") or skill.source,
                "enabled": enabled,
                "category": prev.get("category", "skill"),
            }

        if merged != saved:
            self.skills_config = merged
            self._save_config_file()

    def is_skill_enabled(self, name: str) -> bool:
        entry = self.skills_config.get(name)
        return entry.get("enabled", True) if entry else True

    def set_skill_enabled(self, name: str, enabled: bool) -> None:
        if name not in self.skills_config:
            raise ValueError(f"skill '{name}' not found")
        self.skills_config[name]["enabled"] = enabled
        self._save_config_file()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_skill(self, name: str) -> Optional[SkillEntry]:
        return self.skills.get(name)

    def list_skills(self) -> List[SkillEntry]:
        return list(self.skills.values())

    def filter_skills(
        self,
        skill_filter: Optional[List[str]] = None,
        include_disabled: bool = False,
    ) -> List[SkillEntry]:
        """Return eligible (enabled + requirements met) skills."""
        entries = list(self.skills.values())
        entries = [e for e in entries if should_include_skill(e, self.config)]

        if skill_filter:
            names = set(skill_filter)
            entries = [e for e in entries if e.skill.name in names]

        if not include_disabled:
            entries = [e for e in entries if self.is_skill_enabled(e.skill.name)]

        return entries

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_skills_prompt(
        self,
        skill_filter: Optional[List[str]] = None,
    ) -> str:
        """Build the <available_skills> block for system prompt injection."""
        eligible = self.filter_skills(skill_filter=skill_filter, include_disabled=False)
        return format_skill_entries_for_prompt(eligible)

    def build_skill_snapshot(
        self,
        skill_filter: Optional[List[str]] = None,
        version: Optional[int] = None,
    ) -> SkillSnapshot:
        entries = self.filter_skills(skill_filter=skill_filter, include_disabled=False)
        prompt = format_skill_entries_for_prompt(entries)
        skills_info = [
            {
                "name": e.skill.name,
                "primary_env": e.metadata.primary_env if e.metadata else None,
            }
            for e in entries
        ]
        return SkillSnapshot(
            prompt=prompt,
            skills=skills_info,
            resolved_skills=[e.skill for e in entries],
            version=version,
        )
