"""Format skill entries for system prompt injection."""

from typing import Dict, List

from yiagent.agent.skills.types import SkillEntry


def format_skill_entries_for_prompt(entries: List[SkillEntry]) -> str:
    """Format a list of SkillEntry as XML for the system prompt."""
    if not entries:
        return ""

    lines = ["<available_skills>"]
    for entry in entries:
        skill = entry.skill
        lines.append(f"<skill>")
        lines.append(f"  <name>{skill.name}</name>")
        lines.append(f"  <description>{skill.description}</description>")
        lines.append(f"  <location>{skill.file_path}</location>")
        if entry.metadata and entry.metadata.primary_env:
            lines.append(f"  <primary_env>{entry.metadata.primary_env}</primary_env>")
        lines.append(f"</skill>")
    lines.append("</available_skills>")

    return "\n".join(lines)
