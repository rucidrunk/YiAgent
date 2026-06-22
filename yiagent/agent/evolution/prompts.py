"""
Evolution prompts — system prompt and user message for the review agent.

The review agent is given the full agent context PLUS an evolution task
on top, so it follows user preferences AND knows its job.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

EVOLUTION_MARKER = "[EVOLUTION]"
SILENT_TOKEN = "[SILENT]"

EVOLUTION_SYSTEM_PROMPT = f"""\
## Self-Evolution Task

You are the self-evolution subsystem. Review the conversation transcript
below and decide if any lasting knowledge should be recorded.

**Rules:**
- Be CONSERVATIVE. Most conversations produce NO lasting knowledge.
- If there is nothing to record, respond EXACTLY with: {SILENT_TOKEN}
- Only write when the user clearly states a preference, decision, plan,
  or fact worth remembering long-term.
- Write to the workspace (MEMORY.md for core facts, memory/<date>.md for
  daily notes). Only edit files inside the workspace.
- Never edit built-in skills or knowledge outside the workspace.
- Keep entries concise — 2-5 lines per fact.
- Use the same language as the conversation.

**Allowed tools:** read, write, edit, ls, bash, memory_search, memory_get
**Forbidden:** web_search, web_fetch, browser, scheduler, send
"""


def build_review_user_message(
    transcript: str,
    protected_skills: List[str] = None,
) -> str:
    """Build the user message for the review agent."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        f"Review the following conversation transcript and record any lasting knowledge.",
        f"Today is {today}.",
    ]
    if protected_skills:
        parts.append(
            f"Protected skills (NEVER edit): {', '.join(sorted(protected_skills))}"
        )
    parts.append(f"\n--- Transcript ---\n{transcript}\n--- End Transcript ---")
    return "\n".join(parts)
