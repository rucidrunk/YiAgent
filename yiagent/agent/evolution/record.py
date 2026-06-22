"""
Evolution record — appends evolution results to the evolution log.

Each evolution that produces actual changes is logged with backup_id
for undo support, and the summary is written to the daily memory file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from yiagent.common.log import logger


def append_session_evolution(
    workspace_dir: Path,
    summary: str,
    backup_id: str = "",
    user_id: Optional[str] = None,
) -> None:
    """Append evolution summary to daily memory file."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%H:%M")

    if user_id:
        daily = workspace_dir / "memory" / "users" / user_id / f"{today}.md"
    else:
        daily = workspace_dir / "memory" / f"{today}.md"
    daily.parent.mkdir(parents=True, exist_ok=True)

    entry = f"\n\n## Evolution ({ts})\n{summary}\n"
    if backup_id:
        entry += f"(backup_id: {backup_id})\n"

    try:
        with open(daily, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        logger.warning(f"[Evolution] Failed to write {daily}: {e}")
