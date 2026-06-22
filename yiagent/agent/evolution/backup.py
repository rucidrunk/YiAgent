"""
Backup manager — snapshots workspace files before evolution mutations.

Every evolution pass backs up affected files so changes can be undone.
Backups are stored under memory/.evolution_backups/<backup_id>/.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from yiagent.common.log import logger


def create_backup(workspace_dir: Path, files: List[Path]) -> str:
    """Copy files into a timestamped backup directory. Returns the backup_id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_id = f"evo-{ts}"
    backup_root = workspace_dir / "memory" / ".evolution_backups" / backup_id
    backup_root.mkdir(parents=True, exist_ok=True)

    for src in files:
        if not src.exists():
            continue
        try:
            rel = src.relative_to(workspace_dir)
        except ValueError:
            rel = Path(src.name)

        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            logger.warning(f"[Evolution] Backup failed for {src}: {e}")

    logger.debug(f"[Evolution] Backup {backup_id}: {len(files)} files")
    return backup_id
