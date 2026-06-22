"""
DryRunInterceptor — four-stage write-safety state machine.

Prevents agent hallucination from corrupting filesystem/config by
intercepting every write-tool call and routing it through:

  ① SHADOW → ② ASSERT → ③ HITL → ④ COMMIT

Design:
  - Write tools (write, edit, bash) are always intercepted
  - SHADOW: validate path safety, check no system-path traversal, no credential overwrite
  - ASSERT: run safety rules against ChangePlan
  - HITL: escalate critical operations for human approval
  - COMMIT: only after gate passes, execute real write

Safety rules are in safety_rules.json (configurable patterns + regex).
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from yiagent.common.config import conf
from yiagent.common.log import logger


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ChangePlan:
    """Describes what a write operation will change."""
    tool_name: str
    affected_paths: List[str] = field(default_factory=list)
    diff_preview: str = ""
    risk_level: RiskLevel = RiskLevel.SAFE
    requires_approval: bool = False
    blocked: bool = False
    block_reason: str = ""


@dataclass
class DryRunResult:
    """Result of the four-stage pipeline."""
    allowed: bool
    change_plan: ChangePlan
    result: Any = None
    stage_failed: Optional[str] = None  # which stage blocked it


# ---------------------------------------------------------------------------
# Default safety rules
# ---------------------------------------------------------------------------

DEFAULT_BLOCKED_PATHS = [
    # System paths
    "/etc/", "/proc/", "/sys/", "/boot/", "/dev/",
    "~/.ssh/", "~/.gnupg/",
    # Credential files
    ".env", ".envrc", "credentials.json", "config.json", "secrets.yaml",
    "~/.aws/", "~/.gcloud/", "~/.kube/",
]

DEFAULT_WARNING_PATHS = [
    # System configs that are worth double-checking
    "~/.bashrc", "~/.zshrc", "~/.profile", "~/.gitconfig",
    "/usr/local/",
]

DEFAULT_BULK_DELETE_THRESHOLD = 10  # more than N files → warning


class DryRunInterceptor:
    """
    Write-safety interceptor for tool execution.

    Usage:
        interceptor = DryRunInterceptor()
        result = await interceptor.intercept("write", {"path": "AGENT.md", "content": "..."})
        if result.allowed:
            # execute real write
    """

    def __init__(self, workspace_root: Optional[str] = None,
                 rules_config: Optional[Dict[str, Any]] = None):
        # Resolve workspace root ONCE and store both the raw path (for .. checks)
        # and the real path (for symlink escape detection).
        self._workspace_path = Path(workspace_root or conf().get("workspace_root", "~/yiagent")).expanduser()
        self._workspace_real = self._workspace_path.resolve()
        self._rules = rules_config or self._load_rules()
        self._blocked_patterns = self._rules.get("blocked_paths", DEFAULT_BLOCKED_PATHS)
        self._warning_patterns = self._rules.get("warning_paths", DEFAULT_WARNING_PATHS)
        self._bulk_delete_threshold = self._rules.get("bulk_delete_threshold", DEFAULT_BULK_DELETE_THRESHOLD)

    def _load_rules(self) -> Dict[str, Any]:
        """Load safety_rules.json from workspace."""
        rules_path = self._workspace_path / "safety_rules.json"
        if rules_path.exists():
            import json
            try:
                return json.loads(rules_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Four-stage pipeline
    # ------------------------------------------------------------------

    async def intercept(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        on_hitl: Optional[Callable] = None,
    ) -> DryRunResult:
        """
        Run through the four-stage state machine.

        Args:
            tool_name: Name of the write tool.
            arguments: Tool arguments.
            on_hitl: Optional async callback(change_plan) → bool for HITL approval.
        """
        # ① SHADOW
        change_plan = self._stage_shadow(tool_name, arguments)
        if change_plan.blocked:
            return DryRunResult(
                allowed=False, change_plan=change_plan,
                stage_failed=f"SHADOW: {change_plan.block_reason}",
            )

        # ② ASSERT
        passed, assert_reason = self._stage_assert(change_plan)
        if not passed:
            change_plan.blocked = True
            change_plan.block_reason = assert_reason
            if change_plan.risk_level == RiskLevel.CRITICAL:
                # ③ HITL for critical ops
                if on_hitl:
                    try:
                        approved = await on_hitl(change_plan)
                        if not approved:
                            return DryRunResult(
                                allowed=False, change_plan=change_plan,
                                stage_failed="HITL: user rejected",
                            )
                    except Exception as e:
                        return DryRunResult(
                            allowed=False, change_plan=change_plan,
                            stage_failed=f"HITL: {e}",
                        )
                else:
                    return DryRunResult(
                        allowed=False, change_plan=change_plan,
                        stage_failed="HITL: approval required but no handler",
                    )
            else:
                return DryRunResult(
                    allowed=False, change_plan=change_plan,
                    stage_failed=f"ASSERT: {assert_reason}",
                )

        # ④ COMMIT — allowed
        return DryRunResult(allowed=True, change_plan=change_plan)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_shadow(self, tool_name: str, arguments: Dict[str, Any]) -> ChangePlan:
        """① SHADOW: Validate paths and build ChangePlan without executing."""
        plan = ChangePlan(tool_name=tool_name)

        path = arguments.get("path", "") or arguments.get("file_path", "") or ""
        command = arguments.get("command", "") or ""

        if path:
            try:
                resolved = self._resolve_path(path)
            except ValueError as e:
                plan.blocked = True
                plan.block_reason = str(e)
                plan.risk_level = RiskLevel.CRITICAL
                return plan
            plan.affected_paths = [str(resolved)]

            # Check blocked paths
            for pattern in self._blocked_patterns:
                if self._path_matches(resolved, pattern):
                    plan.blocked = True
                    plan.block_reason = f"Blocked path pattern: {pattern}"
                    plan.risk_level = RiskLevel.CRITICAL
                    return plan

            # Check warning paths
            for pattern in self._warning_patterns:
                if self._path_matches(resolved, pattern):
                    plan.risk_level = RiskLevel.WARNING
                    plan.requires_approval = True

            # Check workspace boundary
            try:
                rel = resolved.relative_to(self._workspace_path)
            except ValueError:
                plan.blocked = True
                plan.block_reason = (
                    f"Path '{path}' is outside workspace '{self._workspace_path}'"
                )
                plan.risk_level = RiskLevel.CRITICAL
                return plan

        if command:
            # Check for dangerous bash commands
            dangerous = self._check_dangerous_command(command)
            if dangerous:
                plan.blocked = True
                plan.block_reason = dangerous
                plan.risk_level = RiskLevel.CRITICAL

        return plan

    def _stage_assert(self, plan: ChangePlan) -> tuple:
        """② ASSERT: Run safety rules against the ChangePlan."""
        # No paths = nothing to assert
        if not plan.affected_paths:
            return (True, "")

        for p in plan.affected_paths:
            p_str = str(p)

            # Rule 1: no credential file overwrite
            filename = Path(p_str).name.lower()
            if filename in {".env", "credentials.json", "secrets.yaml", "config.json"}:
                plan.risk_level = RiskLevel.CRITICAL
                plan.requires_approval = True
                return (False, f"Credential file '{filename}' requires approval")

        return (True, "")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace.

        Checks for ``..`` traversal BEFORE resolution and detects symlink
        escapes by comparing the real (os-level) resolved path against the
        stored real workspace root.
        """
        raw = Path(path)

        # Block .. traversal before resolve() eliminates the evidence
        if ".." in raw.parts:
            raise ValueError(f"Path traversal blocked: {path}")

        p = raw if raw.is_absolute() else (self._workspace_path / raw)
        real = p.resolve()

        # Symlink escape: a symlink inside the workspace pointing outside
        # would cause resolve() to land outside _workspace_real.
        try:
            real.relative_to(self._workspace_real)
        except ValueError:
            raise ValueError(f"Path outside workspace (symlink escape?): {path}")

        return real

    def _path_matches(self, resolved: Path, pattern: str) -> bool:
        """Check if a resolved path matches a pattern."""
        p_str = str(resolved)
        # Expand ~ in pattern
        pattern = os.path.expanduser(pattern)
        if p_str.startswith(pattern):
            return True
        if pattern.endswith("/"):
            return p_str.startswith(pattern) or p_str == pattern[:-1]
        # Regex match
        try:
            if re.search(pattern, p_str):
                return True
        except re.error:
            pass
        return False

    def _check_dangerous_command(self, command: str) -> Optional[str]:
        """Detect dangerous shell patterns."""
        # Forbidden: rm -rf on root, home, parent, or wildcard destructions
        if re.search(r"rm\s+-rf\s+(/|~|\.\.|\.\s|\.$|\*\s|\*$)", command):
            return "Dangerous delete: rm -rf on root/home/parent/wildcard path"

        # Forbidden: raw SQL DELETE/DROP without confirmation
        if re.search(r"(DELETE\s+FROM|DROP\s+(TABLE|DATABASE))", command, re.IGNORECASE):
            return "Dangerous SQL: DELETE/DROP detected"

        # Forbidden: curl/piped execution
        if re.search(r"curl.*\|.*(ba)?sh", command):
            return "Dangerous pipe: curl to shell execution"

        # Warning: bulk deletes
        if re.search(r"rm\s+-rf", command):
            # Count files affected
            parts = command.split()
            if len(parts) > 3:
                return "Bulk delete detected; verify file count"

        return None
