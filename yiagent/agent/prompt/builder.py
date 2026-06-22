"""
PromptBuilder — constructs the full system prompt from modular sections.

Section order (by importance):
  1. Tooling        — available tools and call style
  2. Skills         — skill discovery instructions
  3. Memory         — recall + write guidance
  4. Knowledge      — structured knowledge base
  5. Workspace      — directory layout and path rules
  6. User identity  — name, timezone (optional)
  7. Project context — AGENT.md, USER.md, RULE.md, MEMORY.md
  8. Runtime info   — current time, model, workspace
  9. Response lang  — reply in user's language

Extension point: subclasses can override _build_* methods to inject
custom sections without modifying the core.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from yiagent.common.config import conf
from yiagent.common.log import logger


@dataclass
class ContextFile:
    path: str
    content: str


# ---- public entry point ------------------------------------------------

def build_system_prompt(
    workspace_dir: str,
    language: str = "",
    tools: Optional[List[Any]] = None,
    context_files: Optional[List[ContextFile]] = None,
    skill_manager: Any = None,
    memory_manager: Any = None,
    runtime_info: Optional[Dict[str, Any]] = None,
    skill_filter: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Build the full system prompt. Returns a single string."""
    lang = language or conf().get("agent_language", "zh")

    if context_files is None:
        context_files = _load_context_files(workspace_dir)

    sections: List[str] = []

    if tools:
        sections.extend(_build_tooling(tools, lang))

    if skill_manager:
        sections.extend(_build_skills(skill_manager, tools, lang, skill_filter))

    if memory_manager and tools:
        sections.extend(_build_memory(memory_manager, tools, lang))

    sections.extend(_build_workspace(workspace_dir, lang))

    if context_files:
        sections.extend(_build_context_files(context_files, lang))

    if runtime_info:
        sections.extend(_build_runtime(runtime_info, lang))

    sections.extend(_build_response_language(lang))

    return "\n".join(sections)


# ---- context file loader -----------------------------------------------

def _load_context_files(workspace_dir: str) -> List[ContextFile]:
    """Read AGENT.md, USER.md, RULE.md, MEMORY.md from workspace."""
    files: List[ContextFile] = []
    ws = Path(workspace_dir)
    for name in ("AGENT.md", "USER.md", "RULE.md", "MEMORY.md"):
        path = ws / name
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
            # Cap MEMORY.md to last 200 lines / ~25KB to avoid bloating the prompt
            if name == "MEMORY.md":
                lines = content.split("\n")
                if len(lines) > 200:
                    content = (
                        "...(earlier memory omitted)...\n"
                        + "\n".join(lines[-200:])
                    )
                if len(content) > 25000:
                    content = content[-25000:]
            files.append(ContextFile(path=name, content=content))
        except Exception as e:
            logger.warning(f"[PromptBuilder] Failed to read {path}: {e}")
    return files


# ---- individual sections ------------------------------------------------

def _build_tooling(tools: List[Any], lang: str) -> List[str]:
    is_en = lang == "en"

    summaries: Dict[str, str] = {
        "read": "read file content" if is_en else "读取文件内容",
        "write": "create or overwrite a file" if is_en else "创建或覆盖文件",
        "edit": "make precise edits to a file" if is_en else "精确编辑文件",
        "ls": "list directory contents" if is_en else "列出目录内容",
        "bash": "run shell commands" if is_en else "执行shell命令",
        "web_search": "web search" if is_en else "网络搜索",
        "web_fetch": "fetch URL content" if is_en else "获取URL内容",
        "memory_search": "search memory" if is_en else "搜索记忆",
        "memory_get": "read memory content" if is_en else "读取记忆内容",
        "scheduler": "manage scheduled tasks" if is_en else "管理定时任务",
        "send": "send a local file to the user" if is_en else "发送本地文件给用户",
        "vision": "analyze images" if is_en else "分析图片内容",
    }

    order = [
        "read", "write", "edit", "ls", "bash",
        "web_search", "web_fetch",
        "memory_search", "memory_get",
        "scheduler", "send", "vision",
    ]

    available: Dict[str, str] = {}
    for t in tools:
        name = getattr(t, "name", str(t))
        available[name] = summaries.get(name, "")

    lines_out: List[str] = []
    for name in order:
        if name in available:
            s = available.pop(name)
            lines_out.append(f"- {name}: {s}" if s else f"- {name}")
    for name in sorted(available):
        s = available[name]
        lines_out.append(f"- {name}: {s}" if s else f"- {name}")

    if is_en:
        return [
            "## Tooling",
            "",
            "Available tools (case-sensitive):",
            "\n".join(lines_out),
            "",
            "Style:",
            "- For multi-step tasks or sensitive ops, briefly explain what you're doing.",
            "- Keep going until done, then report the result.",
            "- Redact secrets in replies.",
            "- Put URLs directly in reply text; the system renders them.",
            "",
        ]
    return [
        "## 工具系统",
        "",
        "可用工具（大小写敏感）:",
        "\n".join(lines_out),
        "",
        "调用风格：",
        "- 多步骤任务或敏感操作时简要说明当前在做什么。",
        "- 持续推进直到完成，然后向用户报告结果。",
        "- 回复中涉及密钥必须脱敏。",
        "- URL直接放在回复文本中即可，系统会自动渲染。",
        "",
    ]


def _build_skills(
    skill_manager: Any,
    tools: Optional[List[Any]],
    lang: str,
    skill_filter: Optional[List[str]],
) -> List[str]:
    if not skill_manager:
        return []

    read_name = "read"
    if tools:
        for t in tools:
            if getattr(t, "name", "").lower() == "read":
                read_name = t.name
                break

    if lang == "en":
        header = [
            "## Skills (mandatory)",
            "",
            "Before replying: scan the <description> of every skill below.",
            "",
            f"- If a skill matches the user's need: use `{read_name}` to read its SKILL.md, "
            "then strictly follow the instructions in the file.",
            "- Prefer skills over general tools when one matches.",
            "- If no skill applies: do not read any SKILL.md, use general tools.",
            "",
            f"Skills are not callable tools — use `{read_name}` to load them.",
            "",
        ]
    else:
        header = [
            "## 技能系统（mandatory）",
            "",
            "回复前扫描下方每个技能的描述。",
            "",
            f"- 匹配则用 `{read_name}` 读取其 SKILL.md，严格遵循其中指令。",
            "- 优先使用匹配的技能而非通用工具。",
            "- 无匹配则不读取 SKILL.md，使用通用工具。",
            "",
            f"技能不可直接调用 — 用 `{read_name}` 加载。",
            "",
        ]

    try:
        body = skill_manager.build_skills_prompt(skill_filter=skill_filter)
        if body:
            return header + [body.strip(), ""]
    except Exception as e:
        logger.warning(f"[PromptBuilder] Skills prompt failed: {e}")

    return header


def _build_memory(
    memory_manager: Any, tools: Optional[List[Any]], lang: str
) -> List[str]:
    has_tools = False
    if tools:
        names = {getattr(t, "name", "") for t in tools}
        has_tools = "memory_search" in names or "memory_get" in names
    if not has_tools:
        return []

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    if lang == "en":
        return [
            "## Memory",
            "",
            "### Recall (mandatory)",
            "",
            "When the user asks about past events, references prior decisions, or "
            "mentions preferences — **search memory before answering**.",
            "",
            "1. Unknown location → `memory_search`",
            "2. Known location → `memory_get` to read exact lines",
            "3. Search returns nothing → `memory_get` on last two days",
            "",
            f"Files: `MEMORY.md` (long-term, already loaded), "
            f"`memory/{today}.md` (daily), `knowledge/` (structured)",
            "",
            "### Writing",
            "",
            "Proactively write when the user says 'remember', 'always', 'never', "
            "'prefer', or shares important decisions/plans/preferences.",
            "",
            "- Core info → `MEMORY.md`",
            f"- Today's events → `memory/{today}`",
            "- Append → `edit` with empty oldText",
            "- Never write keys/tokens/secrets.",
            "",
        ]
    return [
        "## 记忆系统",
        "",
        "### Recall（mandatory）",
        "",
        "用户询问过往事件、引用之前决定、提到偏好时——**先检索记忆再回答**。",
        "",
        "1. 不确定位置 → `memory_search`",
        "2. 已知位置 → `memory_get` 精确读取",
        "3. search 无结果 → `memory_get` 读最近两天",
        "",
        f"文件：`MEMORY.md`（长期记忆，已加载），"
        f"`memory/{today}.md`（每日），`knowledge/`（知识库）",
        "",
        "### 写入",
        "",
        "遇到「记住」「以后」「总是」「不要」「偏好」等表达时主动写入。",
        "",
        f"- 核心信息 → `MEMORY.md`",
        f"- 当天进展 → `memory/{today}`",
        "- 追加 → `edit` 工具 oldText 留空",
        "- 禁止写入密钥/令牌等敏感信息",
        "",
    ]


def _build_workspace(workspace_dir: str, lang: str) -> List[str]:
    if lang == "en":
        return [
            "## Workspace",
            "",
            f"Working directory: `{workspace_dir}`",
            "",
            "- Relative paths are relative to this directory.",
            "- Use absolute paths for files outside the workspace.",
            "- Already loaded (no need to read again): `AGENT.md`, `USER.md`, "
            "`RULE.md`, `MEMORY.md`.",
            "- Be genuinely helpful; keep replies structured and focused.",
            "",
        ]
    return [
        "## 工作空间",
        "",
        f"工作目录：`{workspace_dir}`",
        "",
        "- 相对路径均相对于此目录。",
        "- 工作空间外的文件使用绝对路径。",
        "- 已自动加载（无需再读取）：`AGENT.md`、`USER.md`、`RULE.md`、`MEMORY.md`。",
        "- 做真正有帮助的助手，回复结构清晰、重点突出。",
        "",
    ]


def _build_context_files(files: List[ContextFile], lang: str) -> List[str]:
    if not files:
        return []

    has_agent = any("AGENT.md" in f.path for f in files)
    is_en = lang == "en"

    if is_en:
        lines = ["# Project Context", "", "The following files have been loaded:", ""]
        if has_agent:
            lines.append("`AGENT.md` is your soul file — follow the persona, "
                         "tone and settings it defines strictly.")
            lines.append("")
    else:
        lines = ["# 项目上下文", "", "以下文件已被加载：", ""]
        if has_agent:
            lines.append("`AGENT.md` 是你的灵魂文件——严格遵循其中的人格、语气和设定。")
            lines.append("")

    for f in files:
        lines.append(f"## {f.path}")
        lines.append("")
        lines.append(f.content)
        lines.append("")

    return lines


def _build_runtime(runtime_info: Dict[str, Any], lang: str) -> List[str]:
    is_en = lang == "en"
    time_label = "Current time" if is_en else "当前时间"

    lines = ["## Runtime Info" if is_en else "## 运行时信息", ""]

    # Dynamic time via callable
    get_time = runtime_info.get("_get_current_time")
    if callable(get_time):
        try:
            ti = get_time()
            lines.append(f"{time_label}: {ti['time']} {ti.get('weekday', '')} "
                         f"({ti.get('timezone', '')})")
            lines.append("")
        except Exception:
            pass
    elif runtime_info.get("current_time"):
        lines.append(f"{time_label}: {runtime_info['current_time']}")
        lines.append("")

    # Model / workspace tags
    parts = []
    model = runtime_info.get("model") or runtime_info.get("_get_model")
    if callable(model):
        try:
            parts.append(f"model={model()}")
        except Exception:
            pass
    elif model:
        parts.append(f"model={model}")

    if runtime_info.get("workspace"):
        parts.append(f"workspace={runtime_info['workspace']}")

    ch = runtime_info.get("channel", "")
    if ch and ch != "web":
        parts.append(f"channel={ch}")

    if parts:
        prefix = "Runtime: " if is_en else "运行时: "
        lines.append(prefix + " | ".join(parts))
        lines.append("")

    return lines


def _build_response_language(lang: str) -> List[str]:
    if lang == "en":
        return [
            "## Response language",
            "",
            "Reply in the same language as the user's input, "
            "unless the user explicitly asks for another language.",
            "",
        ]
    return [
        "## 回复语言",
        "",
        "默认使用与用户输入相同的语言回复，除非用户明确要求使用其他语言。",
        "",
    ]
