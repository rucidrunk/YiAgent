"""General-purpose utility functions."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict


def expand_path(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path)))


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# CJK character ranges for text analysis
_CJK_RANGES = (
    r'　-ヿ'
    r'㐀-鿿'
    r'가-힯'
    r'豈-﫿'
    r'\U00020000-\U0002fa1f'
)
_RE_CJK = re.compile(f'[{_CJK_RANGES}]')


def contains_cjk(text: str) -> bool:
    return bool(_RE_CJK.search(text))


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK chars ~1.5 tokens, ASCII ~0.25 tokens/char."""
    if not text:
        return 0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ascii_count = len(text) - non_ascii
    return int(non_ascii * 1.5 + ascii_count * 0.25) + 1


def safe_json_dumps(obj: object, max_len: int = 50000) -> str:
    """JSON dump with length cap for tool results.

    Never truncates AFTER serialisation — that produces invalid JSON.
    Instead, if content is too large, builds a clean JSON envelope with
    the truncated original content so the result is always valid JSON.
    """
    import json

    margin = 200  # reserve for truncation marker + envelope overhead

    if isinstance(obj, str):
        if len(obj) <= max_len:
            return json.dumps(obj, ensure_ascii=False)
        return json.dumps(
            obj[: max_len - margin]
            + f"\n\n[Truncated: {len(obj)} total chars]",
            ensure_ascii=False,
        )

    # dict / list / other: serialise, and if too long, wrap in a
    # clean {truncated, preview, original_length} envelope.
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= max_len:
        return s

    preview = s[: max_len - margin] if isinstance(obj, dict) else ""
    envelope = {
        "truncated": True,
        "original_length": len(s),
        "preview": preview,
    }
    return json.dumps(envelope, ensure_ascii=False)


def truncate_str(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
