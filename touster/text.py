"""Shared text helpers: normalisation, code fences and JSON object parsing."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

FENCE_BLOCK_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    no_punct = "".join(ch for ch in text.lower() if not unicodedata.category(ch).startswith("P"))
    return " ".join(no_punct.split())


def strip_code_fence(text: str) -> str:
    """Return the first fenced code block's body, or the whole text when there is no fence."""
    match = FENCE_BLOCK_RE.search(text)
    return match.group(1) if match else text


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Strictly parse text (optionally inside a code fence) as a JSON object; None if it is not one."""
    try:
        value = json.loads(strip_code_fence(text).strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def find_json_object(text: str) -> dict[str, Any] | None:
    """Parse the span from the first "{" to the last "}" as a JSON object, allowing prose around it; None if invalid."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
