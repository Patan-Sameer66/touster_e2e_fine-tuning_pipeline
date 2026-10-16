"""Shared text helpers: normalisation, code fences, strict and lenient JSON parsing."""

from __future__ import annotations

from touster.text import find_json_object, normalize_text, parse_json_object, strip_code_fence


def test_normalize_text() -> None:
    """Case, unicode punctuation and extra whitespace are dropped."""
    assert normalize_text("  Hello,   World! ") == "hello world"
    assert normalize_text("«Positive»…") == "positive"
    assert normalize_text("a+b = c") == "a+b = c"
    assert normalize_text("") == ""


def test_strip_code_fence() -> None:
    """The first fenced block's body is returned; unfenced text comes back unchanged."""
    assert strip_code_fence("Here:\n```python\nx = 1\n```\n```\ny\n```") == "x = 1\n"
    assert strip_code_fence("  plain text ") == "  plain text "


def test_parse_json_object_is_strict() -> None:
    """Bare or fenced objects parse; prose around them, lists and broken JSON do not."""
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure: {"a": 1}') is None
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("name: Bob") is None


def test_find_json_object_is_lenient() -> None:
    """Objects are found inside prose or fences; anything else is None."""
    assert find_json_object('{"a": 1}') == {"a": 1}
    assert find_json_object('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert find_json_object("[1, 2]") is None
    assert find_json_object('{"a": 1') is None
    assert find_json_object("") is None
