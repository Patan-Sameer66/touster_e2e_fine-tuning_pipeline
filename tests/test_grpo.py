"""CPU tests for the GRPO programmatic reward functions."""

from __future__ import annotations

from typing import Any

import pytest

from touster.train.grpo import (
    as_test_lines,
    code_reward,
    completion_text,
    exact_match_reward,
    grpo_rows,
    json_match_reward,
    reward_functions,
)


def chat(text: str) -> list[dict[str, str]]:
    """A conversational completion as TRL passes it."""
    return [{"role": "assistant", "content": text}]


def test_completion_text_handles_both_formats() -> None:
    """Plain-string and chat-format completions both give their text."""
    assert completion_text("plain") == "plain"
    assert completion_text(chat("hi")) == "hi"
    assert completion_text([]) == ""


def test_exact_match_normalises() -> None:
    """Case, whitespace and trailing punctuation are ignored when matching."""
    rewards = exact_match_reward(
        prompts=["p"] * 3, completions=[chat(" Positive. "), "negative", chat("neutral")],
        reference=["positive", "positive", "Neutral"], trainer_state=None,
    )
    assert rewards == [1.0, 0.0, 1.0]


def test_json_reward_validity_and_fields() -> None:
    """Valid JSON scores by matching fields; invalid JSON scores 0."""
    ref = '{"name": "Ana", "age": 30}'
    rewards = json_match_reward(
        completions=['```json\n{"name": "Ana", "age": 30}\n```', '{"name": "Ana", "age": 31}', "not json"],
        reference=[ref, ref, ref],
    )
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(0.6)
    assert rewards[2] == 0.0


def test_code_reward_compile_and_tests() -> None:
    """Passing code scores 1, failing code 0.25, broken code 0, code without tests 1."""
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    wrong = "def add(a, b):\n    return a - b"
    broken = "def add(a, b) return"
    tests = [["assert add(1, 2) == 3"], ["assert add(1, 2) == 3"], [], []]
    rewards = code_reward(completions=[good, wrong, broken, wrong], tests=tests)
    assert rewards == [1.0, 0.25, 0.0, 1.0]


def test_code_reward_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Code that hangs past the timeout gets only the compile reward."""
    from touster.train import grpo

    monkeypatch.setattr(grpo, "CODE_TEST_TIMEOUT_SECONDS", 1)
    looping = "def f():\n    while True:\n        pass"
    assert grpo.code_score(looping, ["f()"]) == grpo.CODE_COMPILE_ONLY_REWARD


def test_as_test_lines() -> None:
    """Test metadata as a string or list becomes non-empty assert lines."""
    assert as_test_lines(None) == []
    assert as_test_lines("assert 1\n\nassert 2") == ["assert 1", "assert 2"]
    assert as_test_lines(["assert 1", " "]) == ["assert 1"]


def test_reward_functions_by_task(make_cfg: Any) -> None:
    """Each task type gets its reward; one without a reward points to DPO."""
    assert reward_functions(make_cfg(task_type="classification")) == [exact_match_reward]
    assert reward_functions(make_cfg(task_type="qa")) == [exact_match_reward]
    assert reward_functions(make_cfg(task_type="extraction")) == [json_match_reward]
    assert reward_functions(make_cfg(task_type="code")) == [code_reward]
    with pytest.raises(ValueError, match="loop_method='dpo'"):
        reward_functions(make_cfg(task_type="summarisation"))


def test_grpo_rows_carry_reference_and_tests() -> None:
    """GRPO rows carry the chat prompt, reference answer and tests."""
    rows = grpo_rows([{"prompt": "p", "response": "r", "metadata": {"tests": "assert 1"}}], "sys")
    assert rows == [{
        "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "p"}],
        "reference": "r", "tests": ["assert 1"],
    }]
