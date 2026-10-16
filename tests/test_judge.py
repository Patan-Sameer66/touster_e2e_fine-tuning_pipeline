"""Pairwise judge swap logic and 1-5 scoring, with a scripted fake LLM client."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from touster.eval.judge import candidate_wins, pairwise_win_rate, parse_score, parse_winner, score_answers


class ScriptedClient:
    """Returns pre-set texts in order and records how it was called."""

    def __init__(self, texts: list[str]) -> None:
        """Store the texts to return."""
        self.texts = texts
        self.calls: list[dict[str, Any]] = []

    def chat_many(self, batch: list, phase: str, model: str | None = None, on_done=None, **kwargs: Any) -> list:
        """Return one fake result per message list."""
        self.calls.append({"batch": batch, "phase": phase, "model": model, **kwargs})
        assert len(batch) == len(self.texts)
        return [SimpleNamespace(text=t) for t in self.texts]


A, B, TIE = '{"winner": "A"}', '{"winner": "B"}', '{"winner": "tie"}'


@pytest.mark.parametrize(
    ("text", "expected"),
    [(A, "A"), (B, "B"), (TIE, "tie"), ('{"winner": "Tie"}', "tie"), ('```json\n{"winner": "B"}\n```', "B"),
     ("A is better", None), ('{"winner": "C"}', None), ("[1]", None), ('{"verdict": "A"}', None)],
)
def test_parse_winner(text: str, expected: str | None) -> None:
    """Valid verdicts parse; anything else is None (never guessed)."""
    assert parse_winner(text) == expected


def test_candidate_wins_needs_both_orders() -> None:
    """Win only when the first order says A and the swapped order says B."""
    assert candidate_wins("A", "B")
    assert not candidate_wins("A", "A")
    assert not candidate_wins("tie", "B")
    assert not candidate_wins(None, "B")


def run_pairs(make_cfg, forward: list[str], swapped: list[str]) -> tuple[float, ScriptedClient]:
    """Run pairwise_win_rate on n dummy prompts with scripted verdicts."""
    cfg = make_cfg(judge_model="ollama_chat/judge")
    client = ScriptedClient(forward + swapped)
    n = len(forward)
    rate = pairwise_win_rate(client, cfg, ["p"] * n, ["cand"] * n, ["base"] * n, ["ref"] * n)
    return rate, client


def test_agreeing_orders_count_as_win(make_cfg) -> None:
    """Both orders prefer the candidate -> win."""
    rate, client = run_pairs(make_cfg, [A, A], [B, B])
    assert rate == 1.0
    call = client.calls[0]
    assert call["phase"] == "judge" and call["model"] == "ollama_chat/judge" and call["temperature"] == 0.0


def test_disagreement_tie_and_malformed_are_not_wins(make_cfg) -> None:
    """Disagreement, tie and malformed output never count as wins."""
    rate, _ = run_pairs(make_cfg, [A, TIE, "garbage", A], [A, TIE, B, B])
    assert rate == pytest.approx(0.25)


def test_swapped_prompt_puts_candidate_second(make_cfg) -> None:
    """The swapped batch shows the base answer as A and includes the task and reference."""
    _, client = run_pairs(make_cfg, [A], [B])
    forward, swapped = client.calls[0]["batch"]
    assert forward[1]["content"].index("cand") < forward[1]["content"].index("base")
    assert swapped[1]["content"].index("base") < swapped[1]["content"].index("cand")
    assert "Classify the sentiment" in forward[1]["content"] and "ref" in forward[1]["content"]


def test_length_mismatch_raises(make_cfg) -> None:
    """Misaligned inputs are an error, not a silent truncation."""
    with pytest.raises(ValueError):
        pairwise_win_rate(ScriptedClient([]), make_cfg(), ["p"], ["a"], [], ["r"])


def test_parse_score_range() -> None:
    """Scores must be numbers within 1..5."""
    assert parse_score('{"score": 4}') == 4.0
    assert parse_score('{"score": 4.5}') == 4.5
    assert parse_score('{"score": 0}') is None
    assert parse_score('{"score": true}') is None
    assert parse_score("four") is None


def test_score_answers_marks_malformed_as_nan(make_cfg) -> None:
    """Unparseable or out-of-range scores become NaN in place, keeping alignment."""
    client = ScriptedClient(['{"score": 4}', "not json", '{"score": 9}'])
    scores = score_answers(client, make_cfg(), "prompt", ["a", "b", "c"], None)
    assert scores[0] == 4.0
    assert math.isnan(scores[1]) and math.isnan(scores[2])
    assert client.calls[0]["phase"] == "judge"
