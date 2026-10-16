"""Synthetic-data filters with a fake judge and a fake embedder (no network, no model downloads)."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from touster.data.filters import (
    decontam_filter,
    dedup_filter,
    parse_score,
    quality_filter,
    quality_messages,
    run_filters,
    schema_filter,
    similarity_filter,
)
from touster.llm import LLMResult
from touster.state import write_jsonl

LONG_A = (
    "I bought this standing desk for my home office last spring and the motor still raises it smoothly "
    "every morning without any wobble at all"
)
LONG_B = "The kettle boils fast but the lid hinge snapped after two weeks and customer service never answered my emails"


class FakeJudge:
    """Stand-in for LLMClient that answers judge prompts from a prompt->text map."""

    def __init__(self, answers: dict[str, str]) -> None:
        """answers maps a record prompt to the judge's raw reply."""
        self.answers = answers
        self.judged: list[str] = []

    def chat_many(self, batch: list[list[dict[str, str]]], phase: str, **_: Any) -> list[LLMResult]:
        """Reply to each judge prompt by finding which known prompt it contains."""
        assert phase == "quality"
        replies = []
        for messages in batch:
            content = messages[-1]["content"]
            prompt = next(p for p in self.answers if f"PROMPT:\n{p}\n" in content)
            self.judged.append(prompt)
            replies.append(LLMResult(self.answers[prompt], "judge", 10, 5, 0.001, 0.0))
        return replies


def rec(rid: str, prompt: str, response: str = "positive", **extra: Any) -> dict[str, Any]:
    """Build a synthetic record."""
    return {"id": rid, "prompt": prompt, "response": response, **extra}


def table_embed(table: dict[str, list[float]]) -> Callable[[list[str]], np.ndarray]:
    """Fake embedder returning fixed vectors per text."""
    return lambda texts: np.array([table[t] for t in texts], dtype=np.float32)


def test_parse_score() -> None:
    """Only integer scores 1-5 count."""
    assert parse_score('{"score": 4, "reason": "ok"}') == 4
    assert parse_score('{"score": 4.0}') == 4
    for bad in ('{"score": 6}', '{"score": 0}', '{"score": "4"}', '{"score": 3.5}', '{"score": true}', "four", ""):
        assert parse_score(bad) is None


def test_schema_filter() -> None:
    """Missing, empty or non-string fields are dropped."""
    records = [rec("s-1", "fine"), rec("s-2", " "), {"id": "s-3", "prompt": "x"}, rec("s-4", "x", None), rec("s-5", "ok")]
    assert [r["id"] for r in schema_filter(records)] == ["s-1", "s-5"]


def test_dedup_removes_planted_near_duplicates() -> None:
    """Exact, case/punctuation variants and one-word extensions of a kept prompt are removed."""
    records = [
        rec("s-1", LONG_A),
        rec("s-2", LONG_B),
        rec("s-3", LONG_A.upper() + "!!"),
        rec("s-4", LONG_A + " honestly"),
        rec("s-5", "Short and sweet"),
        rec("s-6", "short and sweet."),
    ]
    assert [r["id"] for r in dedup_filter(records, threshold=0.85)] == ["s-1", "s-2", "s-5"]


def test_similarity_filter_drops_close_embeddings() -> None:
    """A prompt whose embedding is within the threshold of a kept one is dropped."""
    embed = table_embed({"a": [1.0, 0.0], "b": [0.0, 1.0], "a2": [0.99, 0.05], "c": [0.7, 0.7]})
    records = [rec("s-1", "a"), rec("s-2", "b"), rec("s-3", "a2"), rec("s-4", "c")]
    assert [r["id"] for r in similarity_filter(records, 0.92, embed)] == ["s-1", "s-2", "s-4"]
    assert [r["id"] for r in similarity_filter(records, 0.70, embed)] == ["s-1", "s-2"]


def test_similarity_filter_empty_does_not_load_model() -> None:
    """No records means no embedder call."""
    assert similarity_filter([], 0.9, embed=None) == []


def test_decontam_removes_eval_overlap() -> None:
    """13-gram overlap with an eval prompt or response, or a normalized short-prompt match, is dropped."""
    eval_records = [rec("g-1", LONG_A, "positive"), rec("g-2", "Great value!", LONG_B)]
    thirteen = " ".join(LONG_A.split()[3:16])
    twelve = " ".join(LONG_B.split()[:12])
    records = [
        rec("s-1", f"Someone wrote: {thirteen}. What is the sentiment?"),
        rec("s-2", "Totally different review about socks", f"{LONG_B}"),
        rec("s-3", "great VALUE"),
        rec("s-4", f"{twelve} but otherwise fine"),
        rec("s-5", "Socks were warm"),
    ]
    assert [r["id"] for r in decontam_filter(records, eval_records, 0.85)] == ["s-4", "s-5"]


def test_decontam_catches_short_near_copies() -> None:
    """Short reviews (< 13 words) that nearly copy an eval prompt are dropped; unrelated ones stay."""
    eval_records = [rec("g-1", "The drill battery died after twenty minutes and charging takes four hours", "negative")]
    records = [
        rec("s-1", "the drill battery died after twenty minutes and charging takes four hours!"),
        rec("s-2", "The drill battery died after twenty minutes and charging takes four hours today"),
        rec("s-3", "Lovely soft scarf, warm and the colour matches the photos"),
    ]
    assert [r["id"] for r in decontam_filter(records, eval_records, 0.85)] == ["s-3"]


def test_quality_filter_threshold_and_malformed(make_cfg) -> None:
    """Scores below the threshold and malformed judge output are dropped; scores are attached."""
    cfg = make_cfg(quality_threshold=4)
    judge = FakeJudge({"p5": '{"score": 5}', "p4": '{"score": 4}', "p3": '{"score": 3}', "bad": "not json"})
    records = [rec("s-1", "p5"), rec("s-2", "p4"), rec("s-3", "p3"), rec("s-4", "bad")]
    kept, malformed = quality_filter(cfg, judge, records)
    assert [(r["id"], r["quality_score"]) for r in kept] == [("s-1", 5), ("s-2", 4)]
    assert malformed == 1
    assert "quality_score" not in records[0]


def test_quality_filter_skips_already_scored(make_cfg) -> None:
    """Records scored during calibration are not sent to the judge again."""
    cfg = make_cfg(quality_threshold=3)
    judge = FakeJudge({"new": '{"score": 3}'})
    kept, _ = quality_filter(cfg, judge, [rec("s-1", "old", quality_score=5), rec("s-2", "new")])
    assert judge.judged == ["new"]
    assert [r["quality_score"] for r in kept] == [5, 3]


def test_run_filters_funnel(run) -> None:
    """Every stage runs in order and the funnel counts what is left after each."""
    cfg, paths = run
    write_jsonl(paths.golden_eval, [rec("g-1", LONG_B, "negative")])
    prompts = [LONG_A, LONG_A + " honestly", "close to first", "low quality", LONG_B, "unique and good"]
    records = [rec(f"s-{i}", p) for i, p in enumerate(prompts, 1)] + [{"id": "s-9", "prompt": ""}]
    vectors = {p: [float(i == j) for j in range(6)] for i, p in enumerate(prompts)}
    vectors["close to first"] = vectors[LONG_A]
    judge = FakeJudge({p: json.dumps({"score": 2 if p == "low quality" else 5}) for p in prompts})
    clean, funnel = run_filters(cfg, judge, records, paths, embed=table_embed(vectors))
    assert list(funnel.items()) == [
        ("raw", 7), ("schema", 6), ("dedup", 5), ("similarity", 4), ("decontam", 3), ("quality", 2),
    ]
    assert [r["id"] for r in clean] == ["s-1", "s-6"]
    assert LONG_B not in judge.judged


def test_quality_prompt_scores_instructions_as_1(make_cfg) -> None:
    """The judge is told that an instruction or meta-description instead of a real input scores 1."""
    cfg = make_cfg()
    content = quality_messages(cfg, rec("s-1", "Write about ordering a mid-size but receiving a small one"))[0]["content"]
    assert cfg.task_description in content
    assert "instruction or a description of what to write" in content and "give score 1" in content
