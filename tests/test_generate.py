"""Synthetic generation and the cell 3 flow with a fake teacher (no network, no models)."""

from __future__ import annotations

import builtins
import json
import random
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from touster.budget import BudgetExceeded, BudgetTracker, Calibration
from touster.data import filters
from touster.data.generate import (
    CalibrationRun,
    ask_user,
    fit_target,
    generate_synthetic,
    prompt_messages,
    run_synthetic_cell,
)
from touster.data.golden import run_golden_cell
from touster.llm import LLMResult
from touster.state import create_or_resume_run, load_state, read_jsonl

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "golden_sample.jsonl"
WORDS = "battery lid strap zip screen hinge motor fabric sole handle cable lens pump blade wheel drawer".split()

Responder = Callable[[str], str]


class FakeTeacher:
    """Stand-in for LLMClient: answers by prompt type and records every request."""

    def __init__(self, prompt_reply: Responder, response_reply: Responder | None = None, budget_after: int | None = None) -> None:
        """Replies come from the given functions; budget_after raises BudgetExceeded on that chat_many call."""
        self.prompt_reply = prompt_reply
        self.response_reply = response_reply or (lambda _: json.dumps({"response": "positive"}))
        self.budget_after = budget_after
        self.calls = 0
        self.contents: list[str] = []

    def reply(self, content: str) -> str:
        """Pick the reply for one request."""
        if '"score"' in content:
            return json.dumps({"score": 5})
        if '"draft"' in content:
            return self.prompt_reply(content)
        return self.response_reply(content)

    def chat_many(self, batch: list[list[dict[str, str]]], phase: str, **_: Any) -> list[LLMResult]:
        """Answer a batch in order."""
        self.calls += 1
        if self.budget_after is not None and self.calls > self.budget_after:
            raise BudgetExceeded("fake cap reached")
        contents = [m[-1]["content"] for m in batch]
        self.contents.extend(contents)
        return [LLMResult(self.reply(c), "fake", 100, 50, 0.001, 0.0) for c in contents]


def counter_prompts(bad_every: int = 0) -> Responder:
    """Distinct generated prompts; every bad_every-th answer is malformed JSON."""
    count = {"n": 0}

    def reply(_: str) -> str:
        """Return the next prompt as JSON, or broken output."""
        count["n"] += 1
        n = count["n"]
        if bad_every and n % bad_every == 0:
            return '{"prompt": "unterminated'
        rng = random.Random(n)
        text = " ".join(rng.choice(WORDS) for _ in range(8))
        return json.dumps({"draft": "d", "prompt": f"Review {n}: the {text} part was ok"})

    return reply


def sparse_prompts(every: int) -> Responder:
    """Only one prompt answer in every `every` is valid JSON."""
    count = {"n": 0}

    def reply(_: str) -> str:
        """Return a valid prompt on the 1st, (every+1)th, ... call, junk otherwise."""
        count["n"] += 1
        return json.dumps({"prompt": f"rare prompt {count['n']}"}) if count["n"] % every == 1 else "junk"

    return reply


def planted_first(text: str, then: Responder) -> Responder:
    """Return `text` as the first generated prompt, then defer to another responder."""
    count = {"n": 0}

    def reply(content: str) -> str:
        """First call gives the planted prompt."""
        count["n"] += 1
        return json.dumps({"prompt": text}) if count["n"] == 1 else then(content)

    return reply


def seeds() -> list[dict[str, Any]]:
    """Three seed records."""
    return [{"id": f"g-{i:04d}", "prompt": f"seed review {i}", "response": "neutral"} for i in range(1, 4)]


def hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic pseudo-random unit vectors, one per distinct text."""
    rows = [np.random.default_rng(zlib.crc32(t.encode())).normal(size=64) for t in texts]
    return np.array([r / np.linalg.norm(r) for r in rows], dtype=np.float32)


def test_generation_drops_malformed_and_numbers_ids(make_cfg) -> None:
    """Bad JSON and empty answers are dropped and counted; ids are sequential."""
    cfg = make_cfg()
    empty_every_4th = {"n": 0}

    def response_reply(_: str) -> str:
        """Every fourth response is an empty string."""
        empty_every_4th["n"] += 1
        return json.dumps({"response": "" if empty_every_4th["n"] % 4 == 0 else "negative"})

    teacher = FakeTeacher(counter_prompts(bad_every=3), response_reply)
    records, stats = generate_synthetic(cfg, teacher, seeds(), 10, random.Random(1))
    assert len(records) == 10
    assert [r["id"] for r in records] == [f"s-{i:06d}" for i in range(1, 11)]
    assert stats["malformed"] > 0
    assert all(r["prompt"] and r["response"] == "negative" for r in records)
    assert all(r["seed_id"].startswith("g-") and r["evolution"] and r["teacher"] == cfg.teacher_model for r in records)


def test_every_teacher_prompt_has_task_description(make_cfg) -> None:
    """Prompt and response requests both include the task description and golden style examples."""
    cfg = make_cfg()
    teacher = FakeTeacher(counter_prompts())
    generate_synthetic(cfg, teacher, seeds(), 4, random.Random(1))
    assert teacher.contents and all(cfg.task_description in c for c in teacher.contents)
    response_requests = [c for c in teacher.contents if '"draft"' not in c]
    assert response_requests and all("Answer: neutral" in c and "label only" in c for c in response_requests)


def test_generation_stops_when_round_adds_nothing(make_cfg) -> None:
    """A teacher that never returns valid JSON ends the loop after one round."""
    teacher = FakeTeacher(lambda _: "I cannot do that")
    records, stats = generate_synthetic(make_cfg(), teacher, seeds(), 50, random.Random(1))
    assert records == []
    assert stats["rounds"] == 1 and "nothing" in stats["stop_reason"]


def test_generation_respects_round_cap(make_cfg) -> None:
    """With mostly malformed output the hard round cap ends generation."""
    teacher = FakeTeacher(sparse_prompts(every=32))
    records, stats = generate_synthetic(make_cfg(), teacher, seeds(), 100, random.Random(1), max_rounds=3)
    assert stats["rounds"] == 3 and "cap" in stats["stop_reason"]
    assert len(records) == 3 and stats["malformed"] == 93


def test_generation_is_seeded(make_cfg) -> None:
    """The same rng seed picks the same seeds and evolutions."""
    picks = []
    for _ in range(2):
        records, _ = generate_synthetic(make_cfg(), FakeTeacher(counter_prompts()), seeds(), 6, random.Random(3))
        picks.append([(r["seed_id"], r["evolution"]) for r in records])
    assert picks[0] == picks[1]


def test_budget_stop_saves_records_before_reraising(make_cfg) -> None:
    """Records from finished rounds are saved, then BudgetExceeded propagates."""
    saved: list[list[dict[str, Any]]] = []
    teacher = FakeTeacher(counter_prompts(), budget_after=2)
    with pytest.raises(BudgetExceeded):
        generate_synthetic(make_cfg(), teacher, seeds(), 100, random.Random(1), on_save=saved.append)
    assert len(saved[-1]) == 32


def test_ask_user_non_interactive(monkeypatch) -> None:
    """EOF on input means no confirmation."""

    def no_input(_: str) -> str:
        """Simulate a non-interactive kernel."""
        raise EOFError

    monkeypatch.setattr(builtins, "input", no_input)
    assert ask_user("go?") is False
    monkeypatch.setattr(builtins, "input", lambda _: " Yes ")
    assert ask_user("go?") is True


def test_ask_user_notebook_without_stdin(monkeypatch) -> None:
    """IPython's StdinNotImplementedError (nbconvert, papermill) also means no confirmation."""
    from IPython.core.error import StdinNotImplementedError

    def no_stdin(_: str) -> str:
        """Simulate a kernel that does not support input()."""
        raise StdinNotImplementedError("no stdin")

    monkeypatch.setattr(builtins, "input", no_stdin)
    assert ask_user("go?") is False


def test_prompt_asks_for_real_inputs_not_instructions(make_cfg) -> None:
    """The teacher is told to write the input itself, never an instruction or a description of one."""
    cfg = make_cfg()
    content = prompt_messages(cfg, seeds()[0], "deepen")[0]["content"]
    assert cfg.task_description in content and "seed review 1" in content
    assert "Write ONE new input for this task yourself" in content
    assert "Never write an instruction, a request to write something, or a description" in content
    assert "same kind of text as the example" in content


def test_fit_target_scales_down_to_budget(make_cfg, tmp_path: Path) -> None:
    """With $4 usable and $0.10 per example after 10 calibration examples, about 46 fit."""
    cfg = make_cfg(max_usd=5.0, eval_reserve=0.2)
    budget = BudgetTracker(cfg, tmp_path / "costs.jsonl")
    calib = Calibration(examples=10, usd=1.0, seconds=1.0, prompt_tokens=0, completion_tokens=0)
    quality = Calibration(examples=10, usd=0.0, seconds=0.0, prompt_tokens=0, completion_tokens=0)
    run = CalibrationRun(calib, quality, [{"id": str(i)} for i in range(10)], 0)
    assert fit_target(run, cfg, budget, 100) == 46
    assert fit_target(run, cfg, budget, 30) == 30


def golden_run(make_cfg, **overrides: Any) -> tuple[Any, Any]:
    """Config + run dir with the golden cell already done."""
    cfg = make_cfg(golden_path=EXAMPLE, **overrides)
    paths, _, _ = create_or_resume_run(cfg)
    run_golden_cell(cfg, paths)
    return cfg, paths


def test_synthetic_cell_smoke_clean_set_has_no_eval_overlap(make_cfg, monkeypatch) -> None:
    """Smoke run: no prompt, raw and clean saved, a planted eval prompt never reaches the clean set."""
    monkeypatch.setattr(filters, "load_embedder", lambda *a: hash_embed)
    cfg, paths = golden_run(make_cfg, smoke_test=True)
    eval_prompt = read_jsonl(paths.golden_eval)[0]["prompt"]
    teacher = FakeTeacher(planted_first(eval_prompt, counter_prompts()))
    budget = BudgetTracker(cfg, paths.costs)
    metrics = run_synthetic_cell(cfg, paths, teacher, budget, confirm=lambda _: pytest.fail("smoke must not ask"))
    raw, clean = read_jsonl(paths.synthetic_raw), read_jsonl(paths.synthetic_clean)
    assert len(raw) == 20 and metrics["raw"] == 20
    assert eval_prompt in {r["prompt"] for r in raw}
    eval_prompts = {r["prompt"] for r in read_jsonl(paths.golden_eval)}
    assert clean and not eval_prompts & {r["prompt"] for r in clean}
    assert metrics["funnel"]["decontam"] == len(clean)
    assert "synthetic" in load_state(paths)["completed"]


def test_synthetic_cell_decline_stops_cleanly(make_cfg, monkeypatch) -> None:
    """Declining keeps the calibration examples, spends nothing more, and leaves the cell not done."""
    monkeypatch.setattr(filters, "load_embedder", lambda *a: pytest.fail("filters must not run"))
    cfg, paths = golden_run(make_cfg, target_examples=200)
    teacher = FakeTeacher(counter_prompts())
    metrics = run_synthetic_cell(cfg, paths, teacher, BudgetTracker(cfg, paths.costs), confirm=lambda _: False)
    assert metrics == {"declined": True}
    assert teacher.calls == 3  # calibration only: prompts, responses, quality judging
    raw = read_jsonl(paths.synthetic_raw)
    assert len(raw) == 20 and all(r["quality_score"] == 5 for r in raw)
    state = load_state(paths)
    assert "synthetic" not in state["completed"] and "synthetic" not in state["metrics"]


def test_synthetic_cell_rerun_reuses_saved_records(make_cfg, monkeypatch) -> None:
    """After a decline, a rerun skips calibration, keeps saved records and scores, and only tops up."""
    monkeypatch.setattr(filters, "load_embedder", lambda *a: hash_embed)
    cfg, paths = golden_run(make_cfg, target_examples=30)
    prompts = counter_prompts()
    first = FakeTeacher(prompts)
    run_synthetic_cell(cfg, paths, first, BudgetTracker(cfg, paths.costs), confirm=lambda _: False)
    saved = read_jsonl(paths.synthetic_raw)
    second = FakeTeacher(prompts)
    metrics = run_synthetic_cell(cfg, paths, second, BudgetTracker(cfg, paths.costs), confirm=lambda _: True)
    raw = read_jsonl(paths.synthetic_raw)
    assert raw[:20] == saved
    assert [r["id"] for r in raw] == [f"s-{i:06d}" for i in range(1, 31)]
    assert second.calls == 3  # no calibration: prompts and responses for 10 more, then one judge batch
    judged = [c for c in second.contents if '"score"' in c]
    assert judged and metrics["raw"] == 30
    assert not any(r["prompt"] in c for r in saved for c in judged)
    assert "synthetic" in load_state(paths)["completed"]


def test_valid_response_rejects_wrapped_json_and_long_labels() -> None:
    """A JSON-wrapped answer, or a classification answer that is not a short label, is dropped, never unwrapped."""
    from touster.config import TaskType
    from touster.data.generate import valid_response

    assert valid_response("negative", TaskType.CLASSIFICATION)
    assert not valid_response('{"response": "negative"}', TaskType.CLASSIFICATION)
    assert not valid_response("I think this review is mostly negative overall", TaskType.CLASSIFICATION)
    assert valid_response("A longer free text answer is fine for QA tasks.", TaskType.QA)


def test_show_estimate_prints_training_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training probe's tokens/sec and peak VRAM are shown only when a probe ran."""
    from types import SimpleNamespace

    from rich.console import Console

    from touster.budget import Estimate
    from touster.data import generate

    console = Console(record=True, width=120)
    monkeypatch.setattr(generate, "get_dashboard", lambda: SimpleNamespace(console=console))
    parts = {"generation": Estimate(usd=1.0, minutes=2.0)}
    no_probe = Calibration(examples=10, usd=1.0, seconds=1.0, prompt_tokens=0, completion_tokens=0)
    probed = Calibration(
        examples=10, usd=1.0, seconds=1.0, prompt_tokens=0, completion_tokens=0,
        train_steps=5, train_seconds=10.0, train_tokens_per_sec=1234.0, peak_vram_gb=7.25,
    )
    generate.show_estimate(parts, Estimate(usd=5.0, minutes=60.0), 100, no_probe)
    assert "training probe" not in console.export_text()
    generate.show_estimate(parts, Estimate(usd=5.0, minutes=60.0), 100, probed)
    assert "training probe: 1,234 tokens/sec, peak VRAM 7.2 GB" in console.export_text()
