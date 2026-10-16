"""Task metrics, evaluate() output format and eval-cell resume, with the student and judge mocked."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import pytest

from touster.eval import metrics
from touster.eval.metrics import (
    adapter_context,
    evaluate,
    exact_match,
    field_f1,
    passes_tests,
    primary_metric,
    record_tests,
    resolve_primary,
    rouge_l,
    run_eval_cell,
    score_task,
    token_f1,
)
from touster.state import append_jsonl, read_json, write_json, write_jsonl

SPEC_KEYS = {"stage", "primary_metric", "scores", "judge_win_rate_vs_base", "regression", "n", "cost_usd", "minutes"}


# ---------- metrics on known inputs ----------

def test_exact_match_normalizes_and_is_lenient_for_labels() -> None:
    """Case, punctuation and whitespace are ignored; lenient mode accepts the first line only."""
    assert exact_match("  Positive. ", "positive") == 1.0
    assert exact_match("Positive\nThe review praises the product.", "positive", lenient=True) == 1.0
    assert exact_match("Negative.\nReason: it broke.", "negative", lenient=True) == 1.0
    assert exact_match("positive but sarcastic", "positive", lenient=True) == 0.0
    assert exact_match("Negative, clearly", "negative", lenient=True) == 0.0
    assert exact_match("Negative, clearly", "negative") == 0.0
    assert exact_match("neutral", "positive", lenient=True) == 0.0
    assert exact_match("", "positive", lenient=True) == 0.0


def test_token_f1() -> None:
    """SQuAD-style overlap F1."""
    assert token_f1("the cat sat", "the cat") == pytest.approx(0.8)
    assert token_f1("Paris.", "paris") == 1.0
    assert token_f1("london", "paris") == 0.0
    assert token_f1("", "") == 1.0


def test_field_f1() -> None:
    """F1 counts matching field/value pairs."""
    ref = {"name": "Bob", "age": 30, "address": {"city": "Paris"}}
    assert field_f1(ref, ref) == 1.0
    assert field_f1({"name": "bob", "age": 31}, ref) == pytest.approx(2 * 1 / (2 + 3))


def test_extraction_scores() -> None:
    """json_validity counts parseable outputs; f1 is field-level."""
    refs = ['{"name": "Bob"}', '{"name": "Ann"}']
    scores = score_task("extraction", ['{"name": "Bob"}', "not json"], refs)
    assert scores == {"json_validity": 0.5, "f1": 0.5}


def test_qa_scores() -> None:
    """QA reports token F1 and strict exact match."""
    scores = score_task("qa", ["the cat sat", "Paris"], ["the cat", "paris"])
    assert scores["f1"] == pytest.approx(0.9)
    assert scores["exact_match"] == 0.5


def test_classification_scores() -> None:
    """Classification reports lenient exact match."""
    assert score_task("classification", ["Positive!", "neutral"], ["positive", "negative"]) == {"exact_match": 0.5}


def test_rouge_l() -> None:
    """Identical text scores 1, disjoint text 0."""
    assert rouge_l("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)
    assert rouge_l("alpha beta", "gamma delta") == 0.0
    assert 0.0 < score_task("summarisation", ["the quick fox"], ["the quick brown fox"])["rouge_l"] < 1.0


def test_code_pass_at_1_runs_tests() -> None:
    """Code passing its asserts scores 1, failing code 0; fences are stripped."""
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "def add(a, b):\n    return a - b"
    tests = ["assert add(2, 3) == 5", "assert add(1, 1) == 2"]
    scores = score_task("code", [good, bad], ["", ""], tests)
    assert scores["pass_at_1"] == 0.5 and scores["syntax_valid"] == 1.0


def test_code_timeout_fails() -> None:
    """An infinite loop is killed and counts as a failure."""
    assert passes_tests("while True:\n    pass", "assert True", timeout=1) is False


def test_code_without_tests_falls_back_to_syntax() -> None:
    """With no tests, only syntax_valid is reported and becomes the primary metric."""
    scores = score_task("code", ["def f(:\n  pass", "x = 1"], ["", ""])
    assert scores == {"syntax_valid": 0.5}
    assert resolve_primary("code", scores) == "syntax_valid"


def test_record_tests_from_metadata() -> None:
    """Tests may be a string or list in metadata."""
    assert record_tests({"metadata": {"tests": ["assert 1", "assert 2"]}}) == "assert 1\nassert 2"
    assert record_tests({"metadata": {"source": "human"}}) is None
    assert record_tests({}) is None


@pytest.mark.parametrize(
    ("task", "metric"),
    [("classification", "exact_match"), ("extraction", "f1"), ("qa", "f1"), ("code", "pass_at_1"),
     ("summarisation", "rouge_l"), ("other", "rouge_l")],
)
def test_primary_metric_mapping(task: str, metric: str) -> None:
    """Each task type maps to its headline metric."""
    assert primary_metric(task) == metric
    assert resolve_primary(task, {metric: 0.5}) == metric


def test_score_task_length_mismatch() -> None:
    """Misaligned predictions and references raise."""
    with pytest.raises(ValueError):
        score_task("qa", ["a"], [])


# ---------- evaluate() and the eval cell ----------

EVAL_RECORDS = [
    {"id": "g-0001", "prompt": "Great product!", "response": "positive", "metadata": {"source": "human"}},
    {"id": "g-0002", "prompt": "Broke in a day.", "response": "negative", "metadata": {"source": "human"}},
]


class FakeModel:
    """Stand-in PeftModel that records whether the adapter was disabled during generation."""

    def __init__(self) -> None:
        """Start with the adapter enabled."""
        self.adapter_enabled = True

    @contextmanager
    def disable_adapter(self) -> Iterator[None]:
        """Disable the adapter inside the block."""
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


@pytest.fixture
def fake_sft(monkeypatch) -> SimpleNamespace:
    """Install a fake touster.train.sft whose answers depend on whether the adapter is on."""
    calls = SimpleNamespace(generate=[], load=[], released=[])

    def generate_responses(bundle, prompts, cfg, temperature=0.0, num_samples=1, max_new_tokens=512):
        """Base answers are wrong, adapter answers are right."""
        calls.generate.append({"adapter": bundle.model.adapter_enabled, "temperature": temperature})
        good = {"Great product!": "Positive", "Broke in a day.": "negative"}
        return [[good[p] if bundle.model.adapter_enabled else "neutral"] for p in prompts]

    def load_student(cfg, adapter_path=None, for_training=True):
        """Return a fake bundle."""
        calls.load.append({"adapter_path": adapter_path, "for_training": for_training})
        return SimpleNamespace(model=FakeModel(), tokenizer=None, system_prompt="")

    module = ModuleType("touster.train.sft")
    module.generate_responses = generate_responses
    module.load_student = load_student
    module.release_student = calls.released.append
    monkeypatch.setitem(sys.modules, "touster.train.sft", module)
    monkeypatch.setattr(metrics, "run_regression", lambda bundle, cfg: {"arc_easy": 0.6})
    return calls


class CostLoggingJudge:
    """Fake LLM client: the candidate always wins, and each call logs a judge cost row."""

    def __init__(self, costs: Path) -> None:
        """Log costs to the run's costs.jsonl."""
        self.costs = costs

    def chat_many(self, batch, phase, model=None, on_done=None, **kwargs: Any) -> list:
        """Forward half says A, swapped half says B."""
        half = len(batch) // 2
        for _ in batch:
            append_jsonl(self.costs, {"phase": phase, "cost_usd": 0.01})
        return [SimpleNamespace(text=json.dumps({"winner": "A" if i < half else "B"})) for i in range(len(batch))]


def test_adapter_context_disables_only_when_asked() -> None:
    """use_adapter=False enters disable_adapter; models without it get a no-op context."""
    bundle = SimpleNamespace(model=FakeModel())
    with adapter_context(bundle, use_adapter=False):
        assert bundle.model.adapter_enabled is False
    with adapter_context(bundle, use_adapter=True):
        assert bundle.model.adapter_enabled is True
    with adapter_context(SimpleNamespace(model=object()), use_adapter=False):
        pass


def test_evaluate_writes_spec_format(run, fake_sft) -> None:
    """Base then SFT: spec keys, answers list, judge win rate and judge cost from costs.jsonl."""
    cfg, paths = run
    write_jsonl(paths.golden_eval, EVAL_RECORDS)
    append_jsonl(paths.costs, {"phase": "generation", "cost_usd": 1.0})
    bundle = SimpleNamespace(model=FakeModel(), tokenizer=None, system_prompt="")
    client = CostLoggingJudge(paths.costs)

    base = evaluate(bundle, "base", cfg, paths, client, use_adapter=False)
    sft = evaluate(bundle, "sft", cfg, paths, client)

    assert SPEC_KEYS | {"answers"} == set(base) == set(sft)
    assert base["judge_win_rate_vs_base"] is None and base["cost_usd"] == 0.0
    assert base["scores"] == {"exact_match": 0.0} and sft["scores"] == {"exact_match": 1.0}
    assert sft["primary_metric"] == "exact_match" and sft["n"] == 2
    assert sft["judge_win_rate_vs_base"] == 1.0
    assert sft["cost_usd"] == pytest.approx(0.04)
    assert sft["regression"] == {"arc_easy": 0.6}
    assert sft["answers"][0] == {"id": "g-0001", "prompt": "Great product!", "reference": "positive", "answer": "Positive"}
    assert read_json(paths.eval_result("sft")) == sft
    assert all(c["temperature"] == 0.0 for c in fake_sft.generate)


def test_evaluate_skips_regression_in_smoke_mode(run, make_cfg, fake_sft) -> None:
    """Smoke mode leaves regression empty."""
    _, paths = run
    write_jsonl(paths.golden_eval, EVAL_RECORDS)
    bundle = SimpleNamespace(model=FakeModel(), tokenizer=None, system_prompt="")
    result = evaluate(bundle, "base", make_cfg(smoke_test=True), paths, None, use_adapter=False)
    assert result["regression"] == {}


def test_evaluate_regression_flag_off(run, fake_sft, monkeypatch) -> None:
    """regression=False never calls lm_eval and writes an empty regression dict."""
    cfg, paths = run
    write_jsonl(paths.golden_eval, EVAL_RECORDS)
    monkeypatch.setattr(metrics, "run_regression", lambda bundle, cfg: pytest.fail("lm_eval must not run"))
    bundle = SimpleNamespace(model=FakeModel(), tokenizer=None, system_prompt="")
    assert evaluate(bundle, "base", cfg, paths, None, use_adapter=False, regression=False)["regression"] == {}


def test_eval_cell_reuses_existing_base(run, fake_sft, monkeypatch) -> None:
    """With eval/base.json present the cell only evaluates SFT, from one loaded model."""
    cfg, paths = run
    monkeypatch.setattr(metrics, "require_gpu", lambda cell: {"available": True})
    write_jsonl(paths.golden_eval, EVAL_RECORDS)
    base = {
        "stage": "base", "primary_metric": "exact_match", "scores": {"exact_match": 0.25},
        "judge_win_rate_vs_base": None, "regression": {}, "n": 2, "cost_usd": 0.0, "minutes": 0.1,
        "answers": [{"id": r["id"], "prompt": r["prompt"], "reference": r["response"], "answer": "x"} for r in EVAL_RECORDS],
    }
    write_json(paths.eval_result("base"), base)
    (paths.sft_checkpoints / "final").mkdir(parents=True)

    result = run_eval_cell(cfg, paths, CostLoggingJudge(paths.costs), None)

    assert read_json(paths.eval_result("base")) == base
    assert [c["adapter"] for c in fake_sft.generate] == [True]
    assert fake_sft.load == [{"adapter_path": paths.sft_checkpoints / "final", "for_training": False}]
    assert result["base"] == 0.25 and result["sft"] == 1.0 and result["gain_points"] == 75.0
    assert len(fake_sft.released) == 1
    assert paths.eval_result("sft").is_file()


def test_eval_cell_without_sft_adapter_scores_base_only(run, fake_sft, monkeypatch) -> None:
    """No SFT adapter: base is evaluated with the adapter path None and no sft.json is written."""
    cfg, paths = run
    monkeypatch.setattr(metrics, "require_gpu", lambda cell: {"available": True})
    write_jsonl(paths.golden_eval, EVAL_RECORDS)

    result = run_eval_cell(cfg, paths, None, None)

    assert fake_sft.load == [{"adapter_path": None, "for_training": False}]
    assert result["sft"] is None and result["sft_evaluated"] is False
    assert len(fake_sft.released) == 1
    assert paths.eval_result("base").is_file() and not paths.eval_result("sft").exists()
