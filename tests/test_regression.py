"""Regression subset: drop warnings, result summarising and failure handling (lm_eval mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from touster.eval import regression
from touster.eval.regression import regression_warnings, run_regression


def test_warning_only_above_threshold() -> None:
    """Drops above 5 points warn; exactly 5, small drops and gains do not."""
    base = {"arc_easy": 0.60, "mmlu_subset": 0.50, "other": 0.40}
    current = {"arc_easy": 0.55, "mmlu_subset": 0.44, "other": 0.45}
    warnings = regression_warnings(base, current)
    assert len(warnings) == 1 and warnings[0].startswith("mmlu_subset dropped 6.0 points")


def test_custom_threshold_and_missing_keys() -> None:
    """Only shared keys are compared and the threshold is configurable."""
    assert regression_warnings({"a": 0.5, "b": 0.9}, {"a": 0.48}, max_drop_points=1.0) == [
        "a dropped 2.0 points vs base (50.0 -> 48.0)"
    ]
    assert regression_warnings({}, {"a": 0.1}) == []


def fake_results() -> dict:
    """Shape of lm_eval.simple_evaluate output for the subset."""
    per_task = {"arc_easy": {"acc,none": 0.7}}
    per_task.update({t: {"acc,none": 0.4 + 0.1 * i} for i, t in enumerate(regression.MMLU_TASKS)})
    return {"results": per_task}


def test_run_regression_summarises(monkeypatch, make_cfg) -> None:
    """arc_easy is reported as is; mmlu subtasks are averaged into mmlu_subset."""
    monkeypatch.setattr(regression, "_simple_evaluate", lambda bundle, cfg: fake_results())
    scores = run_regression(SimpleNamespace(model=None, tokenizer=None), make_cfg())
    assert scores["arc_easy"] == pytest.approx(0.7)
    assert scores["mmlu_subset"] == pytest.approx(0.55)


@pytest.mark.parametrize("error", [OSError("dataset download blocked"), ImportError("no lm_eval")])
def test_run_regression_failure_returns_empty(monkeypatch, make_cfg, error) -> None:
    """A failing lm_eval run logs a warning with the reason and returns {}."""

    def boom(bundle, cfg):
        """Raise the parametrized error."""
        raise error

    events: list[tuple[str, str]] = []
    monkeypatch.setattr(regression, "_simple_evaluate", boom)
    monkeypatch.setattr(regression, "log_event", lambda msg, level="info": events.append((msg, level)))
    assert run_regression(SimpleNamespace(model=None, tokenizer=None), make_cfg()) == {}
    assert events and events[0][1] == "warning" and str(error) in events[0][0]


def test_run_regression_reraises_cuda_oom(monkeypatch, make_cfg) -> None:
    """CUDA out-of-memory is raised with a clear message, not hidden as a skipped check."""
    import torch

    def oom(bundle, cfg):
        """Simulate the GPU running out of memory."""
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(regression, "_simple_evaluate", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="REGRESSION_BATCH_SIZE"):
        run_regression(SimpleNamespace(model=None, tokenizer=None), make_cfg())


def test_run_regression_no_results(monkeypatch, make_cfg) -> None:
    """Non-zero ranks return None from simple_evaluate; treat as no scores."""
    monkeypatch.setattr(regression, "_simple_evaluate", lambda bundle, cfg: None)
    assert run_regression(SimpleNamespace(model=None, tokenizer=None), make_cfg()) == {}
