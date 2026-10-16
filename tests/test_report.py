"""Report rendering from fake run directories (full data and almost none)."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from plotly.offline import get_plotlyjs

from touster.config import TousterConfig
from touster.report import build_report, load_eval_results, render_report, report_html, run_report_cell, stage_sort_key
from touster.state import RunPaths, append_jsonl, load_state, mark_done, save_state, write_json

SECTIONS = ("summary", "funnel", "loss", "scores", "regression", "costs", "examples")


def _answers(prefix: str, n: int = 6) -> list[dict[str, str]]:
    """Fake eval answers."""
    return [{"id": f"g-{i:04d}", "prompt": f"prompt {i}", "reference": "positive", "answer": f"{prefix} answer {i}"}
            for i in range(n)]


def _eval(stage: str, score: float, win: float | None, regression: float) -> dict[str, Any]:
    """Fake eval result in the spec format."""
    return {"stage": stage, "primary_metric": "exact_match", "scores": {"exact_match": score},
            "judge_win_rate_vs_base": win, "regression": {"mmlu_subset": regression}, "n": 6,
            "cost_usd": 0.1, "minutes": 1.0, "answers": _answers(stage)}


@pytest.fixture
def fake_sft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in for touster.train.sft with a loss reader."""
    module = types.ModuleType("touster.train.sft")
    module.read_loss_history = lambda folder: [{"step": s, "loss": 2.0 / s} for s in range(1, 6)]
    monkeypatch.setitem(sys.modules, "touster.train.sft", module)


@pytest.fixture
def full_run(run: tuple[TousterConfig, RunPaths], fake_sft: None) -> tuple[TousterConfig, RunPaths]:
    """A run directory with every input the report reads."""
    cfg, paths = run
    funnel = {"raw": 100, "schema": 95, "dedup": 80, "similarity": 70, "quality": 60, "decontam": 58}
    mark_done(paths, "synthetic", {"funnel": funnel})
    mark_done(paths, "loop", {"best_stage": "dpo_round_1", "best_adapter": "x", "rounds": []})
    write_json(paths.eval_result("base"), _eval("base", 0.50, None, 0.52))
    write_json(paths.eval_result("sft"), _eval("sft", 0.70, 0.60, 0.51))
    write_json(paths.eval_result("dpo_round_1"), _eval("dpo_round_1", 0.75, 0.66, 0.40))
    paths.dpo_checkpoints(1).mkdir(parents=True)
    for phase, usd in (("generation", 0.5), ("quality", 0.2), ("judge", 0.1), ("generation", 0.25)):
        append_jsonl(paths.costs, {"phase": phase, "cost_usd": usd})
    return cfg, paths


def test_full_report_has_every_section_and_chart(full_run: tuple[TousterConfig, RunPaths]) -> None:
    """With full data every section renders a chart or table and no placeholder."""
    cfg, paths = full_run
    html = build_report(cfg, paths).read_text(encoding="utf-8")
    for section in SECTIONS:
        assert f'id="{section}"' in html
    assert "not run" not in html
    assert html.count('class="plotly-graph-div"') == 4  # funnel, loss, scores, costs
    assert "+25.0 points" in html
    assert 'id="selection-note"' in html and "slightly optimistic" in html
    assert "dpo_round_1 answer 0" in html and "base answer 0" in html
    assert 'class="num bad"' in html  # 12-point regression drop flagged


def test_plotly_js_embedded_once_and_no_external_resources(full_run: tuple[TousterConfig, RunPaths]) -> None:
    """Plotly is inlined exactly once and nothing is fetched from a CDN."""
    cfg, paths = full_run
    html = render_report(cfg, paths)
    assert html.count(get_plotlyjs()[:400]) == 1
    assert not re.search(r"<script[^>]+src=", html)
    assert not re.search(r"<link[^>]+href=", html)
    assert "fonts.googleapis" not in html


def test_report_renders_with_almost_no_data(run: tuple[TousterConfig, RunPaths]) -> None:
    """A run that stopped early shows placeholders instead of crashing."""
    cfg, paths = run
    html = build_report(cfg, paths).read_text(encoding="utf-8")
    for section in SECTIONS:
        assert f'id="{section}"' in html
    assert html.count("not run") >= 6
    assert 'class="plotly-graph-div"' not in html


def test_report_renders_without_state_file(run: tuple[TousterConfig, RunPaths]) -> None:
    """Missing state.json and eval folder still render."""
    cfg, paths = run
    paths.state.unlink()
    html = render_report(cfg, paths)
    assert "not started" in html
    assert not paths.state.exists()


def test_report_escapes_model_output(full_run: tuple[TousterConfig, RunPaths]) -> None:
    """Autoescape stops model answers injecting HTML."""
    cfg, paths = full_run
    result = _eval("sft", 0.7, 0.6, 0.5)
    result["answers"][0]["answer"] = "<script>alert(1)</script>"
    write_json(paths.eval_result("dpo_round_1"), {**result, "stage": "dpo_round_1"})
    html = render_report(cfg, paths)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_stage_order_and_unreadable_eval_file(run: tuple[TousterConfig, RunPaths]) -> None:
    """Stages sort in pipeline order; a broken JSON file is skipped."""
    _, paths = run
    for stage in ("soup", "dpo_round_10", "dpo_round_2", "sft", "base"):
        write_json(paths.eval_result(stage), {"stage": stage})
    paths.eval_result("broken").write_text("{not json", encoding="utf-8")
    assert list(load_eval_results(paths)) == ["base", "sft", "dpo_round_2", "dpo_round_10", "soup"]
    assert sorted(["x", "base"], key=stage_sort_key) == ["base", "x"]


def test_run_report_cell_marks_done(full_run: tuple[TousterConfig, RunPaths]) -> None:
    """The cell contract records the report path."""
    cfg, paths = full_run
    metrics = run_report_cell(cfg, paths)
    assert Path(metrics["report"]) == paths.report
    assert "report" in load_state(paths)["completed"]
    assert report_html(paths).startswith("<!doctype html>")


def test_report_html_missing_file(run: tuple[TousterConfig, RunPaths]) -> None:
    """Asking for the report before cell 8 gives a clear error."""
    _, paths = run
    with pytest.raises(FileNotFoundError, match="run cell 8"):
        report_html(paths)


def test_bad_funnel_value_is_left_out(run: tuple[TousterConfig, RunPaths]) -> None:
    """A malformed funnel count is logged and the section shows a placeholder."""
    cfg, paths = run
    state = load_state(paths)
    save_state(paths, {**state, "metrics": {"synthetic": {"funnel": {"raw": "many"}}}})
    html = render_report(cfg, paths)
    assert "no synthetic data funnel" in html
