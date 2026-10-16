"""Cell 8: a self-contained HTML report (Jinja2 + Plotly) that renders even when parts of the run are missing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from touster.budget import spend_by_phase
from touster.config import TousterConfig
from touster.dashboard import log_event
from touster.state import RunPaths, read_json, run_cell

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
TEMPLATE_NAME = "report.html.j2"
EXAMPLE_COUNT = 5
MAX_REGRESSION_DROP_POINTS = 5.0
CHART_CONFIG = {"displaylogo": False, "responsive": True}
CHART_FONT_COLOR = "#8a8f98"
CHART_GRID_COLOR = "rgba(138,143,152,0.25)"
STAGE_PATTERN = re.compile(r"^(dpo|grpo)_round_(\d+)$")


def read_state(paths: RunPaths) -> dict[str, Any]:
    """Read state.json without creating it; empty dict if missing or unreadable."""
    if not paths.state.is_file():
        return {}
    try:
        return read_json(paths.state)
    except (json.JSONDecodeError, OSError) as err:
        log_event(f"Could not read {paths.state}: {err}", "warning")
        return {}


def stage_sort_key(stage: str) -> tuple[int, int, str]:
    """Order stages as base, sft, loop rounds by number, soup, then anything else."""
    fixed = {"base": 0, "sft": 1}
    if stage in fixed:
        return fixed[stage], 0, stage
    match = STAGE_PATTERN.match(stage)
    if match:
        return 2, int(match.group(2)), stage
    return (3 if stage == "soup" else 4), 0, stage


def load_eval_results(paths: RunPaths) -> dict[str, dict]:
    """Every eval/<stage>.json keyed by stage, in pipeline order; unreadable files are logged and left out."""
    if not paths.eval.is_dir():
        return {}
    results: dict[str, dict] = {}
    for file in paths.eval.glob("*.json"):
        try:
            payload = read_json(file)
        except (json.JSONDecodeError, OSError) as err:
            log_event(f"Skipping unreadable eval file {file}: {err}", "warning")
            continue
        if isinstance(payload, dict):
            results[str(payload.get("stage") or file.stem)] = payload
    return {stage: results[stage] for stage in sorted(results, key=stage_sort_key)}


def primary_score(result: dict[str, Any]) -> float | None:
    """The primary metric score of one eval result, or None."""
    value = (result.get("scores") or {}).get(result.get("primary_metric"))
    return float(value) if isinstance(value, (int, float)) else None


def _chart(fig: Any) -> Markup:
    """Plotly figure as an HTML div that relies on the single plotly.js bundle in the page head."""
    import plotly.io as pio

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": CHART_FONT_COLOR},
        margin={"l": 50, "r": 20, "t": 30, "b": 50}, height=360,
    )
    fig.update_xaxes(gridcolor=CHART_GRID_COLOR)
    fig.update_yaxes(gridcolor=CHART_GRID_COLOR)
    return Markup(pio.to_html(fig, include_plotlyjs=False, full_html=False, config=CHART_CONFIG))


def _funnel_section(state: dict[str, Any]) -> dict[str, Any] | None:
    """Records left after each filter, from metrics['synthetic']['funnel']."""
    import plotly.graph_objects as go

    funnel = ((state.get("metrics") or {}).get("synthetic") or {}).get("funnel") or {}
    if not funnel:
        return None
    try:
        stages, counts = list(funnel), [int(v) for v in funnel.values()]
    except (TypeError, ValueError) as err:
        log_event(f"Data funnel left out: non-numeric count in state metrics ({err})", "warning")
        return None
    fig = go.Figure(go.Funnel(y=stages, x=counts, textinfo="value+percent initial"))
    return {"chart": _chart(fig), "rows": list(zip(stages, counts))}


def _loss_folders(cfg: TousterConfig, paths: RunPaths) -> list[tuple[str, Path]]:
    """Checkpoint folders that may hold a loss history: SFT then each loop round."""
    folders = [("sft", paths.sft_checkpoints)]
    folders += [(f"{cfg.loop_method.value}_round_{n}", paths.dpo_checkpoints(n)) for n in range(1, cfg.max_rounds + 1)]
    return [(name, folder) for name, folder in folders if folder.is_dir()]


def _loss_section(cfg: TousterConfig, paths: RunPaths) -> dict[str, Any] | None:
    """Training loss curves for SFT and each loop round."""
    import plotly.graph_objects as go

    folders = _loss_folders(cfg, paths)
    if not folders:
        return None
    try:
        from touster.train.sft import read_loss_history
    except ImportError as err:
        log_event(f"Loss curves left out: could not import touster.train.sft ({err})", "warning")
        return None

    fig, plotted = go.Figure(), []
    for name, folder in folders:
        history = read_loss_history(folder)
        if history:
            fig.add_trace(go.Scatter(x=[h["step"] for h in history], y=[h["loss"] for h in history], mode="lines", name=name))
            plotted.append(name)
    if not plotted:
        return None
    fig.update_layout(xaxis_title="step", yaxis_title="loss")
    return {"chart": _chart(fig), "stages": plotted}


def _score_section(evals: dict[str, dict]) -> dict[str, Any] | None:
    """Primary score and judge win rate per stage."""
    import plotly.graph_objects as go

    stages = [s for s in evals if primary_score(evals[s]) is not None]
    if not stages:
        return None
    metric = evals[stages[0]].get("primary_metric", "score")
    fig = go.Figure(go.Bar(x=stages, y=[primary_score(evals[s]) for s in stages], name=metric))
    win = [evals[s].get("judge_win_rate_vs_base") for s in stages]
    if any(isinstance(w, (int, float)) for w in win):
        fig.add_trace(go.Bar(x=stages, y=win, name="judge win rate vs base"))
    fig.update_layout(barmode="group", yaxis={"range": [0, 1], "title": "0..1"})
    rows = [{"stage": s, "score": primary_score(evals[s]), "win": evals[s].get("judge_win_rate_vs_base"),
             "n": evals[s].get("n")} for s in stages]
    return {"chart": _chart(fig), "metric": metric, "rows": rows}


def _regression_section(evals: dict[str, dict]) -> dict[str, Any] | None:
    """Regression benchmark scores per stage, with drops over the limit flagged."""
    base = (evals.get("base") or {}).get("regression") or {}
    staged = [(s, r.get("regression") or {}) for s, r in evals.items() if r.get("regression")]
    if not staged:
        return None
    tasks = sorted({t for _, reg in staged for t in reg})
    rows = []
    for stage, reg in staged:
        cells = []
        for task in tasks:
            value, ref = reg.get(task), base.get(task)
            drop = (ref - value) * 100 if isinstance(value, (int, float)) and isinstance(ref, (int, float)) else None
            cells.append({"value": value, "flag": drop is not None and drop > MAX_REGRESSION_DROP_POINTS})
        rows.append({"stage": stage, "cells": cells})
    return {"tasks": tasks, "rows": rows, "limit": MAX_REGRESSION_DROP_POINTS}


def _cost_section(paths: RunPaths) -> dict[str, Any] | None:
    """Spend per phase from costs.jsonl."""
    import plotly.graph_objects as go

    try:
        spend = spend_by_phase(paths.costs)
    except (json.JSONDecodeError, ValueError) as err:
        log_event(f"Could not read {paths.costs}: {err}", "warning")
        return None
    if not spend:
        return None
    phases = sorted(spend, key=spend.get, reverse=True)
    fig = go.Figure(go.Bar(x=phases, y=[spend[p] for p in phases]))
    fig.update_layout(yaxis_title="USD")
    return {"chart": _chart(fig), "rows": [(p, spend[p]) for p in phases], "total": sum(spend.values())}


def _final_stage(state: dict[str, Any], evals: dict[str, dict]) -> str | None:
    """Stage shipped by export, else the loop's best, else the highest scoring non-base stage."""
    metrics = state.get("metrics") or {}
    for stage in ((metrics.get("export") or {}).get("final_stage"), (metrics.get("loop") or {}).get("best_stage")):
        if stage and stage in evals:
            return str(stage)
    scored = [(primary_score(r), s) for s, r in evals.items() if s != "base" and primary_score(r) is not None]
    return max(scored)[1] if scored else None


def _examples(evals: dict[str, dict], final: str | None) -> list[dict[str, str]]:
    """First few prompts with base and final answers side by side."""
    base_answers = (evals.get("base") or {}).get("answers") or []
    if not base_answers or final is None:
        return []
    final_by_id = {a.get("id"): a for a in (evals[final].get("answers") or [])}
    rows = []
    for item in base_answers:
        match = final_by_id.get(item.get("id"))
        if match is not None:
            rows.append({"prompt": item.get("prompt", ""), "reference": item.get("reference") or "",
                         "base": item.get("answer", ""), "final": match.get("answer", "")})
        if len(rows) == EXAMPLE_COUNT:
            break
    return rows


def _summary_section(cfg: TousterConfig, state: dict[str, Any], evals: dict[str, dict], final: str | None,
                     total_usd: float) -> dict[str, Any]:
    """Headline numbers for the summary block."""
    base = primary_score(evals.get("base") or {})
    final_score = primary_score(evals.get(final) or {}) if final else None
    gain = (final_score - base) * 100 if final_score is not None and base is not None else None
    return {
        "task": cfg.task_name, "task_type": cfg.task_type.value, "description": cfg.task_description,
        "teacher": cfg.teacher_model, "judge": cfg.effective_judge_model, "student": cfg.student_model,
        "metric": (evals.get("base") or evals.get(final or "") or {}).get("primary_metric", "score"),
        "base_score": base, "final_stage": final, "final_score": final_score, "gain_points": gain,
        "total_usd": total_usd, "minutes": float(state.get("elapsed_minutes") or 0.0),
        "run_id": state.get("run_id") or "unknown", "status": state.get("status") or "not started",
        "completed": state.get("completed") or [], "smoke": cfg.smoke_test,
    }


def report_context(cfg: TousterConfig, paths: RunPaths) -> dict[str, Any]:
    """Collect everything the template needs; any missing part becomes None."""
    state = read_state(paths)
    evals = load_eval_results(paths)
    final = _final_stage(state, evals)
    costs = _cost_section(paths)
    return {
        "summary": _summary_section(cfg, state, evals, final, costs["total"] if costs else 0.0),
        "funnel": _funnel_section(state),
        "loss": _loss_section(cfg, paths),
        "scores": _score_section(evals),
        "regression": _regression_section(evals),
        "costs": costs,
        "examples": _examples(evals, final),
        "final_stage": final,
    }


def render_report(cfg: TousterConfig, paths: RunPaths) -> str:
    """Render the report HTML with plotly.js inlined once in the head."""
    from plotly.offline import get_plotlyjs

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template(TEMPLATE_NAME)
    return template.render(plotly_js=Markup(get_plotlyjs()), **report_context(cfg, paths))


def build_report(cfg: TousterConfig, paths: RunPaths) -> Path:
    """Write report.html for the run and return its path."""
    html = render_report(cfg, paths)
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(html, encoding="utf-8")
    log_event(f"Report written to {paths.report}", "success")
    return paths.report


def report_html(paths: RunPaths) -> str:
    """The saved report as a string, for IPython.display.HTML in the notebook."""
    if not paths.report.is_file():
        raise FileNotFoundError(f"{paths.report} does not exist yet; run cell 8 first.")
    return paths.report.read_text(encoding="utf-8")


def run_report_cell(cfg: TousterConfig, paths: RunPaths, force: bool = False) -> dict[str, Any]:
    """Cell 8: build the HTML report under the shared cell contract."""

    def work() -> dict[str, Any]:
        """Build the report and return its path and size."""
        path = build_report(cfg, paths)
        return {"report": str(path), "bytes": path.stat().st_size}

    return run_cell(paths, "report", work, lambda m: f"report at {m.get('report', 'n/a')}", force)
