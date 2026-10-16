"""Task metrics, the reusable evaluate() step, and notebook cell 5 (base vs SFT evaluation)."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from touster.budget import cost_rows
from touster.config import TaskType, TousterConfig
from touster.dashboard import get_dashboard, log_event, set_result
from touster.eval.judge import JUDGE_PHASE, pairwise_win_rate
from touster.eval.regression import regression_warnings, run_regression
from touster.state import RunPaths, read_json, read_jsonl, require_gpu, run_cell, write_json
from touster.text import normalize_text, parse_json_object, strip_code_fence


BASE_STAGE = "base"
SFT_STAGE = "sft"
SFT_FINAL_DIR = "final"
CODE_TIMEOUT_SECONDS = 10
CODE_MEMORY_BYTES = 1 << 30
SYNTAX_VALID = "syntax_valid"
PRIMARY_METRICS = {
    TaskType.CLASSIFICATION: "exact_match",
    TaskType.EXTRACTION: "f1",
    TaskType.QA: "f1",
    TaskType.CODE: "pass_at_1",
    TaskType.SUMMARISATION: "rouge_l",
    TaskType.OTHER: "rouge_l",
}


# ---------- per-item metrics ----------

def _mean(values: list[float]) -> float:
    """Mean of a list, 0.0 when empty."""
    return sum(values) / len(values) if values else 0.0


def exact_match(prediction: str, reference: str, lenient: bool = False) -> float:
    """1.0 if normalized texts match; lenient also accepts an exact match on the first line alone."""
    pred, ref = normalize_text(prediction), normalize_text(reference)
    if pred == ref:
        return 1.0
    if not lenient or not pred:
        return 0.0
    first_line = normalize_text(prediction.strip().splitlines()[0])
    return 1.0 if first_line == ref else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 on normalized text."""
    pred, ref = normalize_text(prediction).split(), normalize_text(reference).split()
    if not pred and not ref:
        return 1.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _flatten_fields(value: Any, prefix: str = "") -> set[tuple[str, str]]:
    """Flatten nested JSON into a set of (dotted.path, normalized value) pairs."""
    if isinstance(value, dict):
        return set().union(*(_flatten_fields(v, f"{prefix}.{k}" if prefix else str(k)) for k, v in value.items()))
    if isinstance(value, list):
        return set().union(*(_flatten_fields(v, f"{prefix}[]") for v in value))
    text = value if isinstance(value, str) else json.dumps(value)
    return {(prefix, normalize_text(text))}


def field_f1(prediction: dict[str, Any], reference: dict[str, Any]) -> float:
    """F1 over (field path, value) pairs of two JSON objects."""
    pred, ref = _flatten_fields(prediction), _flatten_fields(reference)
    if not pred and not ref:
        return 1.0
    return 2 * len(pred & ref) / (len(pred) + len(ref))


def rouge_l(prediction: str, reference: str, scorer: Any = None) -> float:
    """ROUGE-L F-measure from the rouge_score package."""
    if scorer is None:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, prediction)["rougeL"].fmeasure)


def syntax_valid(code: str) -> bool:
    """True if the code parses as Python."""
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def _sandbox_env() -> dict[str, str]:
    """Minimal environment for untrusted code: no API keys or tokens from the parent process."""
    env = {"PATH": os.defpath, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _limit_resources() -> None:
    """POSIX only: cap CPU time, memory and file size of the child process."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (CODE_TIMEOUT_SECONDS, CODE_TIMEOUT_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (CODE_MEMORY_BYTES, CODE_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CODE_MEMORY_BYTES, CODE_MEMORY_BYTES))


def passes_tests(code: str, tests: str, timeout: float = CODE_TIMEOUT_SECONDS) -> bool:
    """Run model-written code plus its asserts in a separate Python process; True if it exits cleanly.

    The child gets a temp working dir, no inherited secrets, no stdin and (on Linux) CPU/memory limits.
    It is not a full sandbox: network and file reads are not blocked.
    """
    with tempfile.TemporaryDirectory(prefix="touster_code_") as folder:
        script = Path(folder) / "candidate.py"
        script.write_text(f"{code}\n\n{tests}\n", encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=folder, env=_sandbox_env(), stdin=subprocess.DEVNULL,
                capture_output=True, timeout=timeout, check=False,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


# ---------- task-level scores ----------

def primary_metric(task_type: TaskType | str) -> str:
    """Name of the headline metric for a task type."""
    return PRIMARY_METRICS[TaskType(task_type)]


def resolve_primary(task_type: TaskType | str, scores: dict[str, float]) -> str:
    """Primary metric actually present in scores (code without tests falls back to syntax_valid)."""
    name = primary_metric(task_type)
    if name not in scores and SYNTAX_VALID in scores:
        return SYNTAX_VALID
    return name


def _classification_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Lenient exact match for labels."""
    return {"exact_match": _mean([exact_match(p, r, lenient=True) for p, r in zip(predictions, references)])}


def _qa_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Token F1 and strict normalized exact match."""
    return {
        "f1": _mean([token_f1(p, r) for p, r in zip(predictions, references)]),
        "exact_match": _mean([exact_match(p, r) for p, r in zip(predictions, references)]),
    }


def _extraction_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """JSON validity rate and field-level F1; records whose reference is not JSON are left out of F1."""
    parsed = [parse_json_object(p) for p in predictions]
    f1s, bad_refs = [], 0
    for pred, ref_text in zip(parsed, references):
        ref = parse_json_object(ref_text)
        if ref is None:
            bad_refs += 1
            continue
        f1s.append(field_f1(pred, ref) if pred is not None else 0.0)
    if bad_refs:
        log_event(f"{bad_refs} eval references are not JSON objects and were left out of field F1", "warning")
    return {"json_validity": _mean([1.0 if p is not None else 0.0 for p in parsed]), "f1": _mean(f1s)}


def _code_scores(predictions: list[str], tests: list[str | None]) -> dict[str, float]:
    """pass@1 on records with tests plus syntax-valid rate on all records."""
    code = [strip_code_fence(p) for p in predictions]
    scores = {SYNTAX_VALID: _mean([1.0 if syntax_valid(c) else 0.0 for c in code])}
    with_tests = [(c, t) for c, t in zip(code, tests) if t]
    if with_tests:
        scores["pass_at_1"] = _mean([1.0 if passes_tests(c, t) else 0.0 for c, t in with_tests])
    else:
        log_event("No eval record has metadata 'tests'; code is scored by syntax-valid rate instead of pass@1", "warning")
    return scores


def _rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Mean ROUGE-L F-measure."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return {"rouge_l": _mean([rouge_l(p, r, scorer) for p, r in zip(predictions, references)])}


def score_task(
    task_type: TaskType | str,
    predictions: list[str],
    references: list[str],
    tests: list[str | None] | None = None,
) -> dict[str, float]:
    """Compute the task metrics (all 0..1); tests are per-record python asserts, used only for code."""
    if len(predictions) != len(references):
        raise ValueError("score_task: predictions and references must have the same length")
    kind = TaskType(task_type)
    if kind == TaskType.CLASSIFICATION:
        return _classification_scores(predictions, references)
    if kind == TaskType.EXTRACTION:
        return _extraction_scores(predictions, references)
    if kind == TaskType.QA:
        return _qa_scores(predictions, references)
    if kind == TaskType.CODE:
        return _code_scores(predictions, tests or [None] * len(predictions))
    return _rouge_scores(predictions, references)


def record_tests(record: dict[str, Any]) -> str | None:
    """Python assert tests from a record's metadata, joined into one string, or None."""
    tests = (record.get("metadata") or {}).get("tests")
    if isinstance(tests, list):
        tests = "\n".join(str(t) for t in tests)
    return tests if isinstance(tests, str) and tests.strip() else None


# ---------- evaluate one stage ----------

def adapter_context(bundle: Any, use_adapter: bool) -> AbstractContextManager[Any]:
    """Context that disables the LoRA adapter (peft PeftModel.disable_adapter) when use_adapter is False."""
    if use_adapter or not hasattr(bundle.model, "disable_adapter"):
        return nullcontext()
    return bundle.model.disable_adapter()


def _generate_answers(bundle: Any, prompts: list[str], cfg: TousterConfig) -> list[str]:
    """Greedy single answer per prompt from the student."""
    from touster.train.sft import generate_responses

    outputs = generate_responses(bundle, prompts, cfg, temperature=0.0, num_samples=1)
    return [samples[0] if samples else "" for samples in outputs]


def _judge_cost_since(costs: Path, start_row: int) -> float:
    """USD of judge-phase calls logged after the first start_row rows of costs.jsonl."""
    rows = cost_rows(costs)[start_row:]
    return sum(float(r.get("cost_usd", 0.0)) for r in rows if r.get("phase") == JUDGE_PHASE)


def _base_answers_from_disk(paths: RunPaths, ids: list[str]) -> list[str] | None:
    """Base answers aligned to ids from eval/base.json, or None if missing or incomplete."""
    path = paths.eval_result(BASE_STAGE)
    if not path.is_file():
        return None
    by_id = {a["id"]: a["answer"] for a in read_json(path).get("answers", [])}
    return [by_id[i] for i in ids] if all(i in by_id for i in ids) else None


def _judge_vs_base(
    stage: str,
    cfg: TousterConfig,
    paths: RunPaths,
    client: Any,
    records: list[dict[str, Any]],
    answers: list[str],
    base_answers: list[str] | None,
) -> float | None:
    """Pairwise win rate of this stage against base; None for base itself or when base answers are missing."""
    if stage == BASE_STAGE:
        return None
    base = base_answers if base_answers is not None else _base_answers_from_disk(paths, [r["id"] for r in records])
    if base is None or len(base) != len(records) or client is None:
        log_event(f"Judge vs base skipped for '{stage}': base answers or LLM client not available", "warning")
        return None
    return pairwise_win_rate(
        client, cfg, [r["prompt"] for r in records], answers, base, [r.get("response") for r in records]
    )


def _regression_for(stage: str, bundle: Any, cfg: TousterConfig, paths: RunPaths, use_adapter: bool) -> dict[str, float]:
    """Regression subset scores (skipped in smoke mode), warning on drops vs base."""
    if cfg.smoke_test:
        return {}
    with adapter_context(bundle, use_adapter):
        scores = run_regression(bundle, cfg)
    base_path = paths.eval_result(BASE_STAGE)
    if stage != BASE_STAGE and scores and base_path.is_file():
        for message in regression_warnings(read_json(base_path).get("regression", {}), scores):
            log_event(f"Regression in '{stage}': {message}", "warning")
    return scores


def load_eval_records(paths: RunPaths) -> list[dict[str, Any]]:
    """Read the held-out eval set (only eval code may do this)."""
    if not paths.golden_eval.is_file():
        raise FileNotFoundError(f"Eval set not found at {paths.golden_eval}. Run cell 2 (golden data) first.")
    records = read_jsonl(paths.golden_eval)
    if not records:
        raise ValueError(f"Eval set {paths.golden_eval} is empty. Add more golden records and rerun cell 2.")
    return records


def evaluate(
    bundle: Any,
    stage: str,
    cfg: TousterConfig,
    paths: RunPaths,
    client: Any,
    base_answers: list[str] | None = None,
    use_adapter: bool = True,
    regression: bool = True,
) -> dict[str, Any]:
    """Generate, score, judge vs base and (if regression) run the lm_eval subset for one stage; write eval/<stage>.json."""
    started, cost_start = time.monotonic(), len(cost_rows(paths.costs))
    records = load_eval_records(paths)
    log_event(f"Evaluating '{stage}' on {len(records)} eval prompts")
    with adapter_context(bundle, use_adapter):
        answers = _generate_answers(bundle, [r["prompt"] for r in records], cfg)
    references = [r.get("response", "") for r in records]
    scores = score_task(cfg.task_type, answers, references, [record_tests(r) for r in records])
    metric = resolve_primary(cfg.task_type, scores)
    result = {
        "stage": stage,
        "primary_metric": metric,
        "scores": scores,
        "judge_win_rate_vs_base": _judge_vs_base(stage, cfg, paths, client, records, answers, base_answers),
        "regression": _regression_for(stage, bundle, cfg, paths, use_adapter) if regression else {},
        "n": len(records),
        "cost_usd": round(_judge_cost_since(paths.costs, cost_start), 6),
        "minutes": round((time.monotonic() - started) / 60.0, 2),
        "answers": [
            {"id": r["id"], "prompt": r["prompt"], "reference": ref, "answer": a}
            for r, ref, a in zip(records, references, answers)
        ],
    }
    write_json(paths.eval_result(stage), result)
    set_result(stage, scores[metric], metric)
    log_event(f"'{stage}' {metric} = {scores[metric]:.4f}", "success")
    return result


# ---------- cell 5 ----------

def _load_bundle(cfg: TousterConfig, adapter: Path | None) -> Any:
    """Load the student for inference, with the SFT adapter when one is given."""
    from touster.train.sft import load_student

    return load_student(cfg, adapter_path=adapter, for_training=False)


def _sft_adapter(paths: RunPaths) -> Path | None:
    """Final SFT adapter folder, or None when SFT did not finish."""
    folder = paths.sft_checkpoints / SFT_FINAL_DIR
    return folder if folder.is_dir() else None


def show_comparison(results: list[dict[str, Any]]) -> None:
    """Print stage results side by side as a rich table."""
    from rich.table import Table

    table = Table(title="evaluation", show_header=True, header_style="bold")
    table.add_column("metric")
    for result in results:
        table.add_column(result["stage"], justify="right")
    metric_names = sorted({name for r in results for name in r["scores"]})
    for name in metric_names:
        table.add_row(name, *(f"{r['scores'][name]:.4f}" if name in r["scores"] else "-" for r in results))
    for name in sorted({name for r in results for name in r.get("regression", {})}):
        table.add_row(f"regression {name}", *(f"{r['regression'].get(name, float('nan')):.4f}" for r in results))
    win_rates = [r.get("judge_win_rate_vs_base") for r in results]
    table.add_row("judge win rate vs base", *("-" if w is None else f"{w:.2f}" for w in win_rates))
    table.add_row("n", *(str(r["n"]) for r in results))
    table.add_row("judge cost USD", *(f"{r['cost_usd']:.4f}" for r in results))
    get_dashboard().console.print(table)


def _cell_metrics(base: dict[str, Any], sft: dict[str, Any] | None) -> dict[str, Any]:
    """Summary stored in state.json for the eval cell."""
    metric = base["primary_metric"]
    base_score = base["scores"][metric]
    sft_score = sft["scores"].get(metric) if sft else None
    return {
        "primary_metric": metric,
        "base": base_score,
        "sft": sft_score,
        "gain_points": None if sft_score is None else round(100.0 * (sft_score - base_score), 2),
        "judge_win_rate_vs_base": sft.get("judge_win_rate_vs_base") if sft else None,
        "sft_evaluated": sft is not None,
    }


def _eval_work(cfg: TousterConfig, paths: RunPaths, client: Any, budget: Any) -> dict[str, Any]:
    """Score base (unless eval/base.json exists) and SFT with one loaded model, then show both."""
    require_gpu("eval")
    if budget is not None:
        budget.check("eval")
    adapter = _sft_adapter(paths)
    base_path = paths.eval_result(BASE_STAGE)
    have_base = base_path.is_file()
    if have_base:
        log_event("eval/base.json exists, reusing the base scores")
    if adapter is None:
        log_event("No SFT adapter found (SFT skipped or stopped on budget): evaluating the base model only", "warning")
    bundle = _load_bundle(cfg, adapter) if (adapter is not None or not have_base) else None
    try:
        base = read_json(base_path) if have_base else evaluate(bundle, BASE_STAGE, cfg, paths, client, use_adapter=False)
        sft = evaluate(bundle, SFT_STAGE, cfg, paths, client, use_adapter=True) if adapter is not None else None
    finally:
        from touster.train.sft import release_student

        release_student(bundle)
    show_comparison([base] if sft is None else [base, sft])
    return _cell_metrics(base, sft)


def _summary(metrics: dict[str, Any]) -> str:
    """One-line summary for the eval cell."""
    if not metrics:
        return "no eval results"
    line = f"{metrics['primary_metric']}: base {metrics['base']:.4f}"
    if metrics.get("sft") is not None:
        line += f", sft {metrics['sft']:.4f} ({metrics['gain_points']:+.1f} points)"
    return line


def run_eval_cell(cfg: TousterConfig, paths: RunPaths, client: Any, budget: Any, force: bool = False) -> dict[str, Any]:
    """Cell 5: evaluate the base model (once, resumable) and the SFT model, saved and shown side by side."""
    return run_cell(paths, "eval", lambda: _eval_work(cfg, paths, client, budget), _summary, force=force)
