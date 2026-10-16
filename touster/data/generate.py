"""Cell 3: Evol-Instruct style synthetic data, calibration with a cost estimate, then filtering."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from rich.table import Table

from touster.budget import (
    BudgetExceeded,
    BudgetTracker,
    Calibration,
    Estimate,
    estimate_generation,
    estimate_training,
    sft_total_steps,
)
from touster.config import SMOKE_SYNTHETIC_RECORDS, TaskType, TousterConfig, is_local_model
from touster.dashboard import get_dashboard, log_event, set_progress
from touster.data.filters import judge_scores, run_filters
from touster.llm import DEFAULT_WORKERS, LOCAL_WORKERS, LLMClient, LLMResult
from touster.state import RunPaths, detect_gpu, load_training_records, read_jsonl, run_cell, write_jsonl
from touster.text import find_json_object

EVOLUTIONS = {
    "deepen": "Make it need more careful reasoning or a harder judgement than the draft.",
    "add_constraints": "Add one or two realistic details, conditions or requirements that make it more specific.",
    "concretise": "Replace general wording with specific, concrete details such as names, numbers or situations.",
    "broaden": "Move it to a different but related topic or situation that the task should also handle.",
}
EVOLUTION_NAMES = tuple(EVOLUTIONS)
ROUND_SIZE = 32
MIN_ROUND_REQUESTS = 4
ROUND_CAP_FACTOR = 2
EXTRA_ROUNDS = 5
STYLE_EXAMPLES = 3
CALIBRATION_EXAMPLES = 20
PROBE_STEPS = 50
PROMPT_TEMPERATURE = 0.9
RESPONSE_TEMPERATURE = 0.3
PROMPT_MAX_TOKENS = 600
RESPONSE_MAX_TOKENS = 1024
ESTIMATE_SAFETY_FACTOR = 1.1
SEARCH_STEPS_CAP = 64
MAX_LABEL_WORDS = 5

Record = dict[str, Any]
SaveFn = Callable[[list[Record]], None]


@dataclass(frozen=True)
class Usage:
    """Money and tokens used by a group of teacher calls."""

    usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Sum two usages."""
        return Usage(
            self.usd + other.usd,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    @classmethod
    def of(cls, results: list[LLMResult]) -> Usage:
        """Total usage of a list of call results."""
        return cls(
            sum(r.cost_usd for r in results),
            sum(r.prompt_tokens for r in results),
            sum(r.completion_tokens for r in results),
        )


@dataclass(frozen=True)
class CalibrationRun:
    """Calibration measurements (generation incl. training probe, quality judging) and the records made."""

    generation: Calibration
    quality: Calibration
    records: list[Record]
    malformed: int


def prompt_messages(cfg: TousterConfig, seed: Record, evolution: str) -> list[dict[str, str]]:
    """Teacher prompt: write a new input from a seed, then evolve it."""
    content = (
        f"Task:\n{cfg.task_description}\n\n"
        f"Here is one example input for this task:\n<example>\n{seed['prompt']}\n</example>\n\n"
        "Write ONE new input for this task yourself: the actual text the model will receive, the same kind "
        "of text as the example (if the example is a product review, write a product review in the "
        "customer's own words). Never write an instruction, a request to write something, or a description "
        "of what the input should contain.\n"
        "First write a draft about something different from the example. Then rewrite the draft with this "
        f"change: {EVOLUTIONS[evolution]} The rewritten input must still be the same kind of text as the "
        "example. Do not include the answer.\n"
        'Reply with JSON only: {"draft": "<first draft>", "prompt": "<rewritten input>"}'
    )
    return [{"role": "user", "content": content}]


def response_messages(cfg: TousterConfig, prompt: str, style: list[Record]) -> list[dict[str, str]]:
    """Teacher prompt: answer one input in the same format as the golden examples."""
    examples = "\n\n".join(f"Input: {r['prompt']}\nAnswer: {r['response']}" for r in style)
    label_rule = (
        "Answer with the label only, written exactly as in the examples.\n"
        if cfg.task_type == TaskType.CLASSIFICATION
        else ""
    )
    content = (
        f"Task:\n{cfg.task_description}\n\n"
        f"Correct answers look like these (match their format and length):\n\n{examples}\n\n"
        f"{label_rule}Now answer this input.\nInput: {prompt}\n\n"
        'Reply with JSON only: {"response": "<your answer>"}'
    )
    return [{"role": "user", "content": content}]


def extract_field(text: str, field: str) -> str | None:
    """Return a non-empty string field from teacher JSON output, or None if malformed."""
    parsed = find_json_object(text)
    value = parsed.get(field) if parsed else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def valid_response(response: str, task_type: TaskType) -> bool:
    """Reject answers that are themselves a JSON wrapper, and classification answers that are not a short label."""
    text = response.strip()
    if extract_field(text, "response") is not None:
        return False
    if task_type == TaskType.CLASSIFICATION:
        return not text.startswith(("{", "[")) and len(text.split()) <= MAX_LABEL_WORDS
    return True


def _new_prompts(
    cfg: TousterConfig, client: LLMClient, plan: list[tuple[Record, str]]
) -> tuple[list[tuple[Record, str, str]], Usage]:
    """Ask the teacher for evolved prompts; return the well-formed (seed, evolution, prompt) items."""
    results = client.chat_many(
        [prompt_messages(cfg, seed, evo) for seed, evo in plan], phase="generation",
        temperature=PROMPT_TEMPERATURE, max_tokens=PROMPT_MAX_TOKENS, json_mode=True,
    )
    items = [(seed, evo, extract_field(r.text, "prompt")) for (seed, evo), r in zip(plan, results)]
    return [(seed, evo, p) for seed, evo, p in items if p is not None], Usage.of(results)


def generate_round(
    cfg: TousterConfig, client: LLMClient, seeds: list[Record], k: int,
    rng: random.Random, style: list[Record], next_index: int,
) -> tuple[list[Record], int, Usage]:
    """Make up to k records; return (records, malformed teacher outputs dropped, usage)."""
    plan = [(rng.choice(seeds), rng.choice(EVOLUTION_NAMES)) for _ in range(k)]
    items, usage = _new_prompts(cfg, client, plan)
    results = client.chat_many(
        [response_messages(cfg, p, style) for _, _, p in items], phase="generation",
        temperature=RESPONSE_TEMPERATURE, max_tokens=RESPONSE_MAX_TOKENS, json_mode=True,
    ) if items else []
    records: list[Record] = []
    for (seed, evo, prompt), result in zip(items, results):
        response = extract_field(result.text, "response")
        if response is not None and valid_response(response, cfg.task_type):
            records.append({
                "id": f"s-{next_index + len(records):06d}", "prompt": prompt, "response": response,
                "seed_id": seed["id"], "evolution": evo, "quality_score": None, "teacher": cfg.teacher_model,
            })
    malformed = k - len(records)
    if malformed:
        log_event(f"Dropped {malformed} malformed teacher answer(s) (bad or empty JSON) this round", "warning")
    return records, malformed, usage + Usage.of(results)


def next_index(records: list[Record]) -> int:
    """Number for the next s-NNNNNN id: one past the highest existing synthetic id."""
    numbers = [int(r["id"][2:]) for r in records if str(r.get("id", "")).startswith("s-") and r["id"][2:].isdigit()]
    return max(numbers, default=0) + 1


def default_max_rounds(n_target: int) -> int:
    """Hard cap on generation rounds: room for about half of each round to be dropped, plus a few top-up rounds."""
    return ROUND_CAP_FACTOR * math.ceil(n_target / ROUND_SIZE) + EXTRA_ROUNDS


def generate_synthetic(
    cfg: TousterConfig, client: LLMClient, seeds: list[Record], n_target: int, rng: random.Random,
    start_index: int = 1, on_save: SaveFn | None = None, max_rounds: int | None = None,
) -> tuple[list[Record], dict[str, Any]]:
    """Generate up to n_target records in rounds; save after each round and before a budget stop re-raises."""
    if not seeds:
        raise ValueError("No seed records; run the golden cell (cell 2) first.")
    style = rng.sample(seeds, min(STYLE_EXAMPLES, len(seeds)))
    cap = max_rounds or default_max_rounds(n_target)
    records: list[Record] = []
    usage, malformed, rounds, stop = Usage(), 0, 0, "target reached"
    started = time.monotonic()
    try:
        while len(records) < n_target:
            if rounds >= cap:
                stop = f"hit the cap of {cap} rounds"
                break
            rounds += 1
            k = min(ROUND_SIZE, max(MIN_ROUND_REQUESTS, n_target - len(records)))
            new, bad, used = generate_round(cfg, client, seeds, k, rng, style, start_index + len(records))
            records, malformed, usage = (records + new)[:n_target], malformed + bad, usage + used
            if on_save is not None:
                on_save(records)
            set_progress("synthetic generation", len(records), n_target)
            if not new:
                stop = "a round added nothing"
                break
    except BudgetExceeded:
        if on_save is not None:
            on_save(records)
        raise
    if len(records) < n_target:
        log_event(f"Generation stopped at {len(records)}/{n_target}: {stop}", "warning")
    stats = {
        "rounds": rounds, "made": len(records), "malformed": malformed, "stop_reason": stop,
        "usd": usage.usd, "seconds": time.monotonic() - started,
        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
    }
    return records, stats


def _probe(cfg: TousterConfig, calib: Calibration, records: list[Record]) -> Calibration:
    """Add training-probe timings to a calibration when a GPU is present."""
    if not detect_gpu()["available"]:
        log_event("No GPU here, so the training time estimate is left out", "warning")
        return calib
    from touster.train.sft import probe_training

    probe = probe_training(cfg, records, steps=PROBE_STEPS)
    return replace(
        calib, train_steps=int(probe["steps"]), train_seconds=float(probe["seconds"]),
        train_tokens_per_sec=float(probe["tokens_per_sec"]), peak_vram_gb=float(probe["peak_vram_gb"]),
    )


def calibrate(
    cfg: TousterConfig, client: LLMClient, seeds: list[Record], n: int = CALIBRATION_EXAMPLES,
    rng: random.Random | None = None, on_save: SaveFn | None = None, existing: list[Record] | None = None,
) -> CalibrationRun:
    """Top up saved records to n, judge the unscored ones, probe training if a GPU exists, and measure costs."""
    existing = existing or []
    save_all = (lambda new: on_save(existing + new)) if on_save is not None else None
    made, stats = generate_synthetic(
        cfg, client, seeds, n - len(existing), rng or random.Random(cfg.seed),
        start_index=next_index(existing), on_save=save_all,
    )
    if not made:
        raise RuntimeError(
            "Calibration produced no usable examples. Check that the teacher model answers in JSON "
            "and that task_description is clear, then rerun the cell."
        )
    records = existing + made
    n_judged = sum(r.get("quality_score") is None for r in records)
    started = time.monotonic()
    scores, quality_usd = judge_scores(cfg, client, records)
    quality_seconds = time.monotonic() - started
    records = [r if s is None else {**r, "quality_score": s} for r, s in zip(records, scores)]
    if on_save is not None:
        on_save(records)
    generation = Calibration(
        examples=len(made), usd=stats["usd"], seconds=stats["seconds"],
        prompt_tokens=stats["prompt_tokens"], completion_tokens=stats["completion_tokens"],
    )
    quality = Calibration(examples=n_judged, usd=quality_usd, seconds=quality_seconds, prompt_tokens=0, completion_tokens=0)
    return CalibrationRun(_probe(cfg, generation, records), quality, records, stats["malformed"])


def _phase_totals(cost_log: Path, phase: str) -> tuple[float, float, int, int]:
    """Sum (usd, call seconds, prompt tokens, completion tokens) over costs.jsonl rows of one phase."""
    rows = [r for r in read_jsonl(cost_log) if r.get("phase") == phase] if cost_log.is_file() else []
    return (
        sum(float(r.get("cost_usd", 0.0)) for r in rows),
        sum(float(r.get("seconds", 0.0)) for r in rows),
        sum(int(r.get("prompt_tokens", 0)) for r in rows),
        sum(int(r.get("completion_tokens", 0)) for r in rows),
    )


def calibration_from_log(cfg: TousterConfig, paths: RunPaths, records: list[Record]) -> CalibrationRun:
    """Rebuild calibration from costs.jsonl for records saved by an earlier attempt (no new teacher calls)."""
    workers = LOCAL_WORKERS if is_local_model(cfg.teacher_model) else DEFAULT_WORKERS
    gen_usd, gen_sec, gen_in, gen_out = _phase_totals(paths.costs, "generation")
    q_usd, q_sec, _, _ = _phase_totals(paths.costs, "quality")
    n_judged = sum(r.get("quality_score") is not None for r in records)
    # Calls ran in parallel, so summed call time over the worker count approximates wall time.
    generation = Calibration(
        examples=len(records), usd=gen_usd, seconds=gen_sec / workers, prompt_tokens=gen_in, completion_tokens=gen_out,
    )
    quality = Calibration(examples=n_judged, usd=q_usd, seconds=q_sec / workers, prompt_tokens=0, completion_tokens=0)
    return CalibrationRun(_probe(cfg, generation, records), quality, records, 0)


def full_run_estimate(run: CalibrationRun, cfg: TousterConfig, n_total: int) -> dict[str, Estimate]:
    """Estimate generation and judging for the examples still to make, plus SFT on n_total examples."""
    n_more = max(0, n_total - len(run.records))
    return {
        "generation": estimate_generation(run.generation, n_more),
        "quality judging": estimate_generation(run.quality, n_more),
        "training": estimate_training(run.generation, sft_total_steps(n_total, cfg)),
    }


def _total(parts: dict[str, Estimate]) -> Estimate:
    """Sum estimate parts."""
    total = Estimate(0.0, 0.0)
    for part in parts.values():
        total = total + part
    return total


def _fits(run: CalibrationRun, cfg: TousterConfig, budget: BudgetTracker, n_total: int) -> bool:
    """True if the padded estimate for n_total examples fits the generation budget."""
    est = _total(full_run_estimate(run, cfg, n_total))
    padded = Estimate(est.usd * ESTIMATE_SAFETY_FACTOR, est.minutes * ESTIMATE_SAFETY_FACTOR)
    return budget.can_afford(padded, "generation")


def fit_target(run: CalibrationRun, cfg: TousterConfig, budget: BudgetTracker, n_target: int) -> int:
    """Largest total example count, between the calibration size and n_target, that fits the budget."""
    low, high = len(run.records), max(n_target, len(run.records))
    if _fits(run, cfg, budget, high):
        return high
    for _ in range(SEARCH_STEPS_CAP):
        if high - low <= 1:
            break
        middle = (low + high) // 2
        low, high = (middle, high) if _fits(run, cfg, budget, middle) else (low, middle)
    return low


def show_estimate(parts: dict[str, Estimate], left: Estimate, n_total: int, calib: Calibration | None = None) -> None:
    """Print the full-run estimate next to what the budget has left, plus the training probe when one ran."""
    table = Table(title=f"Estimate for the full run ({n_total} examples)")
    for column in ("part", "USD", "minutes"):
        table.add_column(column, justify="right" if column != "part" else "left")
    for name, est in {**parts, "total": _total(parts)}.items():
        table.add_row(name, f"{est.usd:.4f}", f"{est.minutes:.1f}")
    table.add_row("budget left", "no cap" if math.isinf(left.usd) else f"{left.usd:.4f}",
                  "no cap" if math.isinf(left.minutes) else f"{left.minutes:.1f}")
    console = get_dashboard().console
    console.print(table)
    if calib is not None and calib.train_steps > 0:
        console.print(
            f"training probe: {calib.train_tokens_per_sec:,.0f} tokens/sec, peak VRAM {calib.peak_vram_gb:.1f} GB"
        )


def plan_size(cfg: TousterConfig, budget: BudgetTracker, run: CalibrationRun, n_target: int) -> int:
    """Show the estimate and return n_target, scaled down when the full run would not fit the budget."""
    n_total = fit_target(run, cfg, budget, n_target)
    if n_total < n_target:
        log_event(
            f"The full run ({n_target} examples) would go over budget; generating {n_total} instead. "
            "Raise max_usd/max_minutes to get more.", "warning",
        )
    show_estimate(full_run_estimate(run, cfg, n_total), budget.remaining("generation"), n_total, run.generation)
    return n_total


class RunDeclined(RuntimeError):
    """Raised when the user does not confirm the full synthetic run."""


def _no_stdin_errors() -> tuple[type[BaseException], ...]:
    """Exceptions input() raises without an interactive stdin (plain EOF, or IPython under nbconvert/papermill)."""
    try:
        from IPython.core.error import StdinNotImplementedError
    except ImportError:
        return (EOFError,)
    return (EOFError, StdinNotImplementedError)


def ask_user(message: str) -> bool:
    """Ask a yes/no question with input(); returns False when there is no interactive input."""
    try:
        answer = input(f"{message} [y/N] ")
    except _no_stdin_errors():
        log_event(
            "No interactive input available, so the run was not confirmed. "
            "Set smoke_test=True to skip the prompt, or run the notebook interactively.", "warning",
        )
        return False
    return answer.strip().lower() in {"y", "yes"}


def _load_seeds(paths: RunPaths) -> list[Record]:
    """Load seed records through the leakage-checked training loader."""
    if not paths.golden_seed.is_file():
        raise FileNotFoundError(f"{paths.golden_seed} not found; run the golden cell (cell 2) first.")
    return load_training_records(paths.golden_seed, paths)


def _calibration_run(
    cfg: TousterConfig, paths: RunPaths, client: LLMClient, seeds: list[Record],
    rng: random.Random, n_target: int, save: SaveFn,
) -> CalibrationRun:
    """Reuse records saved by an earlier attempt; calibrate only if fewer than the calibration size exist."""
    seed_ids = {s["id"] for s in seeds}
    saved = read_jsonl(paths.synthetic_raw) if paths.synthetic_raw.is_file() else []
    existing = [r for r in saved if r.get("seed_id") in seed_ids][:n_target]
    if len(existing) < len(saved):
        log_event(f"Dropped {len(saved) - len(existing)} saved examples whose seed is no longer in the seed set", "warning")
    n_calibration = min(CALIBRATION_EXAMPLES, n_target)
    if existing:
        log_event(f"Reusing {len(existing)} synthetic examples saved by an earlier attempt", "info")
    if len(existing) >= n_calibration:
        return calibration_from_log(cfg, paths, existing)
    return calibrate(cfg, client, seeds, n=n_calibration, rng=rng, on_save=save, existing=existing)


def _synthetic_work(
    cfg: TousterConfig, paths: RunPaths, client: LLMClient, budget: BudgetTracker, confirm: Callable[[str], bool]
) -> dict[str, Any]:
    """Calibrate, confirm, generate the rest, filter and save; returns the cell metrics."""
    seeds = _load_seeds(paths)
    rng = random.Random(cfg.seed)
    n_target = SMOKE_SYNTHETIC_RECORDS if cfg.smoke_test else cfg.target_examples

    def save(records: list[Record]) -> None:
        """Write the raw set so far."""
        write_jsonl(paths.synthetic_raw, records)

    run = _calibration_run(cfg, paths, client, seeds, rng, n_target, save)
    n_total = plan_size(cfg, budget, run, n_target)
    n_more = max(0, n_total - len(run.records))
    question = f"Calibration done ({len(run.records)} examples saved). Generate {n_more} more and filter?"
    confirmed = True
    if not cfg.smoke_test:
        with get_dashboard().pause():
            confirmed = confirm(question)
    if not confirmed:
        raise RunDeclined(
            f"Full run not confirmed; {len(run.records)} examples are saved in {paths.synthetic_raw.name} "
            "and nothing more was spent. Rerun the cell to continue from them."
        )
    more, stats = generate_synthetic(
        cfg, client, seeds, n_more, rng, start_index=next_index(run.records),
        on_save=lambda new: save(run.records + new),
    )
    raw = run.records + more
    save(raw)
    clean, funnel = run_filters(cfg, client, raw, paths)
    write_jsonl(paths.synthetic_clean, clean)
    return {
        "raw": len(raw), "clean": len(clean), "funnel": funnel, "n_target": n_target, "planned": n_total,
        "malformed": run.malformed + stats["malformed"], "rounds": stats["rounds"],
    }


def run_synthetic_cell(
    cfg: TousterConfig, paths: RunPaths, client: LLMClient, budget: BudgetTracker,
    confirm: Callable[[str], bool] = ask_user, force: bool = False,
) -> dict[str, Any]:
    """Cell 3: calibrate, show the estimate, confirm, generate and filter the synthetic set."""

    def work() -> dict[str, Any]:
        """Run the cell body."""
        return _synthetic_work(cfg, paths, client, budget, confirm)

    def summary(metrics: dict[str, Any]) -> str:
        """One-line result."""
        return f"{metrics.get('raw', 0)} raw -> {metrics.get('clean', 0)} clean examples."

    try:
        return run_cell(paths, "synthetic", work, summary, force)
    except RunDeclined as err:
        log_event(str(err), "warning")
        print(f"[synthetic] stopped: {err}")
        return {"declined": True}
