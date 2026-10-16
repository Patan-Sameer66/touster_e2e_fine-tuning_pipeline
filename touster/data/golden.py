"""Cell 2: load and validate the golden dataset, then split it into seed and eval sets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rich.table import Table

from touster.config import SMOKE_GOLDEN_RECORDS, TousterConfig
from touster.dashboard import get_dashboard, log_event
from touster.state import RunPaths, is_done, load_state, run_cell, unmark, update_state, write_jsonl

MIN_GOLDEN_RECORDS = 20
RECOMMENDED_GOLDEN_RECORDS = 50
SAMPLE_ROWS = 3
SAMPLE_CHARS = 80
REQUIRED_FIELDS = ("id", "prompt", "response")
DOWNSTREAM_CELLS = ("synthetic", "sft", "eval", "loop", "export", "report")
MAX_LISTED_PROBLEMS = 20


class GoldenDataError(ValueError):
    """Raised when the golden file is missing, malformed or too small."""


def _record_problems(record: Any, line_no: int) -> list[str]:
    """Return schema problems for one parsed line."""
    if not isinstance(record, dict):
        return [f"line {line_no}: expected a JSON object, got {type(record).__name__}"]
    where = f"line {line_no} (id={record.get('id')!r})"
    problems = []
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if field not in record:
            problems.append(f"{where}: missing field '{field}'")
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"{where}: field '{field}' must be a non-empty string")
    if "metadata" in record and not isinstance(record["metadata"], dict):
        problems.append(f"{where}: field 'metadata' must be an object if present")
    return problems


def _duplicate_problems(records: list[tuple[int, dict[str, Any]]]) -> list[str]:
    """Return one problem per repeated id or repeated prompt, naming both lines."""
    problems = []
    seen_ids: dict[str, int] = {}
    seen_prompts: dict[str, int] = {}
    for line_no, record in records:
        rid, prompt = record["id"], record["prompt"].strip()
        if rid in seen_ids:
            problems.append(f"line {line_no}: duplicate id {rid!r} (first seen on line {seen_ids[rid]})")
        if prompt in seen_prompts:
            problems.append(f"line {line_no} (id={rid!r}): same prompt as line {seen_prompts[prompt]}")
        seen_ids.setdefault(rid, line_no)
        seen_prompts.setdefault(prompt, line_no)
    return problems


def _parse_lines(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    """Parse every non-blank line, returning (line_no, record) pairs and problems."""
    parsed: list[tuple[int, dict[str, Any]]] = []
    problems: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                problems.append(f"line {line_no}: invalid JSON ({err.msg})")
                continue
            found = _record_problems(record, line_no)
            problems.extend(found)
            if not found:
                parsed.append((line_no, record))
    return parsed, problems


def _raise_if_problems(path: Path, problems: list[str]) -> None:
    """Raise GoldenDataError listing the problems, if any."""
    if not problems:
        return
    shown = problems[:MAX_LISTED_PROBLEMS]
    more = len(problems) - len(shown)
    tail = f"\n  ... and {more} more" if more > 0 else ""
    raise GoldenDataError(f"{path} has {len(problems)} problem(s); fix them and rerun:\n  " + "\n  ".join(shown) + tail)


def load_golden(path: Path, smoke: bool = False) -> list[dict[str, Any]]:
    """Load and validate the golden JSONL; in smoke mode keep the first SMOKE_GOLDEN_RECORDS records."""
    path = Path(path)
    if not path.is_file():
        raise GoldenDataError(f"Golden file not found: {path}. Set golden_path to your JSONL file.")
    parsed, problems = _parse_lines(path)
    _raise_if_problems(path, problems)
    _raise_if_problems(path, _duplicate_problems(parsed))
    records = [record for _, record in parsed]
    if smoke:
        return records[:SMOKE_GOLDEN_RECORDS]
    if len(records) < MIN_GOLDEN_RECORDS:
        raise GoldenDataError(
            f"{path} has {len(records)} records; at least {MIN_GOLDEN_RECORDS} are needed "
            f"({RECOMMENDED_GOLDEN_RECORDS}+ recommended). Add more hand-written examples."
        )
    if len(records) < RECOMMENDED_GOLDEN_RECORDS:
        log_event(
            f"Only {len(records)} golden records; {RECOMMENDED_GOLDEN_RECORDS}+ give a more reliable eval set.",
            "warning",
        )
    return records


def split_golden(
    records: list[dict[str, Any]], seed_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle with a fixed seed and split into (seed set, eval set); both sets get at least one record."""
    if len(records) < 2:
        raise GoldenDataError(f"Need at least 2 golden records to split, got {len(records)}")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_seed = min(len(shuffled) - 1, max(1, round(len(shuffled) * seed_fraction)))
    return shuffled[:n_seed], shuffled[n_seed:]


def _clip(text: str) -> str:
    """Shorten text for a table cell."""
    return text if len(text) <= SAMPLE_CHARS else text[: SAMPLE_CHARS - 3] + "..."


def show_golden(seed_set: list[dict[str, Any]], eval_set: list[dict[str, Any]]) -> None:
    """Print split counts and a few sample seed records."""
    table = Table(title=f"golden: {len(seed_set)} seed / {len(eval_set)} eval (samples from seed set)")
    for column in ("id", "prompt", "response"):
        table.add_column(column)
    for record in seed_set[:SAMPLE_ROWS]:
        table.add_row(record["id"], _clip(record["prompt"]), _clip(record["response"]))
    get_dashboard().console.print(table)


def reset_downstream(paths: RunPaths) -> list[str]:
    """Mark every cell that depends on the eval split as not done, so it runs again; return their names."""
    state = load_state(paths)
    redo = [cell for cell in DOWNSTREAM_CELLS if is_done(state, cell)]
    for cell in redo:
        unmark(paths, cell)
    if redo:
        log_event(f"The eval split changed, so these cells will run again: {', '.join(redo)}", "warning")
    return redo


def run_golden_cell(cfg: TousterConfig, paths: RunPaths, force: bool = False) -> dict[str, Any]:
    """Cell 2: validate, split and save the golden data; record the eval ids in state.json."""

    def work() -> dict[str, Any]:
        """Load, split, save and display."""
        records = load_golden(cfg.golden_path, smoke=cfg.smoke_test)
        seed_set, eval_set = split_golden(records, cfg.seed_fraction, cfg.seed)
        write_jsonl(paths.golden_seed, seed_set)
        write_jsonl(paths.golden_eval, eval_set)
        previous_ids = load_state(paths).get("eval_ids", [])
        eval_ids = [r["id"] for r in eval_set]
        update_state(paths, eval_ids=eval_ids)
        if previous_ids and sorted(previous_ids) != sorted(eval_ids):
            reset_downstream(paths)
        show_golden(seed_set, eval_set)
        return {"total": len(records), "seed": len(seed_set), "eval": len(eval_set)}

    def summary(metrics: dict[str, Any]) -> str:
        """One-line result."""
        return f"{metrics.get('total', 0)} golden records: {metrics.get('seed', 0)} seed, {metrics.get('eval', 0)} eval."

    return run_cell(paths, "golden", work, summary, force)
