"""Run directory layout, state.json bookkeeping, JSONL I/O and resume logic."""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from touster.budget import BudgetExceeded
from touster.config import TousterConfig, config_to_dict
from touster.dashboard import get_dashboard

CELLS = ("setup", "golden", "synthetic", "sft", "eval", "loop", "export", "report")
STATUS_RUNNING = "running"
STATUS_STOPPED_BUDGET = "stopped_budget"


class LeakageError(RuntimeError):
    """Raised when eval-set records reach a training loader."""


@dataclass(frozen=True)
class RunPaths:
    """Every path in one run directory; build with RunPaths.at(root)."""

    root: Path
    config: Path
    state: Path
    costs: Path
    data: Path
    golden_seed: Path
    golden_eval: Path
    synthetic_raw: Path
    synthetic_clean: Path
    checkpoints: Path
    sft_checkpoints: Path
    eval: Path
    export: Path
    merged: Path
    gguf: Path
    modelfile: Path
    model_card: Path
    report: Path

    @classmethod
    def at(cls, root: Path) -> RunPaths:
        """Build the full layout rooted at root."""
        root = Path(root)
        data, ckpt, export = root / "data", root / "checkpoints", root / "export"
        return cls(
            root=root,
            config=root / "config.json",
            state=root / "state.json",
            costs=root / "costs.jsonl",
            data=data,
            golden_seed=data / "golden_seed.jsonl",
            golden_eval=data / "golden_eval.jsonl",
            synthetic_raw=data / "synthetic_raw.jsonl",
            synthetic_clean=data / "synthetic_clean.jsonl",
            checkpoints=ckpt,
            sft_checkpoints=ckpt / "sft",
            eval=root / "eval",
            export=export,
            merged=export / "merged",
            gguf=export / "gguf",
            modelfile=export / "Modelfile",
            model_card=export / "MODEL_CARD.md",
            report=root / "report.html",
        )

    def dpo_pairs(self, round_no: int) -> Path:
        """Path of the DPO pair file for one round."""
        return self.data / f"dpo_round_{round_no}.jsonl"

    def dpo_checkpoints(self, round_no: int) -> Path:
        """Checkpoint folder for one DPO/GRPO round."""
        return self.checkpoints / f"dpo_round_{round_no}"

    def eval_result(self, stage: str) -> Path:
        """Path of the eval JSON for one stage (base, sft, dpo_round_<n>)."""
        return self.eval / f"{stage}.json"

    def ensure(self) -> RunPaths:
        """Create every folder in the layout and return self."""
        for folder in (self.data, self.sft_checkpoints, self.eval, self.merged, self.gguf):
            folder.mkdir(parents=True, exist_ok=True)
        return self


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so a disconnect never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL atomically and return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return count


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, naming the line on a parse error."""
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({err.msg})") from None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every record from a JSONL file."""
    return list(iter_jsonl(path))


def make_run_id(task_name: str, now: datetime | None = None) -> str:
    """Build a run id as <task_name>-<YYYYMMDD-HHMM>."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    return f"{task_name}-{stamp}"


def _new_state(run_id: str) -> dict[str, Any]:
    """Return the initial state for a fresh run."""
    return {
        "run_id": run_id,
        "status": STATUS_RUNNING,
        "completed": [],
        "metrics": {},
        "eval_ids": [],
        "elapsed_minutes": 0.0,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _latest_run(base: Path, task_name: str) -> Path | None:
    """Return the newest existing run folder for this task, if any."""
    if not base.is_dir():
        return None
    pattern = re.compile(rf"^{re.escape(task_name)}-\d{{8}}-\d{{4}}(-\d+)?$")
    candidates = sorted(
        (p for p in base.iterdir() if pattern.match(p.name) and (p / "state.json").is_file()),
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def create_or_resume_run(cfg: TousterConfig, new_run: bool = False) -> tuple[RunPaths, dict[str, Any], bool]:
    """Resume the newest run for this task or start a new one; return (paths, state, resumed)."""
    base = Path(cfg.run_dir)
    existing = None if new_run else _latest_run(base, cfg.task_name)
    if existing is not None:
        paths = RunPaths.at(existing).ensure()
        return paths, load_state(paths), True
    run_id = make_run_id(cfg.task_name)
    root = base / run_id
    suffix = 1
    while root.exists():
        suffix += 1
        root = base / f"{run_id}-{suffix}"
    paths = RunPaths.at(root).ensure()
    state = _new_state(root.name)
    write_json(paths.config, config_to_dict(cfg))
    write_json(paths.state, state)
    return paths, state, False


def config_changes(paths: RunPaths, cfg: TousterConfig) -> list[str]:
    """List config fields that differ from the config saved with the run."""
    if not paths.config.is_file():
        return []
    saved = read_json(paths.config)
    current = config_to_dict(cfg)
    return sorted(k for k in current if saved.get(k) != current[k])


def load_state(paths: RunPaths) -> dict[str, Any]:
    """Load state.json, recreating a fresh state if it is missing."""
    if not paths.state.is_file():
        state = _new_state(paths.root.name)
        write_json(paths.state, state)
        return state
    return read_json(paths.state)


def save_state(paths: RunPaths, state: dict[str, Any]) -> dict[str, Any]:
    """Persist state and return it."""
    write_json(paths.state, state)
    return state


def update_state(paths: RunPaths, **changes: Any) -> dict[str, Any]:
    """Return a new state with top-level keys replaced, and save it."""
    state = {**load_state(paths), **changes}
    return save_state(paths, state)


def is_done(state: dict[str, Any], cell: str) -> bool:
    """Return True if a cell already completed in this run."""
    return cell in state.get("completed", [])


def should_run(paths: RunPaths, cell: str, force: bool = False) -> bool:
    """Return True unless the cell already completed and force is off."""
    if cell not in CELLS:
        raise ValueError(f"Unknown cell {cell!r}; expected one of {CELLS}")
    return force or not is_done(load_state(paths), cell)


def mark_done(paths: RunPaths, cell: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record a cell as complete, merge its metrics, and save state."""
    state = load_state(paths)
    completed = [c for c in state.get("completed", []) if c != cell] + [cell]
    merged = {**state.get("metrics", {}), **({cell: metrics} if metrics else {})}
    return save_state(paths, {**state, "completed": completed, "metrics": merged})


def unmark(paths: RunPaths, cell: str) -> dict[str, Any]:
    """Remove a cell from the completed list so it runs again."""
    state = load_state(paths)
    completed = [c for c in state.get("completed", []) if c != cell]
    return save_state(paths, {**state, "completed": completed})


def assert_no_eval_leak(records: Iterable[dict[str, Any]], eval_ids: Iterable[str]) -> None:
    """Raise LeakageError if any training record is, or was seeded from, an eval record."""
    forbidden = set(eval_ids)
    if not forbidden:
        return
    leaked = sorted(
        {r.get("id") for r in records if r.get("id") in forbidden}
        | {r.get("seed_id") for r in records if r.get("seed_id") in forbidden}
    )
    if leaked:
        raise LeakageError(f"Eval-set ids found in training data: {leaked[:10]}")


def load_training_records(path: Path, paths: RunPaths) -> list[dict[str, Any]]:
    """Load records for training, refusing the eval file and any eval ids."""
    if path.resolve() == paths.golden_eval.resolve():
        raise LeakageError("Training code must never read golden_eval.jsonl")
    records = read_jsonl(path)
    assert_no_eval_leak(records, load_state(paths).get("eval_ids", []))
    return records


def set_seeds(seed: int) -> None:
    """Seed python, numpy and torch (when installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class GPUUnavailableError(RuntimeError):
    """Raised by GPU-only cells when no CUDA device is present."""


def detect_gpu() -> dict[str, Any]:
    """Return {'available', 'name', 'vram_gb'} for the first CUDA device."""
    try:
        import torch
    except ImportError:
        return {"available": False, "name": None, "vram_gb": 0.0}
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "vram_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {"available": True, "name": props.name, "vram_gb": round(props.total_memory / 1024**3, 1)}


def require_gpu(cell: str) -> dict[str, Any]:
    """Return GPU info or raise a clear error naming the cell that needs it."""
    info = detect_gpu()
    if not info["available"]:
        raise GPUUnavailableError(
            f"Cell '{cell}' needs an NVIDIA GPU with CUDA and none was found. "
            "On Colab: Runtime > Change runtime type > T4 GPU. Cells 1-3 run without a GPU."
        )
    return info


CELLS_ALLOWED_AFTER_BUDGET_STOP = frozenset({"setup", "golden", "eval", "export", "report"})


def run_cell(
    paths: RunPaths,
    cell: str,
    work: Callable[[], dict[str, Any]],
    summary: Callable[[dict[str, Any]], str],
    force: bool = False,
) -> dict[str, Any]:
    """Run one notebook cell under the shared contract: skip if done, live dashboard, save state, one-line summary."""
    state = load_state(paths)
    dash = get_dashboard()
    if not should_run(paths, cell, force):
        metrics = state.get("metrics", {}).get(cell, {})
        print(f"[{cell}] already done, skipped (pass force=True to redo). {summary(metrics)}")
        return metrics
    if state.get("status") == STATUS_STOPPED_BUDGET and cell not in CELLS_ALLOWED_AFTER_BUDGET_STOP:
        print(f"[{cell}] skipped: run stopped on budget cap. Cells eval, export and report still run.")
        return {}
    started = time.monotonic()
    if dash.budget is not None:
        dash.budget.start_clock()
    try:
        with dash.live(cell):
            metrics = work()
    except BudgetExceeded as err:
        update_state(paths, status=STATUS_STOPPED_BUDGET)
        dash.log(str(err), "error")
        print(f"[{cell}] stopped on budget cap, state saved: {err}")
        return {}
    finally:
        if dash.budget is not None:
            dash.budget.stop_clock()
        minutes = (time.monotonic() - started) / 60.0
        update_state(paths, elapsed_minutes=load_state(paths).get("elapsed_minutes", 0.0) + minutes)
    mark_done(paths, cell, metrics)
    print(f"[{cell}] done in {minutes:.1f} min. {summary(metrics)}")
    return metrics
