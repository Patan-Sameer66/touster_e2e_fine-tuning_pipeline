"""touster: teacher-to-student LLM fine-tuning in one notebook."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from touster.budget import BudgetTracker
    from touster.config import TousterConfig
    from touster.llm import LLMClient
    from touster.state import RunPaths

__version__ = "0.1.0"

INSTALL_MARKER = ".touster_installed"
PRELOADED_WATCH = ("torch", "numpy", "pydantic", "transformers", "huggingface_hub")
RESUME_LOCKED_FIELDS = ("task_type", "golden_path", "seed", "seed_fraction", "student_model", "smoke_test")


@dataclass(frozen=True)
class Run:
    """Everything the notebook cells share after setup."""

    cfg: TousterConfig
    paths: RunPaths
    budget: BudgetTracker
    client: LLMClient
    gpu: dict[str, Any]
    resumed: bool


def install_requirements(requirements: str | Path = "requirements.txt") -> None:
    """Install pinned requirements once (Colab); stop with restart advice if already-imported libraries changed."""
    marker = Path(requirements).parent / INSTALL_MARKER
    if marker.is_file():
        return
    preloaded = [name for name in PRELOADED_WATCH if name in sys.modules]
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True)
    marker.write_text("installed\n", encoding="utf-8")
    if preloaded:
        raise RuntimeError(
            f"Packages were installed while {preloaded} were already loaded. "
            "Restart the runtime (Runtime > Restart session), then run all cells again. The install is skipped next time."
        )


def _print_config(cfg: TousterConfig, gpu: dict[str, Any]) -> None:
    """Show the validated config and hardware as a rich table."""
    from rich.console import Console
    from rich.table import Table

    from touster.config import config_to_dict

    table = Table(title="touster config", show_header=True, header_style="bold")
    table.add_column("field")
    table.add_column("value", overflow="fold")
    for key, value in config_to_dict(cfg).items():
        table.add_row(key, str(value))
    gpu_text = f"{gpu['name']} ({gpu['vram_gb']} GB)" if gpu["available"] else "none (cells 4-7 need a CUDA GPU)"
    table.add_row("GPU", gpu_text, style="cyan")
    Console().print(table)


def _check_models(cfg: TousterConfig) -> None:
    """Fail fast on missing API keys, an unreachable local teacher, or an unpriced model under a money cap."""
    from touster.config import BudgetMode
    from touster.llm import check_local_model, has_known_price, load_api_keys

    models = sorted({cfg.teacher_model, cfg.effective_judge_model})
    missing = load_api_keys(models)
    if missing:
        raise RuntimeError(
            f"Missing API key(s) {missing} for {models}. Set them as environment variables "
            "or in Colab Secrets (key icon in the left sidebar), then re-run this cell."
        )
    for model in models:
        check_local_model(model)
    unpriced = [m for m in models if not has_known_price(cfg, m)]
    if unpriced and cfg.budget_mode in (BudgetMode.MONEY, BudgetMode.TARGET):
        raise RuntimeError(
            f"No price known for {unpriced}, so the money cap cannot be enforced. Add them to price_table as "
            "{model: (input_usd_per_1M, output_usd_per_1M)} or use budget_mode='time'."
        )


def _reconcile_resumed_run(paths: RunPaths, cfg: TousterConfig, budget: BudgetTracker) -> None:
    """On resume: refuse changes that would mix incompatible results, save the rest, and lift a budget stop if caps allow."""
    from touster.config import config_to_dict
    from touster.dashboard import log_event
    from touster.state import STATUS_RUNNING, STATUS_STOPPED_BUDGET, config_changes, load_state, update_state, write_json

    changed = config_changes(paths, cfg)
    locked = [field for field in changed if field in RESUME_LOCKED_FIELDS]
    if locked:
        raise RuntimeError(
            f"{locked} changed since run {paths.root.name} started, so its saved data no longer matches. "
            "Set NEW_RUN = True in cell 1 to start a fresh run, or change them back."
        )
    if changed:
        write_json(paths.config, config_to_dict(cfg))
        log_event(f"Config updated on resume: {changed}. Finished cells keep their results; use force=True to redo one.",
                  "warning")
    if load_state(paths).get("status") == STATUS_STOPPED_BUDGET and not budget.exhausted("generation"):
        update_state(paths, status=STATUS_RUNNING)
        log_event("Budget allows more work now; the run is no longer marked as stopped.", "info")


def _import_unsloth_first(gpu: dict[str, Any]) -> None:
    """Import unsloth before transformers/trl/peft so its patches apply (GPU machines only)."""
    if gpu["available"] and importlib.util.find_spec("unsloth") is not None:
        import unsloth  # noqa: F401


def setup_run(config: dict[str, Any], yaml_path: str | Path | None = None, new_run: bool = False) -> Run:
    """Cell 1: validate config, check keys and teacher, create or resume the run, seed, start the dashboard."""
    from touster.budget import BudgetTracker
    from touster.config import apply_smoke_overrides, load_config
    from touster.dashboard import log_event, start_dashboard
    from touster.llm import LLMClient
    from touster.state import create_or_resume_run, detect_gpu, mark_done, set_seeds

    gpu = detect_gpu()
    _import_unsloth_first(gpu)
    cfg = apply_smoke_overrides(load_config(config, yaml_path))
    if not Path(cfg.golden_path).is_file():
        raise FileNotFoundError(f"golden_path not found: {cfg.golden_path}")
    _check_models(cfg)
    paths, state, resumed = create_or_resume_run(cfg, new_run=new_run)
    set_seeds(cfg.seed)
    budget = BudgetTracker(cfg, paths.costs, prior_minutes=float(state.get("elapsed_minutes", 0.0)))
    start_dashboard(paths.root.name, budget)
    if resumed:
        _reconcile_resumed_run(paths, cfg, budget)
    if "google.colab" in sys.modules and not str(cfg.run_dir).startswith("/content/drive"):
        log_event("run_dir is on the Colab VM and is lost when the runtime is recycled. "
                  "Mount Drive and set run_dir to a /content/drive/... path to keep runs.", "warning")
    _print_config(cfg, gpu)
    mark_done(paths, "setup", {"gpu": gpu["name"], "student_model": cfg.student_model})
    print(f"[setup] {'resumed' if resumed else 'started'} run {paths.root.name} in {paths.root}")
    return Run(cfg=cfg, paths=paths, budget=budget, client=LLMClient(cfg, paths.costs, budget), gpu=gpu, resumed=resumed)
