"""Live terminal dashboard built on rich; modules report through the module-level helpers."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from touster.budget import BudgetTracker

REFRESH_PER_SECOND = 2
EVENT_LOG_SIZE = 8
LEVEL_STYLES = {"info": "white", "warning": "yellow", "error": "bold red", "success": "green"}


@dataclass(frozen=True)
class Event:
    """One line in the event log."""

    clock: str
    level: str
    message: str


def _fmt_minutes(minutes: float) -> str:
    """Format minutes as H:MM:SS."""
    seconds = int(max(0.0, minutes) * 60)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Dashboard:
    """Holds dashboard state and renders it; thread-safe for updates from workers."""

    def __init__(self, run_id: str, budget: BudgetTracker | None = None, console: Console | None = None) -> None:
        """Create an idle dashboard for one run."""
        self.run_id = run_id
        self.budget = budget
        self.console = console or Console()
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._cell = "-"
        self._progress: tuple[str, int, int, float] | None = None
        self._training: dict[str, float] = {}
        self._results: dict[str, float] = {}
        self._metric_name = "score"
        self._events: deque[Event] = deque(maxlen=EVENT_LOG_SIZE)
        self._live: Live | None = None

    def set_cell(self, name: str) -> None:
        """Show which cell is running and clear per-cell progress."""
        with self._lock:
            self._cell = name
            self._progress = None
            self._training = {}

    def log(self, message: str, level: str = "info") -> None:
        """Add a line to the event log; outside a live view it is also printed."""
        event = Event(time.strftime("%H:%M:%S"), level if level in LEVEL_STYLES else "info", message)
        with self._lock:
            self._events.append(event)
            live = self._live
        if live is None:
            self.console.print(Text(f"[{event.clock}] {message}", style=LEVEL_STYLES[event.level]))

    def progress(self, label: str, done: int, total: int) -> None:
        """Update the progress bar; the ETA is measured from the first update for this label."""
        with self._lock:
            started = self._progress[3] if self._progress and self._progress[0] == label else time.monotonic()
            self._progress = (label, done, max(total, 1), started)

    def training(self, **metrics: float) -> None:
        """Merge the latest training metrics (loss, learning_rate, tokens_per_sec, gpu_mem_gb)."""
        with self._lock:
            self._training = {**self._training, **{k: float(v) for k, v in metrics.items() if v is not None}}

    def result(self, stage: str, score: float, metric_name: str | None = None) -> None:
        """Record a stage score for the results table."""
        with self._lock:
            self._results = {**self._results, stage: float(score)}
            if metric_name:
                self._metric_name = metric_name

    def _header(self) -> Text:
        """Run id, current cell and elapsed time."""
        elapsed = (time.monotonic() - self._started) / 60.0
        return Text.assemble(
            ("touster ", "bold cyan"), (self.run_id, "bold"), "  cell: ", (self._cell, "magenta"),
            "  elapsed: ", _fmt_minutes(elapsed),
        )

    def _budget_panel(self) -> Panel:
        """USD and minutes spent against the caps, with the eval reserve marked."""
        table = Table.grid(padding=(0, 1))
        if self.budget is None:
            table.add_row("budget tracking off")
            return Panel(table, title="budget")
        spent, cap = self.budget.spent_usd(), self.budget.usd_limit("eval")
        mins, mins_cap = self.budget.elapsed_minutes(), self.budget.minutes_limit("eval")
        reserve_usd, reserve_min = self.budget.usd_limit("generation"), self.budget.minutes_limit("generation")
        for label, used, limit, reserve, unit in (
            ("USD", spent, cap, reserve_usd, "$"),
            ("min", mins, mins_cap, reserve_min, ""),
        ):
            if limit == float("inf"):
                table.add_row(label, f"{unit}{used:.2f}", "no cap")
                continue
            bar = ProgressBar(total=limit, completed=min(used, limit), width=30)
            table.add_row(label, bar, f"{unit}{used:.2f} / {unit}{limit:.2f}  (reserve from {unit}{reserve:.2f})")
        return Panel(table, title="budget")

    def _progress_panel(self) -> Panel:
        """Current step with ETA."""
        if self._progress is None:
            return Panel(Text("idle", style="dim"), title="progress")
        label, done, total, started = self._progress
        rate = done / max(time.monotonic() - started, 1e-6)
        eta = (total - done) / rate / 60.0 if rate > 0 else 0.0
        grid = Table.grid(padding=(0, 1))
        grid.add_row(label, ProgressBar(total=total, completed=done, width=30), f"{done}/{total}", f"ETA {_fmt_minutes(eta)}")
        return Panel(grid, title="progress")

    def _training_panel(self) -> Panel:
        """Latest loss, learning rate, tokens/sec and GPU memory."""
        grid = Table.grid(padding=(0, 2))
        names = (("loss", "{:.4f}"), ("learning_rate", "{:.2e}"), ("tokens_per_sec", "{:.0f}"), ("gpu_mem_gb", "{:.2f} GB"))
        grid.add_row(*(f"{n}: {fmt.format(self._training[n]) if n in self._training else '-'}" for n, fmt in names))
        return Panel(grid, title="training")

    def _results_panel(self) -> Panel:
        """Score per stage, best highlighted."""
        table = Table(show_header=True, header_style="bold")
        table.add_column("stage")
        table.add_column(self._metric_name, justify="right")
        best = max(self._results.values()) if self._results else None
        for stage, score in self._results.items():
            style = "bold green" if score == best and stage != "base" else ""
            table.add_row(stage, f"{score:.4f}", style=style)
        return Panel(table, title="results")

    def _events_panel(self) -> Panel:
        """Last events, coloured by level."""
        lines = [Text(f"{e.clock} {e.message}", style=LEVEL_STYLES[e.level]) for e in self._events]
        return Panel(Group(*lines) if lines else Text("no events", style="dim"), title="events")

    def render(self) -> Group:
        """Build the full dashboard renderable."""
        with self._lock:
            return Group(
                self._header(), self._budget_panel(), self._progress_panel(),
                self._training_panel(), self._results_panel(), self._events_panel(),
            )

    @contextmanager
    def live(self, cell: str) -> Iterator[Dashboard]:
        """Show the dashboard live while a cell runs, then leave the final frame in the output."""
        self.set_cell(cell)
        with Live(get_renderable=self.render, console=self.console, refresh_per_second=REFRESH_PER_SECOND) as live:
            with self._lock:
                self._live = live
            try:
                yield self
            finally:
                with self._lock:
                    self._live = None

    @contextmanager
    def pause(self) -> Iterator[None]:
        """Stop the live view while the user is asked something, then resume it."""
        with self._lock:
            live = self._live
        if live is None:
            yield
            return
        live.stop()
        try:
            yield
        finally:
            live.start()


class DashboardLogHandler(logging.Handler):
    """Forwards touster warnings and errors from the logging module into the event log."""

    def emit(self, record: logging.LogRecord) -> None:
        """Send a log record to the current dashboard."""
        level = "error" if record.levelno >= logging.ERROR else "warning" if record.levelno >= logging.WARNING else "info"
        get_dashboard().log(record.getMessage(), level)


_current: Dashboard | None = None
_handler = DashboardLogHandler(level=logging.WARNING)


def start_dashboard(run_id: str, budget: BudgetTracker | None = None) -> Dashboard:
    """Create the run's dashboard and route touster warnings into it; safe to call again."""
    global _current
    _current = Dashboard(run_id, budget)
    package_log = logging.getLogger("touster")
    if _handler not in package_log.handlers:
        package_log.addHandler(_handler)
    package_log.setLevel(logging.INFO)
    return _current


def get_dashboard() -> Dashboard:
    """Return the running dashboard, or a print-only one when none was started (tests, scripts)."""
    global _current
    if _current is None:
        _current = Dashboard("no-run")
    return _current


def log_event(message: str, level: str = "info") -> None:
    """Add an event to the current dashboard."""
    get_dashboard().log(message, level)


def set_progress(label: str, done: int, total: int) -> None:
    """Update the current dashboard's progress bar."""
    get_dashboard().progress(label, done, total)


def set_training(**metrics: Any) -> None:
    """Update the current dashboard's training panel."""
    get_dashboard().training(**metrics)


def set_result(stage: str, score: float, metric_name: str | None = None) -> None:
    """Record a stage score on the current dashboard."""
    get_dashboard().result(stage, score, metric_name)
