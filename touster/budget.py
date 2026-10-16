"""Budget caps, spend tracking, calibration-based estimates."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from touster.config import BudgetMode, TousterConfig

log = logging.getLogger(__name__)

RESERVE_PHASES = frozenset({"eval", "judge", "export", "report"})


class BudgetExceeded(RuntimeError):
    """Raised when a phase would go over the configured cap."""


@dataclass(frozen=True)
class Estimate:
    """Predicted cost and time for one phase."""

    usd: float
    minutes: float

    def __add__(self, other: Estimate) -> Estimate:
        """Sum two estimates."""
        return Estimate(self.usd + other.usd, self.minutes + other.minutes)


@dataclass(frozen=True)
class Calibration:
    """Measurements from the small calibration run in cell 3."""

    examples: int
    usd: float
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    train_steps: int = 0
    train_seconds: float = 0.0
    train_tokens_per_sec: float = 0.0
    peak_vram_gb: float = 0.0


def cost_rows(cost_log: Path) -> list[dict]:
    """Read costs.jsonl rows, skipping a line cut short by a disconnect (with a warning)."""
    if not cost_log.is_file():
        return []
    rows = []
    with cost_log.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping unreadable line %d in %s (likely cut off by a disconnect)", line_no, cost_log)
    return rows


def read_spent_usd(cost_log: Path) -> float:
    """Sum cost_usd over every line of costs.jsonl."""
    return sum(float(row.get("cost_usd", 0.0)) for row in cost_rows(cost_log))


def spend_by_phase(cost_log: Path) -> dict[str, float]:
    """Group logged cost by phase."""
    totals: dict[str, float] = {}
    for row in cost_rows(cost_log):
        phase = row.get("phase", "other")
        totals[phase] = totals.get(phase, 0.0) + float(row.get("cost_usd", 0.0))
    return totals


class BudgetTracker:
    """Tracks money and time spent against the caps in the config."""

    def __init__(self, cfg: TousterConfig, cost_log: Path, prior_minutes: float = 0.0) -> None:
        """Start tracking; prior_minutes carries time used before a resume."""
        self._cfg = cfg
        self._cost_log = cost_log
        self._prior_minutes = prior_minutes
        self._active_seconds = 0.0
        self._clock_started: float | None = None
        self._lock = threading.Lock()
        self._spent = read_spent_usd(cost_log)

    @property
    def mode(self) -> BudgetMode:
        """Budget mode from the config."""
        return self._cfg.budget_mode

    def add_spend(self, usd: float) -> None:
        """Record a cost that has already been logged to costs.jsonl."""
        with self._lock:
            self._spent += usd

    def spent_usd(self) -> float:
        """Total USD spent in this run."""
        with self._lock:
            return self._spent

    def start_clock(self) -> None:
        """Start counting time; only time inside running cells counts toward the time cap."""
        with self._lock:
            if self._clock_started is None:
                self._clock_started = time.monotonic()

    def stop_clock(self) -> None:
        """Stop counting time and bank the seconds since start_clock."""
        with self._lock:
            if self._clock_started is not None:
                self._active_seconds += time.monotonic() - self._clock_started
                self._clock_started = None

    def elapsed_minutes(self) -> float:
        """Minutes spent inside cells in this run, including time before any resume."""
        with self._lock:
            running = time.monotonic() - self._clock_started if self._clock_started is not None else 0.0
            return self._prior_minutes + (self._active_seconds + running) / 60.0

    def _money_capped(self) -> bool:
        """Money cap applies in money and target modes."""
        return self.mode in (BudgetMode.MONEY, BudgetMode.TARGET)

    def _time_capped(self) -> bool:
        """Time cap applies only in time mode."""
        return self.mode == BudgetMode.TIME

    def usd_limit(self, phase: str) -> float:
        """USD this phase may spend up to: the full cap for reserve phases, cap minus reserve otherwise."""
        if not self._money_capped():
            return float("inf")
        if phase in RESERVE_PHASES:
            return self._cfg.max_usd
        return self._cfg.max_usd * (1.0 - self._cfg.eval_reserve)

    def minutes_limit(self, phase: str) -> float:
        """Minutes this phase may run up to, using the same reserve rule."""
        if not self._time_capped():
            return float("inf")
        if phase in RESERVE_PHASES:
            return float(self._cfg.max_minutes)
        return self._cfg.max_minutes * (1.0 - self._cfg.eval_reserve)

    def remaining(self, phase: str) -> Estimate:
        """Money and minutes still available to a phase."""
        return Estimate(
            usd=max(0.0, self.usd_limit(phase) - self.spent_usd()),
            minutes=max(0.0, self.minutes_limit(phase) - self.elapsed_minutes()),
        )

    def exhausted(self, phase: str) -> bool:
        """True when the phase has no money or time left."""
        left = self.remaining(phase)
        return left.usd <= 0.0 or left.minutes <= 0.0

    def check(self, phase: str) -> None:
        """Raise BudgetExceeded if the phase has nothing left."""
        if self.exhausted(phase):
            raise BudgetExceeded(
                f"Budget cap reached during '{phase}': spent ${self.spent_usd():.4f}, "
                f"{self.elapsed_minutes():.1f} min (mode={self.mode.value})"
            )

    def can_afford(self, estimate: Estimate, phase: str) -> bool:
        """True if the estimate fits inside what is left for this phase."""
        left = self.remaining(phase)
        return estimate.usd <= left.usd and estimate.minutes <= left.minutes

    def target_reached(self, score: float) -> bool:
        """True in target mode once score meets target_score."""
        return (
            self.mode == BudgetMode.TARGET
            and self._cfg.target_score is not None
            and score >= self._cfg.target_score
        )


def estimate_generation(calib: Calibration, n_examples: int) -> Estimate:
    """Scale calibration cost and time linearly to n_examples."""
    if calib.examples <= 0:
        return Estimate(0.0, 0.0)
    scale = n_examples / calib.examples
    return Estimate(usd=calib.usd * scale, minutes=calib.seconds * scale / 60.0)


def estimate_training(calib: Calibration, total_steps: int) -> Estimate:
    """Scale the training probe to total_steps; zero when no probe ran."""
    if calib.train_steps <= 0:
        return Estimate(0.0, 0.0)
    return Estimate(usd=0.0, minutes=calib.train_seconds / calib.train_steps * total_steps / 60.0)


def sft_total_steps(n_examples: int, cfg: TousterConfig) -> int:
    """Optimizer steps for SFT ignoring packing (an upper bound)."""
    per_step = cfg.batch_size * cfg.grad_accum
    return max(1, -(-n_examples // per_step) * cfg.epochs)
