"""Cost tracking and budget caps."""

from __future__ import annotations

import json

import pytest

from touster.budget import (
    BudgetExceeded,
    BudgetTracker,
    Calibration,
    Estimate,
    estimate_generation,
    estimate_training,
    read_spent_usd,
    spend_by_phase,
)


def write_log(path, rows) -> None:
    """Write a fake costs.jsonl."""
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_cost_log_sums_correctly(run) -> None:
    """Spend adds up from a fake call log, overall and by phase."""
    _, paths = run
    write_log(paths.costs, [
        {"phase": "generation", "cost_usd": 0.10},
        {"phase": "generation", "cost_usd": 0.05},
        {"phase": "judge", "cost_usd": 0.20},
    ])
    assert read_spent_usd(paths.costs) == pytest.approx(0.35)
    assert spend_by_phase(paths.costs) == pytest.approx({"generation": 0.15, "judge": 0.20})


def test_budget_stops_at_cap_and_keeps_reserve(run, make_cfg) -> None:
    """Training phases stop at cap minus reserve; eval phases may use the reserve."""
    _, paths = run
    cfg = make_cfg(max_usd=1.0, eval_reserve=0.2)
    write_log(paths.costs, [{"phase": "generation", "cost_usd": 0.80}])
    budget = BudgetTracker(cfg, paths.costs)
    assert budget.exhausted("generation")
    with pytest.raises(BudgetExceeded, match="generation"):
        budget.check("generation")
    budget.check("eval")
    assert budget.remaining("eval").usd == pytest.approx(0.20)
    budget.add_spend(0.25)
    with pytest.raises(BudgetExceeded):
        budget.check("eval")


def test_can_afford_uses_phase_limit(run, make_cfg) -> None:
    """Estimates are compared with what is left of cap minus reserve."""
    _, paths = run
    budget = BudgetTracker(make_cfg(max_usd=5.0, eval_reserve=0.2), paths.costs)
    assert budget.can_afford(Estimate(3.9, 1.0), "generation")
    assert not budget.can_afford(Estimate(4.1, 1.0), "generation")
    assert budget.can_afford(Estimate(4.1, 1.0), "eval")


def test_time_mode_caps_minutes(run, make_cfg) -> None:
    """Time mode ignores money and stops on minutes, counting prior sessions."""
    _, paths = run
    cfg = make_cfg(budget_mode="time", max_minutes=10, eval_reserve=0.2)
    budget = BudgetTracker(cfg, paths.costs, prior_minutes=8.5)
    assert budget.usd_limit("generation") == float("inf")
    assert budget.exhausted("generation")
    assert not budget.exhausted("eval")


def test_free_mode_has_no_money_cap(run, make_cfg) -> None:
    """Free mode never stops on money."""
    _, paths = run
    budget = BudgetTracker(make_cfg(budget_mode="free"), paths.costs)
    budget.add_spend(100.0)
    budget.check("generation")


def test_target_mode(run, make_cfg) -> None:
    """Target mode reports when the score is reached and still caps money."""
    _, paths = run
    budget = BudgetTracker(make_cfg(budget_mode="target", target_score=0.8, max_usd=1.0), paths.costs)
    assert budget.target_reached(0.81) and not budget.target_reached(0.5)
    assert budget.usd_limit("generation") == pytest.approx(0.8)


def test_estimates_scale_linearly() -> None:
    """Calibration numbers scale to the full run."""
    calib = Calibration(examples=20, usd=0.02, seconds=60, prompt_tokens=1000, completion_tokens=500,
                        train_steps=50, train_seconds=100)
    est = estimate_generation(calib, 2000)
    assert est.usd == pytest.approx(2.0) and est.minutes == pytest.approx(100.0)
    assert estimate_training(calib, 500).minutes == pytest.approx(500 * 2 / 60)
    assert estimate_training(Calibration(20, 0, 0, 0, 0), 500) == Estimate(0.0, 0.0)
