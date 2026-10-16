"""CPU tests for the DPO pair builder and the loop control rules."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest
import torch

from touster.state import GPUUnavailableError, load_state, write_json, write_jsonl
from touster.train import dpo
from touster.train.dpo import (
    Best,
    build_dpo_pairs,
    build_replay_pairs,
    finished_rounds,
    gain_points,
    replay_count,
    stop_reason,
    update_best,
)

NAN = math.nan


def recs(n: int) -> list[dict[str, Any]]:
    """n training records."""
    return [{"id": f"s-{i}", "prompt": f"prompt {i}", "response": f"reference {i}"} for i in range(n)]


def test_pairs_pick_best_and_worst() -> None:
    """The highest-scored sample is chosen and the lowest rejected."""
    pairs = build_dpo_pairs(recs(1), [["a", "b", "c"]], [[3.0, 5.0, 1.0]], round_no=2)
    assert pairs == [{
        "id": "p-r2-0000", "prompt": "prompt 0", "chosen": "b", "rejected": "c",
        "round": 2, "chosen_score": 5.0, "rejected_score": 1.0,
    }]


def test_pairs_drop_low_gap_and_ties() -> None:
    """Prompts whose score gap is below the minimum, including ties, give no pair."""
    samples = [["a", "b"], ["a", "b"], ["a", "b"]]
    scores = [[4.0, 3.5], [3.0, 3.0], [5.0, 4.0]]
    pairs = build_dpo_pairs(recs(3), samples, scores, round_no=1)
    assert [p["prompt"] for p in pairs] == ["prompt 2"]
    assert pairs[0]["id"] == "p-r1-0000"


def test_pairs_handle_empty_and_nan() -> None:
    """Empty sample lists give no pair and NaN scores are ignored."""
    assert build_dpo_pairs([], [], [], round_no=1) == []
    samples = [[], ["a", "b", "c"], ["a", "b"]]
    scores = [[], [NAN, 4.0, 2.0], [NAN, 5.0]]
    pairs = build_dpo_pairs(recs(3), samples, scores, round_no=1)
    assert len(pairs) == 1
    assert (pairs[0]["chosen"], pairs[0]["rejected"]) == ("b", "c")


def test_pairs_skip_identical_texts() -> None:
    """Samples that differ only in whitespace are not paired."""
    assert build_dpo_pairs(recs(1), [["same", "same "]], [[5.0, 1.0]], round_no=1) == []


def test_pairs_length_mismatch_raises() -> None:
    """Records, samples and scores of different lengths raise ValueError."""
    with pytest.raises(ValueError):
        build_dpo_pairs(recs(2), [["a"]], [[1.0]], round_no=1)


def test_replay_pairs_use_reference_and_worst_sample() -> None:
    """Replay pairs take the SFT reference as chosen and the worst sample as rejected."""
    records = recs(2)
    pairs = build_replay_pairs(records, [["x", "y"], ["z", "w"]], [[4.0, 2.0], [5.0, 5.0]], 5, 1, 3, random.Random(0))
    assert pairs == [{
        "id": "p-r1-0003", "prompt": "prompt 0", "chosen": "reference 0", "rejected": "y",
        "round": 1, "chosen_score": None, "rejected_score": 2.0,
    }]


def test_replay_count() -> None:
    """Replay pairs make up the requested share of the mixed set."""
    assert replay_count(0, 0.15) == 0
    assert replay_count(85, 0.15) == 15
    assert replay_count(10, 0.0) == 0


def test_stop_after_two_low_gain_rounds() -> None:
    """Two rounds in a row below min_gain stop the loop."""
    assert stop_reason([0.5], 1, 5, 1.0, False, True) is None
    assert stop_reason([0.5, 2.0], 2, 5, 1.0, False, True) is None
    assert stop_reason([2.0, 0.5, -3.0], 3, 5, 1.0, False, True) == "no_gain"


def test_stop_max_rounds_target_and_budget() -> None:
    """Round cap, target score and an empty budget each stop the loop."""
    assert stop_reason([5.0, 5.0], 2, 2, 1.0, False, True) == "max_rounds"
    assert stop_reason([], 0, 3, 1.0, True, True) == "target_reached"
    assert stop_reason([], 0, 3, 1.0, False, False) == "budget"
    assert stop_reason([], 0, 3, 1.0, False, True) is None


def test_regression_keeps_best() -> None:
    """A lower or equal score never replaces the best checkpoint."""
    best = Best("sft", 0.70, "sft/final")
    best = update_best(best, "dpo_round_1", 0.75, "r1/final")
    assert best == Best("dpo_round_1", 0.75, "r1/final")
    best = update_best(best, "dpo_round_2", 0.60, "r2/final")
    assert best.stage == "dpo_round_1"
    assert update_best(best, "dpo_round_3", 0.75, "r3/final").stage == "dpo_round_1"
    assert gain_points(0.70, 0.75) == pytest.approx(5.0)


def test_finished_rounds_need_eval_file(run: Any) -> None:
    """Only rounds with an eval file count as finished."""
    cfg, paths = run
    state = {"loop_rounds": [{"stage": "dpo_round_1"}, {"stage": "dpo_round_2"}]}
    write_json(paths.eval_result("dpo_round_1"), {"stage": "dpo_round_1"})
    assert finished_rounds(state, paths) == [{"stage": "dpo_round_1"}]


def test_loop_skipped_when_self_improve_off(make_cfg: Any) -> None:
    """With self_improve off the cell returns the SFT checkpoint as best without a GPU."""
    from touster.state import create_or_resume_run

    cfg = make_cfg(self_improve=False)
    paths, _, _ = create_or_resume_run(cfg)
    write_json(paths.eval_result("sft"), {"primary_metric": "accuracy", "scores": {"accuracy": 0.8}})
    metrics = dpo.run_loop_cell(cfg, paths, client=None, budget=None)
    assert metrics["best_stage"] == "sft"
    assert metrics["best_adapter"].endswith("final")
    assert metrics["best_score"] == 0.8
    assert "loop" in load_state(paths)["completed"]


def test_loop_requires_gpu(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a CPU machine the loop cell names itself in the GPU error."""
    cfg, paths = run
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GPUUnavailableError, match="'loop'"):
        dpo.run_loop_cell(cfg, paths, client=None, budget=None)


class FakeBudget:
    """Budget stub with unlimited money and time."""

    def target_reached(self, score: float) -> bool:
        """Never reached."""
        return False

    def exhausted(self, phase: str) -> bool:
        """Never exhausted."""
        return False


def test_loop_keeps_best_through_regression(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worse round is kept out of best, and the next round starts from the best adapter."""
    cfg, paths = run
    cfg = cfg.model_copy(update={"max_rounds": 3})
    monkeypatch.setattr(dpo, "require_gpu", lambda cell: {})
    monkeypatch.setattr(dpo, "_sft_baseline", lambda p: Best("sft", 0.70, "sft/final"))
    monkeypatch.setattr("touster.train.sft.load_sft_records", lambda p: recs(4))
    scores = iter([0.80, 0.65, 0.66])
    started_from: list[str] = []

    def fake_round(cfg: Any, paths: Any, client: Any, budget: Any, round_no: int, best: Best, records: Any) -> dict[str, Any]:
        """Record the starting adapter and return the next scripted score."""
        started_from.append(best.adapter)
        score = next(scores)
        stage = f"dpo_round_{round_no}"
        write_json(paths.eval_result(stage), {"stage": stage})
        return {"round": round_no, "stage": stage, "score": score, "adapter": f"r{round_no}/final",
                "gain_points": gain_points(best.score, score)}

    monkeypatch.setattr(dpo, "run_round", fake_round)
    metrics = dpo.run_loop(cfg, paths, client=None, budget=FakeBudget())
    assert metrics["best_stage"] == "dpo_round_1"
    assert metrics["best_adapter"] == "r1/final"
    assert metrics["stop_reason"] == "no_gain"
    assert started_from == ["sft/final", "r1/final", "r1/final"]
    assert len(load_state(paths)["loop_rounds"]) == 3


def test_saved_pairs_are_reused(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pairs saved by an earlier attempt are returned without sampling again."""
    cfg, paths = run
    saved = [{"id": "p-r1-0000", "prompt": "p", "chosen": "a", "rejected": "b", "round": 1,
              "chosen_score": 5.0, "rejected_score": 1.0}]
    write_jsonl(paths.dpo_pairs(1), saved)

    def no_sampling(*args: Any) -> Any:
        """Fail if the loop tries to sample."""
        raise AssertionError("sampling must be skipped when pairs exist")

    monkeypatch.setattr(dpo, "sample_answers", no_sampling)
    assert dpo.collect_dpo_pairs(cfg, paths, None, FakeBudget(), 1, paths.root, recs(2)) == saved


def test_reset_round_removes_round_files(run: Any) -> None:
    """reset_round deletes the round's pairs and checkpoints."""
    cfg, paths = run
    write_jsonl(paths.dpo_pairs(1), [{"id": "x"}])
    write_json(paths.dpo_checkpoints(1) / "checkpoint-5" / "trainer_state.json", {})
    dpo.reset_round(paths, 1)
    assert not paths.dpo_pairs(1).exists()
    assert not any(paths.dpo_checkpoints(1).iterdir())


def test_budget_hit_mid_round_keeps_best(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hitting the cap during a round stops the loop, keeps SFT as best and marks the run stopped."""
    from touster.budget import BudgetExceeded

    cfg, paths = run
    monkeypatch.setattr(dpo, "require_gpu", lambda cell: {})
    monkeypatch.setattr(dpo, "_sft_baseline", lambda p: Best("sft", 0.70, "sft/final"))
    monkeypatch.setattr("touster.train.sft.load_sft_records", lambda p: recs(4))
    monkeypatch.setattr(dpo, "collect_dpo_pairs", lambda *a: [{"id": "p"}])

    class CapBudget(FakeBudget):
        """Budget that is exhausted once training is about to start."""

        def check(self, phase: str) -> None:
            """Always over the cap."""
            raise BudgetExceeded("cap")

    metrics = dpo.run_loop(cfg, paths, client=None, budget=CapBudget())
    assert metrics["stop_reason"] == "budget"
    assert metrics["best_stage"] == "sft"
    assert load_state(paths)["status"] == "stopped_budget"
