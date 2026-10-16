"""Cell 6: self-improvement loop (iterative DPO with a judge, or GRPO with programmatic rewards)."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from touster.budget import BudgetExceeded, BudgetTracker, Estimate
from touster.config import SMOKE_DPO_ROUNDS, SMOKE_DPO_STEPS, LoopMethod, TousterConfig
from touster.dashboard import log_event, set_progress
from touster.state import (
    STATUS_STOPPED_BUDGET,
    RunPaths,
    load_state,
    read_json,
    read_jsonl,
    require_gpu,
    run_cell,
    update_state,
    write_jsonl,
)

log = logging.getLogger(__name__)

SAMPLE_TEMPERATURE = 0.8
MIN_PAIR_GAP = 1.0
MAX_JUDGE_SCORE = 5.0
ROUND_PROMPTS = 128
SMOKE_ROUND_PROMPTS = 8
JUDGE_CHUNK = 8
DPO_LR_SCALE = 0.1
STATE_KEY = "loop_rounds"


@dataclass(frozen=True)
class Best:
    """The best checkpoint so far."""

    stage: str
    score: float
    adapter: str


# ---------- pure helpers (unit-tested without a GPU) ----------

def valid_indices(scores: list[float]) -> list[int]:
    """Indices whose score is a real number (the judge returns NaN for unparseable output)."""
    return [i for i, s in enumerate(scores) if s is not None and not math.isnan(s)]


def best_and_worst(scores: list[float]) -> tuple[int, int] | None:
    """Indices of the highest and lowest valid score (first one wins ties), or None without two valid scores."""
    idx = valid_indices(scores)
    if len(idx) < 2:
        return None
    best = max(idx, key=lambda i: (scores[i], -i))
    worst = min(idx, key=lambda i: (scores[i], i))
    return best, worst


def pair_id(round_no: int, index: int) -> str:
    """DPO pair id, e.g. p-r1-0042."""
    return f"p-r{round_no}-{index:04d}"


def build_dpo_pairs(
    prompts_with_ids: list[dict[str, Any]],
    samples: list[list[str]],
    scores: list[list[float]],
    round_no: int,
    min_gap: float = MIN_PAIR_GAP,
) -> list[dict[str, Any]]:
    """Best sample = chosen, worst = rejected; drop prompts with a score gap below min_gap or identical texts."""
    if not len(prompts_with_ids) == len(samples) == len(scores):
        raise ValueError("prompts_with_ids, samples and scores must have the same length")
    pairs: list[dict[str, Any]] = []
    for record, texts, marks in zip(prompts_with_ids, samples, scores):
        if len(texts) != len(marks):
            log.warning("Prompt %s: %d samples but %d scores; skipped", record.get("id"), len(texts), len(marks))
            continue
        picked = best_and_worst(marks)
        if picked is None:
            continue
        best, worst = picked
        if marks[best] - marks[worst] < min_gap or texts[best].strip() == texts[worst].strip():
            continue
        pairs.append({
            "id": pair_id(round_no, len(pairs)), "prompt": record["prompt"],
            "chosen": texts[best], "rejected": texts[worst], "round": round_no,
            "chosen_score": float(marks[best]), "rejected_score": float(marks[worst]),
        })
    return pairs


def replay_count(n_pairs: int, fraction: float) -> int:
    """Replay pairs to add so they make up `fraction` of the final mixed set."""
    if n_pairs <= 0 or fraction <= 0:
        return 0
    return round(n_pairs * fraction / (1.0 - fraction))


def build_replay_pairs(
    records: list[dict[str, Any]],
    samples: list[list[str]],
    scores: list[list[float]],
    count: int,
    round_no: int,
    start_index: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """SFT replay pairs: chosen = the reference SFT response, rejected = the model's worst sample for that prompt.

    chosen_score is null because the reference is not judged. These pairs keep the policy anchored to the SFT
    data so DPO does not drift away from what SFT already taught.
    """
    candidates = []
    for record, texts, marks in zip(records, samples, scores):
        idx = valid_indices(marks)
        if not idx or len(texts) != len(marks):
            continue
        worst = min(idx, key=lambda i: (marks[i], i))
        if marks[worst] < MAX_JUDGE_SCORE and texts[worst].strip() != record["response"].strip():
            candidates.append((record, texts[worst], float(marks[worst])))
    chosen = rng.sample(candidates, min(count, len(candidates)))
    return [
        {"id": pair_id(round_no, start_index + k), "prompt": rec["prompt"], "chosen": rec["response"],
         "rejected": worst_text, "round": round_no, "chosen_score": None, "rejected_score": worst_score}
        for k, (rec, worst_text, worst_score) in enumerate(chosen)
    ]


def gain_points(previous_best: float, score: float) -> float:
    """Improvement in points (100 x score difference)."""
    return 100.0 * (score - previous_best)


def update_best(best: Best, stage: str, score: float, adapter: str) -> Best:
    """Return the new best only when score is strictly higher; a regression never replaces it."""
    return Best(stage, score, adapter) if score > best.score else best


def stop_reason(
    gains: list[float], rounds_done: int, max_rounds: int, min_gain: float, target_hit: bool, budget_left: bool
) -> str | None:
    """Why the loop should stop now, or None to run another round."""
    if target_hit:
        return "target_reached"
    if not budget_left:
        return "budget"
    if len(gains) >= 2 and all(g < min_gain for g in gains[-2:]):
        return "no_gain"
    if rounds_done >= max_rounds:
        return "max_rounds"
    return None


def stage_name(round_no: int) -> str:
    """Eval stage name for a loop round."""
    return f"dpo_round_{round_no}"


def stage_score(result: dict[str, Any]) -> float:
    """Primary metric value from an eval result."""
    return float(result["scores"][result["primary_metric"]])


def finished_rounds(state: dict[str, Any], paths: RunPaths) -> list[dict[str, Any]]:
    """Rounds recorded in state whose eval file exists (these are not redone on resume)."""
    return [r for r in state.get(STATE_KEY, []) if paths.eval_result(r["stage"]).is_file()]


def max_rounds_for(cfg: TousterConfig) -> int:
    """Round cap: cfg.max_rounds, or the smoke cap."""
    return min(cfg.max_rounds, SMOKE_DPO_ROUNDS) if cfg.smoke_test else cfg.max_rounds


def select_round_records(records: list[dict[str, Any]], size: int, round_no: int, seed: int) -> list[dict[str, Any]]:
    """A seeded random subset of training records for one round."""
    rng = random.Random(seed + round_no)
    return rng.sample(records, min(size, len(records)))


def scores_rows(records: list[dict[str, Any]], samples: list[list[str]], scores: list[list[float]]) -> list[dict[str, Any]]:
    """Per-prompt samples and judge scores for saving (NaN written as null)."""
    return [
        {"id": r.get("id"), "prompt": r["prompt"], "samples": s,
         "scores": [None if (m is None or math.isnan(m)) else m for m in marks]}
        for r, s, marks in zip(records, samples, scores)
    ]


# ---------- GPU / judge steps ----------

def judge_samples(client: Any, cfg: TousterConfig, budget: BudgetTracker, records: list[dict[str, Any]], samples: list[list[str]]) -> list[list[float]]:
    """Judge samples in chunks while the loop budget can afford the next chunk; returns scores for a prefix."""
    from touster.eval.judge import score_answers

    scores: list[list[float]] = []
    spent0, started = budget.spent_usd(), time.monotonic()
    for start in range(0, len(records), JUDGE_CHUNK):
        done = len(scores)
        if done:
            per_usd = (budget.spent_usd() - spent0) / done
            per_min = (time.monotonic() - started) / 60.0 / done
            chunk = min(JUDGE_CHUNK, len(records) - start)
            if not budget.can_afford(Estimate(per_usd * chunk, per_min * chunk), "loop"):
                log_event(f"Judging stopped after {done} prompts to stay inside the loop budget", "warning")
                break
        else:
            budget.check("loop")
        for record, answers in zip(records[start : start + JUDGE_CHUNK], samples[start : start + JUDGE_CHUNK]):
            scores.append(score_answers(client, cfg, record["prompt"], answers, record.get("response")))
        set_progress("judging samples", len(scores), len(records))
    return scores


def sample_answers(cfg: TousterConfig, adapter: Path, records: list[dict[str, Any]]) -> list[list[str]]:
    """Sample cfg.samples_per_prompt answers per record from the given adapter."""
    from touster.train.sft import generate_responses, load_student, release_student

    bundle = load_student(cfg, adapter, for_training=False)
    try:
        return generate_responses(
            bundle, [r["prompt"] for r in records], cfg,
            temperature=SAMPLE_TEMPERATURE, num_samples=cfg.samples_per_prompt,
        )
    finally:
        release_student(bundle)
        del bundle


def scores_path(paths: RunPaths, round_no: int) -> Path:
    """Where a round's per-prompt samples and judge scores are saved."""
    return paths.dpo_pairs(round_no).with_name(f"dpo_round_{round_no}_scores.jsonl")


def collect_dpo_pairs(cfg: TousterConfig, paths: RunPaths, client: Any, budget: BudgetTracker, round_no: int, adapter: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sample, judge, build judge pairs plus replay pairs; reuse the round's saved pairs if they exist."""
    if paths.dpo_pairs(round_no).is_file():
        log_event(f"Round {round_no}: reusing saved pairs from {paths.dpo_pairs(round_no).name}", "info")
        return read_jsonl(paths.dpo_pairs(round_no))
    samples = sample_answers(cfg, adapter, records)
    budget.check("loop")
    scores = judge_samples(client, cfg, budget, records, samples)
    judged, judged_samples = records[: len(scores)], samples[: len(scores)]
    pairs = build_dpo_pairs(judged, judged_samples, scores, round_no)
    replay = build_replay_pairs(
        judged, judged_samples, scores, replay_count(len(pairs), cfg.replay_fraction),
        round_no, len(pairs), random.Random(cfg.seed + round_no),
    )
    write_jsonl(scores_path(paths, round_no), scores_rows(judged, judged_samples, scores))
    if not pairs:
        return []
    mixed = pairs + replay
    write_jsonl(paths.dpo_pairs(round_no), mixed)
    log_event(f"Round {round_no}: {len(pairs)} judge pairs + {len(replay)} replay pairs from {len(scores)} prompts", "info")
    return mixed


def preference_rows(pairs: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    """Conversational preference rows for TRL's DPOTrainer."""
    from touster.train.sft import prompt_messages

    return [
        {"prompt": prompt_messages(p["prompt"], system_prompt),
         "chosen": [{"role": "assistant", "content": p["chosen"]}],
         "rejected": [{"role": "assistant", "content": p["rejected"]}]}
        for p in pairs
    ]


def _dpo_config(cfg: TousterConfig, out: Path, batch_size: int, grad_accum: int, total_steps: int) -> Any:
    """DPOConfig with beta from the config and a lower learning rate than SFT."""
    from trl import DPOConfig

    from touster.train.sft import WARMUP_RATIO, logging_steps_for, precision_kwargs, save_steps_for

    return DPOConfig(
        output_dir=str(out),
        beta=cfg.dpo_beta,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate * DPO_LR_SCALE,
        warmup_steps=WARMUP_RATIO,
        num_train_epochs=1,
        max_steps=SMOKE_DPO_STEPS if cfg.smoke_test else -1,
        max_length=cfg.max_seq_len,
        gradient_checkpointing=True,
        logging_steps=logging_steps_for(total_steps),
        save_strategy="steps",
        save_steps=save_steps_for(total_steps),
        report_to="none",
        seed=cfg.seed,
        **precision_kwargs(),
    )


def _train_dpo_once(cfg: TousterConfig, start_adapter: Path, rows: list[dict[str, Any]], out: Path, batch_size: int, grad_accum: int) -> dict[str, Any]:
    """One DPO attempt from start_adapter (also the frozen reference); resumes from a checkpoint in out."""
    from touster.train.sft import load_student, make_dashboard_callback, resume_point

    resume = resume_point(out, force=False)
    bundle = load_student(cfg, start_adapter, for_training=True)
    from datasets import Dataset
    from trl import DPOTrainer

    total = SMOKE_DPO_STEPS if cfg.smoke_test else max(1, -(-len(rows) // (batch_size * grad_accum)))
    # With a PeftModel and ref_model=None, TRL 1.15 copies the loaded adapter into a frozen "ref" adapter,
    # so the reference policy is the starting (best) checkpoint, not the bare base model.
    trainer = DPOTrainer(
        model=bundle.model, ref_model=None,
        args=_dpo_config(cfg, out, batch_size, grad_accum, total),
        train_dataset=Dataset.from_list(rows), processing_class=bundle.tokenizer,
        callbacks=[make_dashboard_callback("dpo steps")],
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return {"trainer": trainer, "bundle": bundle}


def train_dpo(cfg: TousterConfig, paths: RunPaths, start_adapter: Path, pairs: list[dict[str, Any]], round_no: int) -> dict[str, Any]:
    """Train one DPO round and save the adapter to dpo_checkpoints(round)/final."""
    from touster.train.sft import run_with_oom_retry, save_round, system_prompt_for

    rows = preference_rows(pairs, system_prompt_for(cfg))
    out = paths.dpo_checkpoints(round_no)
    result, batch, accum = run_with_oom_retry(
        lambda bs, ga: _train_dpo_once(cfg, start_adapter, rows, out, bs, ga), cfg.batch_size, cfg.grad_accum
    )
    return save_round(result, out, batch, accum)


def evaluate_adapter(cfg: TousterConfig, paths: RunPaths, client: Any, adapter: Path, stage: str) -> float:
    """Evaluate an adapter on the held-out set and return its primary score."""
    from touster.eval.metrics import evaluate
    from touster.train.sft import load_student, release_student

    bundle = load_student(cfg, adapter, for_training=False)
    try:
        return stage_score(evaluate(bundle, stage, cfg, paths, client, use_adapter=True, regression=False))
    finally:
        release_student(bundle)
        del bundle


# ---------- loop ----------

def reset_round(paths: RunPaths, round_no: int) -> None:
    """Delete a round's checkpoints, pairs and scores so it is redone from scratch."""
    from touster.train.sft import clear_folder

    clear_folder(paths.dpo_checkpoints(round_no))
    paths.dpo_pairs(round_no).unlink(missing_ok=True)
    scores_path(paths, round_no).unlink(missing_ok=True)


def run_round(cfg: TousterConfig, paths: RunPaths, client: Any, budget: BudgetTracker, round_no: int, best: Best, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One loop round (DPO or GRPO) from the best adapter; None when no usable pairs were found."""
    started = time.monotonic()
    size = SMOKE_ROUND_PROMPTS if cfg.smoke_test else ROUND_PROMPTS
    subset = select_round_records(records, size, round_no, cfg.seed)
    start_adapter = Path(best.adapter)
    if cfg.loop_method == LoopMethod.GRPO:
        from touster.train.grpo import train_grpo_round

        budget.check("loop")
        stats, n_pairs = train_grpo_round(cfg, paths, start_adapter, subset, round_no), 0
    else:
        pairs = collect_dpo_pairs(cfg, paths, client, budget, round_no, start_adapter, subset)
        if not pairs:
            log_event(f"Round {round_no}: no pairs with a score gap >= {MIN_PAIR_GAP}; stopping", "warning")
            return None
        budget.check("loop")
        stats, n_pairs = train_dpo(cfg, paths, start_adapter, pairs, round_no), len(pairs)
    budget.check("loop")
    stage = stage_name(round_no)
    score = evaluate_adapter(cfg, paths, client, Path(stats["adapter"]), stage)
    return {
        "round": round_no, "stage": stage, "score": score, "adapter": stats["adapter"],
        "gain_points": gain_points(best.score, score), "n_pairs": n_pairs, "steps": stats["steps"],
        "batch_size": stats["batch_size"], "minutes": round((time.monotonic() - started) / 60.0, 2),
    }


def _sft_baseline(paths: RunPaths) -> Best:
    """The SFT checkpoint and its eval score, which the loop has to beat."""
    from touster.train.sft import FINAL_DIR_NAME

    adapter = paths.sft_checkpoints / FINAL_DIR_NAME
    if not adapter.is_dir():
        raise FileNotFoundError(f"SFT adapter not found at {adapter}. Run cell 4 (SFT) first.")
    if not paths.eval_result("sft").is_file():
        raise FileNotFoundError(f"{paths.eval_result('sft')} is missing. Run cell 5 (evaluation) first.")
    return Best("sft", stage_score(read_json(paths.eval_result("sft"))), str(adapter))


def _loop_metrics(cfg: TousterConfig, best: Best, rounds: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    """Metrics recorded in state for the loop cell."""
    return {
        "method": cfg.loop_method.value, "best_stage": best.stage, "best_score": best.score,
        "best_adapter": best.adapter, "rounds": rounds, "stop_reason": reason,
    }


def run_loop(cfg: TousterConfig, paths: RunPaths, client: Any, budget: BudgetTracker, force: bool = False) -> dict[str, Any]:
    """Run rounds until a stop rule fires, keeping the best-scoring checkpoint; force redoes every round."""
    from touster.train.sft import load_sft_records

    require_gpu("loop")
    best = _sft_baseline(paths)
    records = load_sft_records(paths)
    if force:
        update_state(paths, **{STATE_KEY: []})
    rounds = finished_rounds(load_state(paths), paths)
    for done in rounds:
        best = update_best(best, done["stage"], done["score"], done["adapter"])
    cap = max_rounds_for(cfg)
    reason: str | None = None
    for round_no in range(len(rounds) + 1, cap + 2):
        reason = stop_reason([r["gain_points"] for r in rounds], len(rounds), cap, cfg.min_gain,
                             budget.target_reached(best.score), not budget.exhausted("loop"))
        if reason:
            break
        try:
            if force:
                reset_round(paths, round_no)
            record = run_round(cfg, paths, client, budget, round_no, best, records)
        except BudgetExceeded as err:
            log_event(f"Loop stopped on budget cap: {err}", "warning")
            update_state(paths, status=STATUS_STOPPED_BUDGET)
            reason = "budget"
            break
        if record is None:
            reason = "no_pairs"
            break
        rounds = [*rounds, record]
        update_state(paths, **{STATE_KEY: rounds})
        best = update_best(best, record["stage"], record["score"], record["adapter"])
        log_event(f"Round {round_no}: score {record['score']:.4f} ({record['gain_points']:+.1f} pts), best = {best.stage}", "info")
    return _loop_metrics(cfg, best, rounds, reason or "max_rounds")


def skipped_loop_metrics(cfg: TousterConfig, paths: RunPaths) -> dict[str, Any]:
    """Loop metrics when self_improve is off: the SFT checkpoint is the best."""
    from touster.train.sft import FINAL_DIR_NAME

    score = stage_score(read_json(paths.eval_result("sft"))) if paths.eval_result("sft").is_file() else None
    best = Best("sft", score if score is not None else float("nan"), str(paths.sft_checkpoints / FINAL_DIR_NAME))
    metrics = _loop_metrics(cfg, best, [], "self_improve_off")
    return {**metrics, "best_score": score}


def _loop_summary(metrics: dict[str, Any]) -> str:
    """One-line summary for the loop cell."""
    if not metrics:
        return "no metrics"
    return f"{len(metrics.get('rounds', []))} round(s), best = {metrics.get('best_stage')} ({metrics.get('stop_reason')})"


def run_loop_cell(cfg: TousterConfig, paths: RunPaths, client: Any, budget: BudgetTracker, force: bool = False) -> dict[str, Any]:
    """Cell 6: self-improvement loop under the shared cell contract."""

    def work() -> dict[str, Any]:
        """Skip cleanly when self_improve is off, else run the loop."""
        if not cfg.self_improve:
            log_event("self_improve is off; the SFT checkpoint is the final model", "info")
            return skipped_loop_metrics(cfg, paths)
        return run_loop(cfg, paths, client, budget, force)

    return run_cell(paths, "loop", work, _loop_summary, force)
