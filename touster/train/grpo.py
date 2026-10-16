"""GRPO round with programmatic rewards for verifiable tasks (no judge model involved)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from touster.config import SMOKE_DPO_STEPS, TaskType, TousterConfig
from touster.dashboard import log_event
from touster.state import RunPaths
from touster.text import normalize_text, strip_code_fence

log = logging.getLogger(__name__)

CODE_TEST_TIMEOUT_SECONDS = 10
CODE_COMPILE_ONLY_REWARD = 0.25
JSON_VALID_REWARD = 0.2
GRPO_LR_SCALE = 0.1
GRPO_TEMPERATURE = 0.8
GRPO_MAX_COMPLETION_TOKENS = 512

RewardFn = Callable[..., list[float]]


def completion_text(completion: Any) -> str:
    """Return the text of a TRL completion: a plain string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return ""


def exact_match_reward(completions: list[Any], reference: list[str], **kwargs: Any) -> list[float]:
    """1.0 when the normalised answer equals the normalised reference, else 0.0."""
    return [
        1.0 if normalize_text(completion_text(c)) == normalize_text(str(ref)) else 0.0
        for c, ref in zip(completions, reference, strict=True)
    ]


def parse_json(text: str) -> Any:
    """Parse JSON from a model answer (fences allowed); raise ValueError when invalid."""
    try:
        return json.loads(strip_code_fence(text).strip())
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid JSON: {err.msg}") from None


def json_score(answer: str, reference: str) -> float:
    """0 for invalid JSON; otherwise 0.2 plus 0.8 times the share of reference key/value pairs matched."""
    try:
        got = parse_json(answer)
    except ValueError:
        return 0.0
    try:
        want = parse_json(reference)
    except ValueError:
        log.warning("Reference answer is not valid JSON; only JSON validity is rewarded")
        return JSON_VALID_REWARD
    if isinstance(want, dict) and isinstance(got, dict):
        matched = sum(1 for k, v in want.items() if k in got and got[k] == v)
        share = matched / len(want) if want else 1.0
    else:
        share = 1.0 if got == want else 0.0
    return JSON_VALID_REWARD + (1.0 - JSON_VALID_REWARD) * share


def json_match_reward(completions: list[Any], reference: list[str], **kwargs: Any) -> list[float]:
    """Reward valid JSON whose keys and values match the reference."""
    return [json_score(completion_text(c), str(ref)) for c, ref in zip(completions, reference, strict=True)]


def as_test_lines(tests: Any) -> list[str]:
    """Normalise record metadata 'tests' (a string or list of assert lines) to a list of lines."""
    if tests is None:
        return []
    if isinstance(tests, str):
        return [line for line in tests.splitlines() if line.strip()]
    return [str(line) for line in tests if str(line).strip()]


def run_python_tests(code: str, tests: list[str], timeout: float = CODE_TEST_TIMEOUT_SECONDS) -> bool:
    """Run code followed by assert lines through the eval sandbox; True when it exits cleanly in time."""
    from touster.eval.metrics import passes_tests

    return passes_tests(code, "\n".join(tests), timeout)


def code_score(answer: str, tests: list[str]) -> float:
    """0 if the code does not compile; 1 if it compiles and has no tests; with tests 0.25 + 0.75 if they pass."""
    code = strip_code_fence(answer).strip()
    try:
        compile(code, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return 0.0
    if not tests:
        return 1.0
    passed = run_python_tests(code, tests)
    return CODE_COMPILE_ONLY_REWARD + (1.0 - CODE_COMPILE_ONLY_REWARD) * float(passed)


def code_reward(completions: list[Any], tests: list[Any] | None = None, **kwargs: Any) -> list[float]:
    """Reward code that compiles and passes the record's assert-line tests."""
    tests = tests if tests is not None else [None] * len(completions)
    return [code_score(completion_text(c), as_test_lines(t)) for c, t in zip(completions, tests, strict=True)]


def reward_functions(cfg: TousterConfig) -> list[RewardFn]:
    """Programmatic reward functions for the task type, in TRL's (completions, **columns) form."""
    if cfg.task_type in (TaskType.CLASSIFICATION, TaskType.QA):
        return [exact_match_reward]
    if cfg.task_type == TaskType.EXTRACTION:
        return [json_match_reward]
    if cfg.task_type == TaskType.CODE:
        return [code_reward]
    raise ValueError(
        f"GRPO needs a programmatic reward, which task_type={cfg.task_type.value!r} does not have. "
        "Use loop_method='dpo' for this task."
    )


def grpo_rows(records: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    """Conversational prompt rows with the reference answer and tests as extra columns for the rewards."""
    return [
        {
            "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["prompt"]}],
            "reference": r["response"],
            "tests": as_test_lines((r.get("metadata") or {}).get("tests")),
        }
        for r in records
    ]


def _grpo_config(cfg: TousterConfig, output_dir: Path, batch_size: int, grad_accum: int, total_steps: int) -> Any:
    """Build the GRPOConfig; per-device batch is a multiple of num_generations."""
    from trl import GRPOConfig

    from touster.train.sft import WARMUP_RATIO, logging_steps_for, precision_kwargs, save_steps_for

    return GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=cfg.samples_per_prompt,
        max_completion_length=min(GRPO_MAX_COMPLETION_TOKENS, cfg.max_seq_len // 2),
        temperature=GRPO_TEMPERATURE,
        learning_rate=cfg.learning_rate * GRPO_LR_SCALE,
        warmup_steps=WARMUP_RATIO,
        num_train_epochs=1,
        max_steps=SMOKE_DPO_STEPS if cfg.smoke_test else -1,
        gradient_checkpointing=True,
        logging_steps=logging_steps_for(total_steps),
        save_strategy="steps",
        save_steps=save_steps_for(total_steps),
        report_to="none",
        seed=cfg.seed,
        **precision_kwargs(),
    )


def _train_grpo_once(cfg: TousterConfig, start_adapter: Path, rows: list[dict[str, Any]], out: Path, batch_size: int, grad_accum: int) -> dict[str, Any]:
    """One GRPO attempt starting from start_adapter; resumes from a checkpoint in out if present."""
    from touster.train.sft import load_student, make_dashboard_callback, resume_point

    resume = resume_point(out, force=False)
    bundle = load_student(cfg, start_adapter, for_training=True)
    from datasets import Dataset
    from trl import GRPOTrainer

    total = SMOKE_DPO_STEPS if cfg.smoke_test else max(1, len(rows) // max(1, batch_size * grad_accum // cfg.samples_per_prompt))
    trainer = GRPOTrainer(
        model=bundle.model,
        reward_funcs=reward_functions(cfg),
        args=_grpo_config(cfg, out, batch_size, grad_accum, total),
        train_dataset=Dataset.from_list(rows),
        processing_class=bundle.tokenizer,
        callbacks=[make_dashboard_callback("grpo steps")],
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return {"trainer": trainer, "bundle": bundle}


def train_grpo_round(cfg: TousterConfig, paths: RunPaths, start_adapter: Path, records: list[dict[str, Any]], round_no: int) -> dict[str, Any]:
    """Train one GRPO round from start_adapter on records; return the new adapter path and training stats."""
    from touster.train.sft import run_with_oom_retry, save_round, system_prompt_for

    reward_functions(cfg)  # fail fast on a task type without a programmatic reward
    rows = grpo_rows(records, system_prompt_for(cfg))
    if not rows:
        raise ValueError("GRPO round has no prompts to train on")
    out = paths.dpo_checkpoints(round_no)
    log_event(f"GRPO round {round_no}: {len(rows)} prompts x {cfg.samples_per_prompt} samples", "info")
    batch = max(cfg.samples_per_prompt, cfg.batch_size - cfg.batch_size % cfg.samples_per_prompt)
    result, used_batch, used_accum = run_with_oom_retry(
        lambda bs, ga: _train_grpo_once(cfg, start_adapter, rows, out, bs, ga), batch, cfg.grad_accum
    )
    return {**save_round(result, out, used_batch, used_accum), "n_prompts": len(rows)}
