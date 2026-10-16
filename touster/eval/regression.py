"""Small lm-evaluation-harness subset to catch general-skill regressions after fine-tuning."""

from __future__ import annotations

import sys
from typing import Any

from touster.config import TousterConfig
from touster.dashboard import log_event

ARC_TASK = "arc_easy"
MMLU_TASKS = (
    "mmlu_high_school_computer_science",
    "mmlu_high_school_mathematics",
    "mmlu_global_facts",
    "mmlu_miscellaneous",
)
REGRESSION_LIMIT = 100
REGRESSION_BATCH_SIZE = 8
ACC_KEY = "acc,none"
MMLU_SUBSET_KEY = "mmlu_subset"
LM_EVAL_ERRORS: tuple[type[BaseException], ...] = (
    ImportError, OSError, ConnectionError, ValueError, RuntimeError, KeyError, AssertionError,
)


def _summarise(results: dict[str, Any]) -> dict[str, float]:
    """Turn simple_evaluate's per-task results into {"arc_easy": acc, "mmlu_subset": mean acc}."""
    per_task = results["results"]
    scores: dict[str, float] = {}
    if ARC_TASK in per_task:
        scores[ARC_TASK] = float(per_task[ARC_TASK][ACC_KEY])
    mmlu = [float(per_task[t][ACC_KEY]) for t in MMLU_TASKS if t in per_task]
    if mmlu:
        scores[MMLU_SUBSET_KEY] = sum(mmlu) / len(mmlu)
    return scores


def _simple_evaluate(bundle: Any, cfg: TousterConfig) -> dict[str, Any] | None:
    """Run lm_eval.simple_evaluate on the already-loaded model with a small sample limit."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=bundle.model,
        tokenizer=bundle.tokenizer,
        batch_size=REGRESSION_BATCH_SIZE,
        max_length=cfg.max_seq_len,
    )
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=[ARC_TASK, *MMLU_TASKS],
        num_fewshot=0,
        limit=REGRESSION_LIMIT,
        bootstrap_iters=0,
        log_samples=False,
        random_seed=cfg.seed,
        numpy_random_seed=cfg.seed,
        torch_random_seed=cfg.seed,
        fewshot_random_seed=cfg.seed,
    )


def _reraise_cuda_oom(err: BaseException) -> None:
    """Re-raise CUDA out-of-memory errors instead of hiding them as a skipped check."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(err, torch.cuda.OutOfMemoryError):
        raise torch.cuda.OutOfMemoryError(
            "Regression check ran out of GPU memory. Restart the runtime, or lower "
            "REGRESSION_BATCH_SIZE in touster/eval/regression.py, then rerun the cell."
        ) from err


def run_regression(bundle: Any, cfg: TousterConfig) -> dict[str, float]:
    """Score the loaded model on the regression subset; return {} with a warning if lm_eval fails."""
    try:
        results = _simple_evaluate(bundle, cfg)
    except LM_EVAL_ERRORS as err:
        _reraise_cuda_oom(err)
        log_event(
            f"Regression check skipped, lm_eval failed ({type(err).__name__}: {err}). "
            "Check that the Hugging Face datasets ai2_arc and cais/mmlu can be downloaded.",
            "warning",
        )
        return {}
    if not results or "results" not in results:
        log_event("Regression check skipped: lm_eval returned no results", "warning")
        return {}
    return _summarise(results)


def regression_warnings(base: dict[str, float], current: dict[str, float], max_drop_points: float = 5.0) -> list[str]:
    """List every benchmark where current is more than max_drop_points (0-100 scale) below base."""
    warnings = []
    for name in sorted(set(base) & set(current)):
        drop = 100.0 * (base[name] - current[name])
        if drop > max_drop_points:
            warnings.append(
                f"{name} dropped {drop:.1f} points vs base ({100 * base[name]:.1f} -> {100 * current[name]:.1f})"
            )
    return warnings
