"""Cell 4: QLoRA SFT with Unsloth + TRL, plus the student loader and batched generation used by later cells."""

from __future__ import annotations

import gc
import importlib.util
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from touster.config import SMOKE_SFT_STEPS, TousterConfig
from touster.dashboard import log_event, set_progress, set_training
from touster.state import (
    RunPaths,
    detect_gpu,
    load_training_records,
    read_json,
    require_gpu,
    run_cell,
    write_json,
)

log = logging.getLogger(__name__)

LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CHECKPOINT_PREFIX = "checkpoint-"
FINAL_DIR_NAME = "final"
TRAINER_STATE_FILE = "trainer_state.json"
MAX_SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
PROBE_STEPS = 50
GENERATION_BATCH_SIZE = 8
VLLM_MIN_CAPABILITY = (7, 0)
VLLM_MAX_LORA_RANK = 64
VLLM_GPU_MEMORY_UTILIZATION = 0.5
# Below this, an engine that is not fully torn down between stages can push the next stage out of memory.
VLLM_MIN_VRAM_GB = 24.0
WARMUP_RATIO = 0.03
OOM_ADVICE = (
    "Training ran out of GPU memory even after halving the batch size. Try one of: a smaller max_seq_len "
    "(e.g. 1024), a smaller student_model (e.g. a 0.5B model), or a lower lora_r (e.g. 8)."
)

T = TypeVar("T")


class TrainingOOMError(RuntimeError):
    """Raised when training still runs out of GPU memory after the single retry."""


@dataclass(frozen=True)
class StudentBundle:
    """A loaded student: model, tokenizer, system prompt, and how to generate with it."""

    model: Any
    tokenizer: Any
    system_prompt: str
    adapter_path: Path | None = None
    use_vllm: bool = False


def system_prompt_for(cfg: TousterConfig) -> str:
    """Build the system prompt used in training, generation and export from the task description."""
    return f"You are a specialised assistant. Task: {cfg.task_description.strip()} Answer directly and concisely."


def to_chat_messages(record: dict[str, Any], system_prompt: str) -> dict[str, list[dict[str, str]]]:
    """Convert a {prompt, response} record into a conversational row for TRL."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    }


def prompt_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    """System + user messages for a prompt, with no assistant turn."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


def valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep records with a non-empty prompt and response; log how many were dropped."""
    kept = [
        r for r in records
        if isinstance(r.get("prompt"), str) and r["prompt"].strip()
        and isinstance(r.get("response"), str) and r["response"].strip()
    ]
    if len(kept) < len(records):
        log.warning("Dropped %d training records with an empty prompt or response", len(records) - len(kept))
    return kept


def load_sft_records(paths: RunPaths) -> list[dict[str, Any]]:
    """Load cleaned synthetic data plus the golden seed split, only through the leak-checked loader."""
    if not paths.golden_seed.is_file():
        raise FileNotFoundError(f"{paths.golden_seed} is missing. Run cell 2 (golden set) first.")
    records = load_training_records(paths.golden_seed, paths)
    if paths.synthetic_clean.is_file():
        records = load_training_records(paths.synthetic_clean, paths) + records
    else:
        log.warning("%s not found; training on the golden seed split only", paths.synthetic_clean)
    return valid_records(records)


def latest_checkpoint(folder: Path) -> Path | None:
    """Return the checkpoint-<step> folder with the highest step number, or None."""
    if not folder.is_dir():
        return None
    pattern = re.compile(rf"^{CHECKPOINT_PREFIX}(\d+)$")
    steps = [(int(m.group(1)), p) for p in folder.iterdir() if p.is_dir() and (m := pattern.match(p.name))]
    return max(steps, key=lambda item: item[0])[1] if steps else None


def read_loss_history(folder: Path) -> list[dict[str, float]]:
    """Read (step, loss) points from trainer_state.json in folder, its final/ subfolder or its latest checkpoint."""
    candidates = [folder / TRAINER_STATE_FILE, folder / FINAL_DIR_NAME / TRAINER_STATE_FILE]
    latest = latest_checkpoint(folder)
    if latest is not None:
        candidates.append(latest / TRAINER_STATE_FILE)
    for path in candidates:
        if path.is_file():
            history = read_json(path).get("log_history", [])
            return [{"step": int(h["step"]), "loss": float(h["loss"])} for h in history if "loss" in h and "step" in h]
    return []


def save_steps_for(total_steps: int) -> int:
    """Checkpoint interval: every 200 steps, or once at the end for shorter runs."""
    return max(1, min(MAX_SAVE_STEPS, total_steps))


def logging_steps_for(total_steps: int) -> int:
    """Log about 20 times per run, at least every 10 steps."""
    return max(1, min(10, total_steps // 20))


def sft_max_steps(cfg: TousterConfig) -> int:
    """max_steps for SFTConfig: the smoke cap, or -1 to train for cfg.epochs."""
    return SMOKE_SFT_STEPS if cfg.smoke_test else -1


def tokens_per_second(prev_tokens: int, prev_time: float | None, tokens: int, now: float) -> float | None:
    """Throughput between two log events, or None when it cannot be measured."""
    if prev_time is None or now <= prev_time or tokens <= prev_tokens:
        return None
    return (tokens - prev_tokens) / (now - prev_time)


def gpu_memory_gb() -> float | None:
    """Peak allocated GPU memory in GB, or None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def free_gpu_memory() -> None:
    """Run the garbage collector and release cached CUDA memory."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def precision_kwargs() -> dict[str, bool]:
    """bf16 on GPUs that support it (Ampere+), fp16 otherwise (e.g. T4)."""
    import torch

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False))
    return {"bf16": bf16, "fp16": not bf16}


def make_dashboard_callback(label: str) -> Any:
    """Build a TrainerCallback that forwards loss, lr, tokens/sec, GPU memory and progress to the dashboard."""
    from transformers import TrainerCallback

    class DashboardCallback(TrainerCallback):
        """Pushes trainer logs to the touster dashboard."""

        def __init__(self) -> None:
            """Start with no throughput reference point."""
            self._prev_time: float | None = None
            self._prev_tokens = 0

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            """Forward the latest metrics."""
            logs = logs or {}
            now, tokens = time.monotonic(), int(getattr(state, "num_input_tokens_seen", 0) or 0)
            tps = tokens_per_second(self._prev_tokens, self._prev_time, tokens, now)
            self._prev_time, self._prev_tokens = now, tokens
            set_training(
                loss=logs.get("loss"), learning_rate=logs.get("learning_rate"),
                tokens_per_sec=tps, gpu_mem_gb=gpu_memory_gb(),
            )

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Advance the progress bar."""
            set_progress(label, state.global_step, max(state.max_steps, 1))

    return DashboardCallback()


def _attempt(train_once: Callable[[int, int], T], batch_size: int, grad_accum: int) -> tuple[bool, T | None]:
    """Run one training attempt; return (False, None) on CUDA OOM so the failed frames are released."""
    import torch

    try:
        return True, train_once(batch_size, grad_accum)
    except torch.cuda.OutOfMemoryError as err:
        log.warning("CUDA out of memory at batch_size=%d: %s", batch_size, err)
        return False, None


def run_with_oom_retry(
    train_once: Callable[[int, int], T], batch_size: int, grad_accum: int, free: Callable[[], None] = free_gpu_memory
) -> tuple[T, int, int]:
    """Run train_once(batch, accum); on CUDA OOM halve batch, double accumulation and retry once."""
    ok, result = _attempt(train_once, batch_size, grad_accum)
    if ok:
        return result, batch_size, grad_accum  # type: ignore[return-value]
    free()
    if batch_size <= 1:
        raise TrainingOOMError(OOM_ADVICE)
    smaller, more_accum = batch_size // 2, grad_accum * 2
    log_event(f"Out of GPU memory; retrying with batch_size={smaller}, grad_accum={more_accum}", "warning")
    ok, result = _attempt(train_once, smaller, more_accum)
    if ok:
        return result, smaller, more_accum  # type: ignore[return-value]
    free()
    raise TrainingOOMError(OOM_ADVICE)


def vllm_supported(cfg: TousterConfig) -> bool:
    """True when vLLM is installed, the GPU is new enough with >= 24 GB VRAM, and the LoRA rank fits vLLM."""
    import torch

    if importlib.util.find_spec("vllm") is None or not torch.cuda.is_available():
        return False
    if detect_gpu()["vram_gb"] < VLLM_MIN_VRAM_GB:
        return False
    return torch.cuda.get_device_capability(0) >= VLLM_MIN_CAPABILITY and cfg.lora_r <= VLLM_MAX_LORA_RANK


def release_student(bundle: StudentBundle | None) -> None:
    """Shut down the bundle's vLLM engine (if any) and free GPU memory; the caller must also drop its reference."""
    if bundle is None:
        return
    engine = getattr(bundle.model, "vllm_engine", None)
    if bundle.use_vllm and engine is not None:
        from unsloth_zoo.vllm_utils import delete_vllm

        delete_vllm(engine)
        bundle.model.vllm_engine = None
    free_gpu_memory()


def checkpoint_finished(checkpoint: Path) -> bool:
    """True when the checkpoint's trainer_state shows the run reached max_steps."""
    state_file = checkpoint / TRAINER_STATE_FILE
    if not state_file.is_file():
        return False
    state = read_json(state_file)
    max_steps = int(state.get("max_steps", 0) or 0)
    return max_steps > 0 and int(state.get("global_step", 0) or 0) >= max_steps


def clear_folder(folder: Path) -> None:
    """Delete a training output folder so the next run starts from scratch."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def resume_point(folder: Path, force: bool) -> Path | None:
    """Checkpoint to resume from; force or an already finished run clears the folder and starts fresh."""
    latest = latest_checkpoint(folder)
    if force or (latest is not None and checkpoint_finished(latest)):
        if latest is not None:
            log_event(f"Starting fresh in {folder} ({'force' if force else 'previous run already finished'})", "info")
        clear_folder(folder)
        return None
    if latest is not None:
        log_event(f"Resuming from {latest}", "info")
    return latest


def _attach_lora(model: Any, cfg: TousterConfig) -> Any:
    """Add LoRA adapters on every linear projection with Unsloth gradient checkpointing."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )


def load_student(cfg: TousterConfig, adapter_path: Path | None = None, for_training: bool = True) -> StudentBundle:
    """Load the student in 4-bit NF4; with an adapter folder the saved LoRA is loaded on top (trainable)."""
    from unsloth import FastLanguageModel

    use_vllm = not for_training and vllm_supported(cfg)
    source = str(adapter_path) if adapter_path is not None else cfg.student_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        fast_inference=use_vllm,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_lora_rank=VLLM_MAX_LORA_RANK,
    )
    if for_training and adapter_path is None:
        model = _attach_lora(model, cfg)
    if not for_training:
        FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return StudentBundle(model, tokenizer, system_prompt_for(cfg), adapter_path, use_vllm)


def truncate_text(tokenizer: Any, text: str, max_tokens: int) -> str:
    """Cut text to at most max_tokens tokens, keeping the start."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[: max(1, max_tokens)], skip_special_tokens=True)


def format_prompts(tokenizer: Any, system_prompt: str, prompts: list[str], max_prompt_tokens: int) -> list[str]:
    """Render prompts with the student's chat template, truncating the user text to fit max_prompt_tokens."""
    template = lambda p: tokenizer.apply_chat_template(  # noqa: E731
        prompt_messages(p, system_prompt), tokenize=False, add_generation_prompt=True
    )
    overhead = len(tokenizer(template(""), add_special_tokens=False)["input_ids"])
    budget = max(1, max_prompt_tokens - overhead)
    return [template(truncate_text(tokenizer, p, budget)) for p in prompts]


def generation_lengths(cfg: TousterConfig, max_new_tokens: int) -> tuple[int, int]:
    """Split max_seq_len into (prompt budget, new tokens); new tokens never exceed half the window."""
    new_tokens = max(1, min(max_new_tokens, cfg.max_seq_len // 2))
    return cfg.max_seq_len - new_tokens, new_tokens


def lora_active(bundle: StudentBundle) -> bool:
    """True when generation should apply the adapter: one is loaded and not switched off by peft's disable_adapter()."""
    return bundle.adapter_path is not None and not getattr(bundle.model, "_adapters_disabled", False)


def _generate_vllm(bundle: StudentBundle, texts: list[str], n: int, temperature: float, new_tokens: int, seed: int) -> list[list[str]]:
    """Generate with Unsloth's vLLM engine, applying the LoRA through a LoRA request."""
    from vllm import SamplingParams

    params = SamplingParams(n=n, temperature=temperature, max_tokens=new_tokens, seed=seed)
    lora = bundle.model.load_lora(str(bundle.adapter_path)) if lora_active(bundle) else None
    outputs = bundle.model.fast_generate(texts, sampling_params=params, lora_request=lora, use_tqdm=False)
    return [[c.text for c in out.outputs] for out in outputs]


def _generate_hf(bundle: StudentBundle, texts: list[str], n: int, temperature: float, new_tokens: int) -> list[list[str]]:
    """Generate with model.generate in left-padded batches; returns n completions per text."""
    import torch

    tokenizer, model = bundle.tokenizer, bundle.model
    tokenizer.padding_side = "left"
    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    results: list[list[str]] = []
    for start in range(0, len(texts), GENERATION_BATCH_SIZE):
        batch = texts[start : start + GENERATION_BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=new_tokens, num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id, **sampling,
            )
        decoded = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        results.extend(decoded[i * n : (i + 1) * n] for i in range(len(batch)))
    return results


def generate_responses(
    bundle: StudentBundle,
    prompts: list[str],
    cfg: TousterConfig,
    temperature: float = 0.0,
    num_samples: int = 1,
    max_new_tokens: int = 512,
) -> list[list[str]]:
    """Generate exactly num_samples answers per prompt (prompt stripped); temperature 0 is greedy."""
    if not prompts:
        return []
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    prompt_budget, new_tokens = generation_lengths(cfg, max_new_tokens)
    texts = format_prompts(bundle.tokenizer, bundle.system_prompt, prompts, prompt_budget)
    # Greedy decoding gives the same answer every time, so generate once and repeat it.
    n = 1 if temperature <= 0 else num_samples
    if bundle.use_vllm:
        raw = _generate_vllm(bundle, texts, n, max(temperature, 0.0), new_tokens, cfg.seed)
    else:
        import torch

        torch.manual_seed(cfg.seed)
        raw = _generate_hf(bundle, texts, n, temperature, new_tokens)
    return [expand_samples([t.strip() for t in row], num_samples) for row in raw]


def expand_samples(samples: list[str], num_samples: int) -> list[str]:
    """Repeat or trim samples so exactly num_samples are returned."""
    if not samples:
        return [""] * num_samples
    repeated = samples * (-(-num_samples // len(samples)))
    return repeated[:num_samples]


def _sft_config(cfg: TousterConfig, output_dir: Path, batch_size: int, grad_accum: int, max_steps: int, save_steps: int) -> Any:
    """Build the SFTConfig: assistant-only loss, no packing, gradient checkpointing, periodic adapter saves."""
    from trl import SFTConfig

    # Packing is off: packed runs take far fewer steps than the per-example estimate, which throws off the
    # LR schedule and save_steps, and T4 GPUs have no FlashAttention. Assistant-only loss stays on.
    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=max_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        max_length=cfg.max_seq_len,
        packing=False,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        logging_steps=logging_steps_for(max_steps if max_steps > 0 else save_steps),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        include_num_input_tokens_seen=True,
        report_to="none",
        seed=cfg.seed,
        **precision_kwargs(),
    )


def planned_steps(n_rows: int, batch_size: int, grad_accum: int, epochs: int, max_steps: int) -> int:
    """Optimizer steps the trainer will run: max_steps when set, else full epochs (no packing)."""
    if max_steps > 0:
        return max_steps
    return max(1, -(-n_rows // (batch_size * grad_accum)) * epochs)


def _train_sft_once(cfg: TousterConfig, rows: list[dict[str, Any]], output_dir: Path, batch_size: int, grad_accum: int, max_steps: int) -> dict[str, Any]:
    """One SFT attempt: load the student, train (resuming an unfinished checkpoint), return trainer and bundle."""
    resume = resume_point(output_dir, force=False)
    bundle = load_student(cfg, for_training=True)
    from datasets import Dataset
    from trl import SFTTrainer

    total = planned_steps(len(rows), batch_size, grad_accum, cfg.epochs, max_steps)
    args = _sft_config(cfg, output_dir, batch_size, grad_accum, max_steps, save_steps_for(total))
    trainer = SFTTrainer(
        model=bundle.model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=bundle.tokenizer,
        callbacks=[make_dashboard_callback("sft steps")],
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return {"trainer": trainer, "bundle": bundle}


def _save_final(trainer: Any, tokenizer: Any, final_dir: Path) -> None:
    """Save the adapter, tokenizer and trainer state (for the loss chart) into final_dir."""
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.state.save_to_json(str(final_dir / TRAINER_STATE_FILE))


def save_round(result: dict[str, Any], out: Path, batch_size: int, grad_accum: int) -> dict[str, Any]:
    """Save a DPO/GRPO round's adapter and tokenizer to out/final, free the student and return the round stats."""
    trainer, bundle = result["trainer"], result["bundle"]
    final_dir = out / FINAL_DIR_NAME
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    bundle.tokenizer.save_pretrained(str(final_dir))
    stats = {"adapter": str(final_dir), "steps": int(trainer.state.global_step), "batch_size": batch_size, "grad_accum": grad_accum}
    release_student(bundle)
    return stats


def loss_summary(log_history: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """First and last logged training loss."""
    losses = [float(h["loss"]) for h in log_history if "loss" in h]
    return (losses[0], losses[-1]) if losses else (None, None)


def train_sft(cfg: TousterConfig, paths: RunPaths, force: bool = False) -> dict[str, Any]:
    """Run QLoRA SFT on the training records and return the cell metrics; force starts from scratch."""
    started = time.monotonic()
    records = load_sft_records(paths)
    if not records:
        raise ValueError("No training records left after validation. Check cells 2 and 3.")
    system_prompt = system_prompt_for(cfg)
    rows = [to_chat_messages(r, system_prompt) for r in records]
    output_dir = paths.sft_checkpoints
    if force:
        clear_folder(output_dir)
    log_event(f"SFT on {len(rows)} examples ({cfg.student_model})", "info")
    result, batch_size, grad_accum = run_with_oom_retry(
        lambda bs, ga: _train_sft_once(cfg, rows, output_dir, bs, ga, sft_max_steps(cfg)), cfg.batch_size, cfg.grad_accum
    )
    trainer, bundle = result["trainer"], result["bundle"]
    final_dir = output_dir / FINAL_DIR_NAME
    _save_final(trainer, bundle.tokenizer, final_dir)
    first_loss, final_loss = loss_summary(trainer.state.log_history)
    metrics = {
        "steps": int(trainer.state.global_step), "first_loss": first_loss, "final_loss": final_loss,
        "adapter": str(final_dir), "batch_size": batch_size, "grad_accum": grad_accum,
        "examples": len(rows), "minutes": round((time.monotonic() - started) / 60.0, 2),
    }
    release_student(bundle)
    del trainer, bundle, result
    free_gpu_memory()
    return metrics


def probe_training(cfg: TousterConfig, records: list[dict[str, Any]], steps: int = PROBE_STEPS) -> dict[str, Any]:
    """Run a short SFT probe in a temp dir; return seconds, steps, tokens_per_sec and peak_vram_gb."""
    import torch

    require_gpu("probe")
    rows = [to_chat_messages(r, system_prompt_for(cfg)) for r in valid_records(records)]
    if not rows:
        raise ValueError("probe_training needs at least one record with a prompt and response")
    tmp = Path(tempfile.mkdtemp(prefix="touster_probe_"))
    result: dict[str, Any] | None = None
    try:
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result, _, _ = run_with_oom_retry(
            lambda bs, ga: _train_sft_once(cfg, rows, tmp, bs, ga, steps), cfg.batch_size, cfg.grad_accum
        )
        seconds = time.monotonic() - started
        state = result["trainer"].state
        done, tokens = int(state.global_step), int(getattr(state, "num_input_tokens_seen", 0) or 0)
        peak = torch.cuda.max_memory_allocated() / 1024**3
    except TrainingOOMError as err:
        raise TrainingOOMError(f"Training probe failed: {err}") from None
    finally:
        del result
        free_gpu_memory()
        shutil.rmtree(tmp, ignore_errors=True)
    return {"seconds": seconds, "steps": done, "tokens_per_sec": tokens / seconds if seconds > 0 else 0.0, "peak_vram_gb": peak}


def _sft_summary(metrics: dict[str, Any]) -> str:
    """One-line summary for the SFT cell."""
    if not metrics:
        return "no metrics"
    return (
        f"{metrics.get('steps')} steps, loss {metrics.get('first_loss')} -> {metrics.get('final_loss')}, "
        f"batch {metrics.get('batch_size')}, adapter at {metrics.get('adapter')}"
    )


def run_sft_cell(cfg: TousterConfig, paths: RunPaths, force: bool = False) -> dict[str, Any]:
    """Cell 4: QLoRA SFT under the shared cell contract."""

    def work() -> dict[str, Any]:
        """Check for a GPU, then train."""
        require_gpu("sft")
        metrics = train_sft(cfg, paths, force)
        write_json(paths.sft_checkpoints / "sft_metrics.json", metrics)
        return metrics

    return run_cell(paths, "sft", work, _sft_summary, force)
