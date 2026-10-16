"""CPU tests for touster.train.sft: chat format, checkpoints, OOM retry, loaders, generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from touster.state import GPUUnavailableError, LeakageError, update_state, write_json, write_jsonl
from touster.train import sft
from touster.train.sft import (
    StudentBundle,
    TrainingOOMError,
    expand_samples,
    format_prompts,
    generation_lengths,
    latest_checkpoint,
    lora_active,
    loss_summary,
    read_loss_history,
    run_with_oom_retry,
    save_steps_for,
    system_prompt_for,
    to_chat_messages,
    tokens_per_second,
)


class FakeTokenizer:
    """Whitespace tokenizer with a trivial chat template."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        """Token ids are word positions."""
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Not used for content; tests check truncation through apply_chat_template."""
        raise AssertionError("decode is patched per test")

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool) -> str:
        """Render as 'SYS <system> USER <user> ASSISTANT'."""
        return f"SYS {messages[0]['content']} USER {messages[1]['content']} ASSISTANT"


def test_chat_format_conversion(make_cfg: Any) -> None:
    """A record becomes system, user and assistant messages with the task in the system prompt."""
    cfg = make_cfg()
    system = system_prompt_for(cfg)
    row = to_chat_messages({"id": "g-1", "prompt": "Is it good?", "response": "positive"}, system)
    assert row == {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Is it good?"},
            {"role": "assistant", "content": "positive"},
        ]
    }
    assert cfg.task_description in system


def test_latest_checkpoint_orders_numerically(tmp_path: Path) -> None:
    """checkpoint-<step> folders sort by step number, ignoring other names."""
    for name in ("checkpoint-200", "checkpoint-1000", "checkpoint-50", "final", "checkpoint-x"):
        (tmp_path / name).mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1000"


def test_latest_checkpoint_none_when_missing(tmp_path: Path) -> None:
    """A missing or empty folder has no latest checkpoint."""
    assert latest_checkpoint(tmp_path / "nope") is None
    assert latest_checkpoint(tmp_path) is None


def test_read_loss_history_from_final(tmp_path: Path) -> None:
    """Loss points are read from final/trainer_state.json, skipping rows without a loss."""
    state = {"log_history": [{"step": 10, "loss": 2.0}, {"step": 20, "loss": 1.5}, {"step": 20, "train_runtime": 3}]}
    write_json(tmp_path / "final" / "trainer_state.json", state)
    assert read_loss_history(tmp_path) == [{"step": 10, "loss": 2.0}, {"step": 20, "loss": 1.5}]
    assert loss_summary(state["log_history"]) == (2.0, 1.5)


def test_save_steps_cap() -> None:
    """Checkpoints are saved every 200 steps, or at the end of shorter runs."""
    assert save_steps_for(10) == 10
    assert save_steps_for(5000) == 200
    assert save_steps_for(0) == 1


def test_oom_retry_halves_batch_and_doubles_accum() -> None:
    """After one OOM the batch is halved, accumulation doubled and memory freed."""
    calls: list[tuple[int, int]] = []

    def train_once(bs: int, ga: int) -> str:
        """Fail with OOM on the first call only."""
        calls.append((bs, ga))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "ok"

    freed: list[int] = []
    result, bs, ga = run_with_oom_retry(train_once, 4, 2, free=lambda: freed.append(1))
    assert (result, bs, ga) == ("ok", 2, 4)
    assert calls == [(4, 2), (2, 4)]
    assert freed == [1]


def test_oom_twice_stops_with_advice() -> None:
    """A second OOM raises TrainingOOMError with advice."""
    def train_once(bs: int, ga: int) -> str:
        """Always run out of memory."""
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(TrainingOOMError, match="max_seq_len"):
        run_with_oom_retry(train_once, 4, 2, free=lambda: None)


def test_oom_at_batch_one_does_not_retry() -> None:
    """At batch size 1 there is nothing to halve, so OOM is raised at once."""
    calls: list[int] = []

    def train_once(bs: int, ga: int) -> str:
        """Record the batch size and run out of memory."""
        calls.append(bs)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(TrainingOOMError):
        run_with_oom_retry(train_once, 1, 2, free=lambda: None)
    assert calls == [1]


def test_other_errors_are_not_retried() -> None:
    """Errors other than OOM pass through without a retry."""
    with pytest.raises(ValueError):
        run_with_oom_retry(lambda bs, ga: (_ for _ in ()).throw(ValueError("bad")), 4, 2, free=lambda: None)


def test_training_loader_rejects_eval_ids(run: Any) -> None:
    """The SFT loader refuses synthetic records seeded from eval ids."""
    cfg, paths = run
    write_jsonl(paths.golden_seed, [{"id": "g-0001", "prompt": "p", "response": "r"}])
    write_jsonl(paths.synthetic_clean, [{"id": "s-1", "prompt": "p2", "response": "r2", "seed_id": "g-0009"}])
    update_state(paths, eval_ids=["g-0009"])
    with pytest.raises(LeakageError):
        sft.load_sft_records(paths)


def test_training_loader_drops_empty_records(run: Any) -> None:
    """Records with a blank prompt are dropped; synthetic records come before the seed set."""
    cfg, paths = run
    write_jsonl(paths.golden_seed, [{"id": "g-1", "prompt": "p", "response": "r"}, {"id": "g-2", "prompt": " ", "response": "r"}])
    write_jsonl(paths.synthetic_clean, [{"id": "s-1", "prompt": "p2", "response": "r2", "seed_id": "g-1"}])
    assert [r["id"] for r in sft.load_sft_records(paths)] == ["s-1", "g-1"]


def test_sft_cell_requires_gpu(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a CPU machine the SFT cell names itself in the GPU error."""
    cfg, paths = run
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GPUUnavailableError, match="'sft'"):
        sft.run_sft_cell(cfg, paths)


def test_tokens_per_second() -> None:
    """Throughput needs a previous log point and new tokens."""
    assert tokens_per_second(0, None, 100, 1.0) is None
    assert tokens_per_second(100, 1.0, 300, 3.0) == 100.0
    assert tokens_per_second(300, 3.0, 300, 4.0) is None


def test_expand_samples_exact_count() -> None:
    """Samples are repeated or trimmed to the requested count."""
    assert expand_samples(["a"], 3) == ["a", "a", "a"]
    assert expand_samples(["a", "b", "c"], 2) == ["a", "b"]
    assert expand_samples([], 2) == ["", ""]


def test_generation_lengths_fit_window(make_cfg: Any) -> None:
    """Prompt budget plus new tokens fill the window; new tokens are capped at half."""
    cfg = make_cfg(max_seq_len=1024)
    prompt_budget, new_tokens = generation_lengths(cfg, 2000)
    assert new_tokens == 512 and prompt_budget + new_tokens == 1024


def test_format_prompts_truncates_user_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long user text is cut to the prompt token budget."""
    tok = FakeTokenizer()
    monkeypatch.setattr(tok, "decode", lambda ids, skip_special_tokens=True: " ".join(f"w{i}" for i in ids))
    long_prompt = " ".join(f"w{i}" for i in range(50))
    [text] = format_prompts(tok, "sys", [long_prompt], max_prompt_tokens=10)
    assert len(text.split()) <= 10
    assert text.startswith("SYS sys USER w0")


def test_lora_skipped_when_adapter_disabled() -> None:
    """The LoRA request is only used when an adapter is loaded and not disabled."""
    class FakePeft:
        """Stand-in PeftModel with the adapter switch."""

        _adapters_disabled = False

    model = FakePeft()
    bundle = StudentBundle(model, None, "sys", adapter_path=Path("adapter"), use_vllm=True)
    assert lora_active(bundle)
    model._adapters_disabled = True
    assert not lora_active(bundle)
    assert not lora_active(StudentBundle(FakePeft(), None, "sys"))


def test_generate_responses_vllm_path_respects_disable_adapter(make_cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the adapter disabled, vLLM runs without LoRA; greedy decoding generates once and repeats the answer."""
    cfg = make_cfg()
    seen: dict[str, Any] = {}

    def fake_vllm(bundle: StudentBundle, texts: list[str], n: int, temperature: float, new_tokens: int, seed: int) -> list[list[str]]:
        """Record whether LoRA is on and return answers with stray spaces."""
        seen["lora"] = lora_active(bundle)
        seen["n"] = n
        return [[" answer "] * n for _ in texts]

    monkeypatch.setattr(sft, "_generate_vllm", fake_vllm)
    monkeypatch.setattr(sft, "format_prompts", lambda tok, system, prompts, budget: list(prompts))

    class FakePeft:
        """Stand-in PeftModel with the adapter switched off."""

        _adapters_disabled = True

    bundle = StudentBundle(FakePeft(), None, "sys", adapter_path=Path("a"), use_vllm=True)
    out = sft.generate_responses(bundle, ["p1", "p2"], cfg, temperature=0.0, num_samples=3)
    assert out == [["answer"] * 3, ["answer"] * 3]
    assert seen == {"lora": False, "n": 1}


def write_state(folder: Path, step: int, max_steps: int) -> None:
    """A fake trainer_state.json for a checkpoint."""
    write_json(folder / "trainer_state.json", {"global_step": step, "max_steps": max_steps, "log_history": []})


def test_resume_point_resumes_unfinished(tmp_path: Path) -> None:
    """An unfinished checkpoint is the resume point."""
    write_state(tmp_path / "checkpoint-200", 200, 500)
    assert sft.resume_point(tmp_path, force=False) == tmp_path / "checkpoint-200"


def test_resume_point_force_clears(tmp_path: Path) -> None:
    """force clears the folder and starts fresh."""
    write_state(tmp_path / "checkpoint-200", 200, 500)
    assert sft.resume_point(tmp_path, force=True) is None
    assert tmp_path.is_dir() and not any(tmp_path.iterdir())


def test_resume_point_finished_run_starts_fresh(tmp_path: Path) -> None:
    """A run that already reached max_steps is cleared and starts fresh."""
    write_state(tmp_path / "checkpoint-500", 500, 500)
    assert sft.resume_point(tmp_path, force=False) is None
    assert latest_checkpoint(tmp_path) is None


def test_planned_steps() -> None:
    """max_steps wins when set; otherwise steps come from epochs."""
    assert sft.planned_steps(100, 2, 4, 2, -1) == 26
    assert sft.planned_steps(100, 2, 4, 2, 10) == 10


def test_vllm_needs_24gb(make_cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM is used only on GPUs with at least 24 GB."""
    monkeypatch.setattr(sft.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 5))
    monkeypatch.setattr(sft, "detect_gpu", lambda: {"available": True, "name": "T4", "vram_gb": 15.0})
    assert not sft.vllm_supported(make_cfg())
    monkeypatch.setattr(sft, "detect_gpu", lambda: {"available": True, "name": "A100", "vram_gb": 40.0})
    assert sft.vllm_supported(make_cfg())


def test_release_student_shuts_down_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """release_student shuts down the vLLM engine only when one is in use."""
    import sys
    import types

    deleted: list[Any] = []
    fake = types.ModuleType("unsloth_zoo.vllm_utils")
    fake.delete_vllm = lambda llm=None: deleted.append(llm)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_utils", fake)
    monkeypatch.setattr(sft, "free_gpu_memory", lambda: None)

    class FakeModel:
        """Stand-in model holding a vLLM engine."""

        vllm_engine = "engine"

    model = FakeModel()
    sft.release_student(StudentBundle(model, None, "sys", use_vllm=True))
    assert deleted == ["engine"] and model.vllm_engine is None
    sft.release_student(StudentBundle(FakeModel(), None, "sys", use_vllm=False))
    assert deleted == ["engine"]


def test_release_student_ignores_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """release_student(None) returns at once without touching GPU memory."""
    freed: list[bool] = []
    monkeypatch.setattr(sft, "free_gpu_memory", lambda: freed.append(True))
    sft.release_student(None)
    assert freed == []


def test_free_gpu_memory_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """free_gpu_memory only collects garbage when torch cannot be imported."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    sft.free_gpu_memory()
