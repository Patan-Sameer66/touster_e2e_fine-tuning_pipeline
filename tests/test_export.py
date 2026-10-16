"""Export cell: adapter selection, Modelfile, model card, load test, Hub push; heavy steps mocked."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import touster.export as export
from touster.config import TousterConfig
from touster.state import (
    GPUUnavailableError,
    RunPaths,
    append_jsonl,
    load_state,
    mark_done,
    update_state,
    write_json,
    write_jsonl,
)

SECRET = "hf_THIS_IS_A_FAKE_SECRET_123"
SYSTEM_PROMPT = "You classify the sentiment of product reviews."
UNSLOTH_STYLE_TEMPLATE = '''
FROM {path}
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """You are Qwen, created by Alibaba Cloud.
You are a helpful assistant."""
'''


@pytest.fixture
def fake_sft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in for touster.train.sft."""
    module = types.ModuleType("touster.train.sft")
    module.system_prompt_for = lambda cfg: SYSTEM_PROMPT
    module.read_loss_history = lambda folder: []
    module.free_gpu_memory = lambda: None
    monkeypatch.setitem(sys.modules, "touster.train.sft", module)


@pytest.fixture
def gpu_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a GPU is present."""
    monkeypatch.setattr(export, "require_gpu", lambda cell: {"available": True, "name": "fake", "vram_gb": 16.0})


def _make_adapter(folder: Path) -> Path:
    """Create a fake adapter folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    return folder


def _eval(stage: str, score: float) -> dict[str, Any]:
    """Fake eval result."""
    return {"stage": stage, "primary_metric": "exact_match", "scores": {"exact_match": score},
            "judge_win_rate_vs_base": 0.6 if stage != "base" else None, "n": 10}


def test_select_prefers_loop_best_adapter(run: tuple[TousterConfig, RunPaths]) -> None:
    """The loop's best adapter wins when it exists."""
    _, paths = run
    _make_adapter(paths.sft_checkpoints / "final")
    best = _make_adapter(paths.dpo_checkpoints(2) / "final")
    state = {"metrics": {"loop": {"best_stage": "dpo_round_2", "best_adapter": str(best)}}}
    choice = export.select_best_adapter(paths, state)
    assert choice == export.AdapterChoice("dpo_round_2", best)


def test_select_falls_back_to_sft(run: tuple[TousterConfig, RunPaths]) -> None:
    """No loop result, or a missing loop adapter, falls back to the SFT final adapter."""
    _, paths = run
    sft = _make_adapter(paths.sft_checkpoints / "final")
    assert export.select_best_adapter(paths, {}) == export.AdapterChoice("sft", sft)
    missing = {"metrics": {"loop": {"best_stage": "dpo_round_1", "best_adapter": str(paths.root / "gone")}}}
    assert export.select_best_adapter(paths, missing).stage == "sft"


def test_select_uses_recorded_rounds_when_loop_missing(run: tuple[TousterConfig, RunPaths]) -> None:
    """A loop stopped on budget has no metrics; the best recorded round with an adapter on disk is used."""
    _, paths = run
    _make_adapter(paths.sft_checkpoints / "final")
    rounds = []
    for n, score in ((1, 0.72), (2, 0.80), (3, 0.90)):
        stage = f"dpo_round_{n}"
        write_json(paths.eval_result(stage), _eval(stage, score))
        rounds.append({"round": n, "stage": stage, "adapter": str(paths.dpo_checkpoints(n) / "final")})
    for n in (1, 2):  # round 3 scored best but its adapter folder is gone
        _make_adapter(paths.dpo_checkpoints(n) / "final")
    choice = export.select_best_adapter(paths, {"metrics": {}, "loop_rounds": rounds})
    assert choice == export.AdapterChoice("dpo_round_2", paths.dpo_checkpoints(2) / "final")


def test_check_disk_space_warns_when_low(run: tuple[TousterConfig, RunPaths], monkeypatch: pytest.MonkeyPatch) -> None:
    """Low free space is logged as a warning."""
    _, paths = run
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(export, "log_event", lambda msg, level="info": events.append((msg, level)))
    monkeypatch.setattr(export.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=2 * 1024**3))
    assert export.check_disk_space(paths) == pytest.approx(2.0)
    assert events[-1][1] == "warning"


def test_select_without_adapter_raises(run: tuple[TousterConfig, RunPaths]) -> None:
    """No adapter at all is a clear error."""
    _, paths = run
    with pytest.raises(export.ExportError, match="Run cell 4"):
        export.select_best_adapter(paths, {})


def test_export_cell_fails_clearly_without_adapter(run: tuple[TousterConfig, RunPaths], gpu_ok: None, fake_sft: None) -> None:
    """The cell stops with the adapter message and is not marked done."""
    cfg, paths = run
    with pytest.raises(export.ExportError, match="No trained adapter"):
        export.run_export_cell(cfg, paths)
    assert "export" not in load_state(paths)["completed"]


def test_export_cell_needs_gpu(run: tuple[TousterConfig, RunPaths], monkeypatch: pytest.MonkeyPatch) -> None:
    """On a CPU machine the cell names itself in the GPU error."""
    cfg, paths = run
    monkeypatch.setattr("touster.state.detect_gpu", lambda: {"available": False, "name": None, "vram_gb": 0.0})
    with pytest.raises(GPUUnavailableError, match="export"):
        export.run_export_cell(cfg, paths)


def test_modelfile_adjusts_unsloth_template() -> None:
    """FROM, SYSTEM and temperature are replaced; chat template and stop tokens kept."""
    text = export.build_modelfile(UNSLOTH_STYLE_TEMPLATE, "./gguf/model.Q4_K_M.gguf", SYSTEM_PROMPT, 0.1)
    assert text.startswith("FROM ./gguf/model.Q4_K_M.gguf\n")
    assert text.count("FROM ") == 1
    assert "<|im_start|>system" in text and 'PARAMETER stop "<|im_end|>"' in text
    assert "PARAMETER min_p 0.1" in text
    assert "temperature 1.5" not in text and "PARAMETER temperature 0.1" in text
    assert "Qwen" not in text and "helpful assistant" not in text
    assert text.rstrip().endswith(f'SYSTEM """{SYSTEM_PROMPT}"""')


def test_modelfile_without_template() -> None:
    """Without an Unsloth template the Modelfile is FROM, temperature and SYSTEM."""
    text = export.build_modelfile(None, "./gguf/m.gguf", 'say """hi"""', 0.7)
    assert text.splitlines() == ["FROM ./gguf/m.gguf", "PARAMETER temperature 0.7", "SYSTEM \"\"\"say '''hi'''\"\"\""]


def _populate(paths: RunPaths) -> None:
    """Fill a run with data the model card reads."""
    write_jsonl(paths.golden_seed, [{"id": f"g-{i}", "prompt": f"review {i}", "response": "positive"} for i in range(3)])
    mark_done(paths, "synthetic", {"funnel": {"raw": 40, "schema": 38, "decontam": 30}})
    update_state(paths, eval_ids=["g-10", "g-11"], elapsed_minutes=42.5)
    write_json(paths.eval_result("base"), _eval("base", 0.5))
    write_json(paths.eval_result("sft"), _eval("sft", 0.7))
    append_jsonl(paths.costs, {"phase": "generation", "cost_usd": 1.25})
    append_jsonl(paths.costs, {"phase": "judge", "cost_usd": 0.25})


def test_model_card_contents_and_no_secrets(run: tuple[TousterConfig, RunPaths], monkeypatch: pytest.MonkeyPatch) -> None:
    """The card has data, settings, evals, cost and the licence note, and never the HF token."""
    cfg, paths = run
    monkeypatch.setenv("HF_TOKEN", SECRET)
    _populate(paths)
    card = export.build_model_card(cfg, paths, load_state(paths), {"final_stage": "sft", "load_test": {"method": "skipped"}})
    assert SECRET not in card
    assert "terms of use" in card and cfg.teacher_model in card
    assert cfg.student_model in card
    assert "Golden seed records: 3" in card and "Golden eval records (held out, never trained on): 2" in card
    assert "raw 40 -> schema 38 -> decontam 30" in card
    assert "| sft | exact_match | 0.7000 | 0.6000 | 10 |" in card
    assert "$1.5000" in card and "42.5 minutes" in card
    assert f"r={cfg.lora_r}" in card


def test_model_card_with_no_results(run: tuple[TousterConfig, RunPaths]) -> None:
    """An almost empty run still produces a card."""
    cfg, paths = run
    card = export.build_model_card(cfg, paths, {}, {})
    assert "No evaluation results." in card and "Synthetic data: not run" in card


def test_soup_candidates_rank_by_score(run: tuple[TousterConfig, RunPaths]) -> None:
    """Top stages with adapters on disk, best first, capped at three."""
    _, paths = run
    _make_adapter(paths.sft_checkpoints / "final")
    for n in (1, 2, 3):
        _make_adapter(paths.dpo_checkpoints(n) / "final")
    evals = {"base": _eval("base", 0.9), "sft": _eval("sft", 0.6), "dpo_round_1": _eval("dpo_round_1", 0.7),
             "dpo_round_2": _eval("dpo_round_2", 0.8), "dpo_round_3": _eval("dpo_round_3", 0.65)}
    ranked = export.soup_candidates(paths, evals, {})
    assert [stage for stage, _, _ in ranked] == ["dpo_round_2", "dpo_round_1", "dpo_round_3"]


def test_soup_merge_config_is_valid_for_mergekit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The linear merge config parses with mergekit's own schema, even with '@' in a path."""
    config = export.soup_merge_config([tmp_path / "a@b", tmp_path / "c"])
    assert config["merge_method"] == "linear" and config["parameters"] == {"normalize": True}
    if importlib.util.find_spec("immutables") is None:  # mergekit is installed --no-deps here
        stub = types.ModuleType("immutables")
        stub.Map = dict
        monkeypatch.setitem(sys.modules, "immutables", stub)
    mergekit_config = pytest.importorskip("mergekit.config")
    parsed = mergekit_config.MergeConfiguration.model_validate(config)
    assert parsed.models[0].model.model.path == str(tmp_path / "a@b")


def test_find_llama_cli_from_unsloth_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A llama.cpp build in UNSLOTH_LLAMA_CPP_PATH is found."""
    suffix = ".exe" if sys.platform == "win32" else ""
    cli = tmp_path / "build" / "bin" / f"llama-cli{suffix}"
    cli.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    assert export.find_llama_cli() == cli


def test_load_test_skipped_without_tools(run: tuple[TousterConfig, RunPaths], monkeypatch: pytest.MonkeyPatch) -> None:
    """No ollama and no llama.cpp: skipped with manual instructions."""
    cfg, paths = run
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    monkeypatch.setattr(export, "find_llama_cli", lambda: None)
    result = export.gguf_load_test(cfg, paths.modelfile, paths.gguf / "m.gguf", "hi")
    assert result["method"] == "skipped" and result["passed"] is None
    assert "ollama create touster-sentiment" in result["detail"]


def test_load_test_with_ollama(run: tuple[TousterConfig, RunPaths], monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama create then run; passes when the model answers."""
    cfg, paths = run
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        """Record the command and answer like ollama."""
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="positive" if cmd[1] == "run" else "", stderr="")

    monkeypatch.setattr(export.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    monkeypatch.setattr(export.subprocess, "run", fake_run)
    result = export.gguf_load_test(cfg, paths.modelfile, paths.gguf / "m.gguf", "great product")
    assert result == {"method": "ollama", "passed": True, "detail": "positive"}
    assert calls[0][:3] == ["ollama", "create", "touster-sentiment"] and calls[1][:2] == ["ollama", "run"]


def test_load_test_llama_cli_timeout_fails(run: tuple[TousterConfig, RunPaths], monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung llama.cpp run is recorded as a failure, not an exception."""
    cfg, paths = run

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        """Simulate a timeout."""
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    monkeypatch.setattr(export, "find_llama_cli", lambda: Path("/opt/llama.cpp/llama-cli"))
    monkeypatch.setattr(export.subprocess, "run", fake_run)
    result = export.gguf_load_test(cfg, paths.modelfile, paths.gguf / "m.gguf", "hi")
    assert result["method"] == "llama.cpp" and result["passed"] is False and "timed out" in result["detail"]


class FakeHfApi:
    """Records Hub calls instead of making them."""

    calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, token: str | None = None) -> None:
        """Keep the token like the real client."""
        self.token = token

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        """Record create_repo."""
        FakeHfApi.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def upload_folder(self, **kwargs: Any) -> None:
        """Record upload_folder."""
        FakeHfApi.calls.append(("upload_folder", kwargs))

    def upload_file(self, **kwargs: Any) -> None:
        """Record upload_file."""
        FakeHfApi.calls.append(("upload_file", kwargs))


def _mock_heavy_steps(monkeypatch: pytest.MonkeyPatch, paths: RunPaths) -> None:
    """Replace merge, GGUF conversion and the load test with fakes."""

    def fake_merge(cfg: TousterConfig, adapter: Path, out_dir: Path) -> Path:
        """Write a fake merged model."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.safetensors").write_text("weights", encoding="utf-8")
        return out_dir

    def fake_gguf(cfg: TousterConfig, model_dir: Path, run_paths: RunPaths) -> tuple[Path, str | None]:
        """Write a fake GGUF file."""
        gguf = run_paths.gguf / "merged.Q4_K_M.gguf"
        gguf.write_text("gguf", encoding="utf-8")
        return gguf, UNSLOTH_STYLE_TEMPLATE

    monkeypatch.setattr(export, "merge_adapter", fake_merge)
    monkeypatch.setattr(export, "export_gguf", fake_gguf)
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    monkeypatch.setattr(export, "find_llama_cli", lambda: None)


def test_export_cell_end_to_end_mocked(make_cfg: Any, gpu_ok: None, fake_sft: None, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    """Full cell with heavy steps mocked: files written, metrics saved, Hub calls made, token never printed."""
    from touster.state import create_or_resume_run

    cfg = make_cfg(push_to_hub=True, hub_repo="someone/touster-sentiment")
    paths, _, _ = create_or_resume_run(cfg)
    _populate(paths)
    _make_adapter(paths.sft_checkpoints / "final")
    _mock_heavy_steps(monkeypatch, paths)
    monkeypatch.setenv("HF_TOKEN", SECRET)
    FakeHfApi.calls = []
    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    metrics = export.run_export_cell(cfg, paths)

    assert metrics["final_stage"] == "sft" and metrics["load_test"]["method"] == "skipped"
    assert metrics["hub"] == {"pushed": True, "repo": "someone/touster-sentiment"}
    assert paths.modelfile.read_text(encoding="utf-8").startswith("FROM ./gguf/merged.Q4_K_M.gguf")
    assert "terms of use" in paths.model_card.read_text(encoding="utf-8")
    assert FakeHfApi.calls[0][1]["private"] is True
    assert [c[0] for c in FakeHfApi.calls] == ["create_repo", "upload_folder", "upload_file", "upload_file", "upload_folder"]
    assert "export" in load_state(paths)["completed"]
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert SECRET not in paths.state.read_text(encoding="utf-8")


def test_ensure_mergekit_install_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed pip install becomes a clear ExportError."""

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        """Fail like pip does."""
        assert "--no-deps" in cmd and export.MERGEKIT_PIN in cmd
        raise subprocess.CalledProcessError(1, cmd, stderr="No matching distribution")

    monkeypatch.setattr(export.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(export.subprocess, "run", fake_run)
    with pytest.raises(export.ExportError, match="model_soup=false"):
        export.ensure_mergekit()


def test_push_network_error_does_not_abort(make_cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error during upload is logged with secrets redacted and recorded, not raised."""
    from touster.state import create_or_resume_run

    cfg = make_cfg(push_to_hub=True, hub_repo="someone/repo")
    paths, _, _ = create_or_resume_run(cfg)
    monkeypatch.setenv("HF_TOKEN", SECRET)
    events: list[str] = []
    monkeypatch.setattr(export, "log_event", lambda msg, level="info": events.append(msg))

    class BrokenApi(FakeHfApi):
        """Fails like a dropped connection."""

        def upload_folder(self, **kwargs: Any) -> None:
            """Raise a network error that quotes the token."""
            raise ConnectionError(f"connection reset (token {SECRET})")

    monkeypatch.setattr("huggingface_hub.HfApi", BrokenApi)
    assert export.push_to_hub(cfg, paths) == {"pushed": False, "reason": "ConnectionError"}
    assert not any(SECRET in e for e in events)


def test_push_without_token_is_skipped(make_cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """No token anywhere: the push is skipped with a reason instead of crashing."""
    from touster.state import create_or_resume_run

    cfg = make_cfg(push_to_hub=True, hub_repo="someone/repo")
    paths, _, _ = create_or_resume_run(cfg)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("touster.llm._colab_secret", lambda name: None)
    assert export.push_to_hub(cfg, paths) == {"pushed": False, "reason": "HF_TOKEN missing"}
