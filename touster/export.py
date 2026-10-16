"""Cell 7: merge the best adapter, optional model soup, GGUF + Ollama Modelfile, model card, Hub push, demo."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from touster.budget import spend_by_phase
from touster.config import TaskType, TousterConfig
from touster.dashboard import log_event
from touster.llm import redact_secrets
from touster.report import load_eval_results, primary_score, read_state
from touster.state import RunPaths, iter_jsonl, require_gpu, run_cell

MERGEKIT_PIN = "mergekit==0.1.4"
LOW_DISK_GB = 10.0
MERGEKIT_EXTRA_DEPS = ("immutables==0.21",)
MAX_SOUP_MODELS = 3
MIN_SOUP_MODELS = 2
PIP_TIMEOUT_SECONDS = 600
MERGE_TIMEOUT_SECONDS = 3600
OLLAMA_TIMEOUT_SECONDS = 600
LLAMA_CLI_TIMEOUT_SECONDS = 300
LOAD_TEST_MAX_TOKENS = 64
DEFAULT_TEST_PROMPT = "Say hello in one short sentence."
TEMPERATURE_BY_TASK = {
    TaskType.CLASSIFICATION: 0.1,
    TaskType.EXTRACTION: 0.1,
    TaskType.CODE: 0.2,
    TaskType.QA: 0.3,
    TaskType.SUMMARISATION: 0.5,
    TaskType.OTHER: 0.7,
}
LLAMA_CLI_NAMES = ("llama-completion", "llama-cli")
LLAMA_CPP_SUBDIRS = ("", "build/bin", "build/bin/Release")
LICENCE_NOTE = (
    "The synthetic training data was written by the teacher model `{teacher}`. Before you publish this model "
    "or use it commercially, check the teacher provider's terms of use: many providers restrict using their "
    "outputs to train other models. The licence of the base model `{student}` also applies."
)


class ExportError(RuntimeError):
    """Raised when the export cell cannot continue; the message says what to do."""


@dataclass(frozen=True)
class AdapterChoice:
    """The adapter picked for export and the stage it came from."""

    stage: str
    adapter: Path


def _best_recorded_round(paths: RunPaths, rounds: list[Any]) -> AdapterChoice | None:
    """Highest scoring loop round from state['loop_rounds'] whose eval file and adapter folder exist."""
    best: tuple[float, AdapterChoice] | None = None
    for entry in rounds:
        if not isinstance(entry, dict) or not entry.get("stage") or not entry.get("adapter"):
            continue
        stage, adapter = str(entry["stage"]), Path(entry["adapter"])
        result_path = paths.eval_result(stage)
        if not adapter.exists() or not result_path.is_file():
            continue
        score = primary_score(load_eval_results(paths).get(stage) or {})
        if score is not None and (best is None or score > best[0]):
            best = (score, AdapterChoice(stage, adapter))
    return best[1] if best else None


def select_best_adapter(paths: RunPaths, state: dict[str, Any]) -> AdapterChoice:
    """Pick the loop's best adapter, else the best round recorded in state, else the SFT final adapter."""
    loop = (state.get("metrics") or {}).get("loop") or {}
    best = loop.get("best_adapter")
    if best:
        best_path = Path(best)
        if best_path.exists():
            return AdapterChoice(str(loop.get("best_stage") or "loop"), best_path)
        log_event(f"Best adapter {best_path} from the loop cell is missing; falling back", "warning")
    if not loop:
        recorded = _best_recorded_round(paths, state.get("loop_rounds") or [])
        if recorded is not None:
            log_event(f"Loop cell did not finish; using its best recorded round {recorded.stage}", "warning")
            return recorded
    sft_final = paths.sft_checkpoints / "final"
    if sft_final.exists():
        return AdapterChoice("sft", sft_final)
    raise ExportError(
        f"No trained adapter found (looked for metrics['loop']['best_adapter'], state['loop_rounds'] and {sft_final}). "
        "Run cell 4 (SFT) before cell 7."
    )


def check_disk_space(paths: RunPaths) -> float:
    """Log free disk space at the run dir; warn under LOW_DISK_GB. Returns free GB."""
    free_gb = shutil.disk_usage(paths.root).free / 1024**3
    if free_gb < LOW_DISK_GB:
        log_event(f"Only {free_gb:.1f} GB free at {paths.root}; merge + GGUF may need about {LOW_DISK_GB:.0f} GB", "warning")
    else:
        log_event(f"{free_gb:.1f} GB free at {paths.root}")
    return free_gb


def merge_adapter(cfg: TousterConfig, adapter: Path, out_dir: Path) -> Path:
    """Merge a LoRA adapter into 16-bit base weights with Unsloth and save to out_dir."""
    from touster.train.sft import load_student, release_student

    log_event(f"Merging {adapter} into the base weights -> {out_dir}")
    # for_training=True loads a plain PeftModel; the inference path may start vLLM, which only wastes VRAM here.
    bundle = load_student(cfg, adapter_path=adapter, for_training=True)
    bundle.model.save_pretrained_merged(str(out_dir), bundle.tokenizer, save_method="merged_16bit")
    release_student(bundle)
    return out_dir


def _adapter_for_stage(paths: RunPaths, stage: str, loop: dict[str, Any]) -> Path | None:
    """Find the adapter folder that produced an eval stage, or None."""
    for round_info in loop.get("rounds") or []:
        if isinstance(round_info, dict) and round_info.get("stage") == stage and round_info.get("adapter"):
            found = Path(round_info["adapter"])
            return found if found.exists() else None
    if stage == "sft":
        candidate = paths.sft_checkpoints / "final"
    elif stage.startswith("dpo_round_") and stage.removeprefix("dpo_round_").isdigit():
        candidate = paths.dpo_checkpoints(int(stage.removeprefix("dpo_round_"))) / "final"
    else:
        return None
    return candidate if candidate.exists() else None


def soup_candidates(paths: RunPaths, evals: dict[str, dict], loop: dict[str, Any]) -> list[tuple[str, Path, float]]:
    """Top sft/dpo stages by primary score that still have an adapter on disk."""
    ranked = []
    for stage, result in evals.items():
        if stage != "sft" and not stage.startswith("dpo_round_"):
            continue
        score = primary_score(result)
        adapter = _adapter_for_stage(paths, stage, loop)
        if score is not None and adapter is not None:
            ranked.append((stage, adapter, score))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked[:MAX_SOUP_MODELS]


def ensure_mergekit() -> None:
    """Install mergekit on demand, or raise ExportError if that fails."""
    if importlib.util.find_spec("mergekit") is not None and importlib.util.find_spec("immutables") is not None:
        return
    # --no-deps on purpose: mergekit 0.1.4 pins pydantic~=2.10, and letting pip resolve that would downgrade
    # pydantic under vLLM (which needs a newer one) and break generation. Its other runtime deps ship with
    # Colab except `immutables`, which has no dependencies of its own, so it is installed alongside.
    log_event(f"Installing {MERGEKIT_PIN} (--no-deps) for the model soup")
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", MERGEKIT_PIN, *MERGEKIT_EXTRA_DEPS]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=PIP_TIMEOUT_SECONDS)
    except subprocess.CalledProcessError as err:
        raise ExportError(
            f"Could not install {MERGEKIT_PIN}: {err.stderr.strip()[-500:]}. "
            "Install it by hand or set model_soup=false."
        ) from None
    except subprocess.TimeoutExpired:
        raise ExportError(f"Installing {MERGEKIT_PIN} timed out; retry or set model_soup=false.") from None


def soup_merge_config(model_dirs: list[Path]) -> dict[str, Any]:
    """mergekit config for an equal-weight linear merge (local paths given as dicts so '@' and '+' are safe)."""
    return {
        "merge_method": "linear",
        "models": [{"model": {"model": {"path": str(d)}}, "parameters": {"weight": 1.0}} for d in model_dirs],
        "parameters": {"normalize": True},
        "dtype": "float16",
    }


def run_mergekit(cfg: TousterConfig, model_dirs: list[Path], out_dir: Path) -> Path:
    """Average merged checkpoints with mergekit's linear merge in a subprocess."""
    config_path = out_dir.parent / "soup_mergekit.yml"
    config_path.write_text(yaml.safe_dump(soup_merge_config(model_dirs)), encoding="utf-8")
    cmd = [sys.executable, "-m", "mergekit.scripts.run_yaml", str(config_path), str(out_dir),
           "--cuda", "--random-seed", str(cfg.seed)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=MERGE_TIMEOUT_SECONDS)
    except subprocess.CalledProcessError as err:
        raise ExportError(f"mergekit linear merge failed: {err.stderr.strip()[-800:]}. Set model_soup=false to skip it.") from None
    except subprocess.TimeoutExpired:
        raise ExportError("mergekit linear merge timed out. Set model_soup=false to skip it.") from None
    return out_dir


def _base_answers(evals: dict[str, dict]) -> list[str] | None:
    """Base-model answers from eval/base.json, for the judge win rate."""
    answers = (evals.get("base") or {}).get("answers")
    return [str(a.get("answer", "")) for a in answers] if answers else None


def _make_client(cfg: TousterConfig, paths: RunPaths) -> Any:
    """Teacher/judge client with a budget tracker that resumes from the run's spend."""
    from touster.budget import BudgetTracker
    from touster.llm import LLMClient

    prior = float(read_state(paths).get("elapsed_minutes", 0.0))
    return LLMClient(cfg, paths.costs, BudgetTracker(cfg, paths.costs, prior_minutes=prior))


def evaluate_model_dir(cfg: TousterConfig, paths: RunPaths, model_dir: Path, stage: str, evals: dict[str, dict]) -> float | None:
    """Load a merged model folder and score it with touster.eval.metrics.evaluate."""
    from unsloth import FastLanguageModel

    from touster.eval.metrics import evaluate
    from touster.train.sft import StudentBundle, release_student, system_prompt_for

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir), max_seq_length=cfg.max_seq_len, load_in_4bit=False
    )
    FastLanguageModel.for_inference(model)
    bundle = StudentBundle(model=model, tokenizer=tokenizer, system_prompt=system_prompt_for(cfg))
    del model
    try:
        result = evaluate(bundle, stage, cfg, paths, _make_client(cfg, paths), base_answers=_base_answers(evals), use_adapter=False)
    finally:
        release_student(bundle)
    return primary_score(result)


def _merge_parts(cfg: TousterConfig, paths: RunPaths, best: AdapterChoice, parts: list[tuple[str, Path, float]]) -> list[Path]:
    """Merged folder for each soup part, reusing paths.merged for the best stage."""
    folders = []
    for stage, adapter, _ in parts:
        if adapter.resolve() == best.adapter.resolve():
            folders.append(paths.merged)
        else:
            folders.append(merge_adapter(cfg, adapter, paths.export / "soup_parts" / stage))
    return folders


def _remove_soup_intermediates(paths: RunPaths) -> None:
    """Delete the per-stage merges, the soup folder and its mergekit config once the soup decision is made."""
    for folder in (paths.export / "soup_parts", paths.export / "soup"):
        if folder.is_dir():
            shutil.rmtree(folder)
    (paths.export / "soup_mergekit.yml").unlink(missing_ok=True)


def make_soup(cfg: TousterConfig, paths: RunPaths, best: AdapterChoice, evals: dict[str, dict], loop: dict[str, Any]) -> dict[str, Any]:
    """Average the top checkpoints, keep the soup in paths.merged only if it beats the best single model."""
    best_score = primary_score(evals.get(best.stage) or {})
    parts = soup_candidates(paths, evals, loop)
    if best_score is None or len(parts) < MIN_SOUP_MODELS:
        log_event("Model soup skipped: need a scored best stage and at least two scored checkpoints", "warning")
        return {"kept": False, "skipped": True, "stages": [p[0] for p in parts]}
    ensure_mergekit()
    soup_dir = run_mergekit(cfg, _merge_parts(cfg, paths, best, parts), paths.export / "soup")
    soup_score = evaluate_model_dir(cfg, paths, soup_dir, "soup", evals)
    kept = soup_score is not None and soup_score > best_score
    if kept:
        shutil.rmtree(paths.merged)
        shutil.copytree(soup_dir, paths.merged)
    _remove_soup_intermediates(paths)
    verdict = "kept" if kept else "dropped, best single model kept"
    log_event(f"Model soup scored {soup_score} vs best {best_score:.4f}: {verdict}", "success" if kept else "info")
    return {"kept": kept, "skipped": False, "stages": [p[0] for p in parts], "score": soup_score, "best_score": best_score}


def _collect_gguf(files: list[Path], target: Path) -> list[Path]:
    """Move GGUF files from Unsloth's sibling output folder into paths.gguf."""
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for file in files:
        destination = target / file.name
        if file.resolve() != destination.resolve():
            destination.unlink(missing_ok=True)
            shutil.move(str(file), str(destination))
        moved.append(destination)
    return moved


def _unsloth_modelfile(result: dict[str, Any], tokenizer: Any, cfg: TousterConfig, gguf_name: str) -> str | None:
    """Unsloth's Modelfile for this model family, if it wrote one or knows a template."""
    location = result.get("modelfile_location")
    if location and Path(location).is_file():
        return Path(location).read_text(encoding="utf-8")
    from unsloth.save import create_ollama_modelfile

    return create_ollama_modelfile(tokenizer, cfg.student_model, gguf_name)


def export_gguf(cfg: TousterConfig, model_dir: Path, paths: RunPaths) -> tuple[Path, str | None]:
    """Convert a merged folder to GGUF at cfg.gguf_quant; return the file and Unsloth's Modelfile text."""
    from unsloth import FastLanguageModel

    from touster.train.sft import free_gpu_memory

    log_event(f"Exporting GGUF ({cfg.gguf_quant}) from {model_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir), max_seq_length=cfg.max_seq_len, load_in_4bit=False
    )
    result = model.save_pretrained_gguf(
        str(model_dir), tokenizer, quantization_method=cfg.gguf_quant.lower(), merge_is_disposable=False
    )
    files = [Path(f) for f in result.get("gguf_files") or []]
    if not files:
        raise ExportError("Unsloth returned no GGUF file. Check the llama.cpp build output above.")
    moved = _collect_gguf(files, paths.gguf)
    template = _unsloth_modelfile(result, tokenizer, cfg, moved[0].name)
    sibling = Path(result.get("gguf_directory") or "")
    if sibling.is_dir() and sibling.resolve() != paths.gguf.resolve():
        shutil.rmtree(sibling)
    del model, tokenizer
    free_gpu_memory()
    return moved[0], template


def _strip_modelfile(text: str) -> list[str]:
    """Drop FROM, SYSTEM and temperature lines (including multi-line SYSTEM blocks) from a Modelfile."""
    kept: list[str] = []
    in_block = skipping = False
    for line in text.strip().splitlines():
        if in_block:
            if not skipping:
                kept.append(line)
            if '"""' in line:
                in_block = skipping = False
            continue
        head = line.strip().upper()
        drop = head.startswith(("FROM ", "SYSTEM ", "PARAMETER TEMPERATURE"))
        if line.count('"""') == 1:
            in_block, skipping = True, drop
        if not drop:
            kept.append(line)
    return kept


def build_modelfile(template: str | None, gguf_ref: str, system_prompt: str, temperature: float) -> str:
    """Ollama Modelfile: FROM the GGUF, Unsloth's chat template if known, our SYSTEM prompt and temperature."""
    body = _strip_modelfile(template) if template else []
    system = system_prompt.replace('"""', "'''")
    lines = [f"FROM {gguf_ref}", *body, f"PARAMETER temperature {temperature}", f'SYSTEM """{system}"""']
    return "\n".join(lines) + "\n"


def temperature_for(cfg: TousterConfig) -> float:
    """Sampling temperature that suits the task type."""
    return TEMPERATURE_BY_TASK.get(cfg.task_type, 0.7)


def _gguf_ref(gguf_file: Path, modelfile: Path) -> str:
    """Path of the GGUF relative to the Modelfile, as Ollama resolves it."""
    return "./" + Path(os.path.relpath(gguf_file, modelfile.parent)).as_posix()


def find_llama_cli() -> Path | None:
    """Locate a llama.cpp CLI built by Unsloth (or on PATH)."""
    suffix = ".exe" if sys.platform == "win32" else ""
    roots = [Path(p) for p in (os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),) if p]
    roots += [Path.home() / ".unsloth" / "llama.cpp", Path("llama.cpp")]
    for name in LLAMA_CLI_NAMES:
        for root in roots:
            for sub in LLAMA_CPP_SUBDIRS:
                candidate = root / sub / f"{name}{suffix}"
                if candidate.is_file():
                    return candidate
        on_path = shutil.which(name)
        if on_path:
            return Path(on_path)
    return None


def _run(cmd: list[str], timeout: int, cwd: Path | None = None) -> tuple[bool, str]:
    """Run a command with a timeout and no stdin; return (exit code was 0, stdout tail or error)."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except OSError as err:
        return False, f"could not start {cmd[0]}: {err}"
    if done.returncode != 0:
        return False, f"exit code {done.returncode}: {(done.stderr or '').strip()[-400:]}"
    return True, (done.stdout or "").strip()[-400:]


def _answer_check(cmd: list[str], timeout: int) -> tuple[bool, str]:
    """Run a generation command; it passes only if it exits cleanly and prints an answer."""
    ok, output = _run(cmd, timeout)
    if ok and not output:
        return False, "the model produced no output"
    return ok, output


def ollama_name(cfg: TousterConfig) -> str:
    """Ollama model name for this task."""
    return f"touster-{cfg.task_name}".lower()


def gguf_load_test(cfg: TousterConfig, modelfile: Path, gguf_file: Path, prompt: str) -> dict[str, Any]:
    """Load the GGUF in Ollama or llama.cpp and answer one prompt; skip with a warning if neither exists."""
    if shutil.which("ollama"):
        name = ollama_name(cfg)
        ok, detail = _run(["ollama", "create", name, "-f", str(modelfile)], OLLAMA_TIMEOUT_SECONDS, cwd=modelfile.parent)
        if ok:
            ok, detail = _answer_check(["ollama", "run", name, prompt], OLLAMA_TIMEOUT_SECONDS)
        result = {"method": "ollama", "passed": ok, "detail": detail}
    elif (cli := find_llama_cli()) is not None:
        cmd = [str(cli), "-m", str(gguf_file), "-p", prompt, "-n", str(LOAD_TEST_MAX_TOKENS), "--temp", "0"]
        if cli.stem == "llama-completion":
            cmd.append("-no-cnv")
        ok, detail = _answer_check(cmd, LLAMA_CLI_TIMEOUT_SECONDS)
        result = {"method": "llama.cpp", "passed": ok, "detail": detail}
    else:
        detail = (f"No ollama or llama.cpp CLI found. Test by hand: ollama create {ollama_name(cfg)} -f {modelfile} "
                  f"&& ollama run {ollama_name(cfg)} \"{prompt}\"")
        log_event(f"GGUF load test skipped. {detail}", "warning")
        return {"method": "skipped", "passed": None, "detail": detail}
    log_event(f"GGUF load test ({result['method']}): {'passed' if result['passed'] else 'FAILED'}",
              "success" if result["passed"] else "error")
    return result


def _test_prompt(paths: RunPaths) -> str:
    """First seed prompt as the load-test prompt, else a generic one."""
    if paths.golden_seed.is_file():
        for record in iter_jsonl(paths.golden_seed):
            if str(record.get("prompt", "")).strip():
                return str(record["prompt"])
    return DEFAULT_TEST_PROMPT


def _count_lines(path: Path) -> int | None:
    """Number of non-empty lines in a JSONL file, or None if it is missing."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _card_data_section(paths: RunPaths, state: dict[str, Any]) -> list[str]:
    """Model card lines describing the golden set and synthetic funnel."""
    seed = _count_lines(paths.golden_seed)
    lines = ["## Data", "", f"- Golden seed records: {seed if seed is not None else 'n/a'}",
             f"- Golden eval records (held out, never trained on): {len(state.get('eval_ids') or [])}"]
    funnel = ((state.get("metrics") or {}).get("synthetic") or {}).get("funnel") or {}
    if funnel:
        lines.append("- Synthetic data funnel: " + " -> ".join(f"{k} {v}" for k, v in funnel.items()))
    else:
        lines.append("- Synthetic data: not run")
    return lines + [""]


def _card_training_section(cfg: TousterConfig) -> list[str]:
    """Model card lines with the training settings."""
    loop = f"{cfg.loop_method.value}, up to {cfg.max_rounds} rounds, beta {cfg.dpo_beta}" if cfg.self_improve else "off"
    return [
        "## Training", "",
        f"- Method: QLoRA SFT (r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout})",
        f"- Epochs: {cfg.epochs}, learning rate: {cfg.learning_rate}, batch size: {cfg.batch_size} x grad accum {cfg.grad_accum}",
        f"- Max sequence length: {cfg.max_seq_len}, seed: {cfg.seed}",
        f"- Self-improvement loop: {loop}", "",
    ]


def _fmt(value: Any) -> str:
    """Format a number for a Markdown table."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _card_eval_section(evals: dict[str, dict]) -> list[str]:
    """Model card eval table: one row per stage."""
    if not evals:
        return ["## Evaluation", "", "No evaluation results.", ""]
    lines = ["## Evaluation", "", "| stage | metric | score | judge win rate vs base | n |", "|---|---|---|---|---|"]
    for stage, result in evals.items():
        lines.append(f"| {stage} | {result.get('primary_metric', 'n/a')} | {_fmt(primary_score(result))} | "
                     f"{_fmt(result.get('judge_win_rate_vs_base'))} | {result.get('n', 'n/a')} |")
    return lines + [""]


def _card_cost_section(paths: RunPaths, state: dict[str, Any]) -> list[str]:
    """Model card lines with spend per phase and total time."""
    spend = spend_by_phase(paths.costs)
    lines = ["## Cost and time", "", f"- Total teacher/judge cost: ${sum(spend.values()):.4f}"]
    lines += [f"  - {phase}: ${usd:.4f}" for phase, usd in sorted(spend.items())]
    lines.append(f"- Total time: {float(state.get('elapsed_minutes', 0.0)):.1f} minutes")
    return lines + [""]


def build_model_card(cfg: TousterConfig, paths: RunPaths, state: dict[str, Any], export: dict[str, Any]) -> str:
    """MODEL_CARD.md text; built only from config, state and result files, never from env or secrets."""
    lines = [
        f"# {ollama_name(cfg)}", "",
        f"Fine-tuned with touster for the task `{cfg.task_name}` ({cfg.task_type.value}).", "",
        f"> {cfg.task_description}", "",
        "## Models", "",
        f"- Base (student) model: `{cfg.student_model}`",
        f"- Teacher model: `{cfg.teacher_model}`", f"- Judge model: `{cfg.effective_judge_model}`",
        f"- Exported stage: `{export.get('final_stage', 'n/a')}`, GGUF quant: `{cfg.gguf_quant}`", "",
    ]
    lines += _card_data_section(paths, state) + _card_training_section(cfg)
    lines += _card_eval_section(load_eval_results(paths)) + _card_cost_section(paths, state)
    load_test = export.get("load_test") or {}
    lines += ["## Usage", "", f"    ollama create {ollama_name(cfg)} -f Modelfile", f"    ollama run {ollama_name(cfg)}",
              "", f"GGUF load test: {load_test.get('method', 'not run')}, passed: {load_test.get('passed', 'n/a')}", ""]
    lines += ["## Licence note", "", LICENCE_NOTE.format(teacher=cfg.teacher_model, student=cfg.student_model), ""]
    return "\n".join(lines)


def _hf_token() -> str | None:
    """HF token from env, or from Colab Secrets via load_api_keys; never printed."""
    if not os.environ.get("HF_TOKEN"):
        from touster.llm import load_api_keys

        load_api_keys([], extra=("HF_TOKEN",))
    return os.environ.get("HF_TOKEN") or None


def push_to_hub(cfg: TousterConfig, paths: RunPaths) -> dict[str, Any]:
    """Create the Hub repo and upload merged weights, model card and GGUF."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    token = _hf_token()
    if token is None:
        log_event("push_to_hub is on but HF_TOKEN is not set (env or Colab Secrets); upload skipped", "error")
        return {"pushed": False, "reason": "HF_TOKEN missing"}
    api = HfApi(token=token)
    try:
        api.create_repo(cfg.hub_repo, exist_ok=True, private=True)
        log_event(f"Hub repo {cfg.hub_repo} is private; make it public in its Settings page on the Hub when ready")
        api.upload_folder(repo_id=cfg.hub_repo, folder_path=paths.merged, commit_message="touster: merged weights")
        api.upload_file(path_or_fileobj=paths.model_card, path_in_repo="README.md", repo_id=cfg.hub_repo)
        api.upload_file(path_or_fileobj=paths.modelfile, path_in_repo="Modelfile", repo_id=cfg.hub_repo)
        if any(paths.gguf.glob("*.gguf")):
            api.upload_folder(repo_id=cfg.hub_repo, folder_path=paths.gguf, path_in_repo="gguf",
                              allow_patterns="*.gguf", commit_message="touster: GGUF")
    except (HfHubHTTPError, OSError, ValueError) as err:
        log_event(f"Hub upload to {cfg.hub_repo} failed: {redact_secrets(str(err))}. "
                  "Check the network and that the token has write access; the export is saved locally.", "error")
        return {"pushed": False, "reason": type(err).__name__}
    log_event(f"Uploaded to https://huggingface.co/{cfg.hub_repo}", "success")
    return {"pushed": True, "repo": cfg.hub_repo}


def _export_work(cfg: TousterConfig, paths: RunPaths) -> dict[str, Any]:
    """Everything cell 7 does, in order."""
    from touster.train.sft import free_gpu_memory, system_prompt_for

    require_gpu("export")
    state = read_state(paths)
    metrics = state.get("metrics") or {}
    best = select_best_adapter(paths, state)
    evals = load_eval_results(paths)
    free_gpu_memory()
    check_disk_space(paths)
    merge_adapter(cfg, best.adapter, paths.merged)
    soup = make_soup(cfg, paths, best, evals, metrics.get("loop") or {}) if cfg.model_soup else None
    free_gpu_memory()
    check_disk_space(paths)
    gguf_file, template = export_gguf(cfg, paths.merged, paths)
    modelfile = build_modelfile(template, _gguf_ref(gguf_file, paths.modelfile), system_prompt_for(cfg), temperature_for(cfg))
    paths.modelfile.write_text(modelfile, encoding="utf-8")
    result: dict[str, Any] = {
        "best_stage": best.stage, "adapter": str(best.adapter), "soup": soup,
        "final_stage": "soup" if soup and soup.get("kept") else best.stage,
        "merged": str(paths.merged), "gguf": str(gguf_file), "gguf_quant": cfg.gguf_quant,
        "modelfile": str(paths.modelfile), "model_card": str(paths.model_card),
        "load_test": gguf_load_test(cfg, paths.modelfile, gguf_file, _test_prompt(paths)),
    }
    paths.model_card.write_text(build_model_card(cfg, paths, state, result), encoding="utf-8")
    if cfg.push_to_hub and cfg.smoke_test:
        log_event("Smoke test: Hub upload skipped", "info")
    elif cfg.push_to_hub:
        result["hub"] = push_to_hub(cfg, paths)
    return result


def _summary(metrics: dict[str, Any]) -> str:
    """One-line summary for the export cell."""
    if not metrics:
        return "nothing exported."
    load = (metrics.get("load_test") or {})
    return (f"exported stage {metrics.get('final_stage')} as {Path(str(metrics.get('gguf'))).name}; "
            f"load test {load.get('method')}: {load.get('passed')}")


def run_export_cell(cfg: TousterConfig, paths: RunPaths, force: bool = False) -> dict[str, Any]:
    """Cell 7: merge, optional soup, GGUF, Modelfile, load test, model card, optional Hub push."""
    return run_cell(paths, "export", lambda: _export_work(cfg, paths), _summary, force)


def _is_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules


def launch_demo(cfg: TousterConfig, paths: RunPaths) -> Any:
    """Gradio side-by-side demo: base vs fine-tuned, one 4-bit model with the adapter toggled."""
    import gradio as gr

    from touster.train.sft import generate_responses, load_student

    best = select_best_adapter(paths, read_state(paths))
    bundle = load_student(cfg, adapter_path=best.adapter, for_training=False)

    def answer(prompt: str) -> tuple[str, str]:
        """Return (base answer, fine-tuned answer) for one prompt."""
        if not prompt.strip():
            return "", ""
        with bundle.model.disable_adapter():
            base = generate_responses(bundle, [prompt], cfg)[0][0]
        tuned = generate_responses(bundle, [prompt], cfg)[0][0]
        return base, tuned

    with gr.Blocks(title=f"touster: {cfg.task_name}") as demo:
        gr.Markdown(f"### {cfg.task_name}: base `{cfg.student_model}` vs fine-tuned (stage `{best.stage}`)")
        prompt = gr.Textbox(label="Prompt", lines=4)
        button = gr.Button("Generate")
        with gr.Row():
            base_out = gr.Textbox(label="Base model", lines=10)
            tuned_out = gr.Textbox(label="Fine-tuned", lines=10)
        button.click(answer, inputs=prompt, outputs=[base_out, tuned_out])
    demo.launch(share=_is_colab())
    return demo
