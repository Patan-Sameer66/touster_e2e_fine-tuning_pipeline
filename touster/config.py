"""Run configuration: one validated pydantic model plus a YAML loader."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_STUDENT_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
SMOKE_STUDENT_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
SMOKE_GOLDEN_RECORDS = 10
SMOKE_SYNTHETIC_RECORDS = 20
SMOKE_SFT_STEPS = 10
SMOKE_DPO_STEPS = 5
SMOKE_DPO_ROUNDS = 1

LOCAL_MODEL_PREFIXES = ("ollama/", "ollama_chat/", "hosted_vllm/", "lm_studio/", "llamafile/")


class ConfigError(ValueError):
    """Raised when a config is invalid; the message lists every bad field."""


class TaskType(str, Enum):
    """Kind of task; picks the eval metric and the GRPO reward."""

    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    QA = "qa"
    CODE = "code"
    SUMMARISATION = "summarisation"
    OTHER = "other"


class LoopMethod(str, Enum):
    """Training method for the self-improvement loop."""

    DPO = "dpo"
    GRPO = "grpo"


class BudgetMode(str, Enum):
    """Which cap stops the run: money, time, a target score, or none (local teachers)."""

    MONEY = "money"
    TIME = "time"
    TARGET = "target"
    FREE = "free"


def is_local_model(model_id: str) -> bool:
    """Return True if a LiteLLM model id points at a locally served model."""
    return model_id.startswith(LOCAL_MODEL_PREFIXES)


class TousterConfig(BaseModel):
    """Every setting for one touster run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # task
    task_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    task_description: str = Field(min_length=10)
    task_type: TaskType
    verifiable: bool = False

    # teacher
    teacher_model: str = Field(min_length=1)
    judge_model: str | None = None

    # student
    student_model: str = DEFAULT_STUDENT_MODEL
    max_seq_len: int = Field(default=2048, ge=128, le=32768)

    # data
    golden_path: Path
    seed_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    target_examples: int = Field(default=2000, ge=1)
    quality_threshold: int = Field(default=4, ge=1, le=5)
    dedup_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.92, gt=0.0, le=1.0)

    # sft
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    epochs: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0, lt=1.0)
    batch_size: int = Field(default=2, ge=1)
    grad_accum: int = Field(default=4, ge=1)

    # loop
    self_improve: bool = True
    loop_method: LoopMethod = LoopMethod.DPO
    max_rounds: int = Field(default=3, ge=1)
    samples_per_prompt: int = Field(default=4, ge=2)
    dpo_beta: float = Field(default=0.1, gt=0.0)
    replay_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    min_gain: float = Field(default=1.0, ge=0.0)

    # budget
    budget_mode: BudgetMode = BudgetMode.MONEY
    max_usd: float = Field(default=5.0, ge=0.0)
    max_minutes: int = Field(default=120, ge=1)
    target_score: float | None = None
    eval_reserve: float = Field(default=0.2, ge=0.0, lt=1.0)
    price_table: dict[str, tuple[float, float]] = Field(default_factory=dict)

    # export
    push_to_hub: bool = False
    hub_repo: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    gguf_quant: str = "Q4_K_M"
    model_soup: bool = False

    # run
    run_dir: Path = Path("runs")
    seed: int = 42
    smoke_test: bool = False

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> TousterConfig:
        """Validate rules that span more than one field."""
        problems: list[str] = []
        if self.loop_method == LoopMethod.GRPO and not self.verifiable:
            problems.append("loop_method='grpo' needs verifiable=true (GRPO uses a programmatic reward)")
        if self.budget_mode == BudgetMode.TARGET and self.target_score is None:
            problems.append("budget_mode='target' needs target_score")
        if self.budget_mode == BudgetMode.FREE:
            cloud = [m for m in (self.teacher_model, self.effective_judge_model) if not is_local_model(m)]
            if cloud:
                problems.append(f"budget_mode='free' allows only local teachers, got cloud model(s): {cloud}")
        if self.push_to_hub and not self.hub_repo:
            problems.append("push_to_hub=true needs hub_repo, e.g. 'your-name/touster-model'")
        for model, prices in self.price_table.items():
            if any(p < 0 for p in prices):
                problems.append(f"price_table[{model!r}] has a negative price")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def effective_judge_model(self) -> str:
        """Judge model, falling back to the teacher."""
        return self.judge_model or self.teacher_model


def _format_validation_error(err: ValidationError) -> str:
    """Turn a pydantic ValidationError into one readable line per problem."""
    lines = ["Invalid touster config:"]
    for item in err.errors():
        where = ".".join(str(p) for p in item["loc"]) or "config"
        lines.append(f"  - {where}: {item['msg']}")
    return "\n".join(lines)


def build_config(values: dict[str, Any]) -> TousterConfig:
    """Validate a dict of settings and return a TousterConfig or raise ConfigError."""
    try:
        return TousterConfig(**values)
    except ValidationError as err:
        raise ConfigError(_format_validation_error(err)) from None


def load_config(defaults: dict[str, Any], yaml_path: str | Path | None = None) -> TousterConfig:
    """Merge Python defaults with an optional YAML override file and validate."""
    merged = dict(defaults)
    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.is_file():
            raise ConfigError(f"Config YAML not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as err:
            raise ConfigError(f"Could not parse {path}: {err}") from None
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a mapping of field: value")
        merged.update(loaded)
    return build_config(merged)


def apply_smoke_overrides(cfg: TousterConfig) -> TousterConfig:
    """Return a copy of cfg shrunk for the smoke test; unchanged if smoke_test is off."""
    if not cfg.smoke_test:
        return cfg
    return cfg.model_copy(
        update={
            "student_model": SMOKE_STUDENT_MODEL,
            "target_examples": SMOKE_SYNTHETIC_RECORDS,
            "max_rounds": SMOKE_DPO_ROUNDS,
            "push_to_hub": False,
            "max_seq_len": min(cfg.max_seq_len, 1024),
        }
    )


def config_to_dict(cfg: TousterConfig) -> dict[str, Any]:
    """Serialise the config to JSON-safe primitives (it never holds secrets)."""
    return cfg.model_dump(mode="json")
