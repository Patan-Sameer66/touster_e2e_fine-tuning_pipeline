"""Config validation accepts good configs and rejects bad ones with readable messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from touster.config import (
    SMOKE_STUDENT_MODEL,
    ConfigError,
    apply_smoke_overrides,
    build_config,
    config_to_dict,
    load_config,
)
from tests.conftest import base_values


def test_good_config_uses_defaults(make_cfg) -> None:
    """Defaults from the spec are applied."""
    cfg = make_cfg()
    assert cfg.seed_fraction == 0.3
    assert cfg.lora_r == 16 and cfg.lora_alpha == 32
    assert cfg.effective_judge_model == cfg.teacher_model


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"task_type": "poetry"}, "task_type"),
        ({"quality_threshold": 7}, "quality_threshold"),
        ({"seed_fraction": 1.5}, "seed_fraction"),
        ({"unknown_field": 1}, "unknown_field"),
        ({"loop_method": "grpo", "verifiable": False}, "verifiable=true"),
        ({"budget_mode": "target"}, "target_score"),
        ({"budget_mode": "free", "teacher_model": "gpt-4o-mini"}, "only local teachers"),
        ({"push_to_hub": True}, "hub_repo"),
        ({"task_name": "bad name!"}, "task_name"),
    ],
)
def test_bad_config_raises_readable_error(tmp_path: Path, override: dict, fragment: str) -> None:
    """Each invalid value names the field in a ConfigError."""
    with pytest.raises(ConfigError) as info:
        build_config({**base_values(tmp_path), **override})
    message = str(info.value)
    assert message.startswith("Invalid touster config:")
    assert fragment in message


def test_missing_required_field(tmp_path: Path) -> None:
    """Dropping a required field is reported by name."""
    values = base_values(tmp_path)
    del values["teacher_model"]
    with pytest.raises(ConfigError, match="teacher_model"):
        build_config(values)


def test_free_mode_accepts_local_teacher(make_cfg) -> None:
    """Local Ollama ids pass free mode."""
    assert make_cfg(budget_mode="free").budget_mode.value == "free"


def test_yaml_override(tmp_path: Path) -> None:
    """YAML values override Python defaults."""
    yaml_file = tmp_path / "override.yaml"
    yaml_file.write_text("epochs: 5\nlearning_rate: 0.0001\n", encoding="utf-8")
    cfg = load_config(base_values(tmp_path), yaml_file)
    assert cfg.epochs == 5 and cfg.learning_rate == 1e-4


def test_yaml_errors(tmp_path: Path) -> None:
    """Missing, malformed and non-mapping YAML files raise ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(base_values(tmp_path), tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(base_values(tmp_path), bad)


def test_smoke_overrides_shrink_run(make_cfg) -> None:
    """Smoke mode swaps in the tiny student and never pushes."""
    cfg = apply_smoke_overrides(make_cfg(smoke_test=True, push_to_hub=True, hub_repo="me/x"))
    assert cfg.student_model == SMOKE_STUDENT_MODEL
    assert cfg.push_to_hub is False
    assert cfg.max_rounds == 1


def test_config_is_frozen_and_serialisable(make_cfg) -> None:
    """Configs are immutable and dump to JSON-safe values."""
    cfg = make_cfg()
    with pytest.raises(Exception):
        cfg.epochs = 9  # type: ignore[misc]
    dumped = config_to_dict(cfg)
    assert dumped["task_type"] == "classification"
    assert isinstance(dumped["golden_path"], str)
