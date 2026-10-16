"""Shared fixtures: a valid config factory and an isolated run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from touster.config import TousterConfig, build_config
from touster.state import RunPaths, create_or_resume_run


def base_values(tmp_path: Path) -> dict[str, Any]:
    """Minimal valid settings for tests."""
    golden = tmp_path / "golden.jsonl"
    golden.touch()
    return {
        "task_name": "sentiment",
        "task_description": "Classify the sentiment of a product review as positive, negative or neutral.",
        "task_type": "classification",
        "teacher_model": "ollama_chat/qwen2.5:3b",
        "golden_path": golden,
        "run_dir": tmp_path / "runs",
    }


@pytest.fixture
def make_cfg(tmp_path: Path) -> Callable[..., TousterConfig]:
    """Build a validated config with overrides."""

    def factory(**overrides: Any) -> TousterConfig:
        """Return base config merged with overrides."""
        return build_config({**base_values(tmp_path), **overrides})

    return factory


@pytest.fixture
def run(make_cfg: Callable[..., TousterConfig]) -> tuple[TousterConfig, RunPaths]:
    """A fresh run directory with the default test config."""
    cfg = make_cfg()
    paths, _, _ = create_or_resume_run(cfg)
    return cfg, paths
