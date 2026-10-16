"""Cell 1: invalid config fails readably, re-running resumes the same run, missing keys fail fast."""

from __future__ import annotations

import pytest

import touster
import touster.llm
from touster import setup_run
from touster.config import ConfigError
from touster.state import STATUS_RUNNING, STATUS_STOPPED_BUDGET, load_state, read_json, update_state
from tests.conftest import base_values


@pytest.fixture(autouse=True)
def no_ollama(monkeypatch) -> None:
    """Skip the live Ollama check in unit tests."""
    monkeypatch.setattr(touster.llm, "check_local_model", lambda model: None)


def test_invalid_config_raises_before_anything_runs(tmp_path) -> None:
    """A bad value stops setup with a readable message and creates no run folder."""
    values = {**base_values(tmp_path), "quality_threshold": 9}
    with pytest.raises(ConfigError, match="quality_threshold"):
        setup_run(values)
    assert not (tmp_path / "runs").exists()


def test_rerun_resumes_same_run(tmp_path) -> None:
    """Running setup twice returns the same run folder and marks setup done."""
    first = setup_run(base_values(tmp_path))
    second = setup_run(base_values(tmp_path))
    assert second.resumed and second.paths.root == first.paths.root
    assert "setup" in load_state(second.paths)["completed"]


def test_missing_cloud_key_fails_fast(tmp_path, monkeypatch) -> None:
    """A cloud teacher without its key stops setup and names the variable, never its value."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        setup_run({**base_values(tmp_path), "teacher_model": "gpt-4o-mini"})


def test_missing_golden_file(tmp_path) -> None:
    """A wrong golden_path is reported before the run starts."""
    with pytest.raises(FileNotFoundError, match="golden_path"):
        setup_run({**base_values(tmp_path), "golden_path": tmp_path / "nope.jsonl"})


def test_locked_field_change_refuses_resume(tmp_path) -> None:
    """Changing the seed after a run started stops setup and points at NEW_RUN."""
    setup_run(base_values(tmp_path))
    with pytest.raises(RuntimeError, match="NEW_RUN"):
        setup_run({**base_values(tmp_path), "seed": 7})


def test_unlocked_change_is_saved(tmp_path) -> None:
    """Raising the cap on resume is accepted and written to config.json."""
    first = setup_run(base_values(tmp_path))
    setup_run({**base_values(tmp_path), "max_usd": 9.0})
    assert read_json(first.paths.config)["max_usd"] == 9.0


def test_budget_stop_lifted_when_cap_raised(tmp_path) -> None:
    """A run stopped on budget runs again once the new cap leaves room."""
    first = setup_run(base_values(tmp_path))
    update_state(first.paths, status=STATUS_STOPPED_BUDGET)
    setup_run({**base_values(tmp_path), "max_usd": 50.0})
    assert load_state(first.paths)["status"] == STATUS_RUNNING


def test_install_requirements_runs_once_and_asks_for_restart(tmp_path, monkeypatch) -> None:
    """pip runs once; a restart is requested when watched libraries were already imported."""
    calls: list[list[str]] = []
    monkeypatch.setattr(touster.subprocess, "run", lambda cmd, check: calls.append(cmd))
    req = tmp_path / "requirements.txt"
    req.write_text("rich==15.0.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Restart the runtime"):
        touster.install_requirements(req)
    touster.install_requirements(req)
    assert len(calls) == 1
