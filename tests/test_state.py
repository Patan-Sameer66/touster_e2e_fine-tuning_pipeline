"""Run directory creation, resume and cell skipping."""

from __future__ import annotations

from datetime import datetime

import pytest

from touster.state import (
    LeakageError,
    assert_no_eval_leak,
    config_changes,
    create_or_resume_run,
    load_training_records,
    make_run_id,
    mark_done,
    read_jsonl,
    should_run,
    unmark,
    update_state,
    write_jsonl,
)


def test_run_id_format() -> None:
    """run_id is <task_name>-<YYYYMMDD-HHMM>."""
    assert make_run_id("sentiment", datetime(2026, 10, 14, 9, 5)) == "sentiment-20261014-0905"


def test_new_run_creates_layout(run) -> None:
    """A new run writes config.json, state.json and every folder."""
    _, paths = run
    assert paths.config.is_file() and paths.state.is_file()
    for folder in (paths.data, paths.sft_checkpoints, paths.eval, paths.merged, paths.gguf):
        assert folder.is_dir()


def test_rerun_resumes_same_run(run) -> None:
    """Calling setup again returns the same run folder."""
    cfg, paths = run
    again, _, resumed = create_or_resume_run(cfg)
    assert resumed and again.root == paths.root


def test_new_run_flag_starts_fresh(run) -> None:
    """new_run=True never reuses a folder."""
    cfg, paths = run
    fresh, _, resumed = create_or_resume_run(cfg, new_run=True)
    assert not resumed and fresh.root != paths.root


def test_resume_skips_completed_cells(run) -> None:
    """Completed cells are skipped unless forced."""
    _, paths = run
    assert should_run(paths, "golden")
    mark_done(paths, "golden", {"seed": 3})
    assert not should_run(paths, "golden")
    assert should_run(paths, "golden", force=True)
    unmark(paths, "golden")
    assert should_run(paths, "golden")


def test_unknown_cell_rejected(run) -> None:
    """Typos in cell names fail loudly."""
    _, paths = run
    with pytest.raises(ValueError, match="Unknown cell"):
        should_run(paths, "golen")


def test_config_changes_detected(run, make_cfg) -> None:
    """Changing a field after the run started is reported."""
    _, paths = run
    assert config_changes(paths, make_cfg()) == []
    assert config_changes(paths, make_cfg(epochs=4)) == ["epochs"]


def test_jsonl_roundtrip(tmp_path) -> None:
    """write_jsonl/read_jsonl keep unicode and order."""
    records = [{"id": "a", "text": "café"}, {"id": "b", "text": "ok"}]
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, records) == 2
    assert read_jsonl(path) == records


def test_bad_jsonl_names_line(tmp_path) -> None:
    """A broken line reports its number."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": 1}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(path)


def test_training_loader_rejects_eval_ids(run) -> None:
    """Training loaders raise if given eval ids or the eval file."""
    _, paths = run
    update_state(paths, eval_ids=["g-0001", "g-0002"])
    write_jsonl(paths.synthetic_clean, [{"id": "s-1", "seed_id": "g-0002", "prompt": "p", "response": "r"}])
    with pytest.raises(LeakageError, match="g-0002"):
        load_training_records(paths.synthetic_clean, paths)
    write_jsonl(paths.golden_eval, [{"id": "g-0001"}])
    with pytest.raises(LeakageError, match="golden_eval"):
        load_training_records(paths.golden_eval, paths)


def test_assert_no_eval_leak_passes_clean_data() -> None:
    """Clean records pass."""
    assert_no_eval_leak([{"id": "s-1", "seed_id": "g-9"}], ["g-1"])
