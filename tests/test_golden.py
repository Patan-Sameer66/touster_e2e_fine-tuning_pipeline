"""Golden dataset: validation, deterministic split, and the cell's saved outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from touster.config import SMOKE_GOLDEN_RECORDS
from touster.data.golden import GoldenDataError, load_golden, run_golden_cell, split_golden
from touster.state import load_state, read_jsonl

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "golden_sample.jsonl"


def make_records(n: int) -> list[dict[str, Any]]:
    """n valid, distinct golden records."""
    return [{"id": f"g-{i:04d}", "prompt": f"review number {i}", "response": "neutral"} for i in range(1, n + 1)]


def write_lines(path: Path, records: list[Any]) -> Path:
    """Write records (or raw strings) one per line."""
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_example_file_is_valid_and_balanced() -> None:
    """The shipped sample has 30 records, 10 per label, ids g-0001..g-0030."""
    records = load_golden(EXAMPLE)
    assert len(records) == 30
    assert Counter(r["response"] for r in records) == {"positive": 10, "negative": 10, "neutral": 10}
    assert [r["id"] for r in records] == [f"g-{i:04d}" for i in range(1, 31)]
    assert all(r["metadata"]["source"] == "human" and r["metadata"]["tags"] for r in records)


def test_split_is_deterministic_and_disjoint() -> None:
    """Same seed gives the same split; sets are disjoint and cover everything."""
    records = make_records(40)
    seed_a, eval_a = split_golden(records, 0.3, seed=7)
    seed_b, eval_b = split_golden(records, 0.3, seed=7)
    assert seed_a == seed_b and eval_a == eval_b
    assert len(seed_a) == 12 and len(eval_a) == 28
    ids_seed, ids_eval = {r["id"] for r in seed_a}, {r["id"] for r in eval_a}
    assert not ids_seed & ids_eval and len(ids_seed | ids_eval) == 40
    other_seed, _ = split_golden(records, 0.3, seed=8)
    assert [r["id"] for r in other_seed] != [r["id"] for r in seed_a]


def test_split_does_not_mutate_input() -> None:
    """The caller's list keeps its order."""
    records = make_records(25)
    before = list(records)
    split_golden(records, 0.5, seed=1)
    assert records == before


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        ({"id": "g-0001", "prompt": "dup id", "response": "x"}, "duplicate id 'g-0001'"),
        ({"id": "g-9999", "prompt": "review number 2", "response": "x"}, "same prompt as line 2"),
        ({"id": "g-9999", "prompt": "fresh", "response": "  "}, "'response' must be a non-empty string"),
        ({"id": "g-9999", "prompt": "fresh"}, "missing field 'response'"),
        ({"id": "g-9999", "prompt": "fresh", "response": "x", "metadata": "tag"}, "'metadata' must be an object"),
        ("{not json", "invalid JSON"),
        ({"id": 5, "prompt": "fresh", "response": "x"}, "'id' must be a non-empty string"),
    ],
)
def test_validation_names_the_problem(tmp_path: Path, bad_line: Any, message: str) -> None:
    """Each kind of bad record is rejected with its line and the reason."""
    path = write_lines(tmp_path / "g.jsonl", [*make_records(25), bad_line])
    with pytest.raises(GoldenDataError, match="line 26") as err:
        load_golden(path)
    assert message in str(err.value)


def test_too_few_records_refused(tmp_path: Path) -> None:
    """Fewer than 20 records is an error outside smoke mode."""
    path = write_lines(tmp_path / "g.jsonl", make_records(19))
    with pytest.raises(GoldenDataError, match="at least 20"):
        load_golden(path)


def test_few_records_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """20-49 records load with a warning."""
    path = write_lines(tmp_path / "g.jsonl", make_records(25))
    assert len(load_golden(path)) == 25
    assert "Only 25 golden records" in capsys.readouterr().out


def test_smoke_takes_first_records_and_skips_minimum(tmp_path: Path) -> None:
    """Smoke mode keeps the first SMOKE_GOLDEN_RECORDS records even from a small file."""
    path = write_lines(tmp_path / "g.jsonl", make_records(12))
    records = load_golden(path, smoke=True)
    assert [r["id"] for r in records] == [f"g-{i:04d}" for i in range(1, SMOKE_GOLDEN_RECORDS + 1)]


def test_missing_file(tmp_path: Path) -> None:
    """A missing file gives a clear error."""
    with pytest.raises(GoldenDataError, match="not found"):
        load_golden(tmp_path / "nope.jsonl")


def test_golden_cell_saves_split_and_eval_ids(make_cfg, tmp_path: Path) -> None:
    """The cell writes both files and records eval ids in state.json."""
    from touster.state import create_or_resume_run

    cfg = make_cfg(golden_path=EXAMPLE)
    paths, _, _ = create_or_resume_run(cfg)
    metrics = run_golden_cell(cfg, paths)
    seed_set, eval_set = read_jsonl(paths.golden_seed), read_jsonl(paths.golden_eval)
    assert metrics == {"total": 30, "seed": 9, "eval": 21}
    assert len(seed_set) == 9 and len(eval_set) == 21
    state = load_state(paths)
    assert state["eval_ids"] == [r["id"] for r in eval_set]
    assert "golden" in state["completed"]


def test_golden_force_resplit_resets_downstream_cells(make_cfg) -> None:
    """A re-split with different eval ids marks downstream cells to run again; the same split leaves them."""
    from touster.state import create_or_resume_run, mark_done

    cfg = make_cfg(golden_path=EXAMPLE)
    paths, _, _ = create_or_resume_run(cfg)
    run_golden_cell(cfg, paths)
    for cell in ("synthetic", "sft"):
        mark_done(paths, cell, {"x": 1})
    run_golden_cell(cfg, paths, force=True)
    assert {"synthetic", "sft"} <= set(load_state(paths)["completed"])
    run_golden_cell(cfg.model_copy(update={"seed": cfg.seed + 1}), paths, force=True)
    completed = load_state(paths)["completed"]
    assert "golden" in completed and "synthetic" not in completed and "sft" not in completed
