"""LiteLLM client: retries, cost logging and budget checks, with litellm mocked."""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest

from touster.budget import BudgetExceeded, BudgetTracker, read_spent_usd
from touster.llm import MAX_TRIES, LLMClient, LLMError, load_api_keys, price_from_table
from touster.state import read_jsonl


def fake_response(text: str = "hello", prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    """Shape-compatible stand-in for a litellm ModelResponse."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def no_price(**_: object) -> float:
    """Simulate litellm having no price for a model."""
    raise Exception("model not mapped")


@pytest.fixture
def cloud_cfg(make_cfg):
    """Config with a cloud teacher and a price_table entry."""
    return make_cfg(teacher_model="acme/model-x", price_table={"acme/model-x": (1.0, 2.0)})


def test_price_table_math(cloud_cfg) -> None:
    """USD per 1M tokens for input and output."""
    assert price_from_table(cloud_cfg, "acme/model-x", 1_000_000, 500_000) == pytest.approx(2.0)
    assert price_from_table(cloud_cfg, "other", 1, 1) is None


def test_chat_logs_cost(monkeypatch, run, cloud_cfg) -> None:
    """Each call writes one costs.jsonl row priced from price_table."""
    _, paths = run
    monkeypatch.setattr(litellm, "completion", lambda **_: fake_response(prompt=1000, completion=1000))
    monkeypatch.setattr(litellm, "completion_cost", no_price)
    client = LLMClient(cloud_cfg, paths.costs)
    result = client.chat([{"role": "user", "content": "hi"}], phase="generation")
    assert result.text == "hello"
    rows = read_jsonl(paths.costs)
    assert len(rows) == 1 and rows[0]["phase"] == "generation"
    assert read_spent_usd(paths.costs) == pytest.approx(0.003)


def test_local_model_costs_zero(monkeypatch, run) -> None:
    """Ollama calls are free."""
    cfg, paths = run
    monkeypatch.setattr(litellm, "completion", lambda **_: fake_response())
    LLMClient(cfg, paths.costs).chat([{"role": "user", "content": "hi"}], phase="judge")
    assert read_jsonl(paths.costs)[0]["cost_usd"] == 0.0


def test_retries_rate_limit_then_succeeds(monkeypatch, run) -> None:
    """Rate limits retry with exponential backoff."""
    cfg, paths = run
    calls = {"n": 0}

    def flaky(**_: object) -> SimpleNamespace:
        """Fail twice, then answer."""
        calls["n"] += 1
        if calls["n"] < 3:
            raise litellm.RateLimitError("slow down", llm_provider="x", model="m")
        return fake_response()

    sleeps: list[float] = []
    monkeypatch.setattr(litellm, "completion", flaky)
    client = LLMClient(cfg, paths.costs, sleep=sleeps.append)
    assert client.chat([{"role": "user", "content": "hi"}], phase="generation").text == "hello"
    assert sleeps == [2.0, 4.0]


def test_gives_up_after_max_tries(monkeypatch, run) -> None:
    """Timeouts stop after MAX_TRIES attempts."""
    cfg, paths = run
    calls = {"n": 0}

    def always_timeout(**_: object) -> None:
        """Always time out."""
        calls["n"] += 1
        raise litellm.Timeout("late", model="m", llm_provider="x")

    monkeypatch.setattr(litellm, "completion", always_timeout)
    with pytest.raises(LLMError, match="gave up"):
        LLMClient(cfg, paths.costs, sleep=lambda _: None).chat([{"role": "user", "content": "x"}], phase="g")
    assert calls["n"] == MAX_TRIES


def test_other_errors_stop_immediately(monkeypatch, run) -> None:
    """Non-retryable errors are raised on the first try."""
    cfg, paths = run
    calls = {"n": 0}

    def broken(**_: object) -> None:
        """Fail with a non-retryable error."""
        calls["n"] += 1
        raise ValueError("bad request")

    monkeypatch.setattr(litellm, "completion", broken)
    with pytest.raises(LLMError, match="bad request"):
        LLMClient(cfg, paths.costs).chat([{"role": "user", "content": "x"}], phase="g")
    assert calls["n"] == 1


def test_budget_blocks_calls(monkeypatch, run, make_cfg) -> None:
    """No call is made once the cap is hit."""
    _, paths = run
    cfg = make_cfg(max_usd=0.1)
    budget = BudgetTracker(cfg, paths.costs)
    budget.add_spend(0.5)
    monkeypatch.setattr(litellm, "completion", lambda **_: pytest.fail("should not call"))
    with pytest.raises(BudgetExceeded):
        LLMClient(cfg, paths.costs, budget).chat([{"role": "user", "content": "x"}], phase="generation")


def test_chat_many_preserves_order(monkeypatch, run) -> None:
    """Parallel batches return results in input order and report progress."""
    cfg, paths = run
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response(kw["messages"][0]["content"]))
    seen: list[int] = []
    batch = [[{"role": "user", "content": str(i)}] for i in range(6)]
    results = LLMClient(cfg, paths.costs).chat_many(batch, phase="generation", on_done=seen.append)
    assert [r.text for r in results] == [str(i) for i in range(6)]
    assert sorted(seen) == list(range(1, 7))


def test_load_api_keys_reports_missing_names(monkeypatch) -> None:
    """Missing keys are reported by name and local models need none."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_api_keys(["ollama_chat/qwen2.5:3b"]) == []
    assert "OPENAI_API_KEY" in load_api_keys(["gpt-4o-mini"])
