"""One LiteLLM client for every teacher and judge call: retries, parallelism, cost logging."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402  (env var above must be set before import)

from touster.budget import BudgetTracker
from touster.config import TousterConfig, is_local_model
from touster.state import append_jsonl

log = logging.getLogger(__name__)

MAX_TRIES = 5
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_WORKERS = 8
OLLAMA_DEFAULT_BASE = "http://localhost:11434"
MAX_ERROR_CHARS = 500
SECRET_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9_\-.=]{8,})")
LOCAL_WORKERS = 2
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (litellm.RateLimitError, litellm.Timeout)

Messages = list[dict[str, str]]


def redact_secrets(text: str) -> str:
    """Mask API keys and tokens (env values and common key formats) in text, and cap its length."""
    for name, value in os.environ.items():
        if value and len(value) >= 8 and any(hint in name.upper() for hint in SECRET_NAME_HINTS):
            text = text.replace(value, "[redacted]")
    text = SECRET_PATTERN.sub("[redacted]", text)
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS] + "..."


class LLMError(RuntimeError):
    """Raised when a teacher or judge call fails for a non-retryable reason."""


@dataclass(frozen=True)
class LLMResult:
    """Text and accounting for one completed call."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    seconds: float


def price_from_table(cfg: TousterConfig, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Price a call from price_table (USD per 1M tokens); None if the model is not listed."""
    prices = cfg.price_table.get(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


class LLMClient:
    """Thread-safe wrapper around litellm.completion used by every phase."""

    def __init__(
        self,
        cfg: TousterConfig,
        cost_log: Path,
        budget: BudgetTracker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client that logs every call to cost_log and checks budget before calls."""
        self._cfg = cfg
        self._cost_log = cost_log
        self._budget = budget
        self._sleep = sleep
        self._lock = threading.Lock()
        self._unpriced_warned: set[str] = set()

    def _cost(self, response: Any, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Use LiteLLM's cost map first, then price_table, then 0 with a one-time warning."""
        if is_local_model(model):
            return 0.0
        try:
            cost = litellm.completion_cost(completion_response=response)
            if cost:
                return float(cost)
        except Exception as err:  # litellm raises plain Exception for unmapped models
            log.debug("litellm has no price for %s: %s", model, err)
        table_cost = price_from_table(self._cfg, model, prompt_tokens, completion_tokens)
        if table_cost is not None:
            return table_cost
        if model not in self._unpriced_warned:
            self._unpriced_warned.add(model)
            log.warning("No price known for %s; add it to price_table or costs will read $0", model)
        return 0.0

    def _log_call(self, phase: str, result: LLMResult) -> None:
        """Append one call to costs.jsonl and update the budget."""
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "phase": phase,
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": round(result.cost_usd, 8),
            "seconds": round(result.seconds, 3),
        }
        with self._lock:
            append_jsonl(self._cost_log, row)
        if self._budget is not None:
            self._budget.add_spend(result.cost_usd)

    def _call_with_retries(self, kwargs: dict[str, Any]) -> Any:
        """Call litellm.completion, retrying rate limits and timeouts with exponential backoff."""
        attempt = 1
        while True:
            try:
                return litellm.completion(**kwargs)
            except RETRYABLE_ERRORS as err:
                if attempt >= MAX_TRIES:
                    raise LLMError(f"{kwargs['model']}: gave up after {MAX_TRIES} tries ({type(err).__name__})") from None
                delay = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * 2 ** (attempt - 1))
                log.warning("%s on %s, retry %d/%d in %.0fs", type(err).__name__, kwargs["model"], attempt, MAX_TRIES - 1, delay)
                self._sleep(delay)
                attempt += 1
            except litellm.AuthenticationError:
                raise LLMError(f"{kwargs['model']}: authentication failed, check the API key env var") from None
            except Exception as err:
                raise LLMError(f"{kwargs['model']}: {type(err).__name__}: {redact_secrets(str(err))}") from None

    def chat(
        self,
        messages: Messages,
        phase: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
        check_budget: bool = True,
    ) -> LLMResult:
        """Send one chat request and return its text and cost."""
        model = model or self._cfg.teacher_model
        if check_budget and self._budget is not None:
            self._budget.check(phase)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": DEFAULT_TIMEOUT_SECONDS,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        response = self._call_with_retries(kwargs)
        seconds = time.monotonic() - started
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        text = response.choices[0].message.content or ""
        result = LLMResult(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._cost(response, model, prompt_tokens, completion_tokens),
            seconds=seconds,
        )
        self._log_call(phase, result)
        return result

    def chat_many(
        self,
        batch: list[Messages],
        phase: str,
        model: str | None = None,
        max_workers: int | None = None,
        on_done: Callable[[int], None] | None = None,
        **kwargs: Any,
    ) -> list[LLMResult]:
        """Run a batch in parallel; the first non-retryable error stops the batch and is raised."""
        model = model or self._cfg.teacher_model
        if self._budget is not None:
            self._budget.check(phase)
        workers = max_workers or (LOCAL_WORKERS if is_local_model(model) else DEFAULT_WORKERS)
        results: list[LLMResult | None] = [None] * len(batch)
        done_count = 0
        count_lock = threading.Lock()

        def run(index: int) -> None:
            """Fill results[index] and report progress."""
            nonlocal done_count
            results[index] = self.chat(batch[index], phase, model=model, check_budget=False, **kwargs)
            with count_lock:
                done_count += 1
                current = done_count
            if on_done is not None:
                on_done(current)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(len(batch))))
        return [r for r in results if r is not None]


def _colab_secret(name: str) -> str | None:
    """Read a secret from Colab Secrets when running on Colab, else None."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return userdata.get(name)
    except Exception:  # Colab raises its own SecretNotFoundError / NotebookAccessError
        return None


def load_api_keys(models: list[str], extra: tuple[str, ...] = ()) -> list[str]:
    """Copy needed keys from Colab Secrets into env if absent; return names still missing (never values)."""
    needed: set[str] = set(extra)
    for model in models:
        if is_local_model(model):
            continue
        needed.update(litellm.validate_environment(model=model).get("missing_keys", []))
    missing = []
    for name in sorted(needed):
        if os.environ.get(name):
            continue
        value = _colab_secret(name)
        if value:
            os.environ[name] = value
        else:
            missing.append(name)
    return missing


def check_local_model(model: str) -> None:
    """For Ollama model ids, confirm the server is up and the model is pulled; raise LLMError otherwise."""
    if not model.startswith(("ollama/", "ollama_chat/")):
        return
    base = (os.environ.get("OLLAMA_API_BASE") or OLLAMA_DEFAULT_BASE).rstrip("/")
    name = model.split("/", 1)[1]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            tags = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        raise LLMError(f"Ollama is not reachable at {base} ({err}). Start it with `ollama serve`.") from None
    pulled = {m.get("name", "") for m in tags.get("models", [])}
    wanted = {name, f"{name}:latest"}
    if not pulled & wanted:
        raise LLMError(f"Ollama model '{name}' is not pulled. Run: ollama pull {name}")


def has_known_price(cfg: TousterConfig, model: str) -> bool:
    """True if the model is local, listed in price_table, or priced in LiteLLM's cost map."""
    if is_local_model(model) or model in cfg.price_table:
        return True
    try:
        litellm.cost_per_token(model=model, prompt_tokens=1, completion_tokens=1)
    except Exception:  # litellm raises ModelNotMappedError/BadRequestError for unknown models
        return False
    return True
