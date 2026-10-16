"""LLM judge: pairwise win rate with order swapping, and 1-5 scores for the DPO loop."""

from __future__ import annotations

import logging
import math

from touster.config import TousterConfig
from touster.dashboard import log_event, set_progress
from touster.llm import LLMClient, Messages
from touster.text import find_json_object

log = logging.getLogger(__name__)

JUDGE_PHASE = "judge"
JUDGE_MAX_TOKENS = 256
MIN_SCORE, MAX_SCORE = 1, 5
VALID_WINNERS = ("A", "B", "tie")
NO_REFERENCE = "(no reference answer given)"

PAIRWISE_SYSTEM = (
    "You are a strict evaluator. Compare two answers to the same prompt for the task below. "
    "Judge correctness against the reference answer first, then how well the answer follows the task. "
    "Ignore answer length and style unless the task asks for them. "
    'Reply with JSON only, exactly one of: {"winner": "A"}, {"winner": "B"}, {"winner": "tie"}.'
)
SCORE_SYSTEM = (
    "You are a strict evaluator. Rate one answer to a prompt for the task below on a 1 to 5 scale: "
    "5 = fully correct and follows the task, 3 = partly correct, 1 = wrong or off task. "
    "Use the reference answer when one is given. "
    'Reply with JSON only, for example {"score": 4}.'
)


def parse_winner(text: str) -> str | None:
    """Return "A", "B" or "tie" from judge output, or None when the output is malformed."""
    payload = find_json_object(text)
    winner = payload.get("winner") if payload else None
    if not isinstance(winner, str):
        return None
    winner = winner.strip()
    if winner.lower() == "tie":
        return "tie"
    return winner if winner in VALID_WINNERS else None


def parse_score(text: str) -> float | None:
    """Return a 1-5 score from judge output, or None when it is malformed or out of range."""
    payload = find_json_object(text)
    score = payload.get("score") if payload else None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    if not MIN_SCORE <= score <= MAX_SCORE:
        return None
    return float(score)


def candidate_wins(first_order: str | None, second_order: str | None) -> bool:
    """True only if the candidate wins in both orders (A in the first, B in the swapped one)."""
    return first_order == "A" and second_order == "B"


def pairwise_messages(cfg: TousterConfig, prompt: str, reference: str | None, first: str, second: str) -> Messages:
    """Build the judge messages for one ordered pair."""
    user = (
        f"Task: {cfg.task_description}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Reference answer:\n{reference or NO_REFERENCE}\n\n"
        f"Answer A:\n{first}\n\n"
        f"Answer B:\n{second}\n\n"
        "Which answer is better? Reply with JSON only."
    )
    return [{"role": "system", "content": PAIRWISE_SYSTEM}, {"role": "user", "content": user}]


def score_messages(cfg: TousterConfig, prompt: str, answer: str, reference: str | None) -> Messages:
    """Build the judge messages for scoring one answer."""
    user = (
        f"Task: {cfg.task_description}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Reference answer:\n{reference or NO_REFERENCE}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score the answer. Reply with JSON only."
    )
    return [{"role": "system", "content": SCORE_SYSTEM}, {"role": "user", "content": user}]


def _judge_texts(client: LLMClient, cfg: TousterConfig, batch: list[Messages], label: str) -> list[str]:
    """Send a judge batch at temperature 0 and return the raw texts in order."""
    results = client.chat_many(
        batch,
        JUDGE_PHASE,
        model=cfg.effective_judge_model,
        on_done=lambda done: set_progress(label, done, len(batch)),
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    return [r.text for r in results]


def _log_malformed(kind: str, index: int, text: str) -> None:
    """Log one malformed judge output, shortened."""
    log.warning("Malformed %s judge output for item %d, counted as no result: %r", kind, index, text[:200])


def pairwise_win_rate(
    client: LLMClient,
    cfg: TousterConfig,
    prompts: list[str],
    answers_a: list[str],
    answers_b: list[str],
    references: list[str | None],
) -> float:
    """Fraction of prompts where answers_a beats answers_b in both judge orders; ties, disagreements and malformed output are not wins."""
    n = len(prompts)
    if not (len(answers_a) == len(answers_b) == len(references) == n):
        raise ValueError("pairwise_win_rate: prompts, answers_a, answers_b and references must have the same length")
    if n == 0:
        log_event("Judge skipped: no prompts to compare", "warning")
        return 0.0
    forward = [pairwise_messages(cfg, p, r, a, b) for p, r, a, b in zip(prompts, references, answers_a, answers_b)]
    swapped = [pairwise_messages(cfg, p, r, b, a) for p, r, a, b in zip(prompts, references, answers_a, answers_b)]
    texts = _judge_texts(client, cfg, forward + swapped, "judge pairs")
    wins = malformed = 0
    for i in range(n):
        first, second = parse_winner(texts[i]), parse_winner(texts[n + i])
        for offset, verdict in ((0, first), (n, second)):
            if verdict is None:
                malformed += 1
                _log_malformed("pairwise", i, texts[offset + i])
        wins += candidate_wins(first, second)
    if malformed:
        log_event(f"Judge: {malformed} of {2 * n} verdicts were malformed and counted as no-win", "warning")
    return wins / n


def score_answers(
    client: LLMClient,
    cfg: TousterConfig,
    prompt: str,
    answers: list[str],
    reference: str | None,
) -> list[float]:
    """Judge scores 1..5, one per answer and in the same order; malformed output gives float('nan'), which the caller must drop (math.isnan)."""
    if not answers:
        return []
    texts = _judge_texts(client, cfg, [score_messages(cfg, prompt, a, reference) for a in answers], "judge scores")
    scores: list[float] = []
    for i, text in enumerate(texts):
        score = parse_score(text)
        if score is None:
            _log_malformed("score", i, text)
            scores.append(math.nan)
        else:
            scores.append(score)
    return scores
