"""Cell 3 filters: schema, MinHash dedup, embedding similarity, LLM quality, eval decontamination."""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from touster.config import TousterConfig
from touster.dashboard import log_event, set_progress
from touster.llm import LLMClient
from touster.state import RunPaths, read_jsonl
from touster.text import find_json_object, normalize_text

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINHASH_PERMUTATIONS = 128
SHINGLE_WORDS = 3
DECONTAM_NGRAM = 13
QUALITY_BATCH_SIZE = 32
QUALITY_MAX_TOKENS = 200
SYNTHETIC_FIELDS = ("id", "prompt", "response")
FUNNEL_STAGES = ("raw", "schema", "dedup", "similarity", "decontam", "quality")

Embedder = Callable[[list[str]], np.ndarray]
_WORD_RE = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Lowercase word tokens used by dedup and decontamination."""
    return _WORD_RE.findall(text.lower())


def schema_filter(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep records whose id, prompt and response are non-empty strings."""
    return [
        r for r in records
        if all(isinstance(r.get(f), str) and r[f].strip() for f in SYNTHETIC_FIELDS)
    ]


def _shingles(text: str) -> set[str]:
    """Word shingles of SHINGLE_WORDS words; short texts fall back to their whole normalized form."""
    tokens = words(text)
    if len(tokens) < SHINGLE_WORDS:
        return {" ".join(tokens) or text.strip().lower()}
    return {" ".join(tokens[i : i + SHINGLE_WORDS]) for i in range(len(tokens) - SHINGLE_WORDS + 1)}


def _minhash(text: str) -> Any:
    """MinHash signature of a text's word shingles."""
    from datasketch import MinHash

    signature = MinHash(num_perm=MINHASH_PERMUTATIONS)
    signature.update_batch([s.encode("utf-8") for s in _shingles(text)])
    return signature


def _new_lsh(threshold: float) -> Any:
    """Empty MinHash LSH index for Jaccard >= threshold."""
    from datasketch import MinHashLSH

    return MinHashLSH(threshold=threshold, num_perm=MINHASH_PERMUTATIONS)


def dedup_filter(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Drop records whose prompt is a near duplicate (MinHash Jaccard >= threshold) of a kept prompt."""
    lsh = _new_lsh(threshold)
    kept = []
    for index, record in enumerate(records):
        signature = _minhash(record["prompt"])
        if lsh.query(signature):
            continue
        lsh.insert(index, signature)
        kept.append(record)
    return kept


def load_embedder(model_name: str = EMBED_MODEL) -> Embedder:
    """Load a sentence-transformers model (downloads on first use) and return an embed function."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        """Return one unit-length embedding per text."""
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    return embed


def similarity_filter(
    records: list[dict[str, Any]], threshold: float, embed: Embedder | None = None
) -> list[dict[str, Any]]:
    """Drop records whose prompt embedding has cosine >= threshold with an already kept prompt."""
    if not records:
        return []
    embed = embed or load_embedder()
    vectors = np.asarray(embed([r["prompt"] for r in records]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0.0, 1.0, norms)
    kept_vectors = np.empty_like(vectors)
    kept_idx: list[int] = []
    for index in range(len(records)):
        n_kept = len(kept_idx)
        if n_kept and float(np.max(kept_vectors[:n_kept] @ vectors[index])) >= threshold:
            continue
        kept_vectors[n_kept] = vectors[index]
        kept_idx.append(index)
    return [records[i] for i in kept_idx]


def quality_messages(cfg: TousterConfig, record: dict[str, Any]) -> list[dict[str, str]]:
    """Judge prompt asking for a 1-5 quality score as JSON."""
    content = (
        f"Task the training data is for:\n{cfg.task_description}\n\n"
        "Rate this training example from 1 (unusable) to 5 (excellent). Consider whether the prompt is a "
        "realistic input for the task and whether the response is correct, complete and in the right format.\n"
        "If the PROMPT is an instruction or a description of what to write (for example 'Write a review about "
        "...' or 'Make it sound like a customer') instead of a real input of the kind the task receives, "
        "give score 1.\n\n"
        f"PROMPT:\n{record['prompt']}\n\nRESPONSE:\n{record['response']}\n\n"
        'Reply with JSON only: {"score": <integer 1-5>, "reason": "<one short sentence>"}'
    )
    return [{"role": "user", "content": content}]


def parse_score(text: str) -> int | None:
    """Return an integer score 1-5 from judge output, or None if the output is malformed."""
    parsed = find_json_object(text)
    score = parsed.get("score") if parsed else None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score != int(score):
        return None
    return int(score) if 1 <= score <= 5 else None


def judge_scores(
    cfg: TousterConfig, client: LLMClient, records: list[dict[str, Any]]
) -> tuple[list[int | None], float]:
    """Return one score per record (None if malformed) and USD spent; records already scored are not re-judged."""
    scores: list[int | None] = []
    usd = 0.0
    for start in range(0, len(records), QUALITY_BATCH_SIZE):
        batch = records[start : start + QUALITY_BATCH_SIZE]
        todo = [i for i, r in enumerate(batch) if r.get("quality_score") is None]
        results = client.chat_many(
            [quality_messages(cfg, batch[i]) for i in todo], phase="quality", model=cfg.effective_judge_model,
            temperature=0.0, max_tokens=QUALITY_MAX_TOKENS, json_mode=True,
        ) if todo else []
        fresh = {i: parse_score(res.text) for i, res in zip(todo, results)}
        usd += sum(res.cost_usd for res in results)
        scores.extend(fresh[i] if i in fresh else r["quality_score"] for i, r in enumerate(batch))
        set_progress("quality judging", min(start + QUALITY_BATCH_SIZE, len(records)), len(records))
    return scores, usd


def quality_filter(
    cfg: TousterConfig, client: LLMClient, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Keep records scored >= cfg.quality_threshold; return (kept, malformed judge outputs)."""
    scores, _ = judge_scores(cfg, client, records)
    kept: list[dict[str, Any]] = []
    malformed = 0
    for record, score in zip(records, scores):
        if score is None:
            malformed += 1
            log_event(f"Judge gave a malformed score for {record['id']}; record dropped", "warning")
        elif score >= cfg.quality_threshold:
            kept.append({**record, "quality_score": score})
    return kept, malformed


def _ngrams(text: str, n: int = DECONTAM_NGRAM) -> set[tuple[str, ...]]:
    """All word n-grams of a text."""
    tokens = words(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def decontam_filter(
    records: list[dict[str, Any]], eval_records: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Drop records sharing a 13-gram with eval data, or whose prompt matches or near-matches an eval prompt."""
    eval_grams: set[tuple[str, ...]] = set()
    eval_lsh = _new_lsh(threshold)
    for index, record in enumerate(eval_records):
        eval_grams |= _ngrams(record["prompt"]) | _ngrams(record["response"])
        eval_lsh.insert(index, _minhash(record["prompt"]))
    eval_prompts = {normalize_text(r["prompt"]) for r in eval_records}
    kept = []
    for record in records:
        if normalize_text(record["prompt"]) in eval_prompts or eval_lsh.query(_minhash(record["prompt"])):
            continue
        if (_ngrams(record["prompt"]) | _ngrams(record["response"])) & eval_grams:
            continue
        kept.append(record)
    return kept


def load_eval_for_decontam(paths: RunPaths) -> list[dict[str, Any]]:
    """Read the eval set; only the decontamination filter may call this."""
    if not paths.golden_eval.is_file():
        raise FileNotFoundError(f"{paths.golden_eval} not found; run the golden cell (cell 2) first.")
    return read_jsonl(paths.golden_eval)


def run_filters(
    cfg: TousterConfig,
    client: LLMClient,
    records: list[dict[str, Any]],
    paths: RunPaths,
    embed: Embedder | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run every filter in order and return (clean records, records left after each stage)."""
    funnel = {"raw": len(records)}
    current = schema_filter(records)
    funnel["schema"] = len(current)
    current = dedup_filter(current, cfg.dedup_threshold)
    funnel["dedup"] = len(current)
    current = similarity_filter(current, cfg.similarity_threshold, embed)
    funnel["similarity"] = len(current)
    # Decontamination runs before quality (the spec lists it last) so the paid judge never scores dropped items.
    current = decontam_filter(current, load_eval_for_decontam(paths), cfg.dedup_threshold)
    funnel["decontam"] = len(current)
    current, malformed = quality_filter(cfg, client, current)
    funnel["quality"] = len(current)
    for previous, stage in zip(FUNNEL_STAGES, FUNNEL_STAGES[1:]):
        log_event(f"filter {stage}: removed {funnel[previous] - funnel[stage]}, {funnel[stage]} left")
    if malformed:
        log_event(f"quality judge gave {malformed} malformed answer(s); those records were dropped", "warning")
    return current, funnel
