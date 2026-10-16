# Contributing

Bug reports and pull requests are welcome. Please open an issue first for larger changes so we can agree on
the approach.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a machine without a CUDA GPU, `unsloth`, `bitsandbytes` and `vllm` may fail to install. The unit tests do
not need them; install the rest and skip those three.

## Unit tests

```bash
pytest -q tests
```

They run on CPU, need no network, and mock every LLM call and model. Add a test with every change in
behaviour; the tests in `tests/` show the patterns (fake LiteLLM responses, fake student models).

## Smoke test

The smoke test runs the whole notebook end to end with a tiny student, 10 golden records, 20 synthetic records,
10 SFT steps and one 5-step DPO round. It should finish in under 15 minutes on a T4 and produce every file in
the run folder, even though the scores mean nothing.

1. Open `touster.ipynb` on a GPU machine or Colab T4.
2. In cell 1 set `"smoke_test": True` and pick a cheap teacher (a local Ollama model, or the cheapest model of
   your API provider).
3. Run All and check that `runs/<run_id>/` contains `report.html`, `export/gguf/`, `export/Modelfile`,
   `export/MODEL_CARD.md`, `eval/base.json`, `eval/sft.json` and `eval/dpo_round_1.json`.

Smoke mode skips the lm-eval regression check and the Hub push.

## Style

- Notebook cells stay short; logic lives in the `touster/` package.
- Every function has type hints and a one-line docstring.
- Every LLM call goes through `touster/llm.py` so cost is logged and the budget is enforced.
- Training code loads data only through `touster.state.load_training_records`, which refuses eval data.
- Never print, log or save API keys or tokens.
- Pin exact versions in `requirements.txt`, and check compatibility between Unsloth, TRL, PEFT,
  transformers and torch before changing any of them.
