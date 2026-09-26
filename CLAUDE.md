# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tokenfold` is an HTTP decision API (`POST /v1/tokenfold`, also served at TypeSafe's own path `/v1/systemone`)
that mimics TypeSafe's wire format on top of an open-weights model. A request carries a `state` (string or JSON)
and a map of typed `questions` (`choice`, `score`, `noul`). The response has typed answers with calibrated
probabilities, keyed by the caller's question ids. The model never generates text. It only scores the
log-likelihood of each candidate answer.

The project is always called **tokenfold**. Earlier drafts called it "systemone", so rename any leftover occurrence,
except the `/v1/systemone` route alias in `server.py`: that is TypeSafe's API path, kept so TypeSafe clients work
unchanged.

## Commands

The project uses uv, Python 3.14, and a `src/` layout. Its tooling follows `~/Projects/felix`: ruff at 110
columns, ty, pytest, and pre-commit.

    make install          # uv sync --dev  (NOTE: removes the llama extra; use install-llama to keep it)
    make install-llama    # + llama-cpp-python built with Metal
    make check            # ruff check + ty check src + pytest --cov + ruff format --check
    make test             # uv run pytest -q
    uv run pytest tests/unit/test_answers.py::test_noul_is_p_yes   # single test
    make fmt              # ruff format
    make dev              # mock backend on $TOKENFOLD_PORT (default 8080), no GPU
    make dev-vllm         # VLLM_BASE_URL / VLLM_MODEL / VLLM_API_KEY, loads calibration.json
    make dev-llama        # local GGUF via llama.cpp ($TOKENFOLD_LLAMA_MODEL, default Qwen3-4B-Instruct Q4_K_M)
    make calibrate-mock   # random dataset -> tokenfold-evaluate, writes into data/ (gitignored)
    make build-eval       # 26 public tasks -> data/eval/*.jsonl (N=100/task; pages cached in data/raw/)
    make baseline         # llama backend over data/eval/all.jsonl -> data/reports/, ~20 min on an M4 Pro
    make recalibrate      # refit + re-report from data/logits/baseline-llama.json in seconds, no model
    TOKENFOLD_TEST_LLAMA=1 uv run pytest tests/integration/test_llama_backend.py   # opt-in real-model test
    TOKENFOLD_URL=http://localhost:8765 uv run python examples/client.py

The console scripts are `tokenfold-server`, `tokenfold-evaluate`, and `tokenfold-build-eval`. `--api-key` or `$TOKENFOLD_API_KEY` turns on
Bearer auth. The tests never contact a real model: they build their own engines on `MockBackend`, and the
integration tests bind `127.0.0.1:0`. On this machine, port 8080 is often taken by Docker.

## Request pipeline

`server.py` → `schema.parse_request` → `Engine.evaluate` → the backend's `score` → `Calibrator.apply` → `answers.*`

- **schema.py** defines the dataclasses `Choice` (criteria is a dict of option→description, 2–255 options),
  `Score` (criteria is an ordered list of 2–10 levels), and `Noul` (optional `{"true","false"}` criteria).
  Any validation failure raises `ValidationError`, and the server maps it to HTTP 422. A backend exception maps to HTTP 529.
- **prompt.py**: `build_prompt(state, q)` returns `(prefix ending in "ANSWER:", candidates)`. Each candidate has a
  **leading space** (`" billing"`, `" 0"`, `" yes"`/`" no"`) so it tokenizes separately from the prefix.
  Candidate order must match `q.options`, because answers are built by index.
- **backends/base.py** defines the `Backend` Protocol: `score(items: list[(prompt, candidates)]) -> Scores`, where
  `Scores` holds `logits` (one per candidate) plus the `input_tokens`/`output_tokens` it took, and a `.name`
  attribute. `Engine` sends every question in a request as **one batch**. The token counts come back with the
  logits, not as backend state, because the threaded server shares one backend. The response reports them as
  TypeSafe's `usage.input_tokens`/`output_tokens`, which Felix meters. The model is hosted, so these tokens cost
  money: vLLM's counts are its own billed `usage` summed over every request, including the boundary pass.
  The `backends` package exports every backend, and `server.make_engine` picks one by name.
  - **mock.py** returns deterministic hash-based logits in (−8, 0]. It exists for wiring and tests, and its
    output means nothing.
  - **llama_cpp.py** does exact scoring in-process with llama-cpp-python. `llama-server` has no prompt
    logprobs, so it can't stand in for vLLM. `logits_all=True` keeps a logits row per cached position, and
    `_load` evaluates only the tokens past the longest cached prefix. The prompt is prefilled once per
    question and each candidate costs its own tokens. Cached scores match cold scoring exactly when the batch
    shapes match. Different batch shapes shift logprobs by ~0.06 through Metal quantized-kernel numerics, so
    compare like with like. A lock serializes access, because one llama context can't be shared by the
    threaded server.
  - **vllm_logprob.py** does exact sequence scoring against an OpenAI-compatible `/v1/completions` endpoint with
    `prompt_logprobs`. It first scores the bare prompts to get the token-boundary length, then scores
    prompt+candidate and sums the logprobs past that boundary. If the tokenizer merges the candidate into the
    prefix, it falls back to the last token. Cost is N forward passes per question.
- **calibration.py** fits a temperature curve per question type, `T(n) = T·(n/2)^slope` over the option count n,
  by minimizing NLL. T(n) is clamped to 0.05–20 and the slope to ±1. The slope is fit only when the calibration
  rows span more than one option count, so noul keeps slope 0. It can also apply a **prior weight** per type
  (contextual calibration): the engine scores each question once against `prompt.CONTENT_FREE_STATE` (`"N/A"`),
  and `prior_weight · prior_logits` is subtracted before the temperature. That is the one thing that can change
  the argmax. It is **off by default and stays at 0 in the shipped calibration**: on the Qwen3-4B baseline it did
  not transfer to the test halves (flat NLL, −1 point choice accuracy, no heldout gain) under any content-free
  state tried, and an end-level penalty for score didn't help either. The observed skews (end levels, "neutral")
  are not separable from the state. Don't re-enable it without a harness result; `--prior-weight fit` tries it.
  The file is `{"temperatures": {...}, "slopes": {...}, "prior_weights": {...}}` JSON; older files load with slope
  0 and no prior. Temperature is used rather than Platt or isotonic because it works for any option count, which
  matters because callers define option sets per request. Don't swap those in.
- **engine.py** caches prior logits by content-free prompt text (`PRIOR_CACHE_SIZE` entries), so a question's
  prior is scored once and only when the calibrator's `needs_prior` is true. Uncached priors ride in the same
  backend batch as the request's real prompts. `raw_logits` and `prior_logits` feed the harness separately.
- **answers.py** is pure numpy.
  - `confidence = (n·p_max − 1)/(n − 1)`
  - A score answer is `Σ i·p_i`, the probability-weighted level, so it can fall between levels.
  - `noul = p(yes)`, with backend order `[yes, no]`.
- **tasks/**: the `Task` registry (`__init__.py`) turns public HF datasets into
  `{task, heldout, state, question, label}` rows. `hf.py` samples random pages from the datasets-server
  `/rows` API and caches them on disk. Use a small `page` for label-sorted splits, as clinc and amazon do.
  `expect_names` fails the build if an upstream ClassLabel order changes. `build.py` validates every
  question with `parse_question`. `heldout` tasks are never used to fit calibration, and must never be
  used for training either.
- **evaluate.py**: `Engine.raw_logits` (uncalibrated) feeds the harness. Rows that share a state are batched
  into one request, just like production. The harness splits **each task** by `--cal-frac`, seeded per task so
  adding a task never reshuffles another's split, then fits one temperature curve per type on the non-heldout
  calibration rows. `--save-logits` writes the raw logits after scoring, and `--from-logits` refits from them
  without a model (the report then has no latency). The logits file also holds each row's content-free prior,
  so `--prior-weight fit` can A/B the prior correction offline. It reports each task's test half raw and calibrated,
  covering acc, NLL, ECE, MAE of the expected level for score, and per-request latency. Rows without `task` or
  `heldout` default to `default` and `False`. `label` is an option key (choice), a level index (score), or a
  bool (noul).

## Known caveats

- Length bias: multi-token option keys collect more negative logprob. Keep keys short and uniform, or set `length_norm=True`.
- Raw logprobs are overconfident, so expect T well above 1. Don't trust probabilities without a calibration file.
  With the mock backend, T hits the 20.0 upper bound because the labels are random.
- `vllm_logprob.py` is tested only against a fake `/completions` endpoint (`tests/unit/test_vllm_backend.py`), never
  a live one. Real vLLM response shapes can drift from the fake.
- The internal disk is nearly full, so model weights must never land on it. The Makefile exports
  `HF_HUB_CACHE=/Volumes/External SSD 1TB/huggingface/hub`, and `dev-llama` and `baseline` fail fast if the SSD
  isn't mounted. When you call `uv run` outside make (for example, the opt-in llama test), export the same
  `HF_HUB_CACHE` first, or `hf_hub_download` falls back to `~/.cache/huggingface`.
