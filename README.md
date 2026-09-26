# tokenfold — a TypeSafe-style decision API on an open-weights model

`POST /v1/tokenfold` takes a `state` and a map of typed `questions` (`choice`, `score`, `noul`) and
returns typed answers with calibrated probabilities and confidence under the same ids. Same wire
shape as TypeSafe's endpoint, so their SDK examples map 1:1.

The same handler also answers at TypeSafe's own path, `/v1/systemone`, so a TypeSafe client only needs
its base URL changed. For Felix, no code change is needed:

    FELIX_MODEL_PROVIDER_OPTIONS='{"typesafe": {"base_url": "http://<tokenfold-host>/v1", "api_key": "<TOKENFOLD_API_KEY>"}}'

Manifests whose `spec.decider.id` is `jev` then decide through tokenfold. The response's
`usage.input_tokens`/`output_tokens` are the tokens the backend was billed for, so Felix counts them
against `limits.max_cost_usd`. Felix prices them at Jev's catalog rate, not your host's. This takes over the
`typesafe` entry, so that deployment can't reach TypeSafe's hosted Jev at the same time.

## How it works
1. `prompt.py` renders (state, question) into a scoring prompt ending in `ANSWER:` and a list of
   candidate answer strings (option keys / level indices / yes,no).
2. A backend scores each candidate's log-likelihood as a continuation, exactly (summed token
   logprobs, no top-k truncation). All questions in a request go in one batch.
   - `VLLMLogprobBackend`: `prompt_logprobs` on an OpenAI-compatible `/v1/completions` (vLLM).
   - `LlamaCppBackend`: in-process llama.cpp. `llama-server` has no `prompt_logprobs`, so it can't
     be used through the vLLM backend. Reuses the KV cache so the prompt is prefilled once per
     question and each candidate only costs its own tokens.
3. `calibration.py` applies a per-type temperature that scales with the option count,
   `T(n) = T·(n/2)^slope` (fit on labeled data), to turn logits into calibrated probabilities.
   Temperature never changes the argmax and generalizes to any option count.
4. `answers.py` shapes the typed answer: `confidence = (n·p_max − 1)/(n − 1)`,
   `score = Σ i·p_i` with a `legend`, `noul = p(yes)`.

## Run
    make install                                   # uv sync --dev (Python 3.14)
    make dev                                       # mock backend on :8080, no GPU, wiring only
    VLLM_BASE_URL=http://gpu:8000/v1 make dev-vllm # real model, loads calibration.json
    make install-llama && make dev-llama           # local GGUF via llama.cpp (Metal), no GPU server
    uv run python examples/client.py               # TOKENFOLD_URL overrides http://localhost:8080

`make install` is a plain `uv sync --dev` and **uninstalls** the llama extra; use `make install-llama`
to keep it. The default local model is Qwen3-4B-Instruct-2507 Q4_K_M (~2.5 GB, downloaded on first use);
set `TOKENFOLD_LLAMA_MODEL` to a `.gguf` path or `<hf_repo>:<filename>` to change it.

`make check` runs ruff, ty, and pytest with coverage. `make calibrate-mock` exercises the calibration
harness end to end on random data.

## Eval suite and baseline
    make build-eval      # 26 public tasks -> data/eval/<task>.jsonl + all.jsonl (N=100 rows/task)
    make baseline        # score with the llama backend -> data/reports/baseline-llama.json
    make recalibrate     # refit from the saved baseline logits, no model needed

You don't need data for one domain, you need *breadth of question shapes*. `tokenfold.tasks`
reshapes public datasets into `{task, heldout, state, question, label}` rows: NLI/paraphrase/QA →
noul, topic/intent/emotion → choice, rating scales → score. Rows are sampled from the Hugging Face
datasets-server API (no `datasets` dependency) and cached under `data/raw/`, so rebuilds are
offline and deterministic for a given seed.

The harness splits each task into calibration and test halves, fits one temperature curve per
question type on the calibration halves of the **non-heldout** tasks, and reports acc / NLL / ECE (and MAE of
the expected level for score) per task and per type, raw and calibrated, plus per-request latency.
Heldout tasks (dbpedia, clinc_oos, subj, amazon, stsb, paws, scitail) show whether the fitted
temperatures transfer to question shapes they weren't fit on, which is the production case.
Raw LLM logprobs are badly overconfident; expect T well above 1.

## Known limits of the logprob backend
- Length bias: multi-token option keys accumulate more negative logprob. Keep keys short and
  uniform, or set `length_norm=True`.
- Token boundary: candidates start with a space so they don't merge with `ANSWER:`. If a
  tokenizer merges anyway, the backend falls back to the last token — check with a real model.
- Cost: N candidates = N forward passes per question (batched). Fine for ≤20 options; for 255-option
  choices you want a fine-tuned cross-encoder (next step) or a two-stage rank-then-recheck.

## Layout
    src/tokenfold/schema.py         request/question validation (422 on failure)
    src/tokenfold/prompt.py         prompt rendering
    src/tokenfold/backends/         Backend protocol (base), vllm_logprob, llama_cpp, mock
    src/tokenfold/calibration.py    temperature scaling
    src/tokenfold/answers.py        typed answer math
    src/tokenfold/engine.py         request -> response
    src/tokenfold/server.py         FastAPI app + uvicorn entry point (GET /healthz, POST /v1/tokenfold)
    src/tokenfold/evaluate.py       calibration + metrics harness
    src/tokenfold/tasks/            public-dataset task registry, datasets-server sampler, eval builder
    examples/                       client, mock dataset generator
    tests/unit/                     answer math, mock backend, engine, tasks, eval harness
    tests/integration/              HTTP round trips; llama backend (opt-in: TOKENFOLD_TEST_LLAMA=1)
