"""In-process llama.cpp backend (llama-cpp-python, Metal/CUDA/CPU). Exact sequence scoring, like
vllm_logprob, but local: llama-server has no prompt_logprobs, so it can't score a given continuation.

KV-cache reuse is what keeps this affordable. Each call makes the cache hold `prompt + candidate` and only
evaluates the tokens past the longest prefix already cached, so the shared prompt (system text + state +
question) is prefilled once per question and each candidate costs just its own few tokens. Consecutive
questions about the same state also reuse the state's prefix.

Model spec (arg or $TOKENFOLD_LLAMA_MODEL): a local .gguf path, or `<hf_repo>:<filename>` fetched into the
Hugging Face cache. Install with `uv sync --extra llama` (CMAKE_ARGS="-DGGML_METAL=on" on Apple silicon).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .base import Scores

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


def _common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class LlamaCppBackend:
    def __init__(
        self,
        model: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        length_norm: bool = False,
    ):
        from llama_cpp import Llama  # optional extra; imported here so other backends don't need it

        spec = model or os.environ.get("TOKENFOLD_LLAMA_MODEL", DEFAULT_MODEL)
        path = spec
        if not os.path.exists(spec):
            from huggingface_hub import hf_hub_download

            repo, sep, fname = spec.partition(":")
            if not sep:
                raise ValueError(f"model spec {spec!r} is neither a file nor '<hf_repo>:<filename>'")
            path = hf_hub_download(repo, fname)  # cached after the first call
        # logits_all keeps one logits row per cached position, so reused prefix positions stay scoreable.
        # Costs n_ctx * n_vocab float32 of RAM (~2.5 GB for Qwen3 at 4096).
        self.llm = Llama(
            model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=True, verbose=False
        )
        self.n_ctx = n_ctx
        self.length_norm = length_norm
        self.name = f"llama-cpp:{Path(path).stem}"
        self._lock = threading.Lock()  # one llama context; the threaded server must not interleave evals

    def _tok(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=True, special=False)

    def _load(self, toks: list[int]) -> int:
        """Make the KV cache hold exactly `toks`, evaluating only what isn't already cached.
        Always re-evaluates at least the last token so its logits row is guaranteed fresh.
        Returns how many tokens were evaluated."""
        if len(toks) > self.n_ctx:
            raise ValueError(f"prompt is {len(toks)} tokens, n_ctx is {self.n_ctx}")
        cached = self.llm.input_ids[: self.llm.n_tokens]
        n = min(_common_prefix(cached, toks), len(toks) - 1)
        self.llm.n_tokens = n
        self.llm.eval(toks[n:])
        return len(toks) - n

    def _score_one(self, prompt_toks: list[int], full: list[int]) -> tuple[float, int]:
        # Split point: where prompt tokens end. If the tokenizer merged across the boundary, the
        # continuation starts at the first differing token (same fallback as vllm_logprob).
        start = _common_prefix(prompt_toks, full)
        start = max(1, min(start, len(full) - 1))
        evaluated = self._load(full)
        rows = np.asarray(self.llm.scores[start - 1 : len(full) - 1], dtype=np.float64)
        rows -= rows.max(axis=1, keepdims=True)
        logprobs = rows - np.log(np.exp(rows).sum(axis=1, keepdims=True))
        cont = np.asarray(full[start:])
        lp = float(logprobs[np.arange(len(cont)), cont].sum())
        return (lp / len(cont) if self.length_norm else lp), evaluated

    def score(self, items: list[tuple[str, list[str]]]) -> Scores:
        with self._lock:
            out, evaluated = [], 0
            for prompt, cands in items:
                p = self._tok(prompt)
                row = []
                for c in cands:
                    lp, n = self._score_one(p, self._tok(prompt + c))
                    row.append(lp)
                    evaluated += n
                out.append(row)
            return Scores(logits=out, input_tokens=evaluated)
