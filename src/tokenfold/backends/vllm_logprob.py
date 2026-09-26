"""Logprob-scoring backend against a vLLM (or any OpenAI-compatible) /v1/completions endpoint.

For each (prompt, candidates) we submit prompt+candidate for every candidate with
`prompt_logprobs` enabled and sum the log-probabilities of the candidate's tokens.
This is exact sequence scoring: no first-token tricks, no top-k truncation, works for 255 options.

Token boundary: we send the bare prompt too and use its token count L0 as the split point,
so candidates must start with a space/newline to avoid merging with the last prompt token
(prompt.py does this). If your tokenizer still merges, set `boundary_token="\\n"`.

Length bias: longer option keys accumulate more negative logprob. `length_norm=True` divides by
token count; leave it off and let calibration absorb it unless option keys vary wildly in length.

Run vLLM with something like:
  vllm serve Qwen/Qwen2.5-7B-Instruct --max-logprobs 0
(prompt_logprobs is a per-request sampling param; no special server flag needed on recent vLLM).
"""

from __future__ import annotations

import os

import requests

from .base import Scores


class VLLMLogprobBackend:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        length_norm: bool = False,
        batch_size: int = 64,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("VLLM_MODEL", "")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.length_norm = length_norm
        self.batch_size = batch_size
        self.timeout = timeout
        self.name = f"vllm-logprob:{self.model}"
        if not self.model:
            self.model = self._first_model()

    # ---- HTTP ----------------------------------------------------------------
    def _first_model(self) -> str:
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()["data"][0]["id"]

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _prompt_logprobs(self, prompts: list[str], usage: list[int]) -> list[list[float]]:
        """Returns per-prompt list of token logprobs (first token has none -> skipped). Adds the
        billed [prompt_tokens, completion_tokens] of every request sent to `usage`."""
        out: list[list[float]] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            body = {
                "model": self.model,
                "prompt": chunk,
                "max_tokens": 1,
                "temperature": 0,
                "prompt_logprobs": 0,
                "logprobs": 0,
            }
            r = requests.post(
                f"{self.base_url}/completions", json=body, headers=self._headers(), timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()
            choices = sorted(data["choices"], key=lambda c: c["index"])
            billed = data.get("usage") or {}
            # the server's own count is what's billed; without one, prompt_logprobs has one
            # entry per prompt token and max_tokens=1 generates one token per prompt
            usage[0] += billed.get("prompt_tokens", sum(len(c.get("prompt_logprobs") or []) for c in choices))
            usage[1] += billed.get("completion_tokens", len(choices))
            for c in choices:
                toks = []
                for entry in c.get("prompt_logprobs") or []:
                    if entry is None:  # first token has no logprob
                        continue
                    # entry: {token_id: {"logprob": float, "rank": int, "decoded_token": str}}
                    # with prompt_logprobs=0 there is exactly one key: the actual token.
                    lp = max(v["logprob"] for v in entry.values())
                    toks.append(float(lp))
                out.append(toks)
        return out

    # ---- Backend API ---------------------------------------------------------
    def score(self, items: list[tuple[str, list[str]]]) -> Scores:
        usage = [0, 0]
        # 1) base prompts -> token counts (boundary)
        bases = [p for p, _ in items]
        base_lens = [len(t) for t in self._prompt_logprobs(bases, usage)]
        # 2) all prompt+candidate strings in one flat batch
        flat, owner = [], []
        for i, (p, cands) in enumerate(items):
            for c in cands:
                flat.append(p + c)
                owner.append(i)
        scored = self._prompt_logprobs(flat, usage)
        # 3) sum continuation logprobs
        results: list[list[float]] = [[] for _ in items]
        for i, toks in zip(owner, scored, strict=True):
            cont = toks[base_lens[i] :]
            if not cont:  # tokenizer merged everything; fall back to last token
                cont = toks[-1:]
            s = sum(cont)
            if self.length_norm:
                s /= len(cont)
            results[i].append(s)
        return Scores(logits=results, input_tokens=usage[0], output_tokens=usage[1])
