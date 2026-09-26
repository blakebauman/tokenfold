"""Deterministic fake backend for wiring and tests. No model: logits are a stable hash of
(prompt, candidate), scaled so they look as overconfident as raw LLM logprobs. It has no tokenizer, so
it counts whitespace-separated words as tokens: enough to exercise usage wiring, not a real count."""

from __future__ import annotations

import hashlib

from .base import Scores


class MockBackend:
    def __init__(self, scale: float = 8.0):
        self.scale = scale
        self.name = "mock"

    def _logit(self, prompt: str, cand: str) -> float:
        h = hashlib.sha256(f"{prompt}\x00{cand}".encode()).digest()
        return -self.scale * int.from_bytes(h[:8], "big") / 2**64  # in (-scale, 0], like a logprob

    def score(self, items: list[tuple[str, list[str]]]) -> Scores:
        return Scores(
            logits=[[self._logit(p, c) for c in cands] for p, cands in items],
            input_tokens=sum(len((p + c).split()) for p, cands in items for c in cands),
        )
