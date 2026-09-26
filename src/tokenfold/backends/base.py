"""Backend protocol. A backend scores candidate continuations; it knows nothing about question types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Scores:
    """One logit per candidate, plus the tokens it took to get them.

    Returned together rather than kept on the backend, so concurrent requests on the threaded server
    can't read each other's counts. input_tokens is what the backend processed or was billed for:
    vLLM's own `usage` over every request it sent (the boundary pass included), llama.cpp's tokens
    actually evaluated after KV-cache reuse."""

    logits: list[list[float]]
    input_tokens: int = 0
    output_tokens: int = 0


class Backend(Protocol):
    name: str

    def score(self, items: list[tuple[str, list[str]]]) -> Scores:
        """items: (prompt_prefix, candidates) per question. Logits come back in the same order."""
        ...
