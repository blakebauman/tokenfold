"""Render (state, question) into a scoring prompt. The backend appends each candidate
option key to `prompt` and scores its log-likelihood; nothing here is model-specific."""

from __future__ import annotations

import json

from .schema import Choice, Json, Noul, Question, Score


def _render(x: Json) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2)


SYSTEM = (
    "You are a decision model. You are given a STATE and one QUESTION about it. "
    "Judge the state against the question and answer with exactly one of the allowed answers. "
    "Refer to parts of the state by the backticked paths named in the question."
)


# Scoring a question against this state measures each candidate's state-independent prior (see
# calibration.py). "N/A" rather than "" so the prompt still reads as a filled-in template.
CONTENT_FREE_STATE = "N/A"


def build_prompt(state: Json, q: Question) -> tuple[str, list[str]]:
    """Returns (prompt_prefix, candidate_answer_strings). Candidates are what the backend scores."""
    parts = [SYSTEM, "", "STATE:", _render(state), "", "QUESTION:", _render(q.instructions), ""]
    if isinstance(q, Choice):
        parts.append("ALLOWED ANSWERS (answer with the key exactly):")
        for k, desc in q.criteria.items():
            parts.append(f"- {k}" + (f": {_render(desc)}" if desc is not None else ""))
        cands = [f" {k}" for k in q.criteria]
    elif isinstance(q, Score):
        parts.append("Rate on this ordered scale. Answer with the level number exactly:")
        for i, desc in enumerate(q.criteria):
            parts.append(f"- {i}: {_render(desc)}")
        cands = [f" {i}" for i in range(len(q.criteria))]
    elif isinstance(q, Noul):
        parts.append("Answer yes or no.")
        if q.criteria:
            if "true" in q.criteria:
                parts.append(f"- yes means: {_render(q.criteria['true'])}")
            if "false" in q.criteria:
                parts.append(f"- no means: {_render(q.criteria['false'])}")
        cands = [" yes", " no"]
    else:
        raise TypeError(type(q))
    parts += ["", "ANSWER:"]
    return "\n".join(parts), cands
