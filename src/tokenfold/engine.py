"""Engine: Request -> response dict. Every question is independent: build all prompts, score
them in one backend batch, calibrate per type, and shape typed answers under the caller's ids.

When the calibrator has a prior weight, each question is also scored against a content-free state
(prompt.CONTENT_FREE_STATE) so the calibrator can subtract the candidates' state-independent prior. That
prompt depends only on the question, so its logits are cached by prompt text: the first request with a
new question pays for it, later ones don't. Uncached priors ride in the same backend batch as the real
prompts, so a request still costs one backend call."""

from __future__ import annotations

from collections import OrderedDict

from .answers import choice_answer, noul_answer, score_answer
from .backends.base import Backend, Scores
from .calibration import Calibrator
from .prompt import CONTENT_FREE_STATE, build_prompt
from .schema import Choice, Noul, Question, Request, Score

PRIOR_CACHE_SIZE = 4096


class Engine:
    def __init__(self, backend: Backend, calibrator: Calibrator | None = None):
        self.backend = backend
        self.cal = calibrator or Calibrator()
        self._priors: OrderedDict[str, list[float]] = OrderedDict()

    def _prior_items(
        self, questions: dict[str, Question]
    ) -> tuple[dict[str, list[float]], dict[str, tuple[str, list[str]]]]:
        """Split questions into (cached prior logits, content-free prompts still to score)."""
        cached: dict[str, list[float]] = {}
        todo: dict[str, tuple[str, list[str]]] = {}
        for qid, q in questions.items():
            prompt, cands = build_prompt(CONTENT_FREE_STATE, q)
            if prompt in self._priors:
                cached[qid] = self._priors[prompt]
                self._priors.move_to_end(prompt)
            else:
                todo[qid] = (prompt, cands)
        return cached, todo

    def _store_priors(
        self, todo: dict[str, tuple[str, list[str]]], logits: list[list[float]]
    ) -> dict[str, list[float]]:
        out = {}
        for (qid, (prompt, _)), z in zip(todo.items(), logits, strict=True):
            self._priors[prompt] = z
            out[qid] = z
        while len(self._priors) > PRIOR_CACHE_SIZE:
            self._priors.popitem(last=False)
        return out

    def _score(
        self, req: Request, with_priors: bool
    ) -> tuple[dict[str, list[float]], dict[str, list[float]] | None, Scores]:
        ids = list(req.questions)
        items = [build_prompt(req.state, req.questions[i]) for i in ids]
        priors, todo = self._prior_items(req.questions) if with_priors else ({}, {})
        scores = self.backend.score(items + list(todo.values()))
        logits = dict(zip(ids, scores.logits[: len(ids)], strict=True))
        priors.update(self._store_priors(todo, scores.logits[len(ids) :]))
        return logits, (priors if with_priors else None), scores

    def raw_logits(self, req: Request) -> dict[str, list[float]]:
        """Uncalibrated, uncorrected logits per question id. Used by the eval/calibration harness."""
        return self._score(req, with_priors=False)[0]

    def prior_logits(self, questions: dict[str, Question]) -> dict[str, list[float]]:
        """Content-free logits per question id, from the cache or one backend batch for the rest."""
        priors, todo = self._prior_items(questions)
        if todo:
            priors.update(self._store_priors(todo, self.backend.score(list(todo.values())).logits))
        return priors

    def evaluate(self, req: Request) -> dict:
        logits, priors, scores = self._score(req, with_priors=self.cal.needs_prior)
        answers = {}
        for qid, q in req.questions.items():
            p = self.cal.apply(q.type, logits[qid], priors[qid] if priors else None)
            if isinstance(q, Choice):
                answers[qid] = choice_answer(q.options, p)
            elif isinstance(q, Score):
                answers[qid] = score_answer(q.criteria, p)
            elif isinstance(q, Noul):
                answers[qid] = noul_answer(p)
        return {
            "model": self.backend.name,
            "answers": answers,
            # input_tokens/output_tokens are TypeSafe's usage fields, which clients such as Felix meter
            "usage": {
                "input_tokens": scores.input_tokens,
                "output_tokens": scores.output_tokens,
                "questions": len(answers),
                "candidates": sum(len(v) for v in logits.values()),
            },
        }
