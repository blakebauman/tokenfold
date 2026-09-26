"""Engine: Request -> response dict. Every question is independent: build all prompts, score
them in one backend batch, calibrate per type, and shape typed answers under the caller's ids."""

from __future__ import annotations

from .answers import choice_answer, noul_answer, score_answer
from .backends.base import Backend, Scores
from .calibration import Calibrator
from .prompt import build_prompt
from .schema import Choice, Noul, Request, Score


class Engine:
    def __init__(self, backend: Backend, calibrator: Calibrator | None = None):
        self.backend = backend
        self.cal = calibrator or Calibrator()

    def _score(self, req: Request) -> tuple[dict[str, list[float]], Scores]:
        ids = list(req.questions)
        items = [build_prompt(req.state, req.questions[i]) for i in ids]
        scores = self.backend.score(items)
        return dict(zip(ids, scores.logits, strict=True)), scores

    def raw_logits(self, req: Request) -> dict[str, list[float]]:
        """Uncalibrated logits per question id. Used by the eval/calibration harness."""
        return self._score(req)[0]

    def evaluate(self, req: Request) -> dict:
        logits, scores = self._score(req)
        answers = {}
        for qid, q in req.questions.items():
            p = self.cal.apply(q.type, logits[qid])
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
