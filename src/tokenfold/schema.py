"""Request/response contract. Mirrors the TypeSafe wire format:
state + {id: question} in, {id: answer} out. Validation errors raise ValidationError (-> HTTP 422)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Json = str | dict | list | None
MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = (2, 10)


class ValidationError(ValueError):
    pass


@dataclass
class Choice:
    instructions: Json
    criteria: dict[str, Json]  # option -> description (None allowed)
    type: str = "choice"

    def validate(self, qid: str):
        if not isinstance(self.criteria, dict) or not self.criteria:
            raise ValidationError(f"{qid}: choice.criteria must be a non-empty map")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValidationError(f"{qid}: choice has {len(self.criteria)} options, max {MAX_CHOICE_OPTIONS}")
        if len(self.criteria) < 2:
            raise ValidationError(f"{qid}: choice needs at least 2 options")

    @property
    def options(self) -> list[str]:
        return list(self.criteria)


@dataclass
class Score:
    instructions: Json
    criteria: list[Json]  # ordered level descriptions
    type: str = "score"

    def validate(self, qid: str):
        lo, hi = SCORE_LEVELS
        if not isinstance(self.criteria, list) or not (lo <= len(self.criteria) <= hi):
            raise ValidationError(f"{qid}: score.criteria must be a list of {lo}-{hi} levels")

    @property
    def options(self) -> list[str]:
        return [str(i) for i in range(len(self.criteria))]


@dataclass
class Noul:
    instructions: Json
    criteria: dict[str, Json] | None = None  # optional {"true": ..., "false": ...}
    type: str = "noul"

    def validate(self, qid: str):
        if self.criteria is not None:
            extra = set(self.criteria) - {"true", "false"}
            if extra:
                raise ValidationError(
                    f"{qid}: noul.criteria only accepts 'true'/'false', got {sorted(extra)}"
                )

    @property
    def options(self) -> list[str]:
        return ["yes", "no"]


Question = Choice | Score | Noul


@dataclass
class Request:
    state: Json
    questions: dict[str, Question]
    model: str = "local-latest"


def parse_question(qid: str, raw: Any) -> Question:
    if not isinstance(raw, dict) or "type" not in raw or "instructions" not in raw:
        raise ValidationError(f"{qid}: question needs 'type' and 'instructions'")
    t = raw["type"]
    if t == "choice":
        if "criteria" not in raw:
            raise ValidationError(f"{qid}: choice requires criteria")
        q = Choice(raw["instructions"], raw["criteria"])
    elif t == "score":
        if "criteria" not in raw:
            raise ValidationError(f"{qid}: score requires criteria")
        q = Score(raw["instructions"], raw["criteria"])
    elif t == "noul":
        q = Noul(raw["instructions"], raw.get("criteria"))
    else:
        raise ValidationError(f"{qid}: unknown type {t!r}")
    q.validate(qid)
    return q


def parse_request(body: Any) -> Request:
    if not isinstance(body, dict):
        raise ValidationError("body must be an object")
    for k in ("state", "questions"):
        if k not in body:
            raise ValidationError(f"missing required field '{k}'")
    if not isinstance(body["questions"], dict) or not body["questions"]:
        raise ValidationError("questions must be a non-empty map")
    qs = {qid: parse_question(qid, raw) for qid, raw in body["questions"].items()}
    return Request(state=body["state"], questions=qs, model=body.get("model", "local-latest"))
