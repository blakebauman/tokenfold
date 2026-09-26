"""Eval harness on synthetic logits: split/heldout bookkeeping and metric shapes."""

import numpy as np
import pytest

from tokenfold.evaluate import evaluate, load_logits, save_logits, split_by_task
from tokenfold.schema import parse_question

CHOICE = parse_question("q", {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}})
SCORE = parse_question("q", {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]})


def rows_for(task, q, n, heldout=False, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y = int(rng.integers(len(q.options)))
        z = np.full(len(q.options), -8.0)
        z[y if rng.random() < 0.7 else int(rng.integers(len(q.options)))] = 0.0  # overconfident
        out.append({"task": task, "heldout": heldout, "q": q, "y": y, "logits": list(z)})
    return out


def test_split_is_per_task():
    rows = rows_for("a", CHOICE, 40) + rows_for("b", CHOICE, 20)
    fit, test = split_by_task(rows, 0.5, seed=0)
    assert sum(r["task"] == "a" for r in fit) == 20 and sum(r["task"] == "b" for r in fit) == 10
    assert len(fit) + len(test) == 60


def test_split_of_a_task_does_not_depend_on_other_tasks():
    a = rows_for("a", CHOICE, 40)
    for r, i in zip(a, range(40), strict=True):
        r["i"] = i
    alone, _ = split_by_task(a, 0.5, seed=0)
    mixed, _ = split_by_task(rows_for("z", CHOICE, 30) + a, 0.5, seed=0)
    assert [r["i"] for r in alone] == [r["i"] for r in mixed if r["task"] == "a"]


def test_heldout_rows_never_fit_temperature():
    base = rows_for("seen", CHOICE, 100)
    held = rows_for("held", CHOICE, 100, heldout=True, seed=1)
    report, _ = evaluate(base + held)
    assert report["per_type"]["choice"]["fit_n"] == 50  # only the non-heldout calibration half
    assert report["per_task"]["held"]["heldout"] is True
    assert report["per_type"]["choice"]["T"] > 1.0  # overconfident logits -> flatten


def test_calibration_improves_nll_and_score_reports_mae():
    report, cal = evaluate(rows_for("s", SCORE, 200))
    m = report["per_task"]["s"]
    assert m["after"]["nll"] < m["before"]["nll"]
    assert "mae" in m["after"] and "mae" not in report["per_type"].get("choice", {}).get("after", {})
    assert abs(cal.T["score"] - report["per_type"]["score"]["T"]) < 1e-3


def test_saved_logits_match_by_key_and_may_cover_more_rows(tmp_path):
    def row(text):
        return {"state": text, "question": {"type": "noul", "instructions": "?"}, "logits": [0.0, -1.0]}

    p = tmp_path / "logits.json"
    save_logits(str(p), "m", [row("a"), row("b"), row("c")])
    subset = [{k: v for k, v in row(t).items() if k != "logits"} for t in ("c", "a")]
    assert load_logits(str(p), subset) == "m" and subset[0]["logits"] == [0.0, -1.0]

    with pytest.raises(SystemExit, match="1 of 1"):
        load_logits(str(p), [{"state": "new", "question": {"type": "noul", "instructions": "?"}}])
