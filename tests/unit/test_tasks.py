"""Task registry and builder, offline: a fake HFRows stands in for the datasets-server API."""

import pytest

from tokenfold.schema import parse_question
from tokenfold.tasks import BY_NAME, TASKS
from tokenfold.tasks.build import build_task


class FakeHF:
    def __init__(self, rows, names):
        self.rows, self.names = rows, names

    def sample(self, dataset, config, split, n, seed, page=20):
        feats = {"label": {"names": self.names, "_type": "ClassLabel"}} if self.names else {}
        return self.rows, feats


def test_every_task_question_is_valid():
    for t in TASKS:
        names = ["oos", "a", "Refund_x", "b?"] if callable(t.criteria) else None
        parse_question(t.name, t.question(names))


def test_task_names_unique_and_types_covered():
    assert len(BY_NAME) == len(TASKS)
    assert {t.type for t in TASKS} == {"choice", "score", "noul"}
    assert any(t.heldout for t in TASKS) and not all(t.heldout for t in TASKS)


def test_build_choice_maps_index_to_key_and_skips_unlabeled():
    t = BY_NAME["ag_news"]
    rows = [
        {"text": "Stocks fell", "label": 2},
        {"text": "hidden", "label": -1},
        {"text": "Goal!", "label": 1},
    ]
    out = build_task(t, FakeHF(rows, list(t.expect_names)), n=10, seed=0)
    assert [(r["state"], r["label"]) for r in out] == [
        ({"article": "Stocks fell"}, "business"),
        ({"article": "Goal!"}, "sports"),
    ]
    assert out[0]["task"] == "ag_news" and out[0]["heldout"] is False


def test_build_noul_label_is_bool():
    t = BY_NAME["rte"]
    rows = [
        {"sentence1": "p", "sentence2": "h", "label": 0},
        {"sentence1": "p", "sentence2": "h", "label": 1},
    ]
    out = build_task(t, FakeHF(rows, list(t.expect_names)), n=10, seed=0)
    assert [r["label"] for r in out] == [True, False]


def test_build_respects_n_and_clips_long_text():
    t = BY_NAME["sst5"]
    rows = [{"text": "x" * 5000, "label": 4}] * 5
    out = build_task(t, FakeHF(rows, None), n=3, seed=0)
    assert len(out) == 3 and len(out[0]["state"]["review"]) < 1600


def test_upstream_label_order_change_fails_loudly():
    t = BY_NAME["ag_news"]
    with pytest.raises(ValueError, match="label names changed"):
        build_task(t, FakeHF([], ["Sports", "World", "Business", "Sci/Tech"]), n=10, seed=0)


def test_stsb_rounds_to_levels():
    t = BY_NAME["stsb"]
    out = build_task(t, FakeHF([{"sentence1": "a", "sentence2": "b", "label": 3.6}], None), n=1, seed=0)
    assert out[0]["label"] == 4


def test_banking_keys_are_normalized():
    t = BY_NAME["banking77"]
    names = ["activate_my_card", "Refund_not_showing_up", "reverted_card_payment?"]
    q = parse_question("q", t.question(names))
    assert q.options == ["activate_my_card", "refund_not_showing_up", "reverted_card_payment"]
    out = build_task(t, FakeHF([{"text": "where is my refund", "label": 1}], names), n=5, seed=0)
    assert out[0]["label"] == "refund_not_showing_up"
