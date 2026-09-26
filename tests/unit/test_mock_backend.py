from tokenfold.backends import MockBackend
from tokenfold.engine import Engine
from tokenfold.schema import parse_request


def test_mock_is_deterministic_and_shaped_like_logprobs():
    items = [("prompt A", [" x", " y", " z"]), ("prompt B", [" yes", " no"])]
    a, b = MockBackend().score(items).logits, MockBackend().score(items).logits
    assert a == b
    assert [len(r) for r in a] == [3, 2]
    assert all(-8.0 < v <= 0.0 for row in a for v in row)


def test_engine_keeps_caller_ids_and_types():
    req = parse_request(
        {
            "state": "s",
            "questions": {
                "dept": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": "desc"}},
                "sev": {"type": "score", "instructions": "?", "criteria": ["lo", "hi"]},
                "urgent": {"type": "noul", "instructions": "?"},
            },
        }
    )
    out = Engine(MockBackend()).evaluate(req)
    assert {k: v["type"] for k, v in out["answers"].items()} == {
        "dept": "choice",
        "sev": "score",
        "urgent": "noul",
    }
    usage = out["usage"]
    assert (usage["questions"], usage["candidates"]) == (3, 6)
    assert usage["input_tokens"] > 0 and usage["output_tokens"] == 0


def test_engine_scores_priors_once_per_question_and_corrects_logits():
    from tokenfold.calibration import Calibrator
    from tokenfold.prompt import CONTENT_FREE_STATE, build_prompt

    class Counting(MockBackend):
        def __init__(self):
            super().__init__()
            self.items = []

        def score(self, items):
            self.items.append(items)
            return super().score(items)

    be = Counting()
    cal = Calibrator({"choice": 1.0}, prior_weights={"choice": 1.0})
    engine = Engine(be, cal)
    body = {
        "state": "s1",
        "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}},
    }
    req = parse_request(body)
    q = req.questions["q"]
    out1 = engine.evaluate(req)
    assert len(be.items) == 1 and len(be.items[0]) == 2  # real prompt + content-free prompt, one batch
    assert be.items[0][1] == build_prompt(CONTENT_FREE_STATE, q)
    # second request with the same question but a new state: prior comes from the cache
    engine.evaluate(parse_request({**body, "state": "s2"}))
    assert len(be.items[1]) == 1
    # the served probabilities are softmax(logits - prior), not softmax(logits)
    import numpy as np

    from tokenfold.answers import softmax

    raw = engine.raw_logits(req)["q"]
    prior = engine.prior_logits({"q": q})["q"]
    expect = softmax(np.array(raw) - np.array(prior))
    got = [out1["answers"]["q"]["probabilities"][k] for k in ("a", "b")]
    assert np.allclose(got, expect, atol=1e-4)
    # without a prior weight, no content-free prompt is scored at all
    be2 = Counting()
    Engine(be2).evaluate(req)
    assert len(be2.items[0]) == 1
