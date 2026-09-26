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
