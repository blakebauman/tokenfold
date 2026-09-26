"""vLLM backend against a fake /completions endpoint: continuation scoring and billed-token usage."""

import pytest

from tokenfold.backends import VLLMLogprobBackend

# Fake tokenizer: one token per whitespace-separated word, logprob -0.1 for prompt words and -1.0
# per word of a candidate starting with "no" (so " yes" beats " no").
LP = {"yes": -0.5, "no": -2.0}


def _entry(word: str) -> dict:
    return {"1": {"logprob": LP.get(word, -0.1), "rank": 1, "decoded_token": word}}


class FakeCompletions:
    def __init__(self, report_usage: bool):
        self.report_usage = report_usage
        self.requests: list[list[str]] = []

    def __call__(self, url, json, headers, timeout):
        self.requests.append(json["prompt"])
        choices = []
        for i, text in enumerate(json["prompt"]):
            words = text.split()
            choices.append({"index": i, "prompt_logprobs": [None] + [_entry(w) for w in words[1:]]})
        body = {"choices": choices}
        if self.report_usage:
            body["usage"] = {
                "prompt_tokens": 100 * len(json["prompt"]),  # distinct from the word count on purpose
                "completion_tokens": len(json["prompt"]),
            }
        return type("R", (), {"json": lambda self: body, "raise_for_status": lambda self: None})()


@pytest.fixture
def backend():
    return VLLMLogprobBackend(base_url="http://fake/v1", model="m", batch_size=2)


def test_scores_the_continuation_only(backend, monkeypatch):
    monkeypatch.setattr("requests.post", FakeCompletions(report_usage=True))
    scores = backend.score([("is it ANSWER:", [" yes", " no"])])
    assert scores.logits == [[-0.5, -2.0]]


def test_usage_is_the_servers_billed_count_over_every_request(backend, monkeypatch):
    fake = FakeCompletions(report_usage=True)
    monkeypatch.setattr("requests.post", fake)
    scores = backend.score([("a ANSWER:", [" yes", " no"]), ("b ANSWER:", [" yes", " no", " maybe"])])
    sent = sum(len(r) for r in fake.requests)
    assert sent == 2 + 5  # the boundary pass over the bare prompts is billed too
    assert len(fake.requests) == 1 + 3  # batch_size=2: one boundary request, three candidate requests
    assert (scores.input_tokens, scores.output_tokens) == (100 * sent, sent)


def test_usage_falls_back_to_counting_prompt_logprobs(backend, monkeypatch):
    monkeypatch.setattr("requests.post", FakeCompletions(report_usage=False))
    scores = backend.score([("a ANSWER:", [" yes", " no"])])
    # boundary: "a ANSWER:" = 2 tokens; candidates: 3 tokens each; one generated token per prompt
    assert (scores.input_tokens, scores.output_tokens) == (2 + 3 + 3, 3)
