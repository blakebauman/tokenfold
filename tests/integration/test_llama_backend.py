"""Real-model check of the llama.cpp backend's KV-cache reuse. Opt-in: loads a multi-GB model.
TOKENFOLD_TEST_LLAMA=1 uv run pytest tests/integration/test_llama_backend.py"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENFOLD_TEST_LLAMA"), reason="set TOKENFOLD_TEST_LLAMA=1"
)


def test_cached_scores_match_cold_scores_with_same_batch_shape():
    from tokenfold.backends import LlamaCppBackend

    be = LlamaCppBackend()
    prompt = "Question: is the sky blue on a clear day?\nANSWER:"
    cands = [" yes", " no", " maybe_later"]
    scores = be.score([(prompt, cands)])
    cached = scores.logits[0]
    # the prompt is prefilled once; each later candidate only evaluates its own tokens
    assert scores.input_tokens < len(cands) * len(be._tok(prompt))
    for c, want in zip(cands, cached, strict=True):
        # cold cache, but prompt and candidate evaluated as separate batches like the cached path;
        # different batch shapes change quantized-kernel numerics, so compare like with like
        be.llm.n_tokens = 0
        be.llm.eval(be._tok(prompt))
        assert abs(be.score([(prompt, [c])]).logits[0][0] - want) < 1e-4
    assert cached[0] > cached[1]
