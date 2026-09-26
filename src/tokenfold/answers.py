"""Answer math. Logits -> probabilities -> typed answer. Pure numpy, no model here."""

from __future__ import annotations

import numpy as np


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def choice_confidence(probs) -> float:
    """(n * p_max - 1) / (n - 1): 1.0 when all mass on one option, 0.0 when uniform."""
    p = np.asarray(probs, dtype=float)
    n = len(p)
    if n < 2:
        return 1.0
    return float(np.clip((n * p.max() - 1) / (n - 1), 0, 1))


def _round(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def choice_answer(options: list[str], probs) -> dict:
    p = np.asarray(probs, dtype=float)
    return {
        "type": "choice",
        "choice": options[int(p.argmax())],
        "probabilities": {o: _round(v) for o, v in zip(options, p, strict=True)},
        "confidence": _round(choice_confidence(p)),
    }


def score_answer(levels: list, probs) -> dict:
    p = np.asarray(probs, dtype=float)
    idx = np.arange(len(levels))
    return {
        "type": "score",
        "score": _round(float((idx * p).sum())),  # probability-weighted level, can fall between levels
        "legend": {str(i): lv for i, lv in enumerate(levels)},
        "probabilities": {str(i): _round(v) for i, v in enumerate(p)},
        "confidence": _round(choice_confidence(p)),
    }


def noul_answer(probs) -> dict:
    """probs are [p_yes, p_no]."""
    return {"type": "noul", "noul": _round(float(probs[0]))}
