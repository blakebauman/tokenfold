"""Answer-math checks. python tests_answers.py  (or pytest tests_answers.py)"""

import numpy as np

from tokenfold.answers import choice_answer, choice_confidence, noul_answer, score_answer, softmax
from tokenfold.calibration import Calibrator


def test_softmax_sums_to_one_and_temperature_keeps_argmax():
    z = [-1.0, -3.0, -0.5]
    for T in (0.5, 1.0, 5.0):
        p = softmax(z, T)
        assert abs(p.sum() - 1) < 1e-9
        assert p.argmax() == 2
    assert softmax(z, 5.0).max() < softmax(z, 1.0).max()  # higher T flattens


def test_confidence_bounds():
    assert choice_confidence([1, 0, 0]) == 1.0
    assert choice_confidence([1 / 3] * 3) == 0.0
    assert abs(choice_confidence([0.5, 0.25, 0.25]) - 0.25) < 1e-9
    assert choice_confidence([1.0]) == 1.0


def test_choice_answer():
    a = choice_answer(["a", "b"], [0.2, 0.8])
    assert a["choice"] == "b" and a["probabilities"] == {"a": 0.2, "b": 0.8}
    assert abs(a["confidence"] - 0.6) < 1e-9


def test_score_is_expected_level():
    a = score_answer(["lo", "mid", "hi"], [0.0, 0.5, 0.5])
    assert a["score"] == 1.5 and a["legend"] == {"0": "lo", "1": "mid", "2": "hi"}


def test_noul_is_p_yes():
    assert noul_answer([0.7, 0.3]) == {"type": "noul", "noul": 0.7}


def test_calibrator_fits_overconfident_logits_to_T_above_one():
    rng = np.random.default_rng(0)
    rows, ys = [], []
    for _ in range(400):
        y = int(rng.integers(3))
        pred = y if rng.random() < 0.6 else int(rng.integers(3))  # 60%+ accurate
        z = np.full(3, -10.0)
        z[pred] = 0.0  # but claims ~100% confidence
        rows.append(z)
        ys.append(y)
    assert Calibrator().fit("choice", rows, ys) > 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
