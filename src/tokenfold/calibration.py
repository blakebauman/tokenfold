"""Per-question-type temperature scaling, fit by minimizing NLL on a labeled set. Saved as JSON so the
engine can load it at startup.

The temperature depends on the option count n: T(n) = T * (n/2)**slope, clamped to [0.05, 20]. Raw
logprobs over many options need less flattening than over a few, and one scalar per type can't do both
(on the Qwen3-4B baseline, 3-10 option tasks wanted T~5-7 while dbpedia's 14 and clinc's 151 wanted
T~2.5-3). The slope is only fit when the calibration rows span more than one option count; otherwise it
stays 0 and T(n) = T.

Why temperature only: whatever T(n) is, it rescales a question's logits uniformly, so it preserves the
argmax (the chosen answer never changes) and still applies to option sets callers invent at request
time. Isotonic/Platt would need per-option-count fitting and can flip the argmax."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .answers import softmax

TYPES = ("choice", "score", "noul")
T_MIN, T_MAX = 0.05, 20.0
SLOPE_MAX = 1.0


def _temperature(T: float, slope: float, n: int) -> float:
    return float(np.clip(T * (n / 2) ** slope, T_MIN, T_MAX))


class Calibrator:
    def __init__(self, temperatures: dict[str, float] | None = None, slopes: dict[str, float] | None = None):
        self.T = {t: 1.0 for t in TYPES}
        self.slope = {t: 0.0 for t in TYPES}
        if temperatures:
            self.T.update(temperatures)
        if slopes:
            self.slope.update(slopes)

    def temperature(self, qtype: str, n: int) -> float:
        return _temperature(self.T.get(qtype, 1.0), self.slope.get(qtype, 0.0), n)

    def apply(self, qtype: str, logits) -> np.ndarray:
        return softmax(logits, self.temperature(qtype, len(logits)))

    @staticmethod
    def _nll(T: float, slope: float, logit_rows: list, labels: list) -> float:
        total = 0.0
        for z, y in zip(logit_rows, labels, strict=True):
            p = softmax(z, _temperature(T, slope, len(z)))
            total -= np.log(max(p[y], 1e-12))
        return total / len(labels)

    def fit(self, qtype: str, logit_rows: list, labels: list) -> float:
        """logit_rows: list of logit vectors (ragged ok); labels: index of the correct option.
        Returns the base temperature T; the fitted slope is in self.slope[qtype]."""
        res = minimize_scalar(
            self._nll, bounds=(T_MIN, T_MAX), args=(0.0, logit_rows, labels), method="bounded"
        )
        T, slope = float(res.x), 0.0
        if len({len(z) for z in logit_rows}) > 1:
            # optimize log T so the search is scale-free; start from the slope-0 fit
            res2 = minimize(
                lambda x: self._nll(math.exp(x[0]), x[1], logit_rows, labels),
                x0=[math.log(T), 0.0],
                bounds=[(math.log(T_MIN), math.log(T_MAX)), (-SLOPE_MAX, SLOPE_MAX)],
                method="L-BFGS-B",
            )
            if res2.fun < res.fun:
                T, slope = math.exp(res2.x[0]), float(res2.x[1])
        self.T[qtype], self.slope[qtype] = T, slope
        return T

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"temperatures": self.T, "slopes": self.slope}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Calibrator:
        with open(path) as f:
            data = json.load(f)
        return cls(data["temperatures"], data.get("slopes"))  # files without slopes: T(n) = T
