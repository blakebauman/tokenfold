"""Per-question-type temperature scaling, fit by minimizing NLL on a labeled set. Saved as JSON so the
engine can load it at startup.

The temperature depends on the option count n: T(n) = T * (n/2)**slope, clamped to [0.05, 20]. Raw
logprobs over many options need less flattening than over a few, and one scalar per type can't do both
(on the Qwen3-4B baseline, 3-10 option tasks wanted T~5-7 while dbpedia's 14 and clinc's 151 wanted
T~2.5-3). The slope is only fit when the calibration rows span more than one option count; otherwise it
stays 0 and T(n) = T.

Why temperature (and not Platt/isotonic): whatever T(n) is, it rescales a question's logits uniformly, so
it applies to option sets callers invent at request time. Isotonic/Platt would need per-option-count
fitting.

Prior correction (contextual calibration, Zhao et al. 2021) is available but off by default. The same
question scored against a content-free state (prompt.CONTENT_FREE_STATE) measures each candidate's
state-independent prior, and `prior_weight` * prior is subtracted before the temperature; unlike temperature
this can change the argmax. It was added because the Qwen3-4B baseline pulled score answers to the end
levels and under-predicted "neutral" 15:53, but on that baseline it did not transfer: the NLL-fit weight of
0.25 left test NLL flat and cost 1 point of choice accuracy, and "", "[MASK]", null and an average of them
did no better than "N/A". The content-free prompt has its own semantics (an empty state reads as neutral),
so it does not measure the bias seen on real states. Kept so the hosted model can be checked with
`tokenfold-evaluate --prior-weight fit`; a weight above 0 costs one extra prompt per new question."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .answers import softmax

TYPES = ("choice", "score", "noul")
T_MIN, T_MAX = 0.05, 20.0
SLOPE_MAX = 1.0
PRIOR_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _temperature(T: float, slope: float, n: int) -> float:
    return float(np.clip(T * (n / 2) ** slope, T_MIN, T_MAX))


class Calibrator:
    def __init__(
        self,
        temperatures: dict[str, float] | None = None,
        slopes: dict[str, float] | None = None,
        prior_weights: dict[str, float] | None = None,
    ):
        self.T = {t: 1.0 for t in TYPES}
        self.slope = {t: 0.0 for t in TYPES}
        self.prior_weight = {t: 0.0 for t in TYPES}
        if temperatures:
            self.T.update(temperatures)
        if slopes:
            self.slope.update(slopes)
        if prior_weights:
            self.prior_weight.update(prior_weights)

    @property
    def needs_prior(self) -> bool:
        """Whether the engine must score content-free priors for any type."""
        return any(w > 0 for w in self.prior_weight.values())

    def correct(self, qtype: str, logits, prior=None) -> np.ndarray:
        """Subtract the weighted content-free prior. No prior (or weight 0) leaves the logits as they are."""
        z = np.asarray(logits, dtype=float)
        w = self.prior_weight.get(qtype, 0.0)
        if prior is None or w == 0.0:
            return z
        return z - w * np.asarray(prior, dtype=float)

    def temperature(self, qtype: str, n: int) -> float:
        return _temperature(self.T.get(qtype, 1.0), self.slope.get(qtype, 0.0), n)

    def apply(self, qtype: str, logits, prior=None) -> np.ndarray:
        z = self.correct(qtype, logits, prior)
        return softmax(z, self.temperature(qtype, len(z)))

    @staticmethod
    def _nll(T: float, slope: float, logit_rows: list, labels: list) -> float:
        total = 0.0
        for z, y in zip(logit_rows, labels, strict=True):
            p = softmax(z, _temperature(T, slope, len(z)))
            total -= np.log(max(p[y], 1e-12))
        return total / len(labels)

    def _fit_temperature(self, logit_rows: list, labels: list) -> tuple[float, float, float]:
        """Returns (T, slope, nll) for these logits."""
        res = minimize_scalar(
            self._nll, bounds=(T_MIN, T_MAX), args=(0.0, logit_rows, labels), method="bounded"
        )
        T, slope, nll = float(res.x), 0.0, float(res.fun)
        if len({len(z) for z in logit_rows}) > 1:
            # optimize log T so the search is scale-free; start from the slope-0 fit
            res2 = minimize(
                lambda x: self._nll(math.exp(x[0]), x[1], logit_rows, labels),
                x0=[math.log(T), 0.0],
                bounds=[(math.log(T_MIN), math.log(T_MAX)), (-SLOPE_MAX, SLOPE_MAX)],
                method="L-BFGS-B",
            )
            if res2.fun < res.fun:
                T, slope, nll = math.exp(res2.x[0]), float(res2.x[1]), float(res2.fun)
        return T, slope, nll

    def fit(
        self,
        qtype: str,
        logit_rows: list,
        labels: list,
        priors: list | None = None,
        prior_weights: tuple[float, ...] = PRIOR_WEIGHTS,
    ) -> float:
        """logit_rows: list of logit vectors (ragged ok); labels: index of the correct option; priors: the
        content-free logits per row, or None to fit temperature only. With priors, each weight on the grid
        gets its own temperature fit and the lowest NLL wins. Returns the base temperature T; the fitted
        slope and prior weight are in self.slope[qtype] / self.prior_weight[qtype]."""
        candidates: list[tuple[float, float, float, float]] = []  # (nll, w, T, slope)
        for w in prior_weights if priors is not None else (0.0,):
            rows = logit_rows
            if w and priors is not None:
                rows = [
                    np.asarray(z, float) - w * np.asarray(pr, float)
                    for z, pr in zip(logit_rows, priors, strict=True)
                ]
            T, slope, nll = self._fit_temperature(rows, labels)
            candidates.append((nll, w, T, slope))
        _, w, T, slope = min(candidates, key=lambda c: (round(c[0], 9), c[1]))  # ties go to the smaller w
        self.T[qtype], self.slope[qtype], self.prior_weight[qtype] = T, slope, w
        return T

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"temperatures": self.T, "slopes": self.slope, "prior_weights": self.prior_weight},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> Calibrator:
        with open(path) as f:
            data = json.load(f)
        # files without slopes: T(n) = T; without prior_weights: no prior correction
        return cls(data["temperatures"], data.get("slopes"), data.get("prior_weights"))
