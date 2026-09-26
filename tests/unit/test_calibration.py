"""Option-count temperature: T(n) = T * (n/2)**slope, fit jointly, clamped, backward-compatible JSON."""

import json

import numpy as np

from tokenfold.answers import softmax
from tokenfold.calibration import T_MAX, Calibrator


def rows_needing(scale_for_n, ns, per_n=400, seed=0):
    """Logits that are calibrated once divided by scale_for_n(n): labels are sampled from softmax(g),
    then the logits are sharpened by that scale, so the optimal temperature for n options is the scale."""
    rng = np.random.default_rng(seed)
    rows, ys = [], []
    for n in ns:
        for _ in range(per_n):
            g = rng.normal(0, 2, n)
            ys.append(int(rng.choice(n, p=softmax(g))))
            rows.append(g * scale_for_n(n))
    return rows, ys


def test_fits_slope_when_more_options_need_more_flattening():
    rows, ys = rows_needing(lambda n: 2.0 * (n / 2) ** 0.5, ns=(2, 4, 16))
    cal = Calibrator()
    T = cal.fit("choice", rows, ys)
    assert abs(T - 2.0) < 0.4
    assert abs(cal.slope["choice"] - 0.5) < 0.1
    assert cal.temperature("choice", 16) > cal.temperature("choice", 2)


def test_single_option_count_keeps_slope_zero():
    rows, ys = rows_needing(lambda n: 3.0, ns=(2,))
    cal = Calibrator()
    assert abs(cal.fit("noul", rows, ys) - 3.0) < 0.5
    assert cal.slope["noul"] == 0.0


def test_temperature_is_clamped_and_keeps_argmax():
    cal = Calibrator({"choice": 5.0}, {"choice": 1.0})
    assert cal.temperature("choice", 255) == T_MAX
    z = np.random.default_rng(1).normal(size=255)
    assert cal.apply("choice", z).argmax() == z.argmax()


def test_save_load_roundtrip_and_old_files_without_slopes(tmp_path):
    p = tmp_path / "cal.json"
    Calibrator({"choice": 4.0}, {"choice": 0.3}).save(str(p))
    loaded = Calibrator.load(str(p))
    assert loaded.T["choice"] == 4.0 and loaded.slope["choice"] == 0.3

    p.write_text(json.dumps({"temperatures": {"choice": 4.0}}))
    old = Calibrator.load(str(p))
    assert old.temperature("choice", 2) == old.temperature("choice", 150) == 4.0


def test_prior_weight_is_fit_when_a_prior_bias_is_present():
    """Logits = signal + per-option prior. Subtracting the prior (w=1) should win; w=0 loses NLL."""
    rng = np.random.default_rng(0)
    prior = np.array([0.0, 3.0, -3.0])  # a strong pull toward option 1
    rows, ys, priors = [], [], []
    for _ in range(600):
        g = rng.normal(0, 2, 3)
        ys.append(int(rng.choice(3, p=softmax(g))))
        rows.append(g * 2 + prior)
        priors.append(prior)
    cal = Calibrator()
    cal.fit("choice", rows, ys, priors)
    assert cal.prior_weight["choice"] >= 0.75
    plain = Calibrator()
    plain.fit("choice", rows, ys)  # no priors -> temperature only
    assert plain.prior_weight["choice"] == 0.0
    # the correction can move the argmax; temperature alone never does
    z = [1.0, 2.0, -5.0]
    assert int(np.argmax(cal.apply("choice", z, prior))) == 0
    assert int(np.argmax(plain.apply("choice", z))) == 1


def test_prior_weight_round_trips_and_old_files_load_without_it(tmp_path):
    cal = Calibrator({"noul": 3.0}, prior_weights={"noul": 0.5})
    assert cal.needs_prior
    p = tmp_path / "cal.json"
    cal.save(str(p))
    back = Calibrator.load(str(p))
    assert back.prior_weight["noul"] == 0.5 and back.T["noul"] == 3.0
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"temperatures": {"noul": 3.0}}))
    legacy = Calibrator.load(str(old))
    assert not legacy.needs_prior and list(legacy.apply("noul", [0.0, -1.0], [5.0, 0.0])) == list(
        legacy.apply("noul", [0.0, -1.0])
    )
