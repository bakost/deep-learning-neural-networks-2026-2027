"""Эталонные решения и границы устойчивости."""

from __future__ import annotations

import numpy as np
import pytest

from sgd_lab import MODELS, X, Y, design_matrix, linear_solution, loo_press, minibatch_lr_bound, stability
from sgd_lab.report import baselines, classify


@pytest.mark.parametrize("key", ["linear", "quadratic"])
def test_least_squares_matches_numpy(key):
    model = MODELS[key]
    w = linear_solution(model, X, Y)
    assert np.allclose(w, np.polyfit(X, Y, 1 if key == "linear" else 2))
    assert np.allclose(model(w, X), design_matrix(model, X) @ w)


def test_ridge_shrinks_weights_not_bias():
    model = MODELS["quadratic"]
    w0, w1 = linear_solution(model, X, Y), linear_solution(model, X, Y, 1.0)
    assert np.sum(w1[:2] ** 2) < np.sum(w0[:2] ** 2)
    big = linear_solution(model, X, Y, 1e9)
    assert np.allclose(big[:2], 0, atol=1e-7) and big[2] == pytest.approx(Y.mean())


def test_stability_numbers():
    st = stability(MODELS["quadratic"], X)
    assert st["kappa"] == pytest.approx(278.37, rel=1e-3)
    assert st["lr_opt"] < st["lr_max"] and st["rate_opt"] == pytest.approx((st["kappa"] - 1) / (st["kappa"] + 1))
    # штраф улучшает обусловленность
    assert stability(MODELS["quadratic"], X, 0.1)["kappa"] < st["kappa"]


def test_minibatch_bound_decreases_with_batch():
    b = [minibatch_lr_bound(MODELS["linear"], X, m) for m in (1, 2, 4, 12)]
    assert b == sorted(b) and b[-1] == pytest.approx(stability(MODELS["linear"], X)["lr_max"])


def test_loo_press_bruteforce():
    model = MODELS["linear"]
    errs = []
    for i in range(12):
        keep = np.arange(12) != i
        c = np.polyfit(X[keep], Y[keep], 1)
        errs.append((np.polyval(c, X[i]) - Y[i]) ** 2)
    assert loo_press(model, X, Y) == pytest.approx(sum(errs))


def test_classify_outcomes():
    base = baselines()
    sse = np.array([0.0645, 0.08, 0.13, base["linear"], 0.5, base["constant"], base["zero"], np.inf])
    assert classify(sse, 0.0644, base).tolist() == [
        "global", "better_quad", "between", "linear", "worse", "constant", "dead", "dead"]
