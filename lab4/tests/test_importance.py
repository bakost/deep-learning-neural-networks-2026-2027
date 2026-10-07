"""Вклад регрессоров."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_mlp import Params, ensemble_predict, ols_standardized, permutation_importance, sensitivity


def test_sensitivity_of_linear_regime():
    """Если все нейроны активны, сеть линейна и производная равна (w¹)ᵀw²."""
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(3, 4))
    p = Params(W1, np.full(3, 100.0), rng.normal(size=3), 0.0)
    Z = rng.normal(size=(20, 4))
    s = sensitivity([p], Z)
    assert np.allclose(s["mean"], W1.T @ p.w2)
    assert np.allclose(s["mean_abs"], np.abs(W1.T @ p.w2))


def test_ols_standardized(std):
    Z, t, _ = std
    beta = ols_standardized(Z, t)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(t)), Z]), t, rcond=None)
    assert np.allclose(beta, coef[1:]) and beta.shape == (24,)


def test_permutation_importance_detects_relevant_feature():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(300, 3))
    t = 2 * Z[:, 0]
    imp = permutation_importance(lambda M: 2 * M[:, 0], Z, t, repeats=5, seed=0)
    assert imp[0] == pytest.approx(8.0, rel=0.15) and np.allclose(imp[1:], 0)


def test_ensemble_predict():
    p = Params(np.zeros((1, 2)), np.zeros(1), np.zeros(1), 1.0)
    q = Params(np.zeros((1, 2)), np.zeros(1), np.zeros(1), 3.0)
    assert np.allclose(ensemble_predict([p, q])(np.zeros((4, 2))), 2.0)
