"""Скользящий контроль, перекрёстная проверка, пошаговое исключение."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_ols import (
    backward_elimination,
    final_model,
    fit_ols_data,
    kfold_cv,
    leverage,
    loo_predictions,
    press_statistics,
)


def test_loo_matches_closed_form(fit):
    """y_i − ŷ_(i) = e_i / (1 − h_i) для точек с h_i < 1."""
    pred = loo_predictions(fit.X, fit.y)
    h = leverage(fit.X)
    ok = h < 1 - 1e-8
    assert np.allclose((fit.y - pred)[ok], fit.residuals[ok] / (1 - h[ok]), atol=1e-8)


def test_loo_for_single_carrier(data, fit):
    """Без бромбензола столбец RBr нулевой: прогноз = модель без вклада RBr."""
    pred = loo_predictions(fit.X, fit.y)
    i = data.index("Bromobenzene")
    mask = np.arange(data.n) != i
    j = fit.labels.index("RBr")
    X_reduced = np.delete(fit.X, j, axis=1)
    coef, *_ = np.linalg.lstsq(X_reduced[mask], fit.y[mask], rcond=None)
    assert pred[i] == pytest.approx(X_reduced[i] @ coef)


def test_press_is_worse_than_training(fit):
    press = press_statistics(fit.X, fit.y)
    assert press["press"] > fit.ssr and press["q2"] < fit.r2
    assert press["rmse"] == pytest.approx(np.sqrt(press["press"] / fit.n))


def test_kfold_reproducible(fit):
    a = kfold_cv(fit.X, fit.y, k=5, repeats=20, seed=1)
    b = kfold_cv(fit.X, fit.y, k=5, repeats=20, seed=1)
    assert np.array_equal(a.rmse, b.rmse) and a.rmse.shape == (20,)
    assert a.rmse_mean > fit.rmse and a.rmse_std > 0


def test_kfold_with_n_folds_is_loo(fit):
    cv = kfold_cv(fit.X, fit.y, k=fit.n, repeats=1, seed=0)
    assert cv.rmse[0] == pytest.approx(press_statistics(fit.X, fit.y)["rmse"])
    with pytest.raises(ValueError):
        kfold_cv(fit.X, fit.y, k=1)


def test_backward_elimination(data):
    steps = backward_elimination(data, alpha=0.05)
    assert steps[0].removed is None and len(steps[0].columns) == 24
    assert steps[1].removed == "RCHO"
    assert all(len(b.columns) == len(a.columns) - 1 for a, b in zip(steps, steps[1:]))
    final = final_model(data)
    assert np.all(final.p_values[1:] < 0.05)
    assert set(final.labels[1:]) == {"R3N", "RCOR", "ROR", "C", "HBA1", "HBA2", "PSA"}


def test_reduced_model_generalizes_better(data, fit):
    reduced = final_model(data)
    full_cv = kfold_cv(fit.X, fit.y, repeats=50, seed=5)
    red_cv = kfold_cv(reduced.X, reduced.y, repeats=50, seed=5)
    assert reduced.r2 < fit.r2 and red_cv.rmse_mean < full_cv.rmse_mean


def test_dropped_coefficients_jointly_insignificant(data, fit):
    dropped = [s.removed for s in backward_elimination(data)[1:]]
    assert len(dropped) == 17 and not fit.zero_test(dropped).rejected


def test_elimination_from_custom_start(data):
    steps = backward_elimination(data, start=["C", "PSA", "RCHO"])
    assert steps[-1].columns == ("C", "PSA")
    assert fit_ols_data(data, steps[-1].columns).K == 3
