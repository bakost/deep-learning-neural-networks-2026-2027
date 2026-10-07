"""Условия применимости МНК."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from lipo_ols import (
    breusch_pagan,
    condition_number,
    cooks_distance,
    durbin_watson,
    fit_ols,
    jarque_bera,
    leverage,
    normality_tests,
    pure_error,
    reset_test,
    studentized_residuals,
    vif,
)


def test_leverage_properties(fit):
    h = leverage(fit.X)
    P = fit.X @ np.linalg.inv(fit.X.T @ fit.X) @ fit.X.T
    assert np.allclose(h, np.diag(P))
    assert h.sum() == pytest.approx(fit.K)
    assert np.allclose(P @ P, P, atol=1e-10)          # проектор идемпотентен


def test_exact_fit_points(data, fit):
    """Единственные носители RCHO, RBr, ROPO3 подогнаны точно: h = 1, e = 0."""
    h = leverage(fit.X)
    exact = {data.names[i] for i in np.where(h > 1 - 1e-8)[0]}
    assert exact == {"Benzaldehyde", "Bromobenzene", "Fosamprenavir"}
    for name in exact:
        assert abs(fit.residuals[data.index(name)]) < 1e-9
    assert np.isnan(studentized_residuals(fit)[data.index("Bromobenzene")])
    assert np.isnan(cooks_distance(fit)[data.index("Bromobenzene")])


def test_studentized_residuals_against_deletion(synthetic):
    """Внешний стьюдентизированный остаток = e_i / (s_(i) √(1−h_i)), s_(i) — без i-го наблюдения."""
    X, y, _, f = synthetic
    t = studentized_residuals(f)
    h = leverage(X)
    for i in (0, 17, 123):
        mask = np.arange(len(y)) != i
        s_i = fit_ols(X[mask], y[mask]).s
        assert t[i] == pytest.approx(f.residuals[i] / (s_i * np.sqrt(1 - h[i])))


def test_cooks_distance_against_deletion(synthetic):
    X, y, _, f = synthetic
    D = cooks_distance(f)
    i = 42
    mask = np.arange(len(y)) != i
    b_i = fit_ols(X[mask], y[mask]).b
    d = b_i - f.b
    assert D[i] == pytest.approx(float(d @ (X.T @ X) @ d) / (f.K * f.s2))


def test_condition_number(fit):
    assert condition_number(fit.X, scale=False) == pytest.approx(np.linalg.cond(fit.X))
    assert condition_number(fit.X) > 30           # сильная мультиколлинеарность
    assert condition_number(np.eye(4)) == pytest.approx(1.0)


def test_vif(data):
    v = vif(data)
    assert v["MR"] > 1000 and v["C"] > 500 and v["HBA2"] > 100
    assert v["RCHO"] < 2
    # VIF_j = j-й диагональный элемент обратной корреляционной матрицы
    corr_inv = np.linalg.inv(np.corrcoef(data.X, rowvar=False))
    assert np.allclose(list(v.values()), np.diag(corr_inv), rtol=1e-6)


def test_normality(fit):
    res = normality_tests(fit.residuals)
    assert res["shapiro"].p > 0.05 and res["jarque_bera"].p > 0.05
    jb = stats.jarque_bera(fit.residuals)
    assert jarque_bera(fit.residuals).statistic == pytest.approx(jb.statistic)


def test_breusch_pagan_detects_heteroscedasticity():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(1, 5, n)
    X = np.column_stack([np.ones(n), x])
    homo = fit_ols(X, 1 + x + rng.normal(size=n))
    hetero = fit_ols(X, 1 + x + rng.normal(size=n) * x**1.5)
    assert breusch_pagan(homo).p > 0.01 and breusch_pagan(hetero).p < 1e-6


def test_reset_detects_nonlinearity():
    rng = np.random.default_rng(4)
    n = 200
    x = rng.uniform(-2, 2, n)
    X = np.column_stack([np.ones(n), x])
    assert reset_test(fit_ols(X, 1 + x + 0.3 * rng.normal(size=n))).p > 0.01
    assert reset_test(fit_ols(X, np.exp(x) + 0.3 * rng.normal(size=n))).p < 1e-6


def test_assumptions_hold_for_lipo(fit):
    assert breusch_pagan(fit).p > 0.05
    assert reset_test(fit).p > 0.05
    assert 1.5 < durbin_watson(fit.residuals) < 2.5


def test_durbin_watson_extremes():
    assert durbin_watson(np.ones(10)) == 0.0
    assert durbin_watson(np.array([1.0, -1.0] * 5)) == pytest.approx(3.6)


def test_pure_error(data, fit):
    pe = pure_error(fit, data)
    assert pe.df_pe == 4 and pe.ss_pe == pytest.approx(1.25552, abs=1e-5)
    assert pe.ss_pe + pe.ss_lof == pytest.approx(fit.ssr)
    assert pe.df_lof == 53 and pe.p > 0.05
    assert pe.sigma_pe == pytest.approx(np.sqrt(1.25552 / 4), abs=1e-5)
