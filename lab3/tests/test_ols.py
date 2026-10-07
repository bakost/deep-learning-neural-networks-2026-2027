"""МНК-оценки, остатки, s², R², интервалы и F-тесты (пункты 2–6)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from lipo_ols import METHODS, fit_ols, fit_ols_data, solve_ols


@pytest.mark.parametrize("method", METHODS)
def test_solvers_agree(data, fit, method):
    b = solve_ols(data.design(), data.y, method)
    ref, *_ = np.linalg.lstsq(data.design(), data.y, rcond=None)
    assert np.allclose(b, ref, atol=1e-9) and np.allclose(fit.b, ref, atol=1e-9)


def test_unknown_method(data):
    with pytest.raises(ValueError):
        solve_ols(data.design(), data.y, "magic")


def test_normal_equations_are_orthogonality(fit):
    """Xᵀe = 0: остатки ортогональны всем регрессорам, в том числе константе (сумма остатков 0)."""
    assert np.max(np.abs(fit.X.T @ fit.residuals)) < 1e-9
    assert abs(fit.residuals.sum()) < 1e-10


def test_residual_decomposition(fit):
    assert np.allclose(fit.fitted + fit.residuals, fit.y)
    assert fit.ssr == pytest.approx(float(fit.residuals @ fit.residuals))
    assert fit.tss == pytest.approx(fit.ssr + float(np.sum((fit.fitted - fit.y.mean()) ** 2)))


def test_variance_and_r2(fit):
    assert fit.df_resid == 57
    assert fit.s2 == pytest.approx(fit.ssr / 57)
    assert fit.s == pytest.approx(np.sqrt(fit.s2))
    assert fit.sigma2_ml == pytest.approx(fit.s2 * 57 / 82)
    assert fit.r2 == pytest.approx(1 - fit.ssr / fit.tss)
    assert fit.r2_uncentered == pytest.approx(float(fit.fitted @ fit.fitted / (fit.y @ fit.y)))
    assert fit.r2_adj < fit.r2 < fit.r2_uncentered < 1


def test_reference_numbers(fit):
    assert fit.ssr == pytest.approx(33.6849, abs=1e-4)
    assert fit.s2 == pytest.approx(0.59096, abs=1e-5)
    assert fit.r2 == pytest.approx(0.83799, abs=1e-5)
    assert fit.r2_adj == pytest.approx(0.76977, abs=1e-5)


def test_standard_errors_and_t(fit):
    cov = fit.s2 * np.linalg.inv(fit.X.T @ fit.X)
    assert np.allclose(fit.se, np.sqrt(np.diag(cov)))
    assert np.allclose(fit.t, fit.b / fit.se)
    assert np.allclose(fit.p_values, 2 * stats.t.sf(np.abs(fit.t), 57))


def test_confidence_intervals(fit):
    ci = fit.conf_int(0.95)
    tq = stats.t.ppf(0.975, 57)
    assert np.allclose(ci[:, 1] - fit.b, tq * fit.se) and np.allclose(fit.b - ci[:, 0], tq * fit.se)
    # интервал не содержит 0 ⇔ p < 0.05
    excludes_zero = (ci[:, 0] > 0) | (ci[:, 1] < 0)
    assert np.array_equal(excludes_zero, fit.p_values < 0.05)
    assert set(np.array(fit.labels)[excludes_zero]) == {"HBA1", "HBA2", "PSA"}
    assert np.all(fit.conf_int(0.99)[:, 1] > ci[:, 1])


def test_ci_coverage_on_synthetic_data():
    """95%-е интервалы накрывают истинное β примерно в 95 % повторений."""
    rng = np.random.default_rng(11)
    n, reps = 40, 2000
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    beta = np.array([0.5, 1.0, -2.0])
    hits = np.zeros(3)
    for _ in range(reps):
        f = fit_ols(X, X @ beta + rng.normal(size=n))
        ci = f.conf_int(0.95)
        hits += (ci[:, 0] <= beta) & (beta <= ci[:, 1])
    assert np.all(np.abs(hits / reps - 0.95) < 0.015)


def test_synthetic_recovers_beta(synthetic):
    _, _, beta, f = synthetic
    assert np.all(np.abs(f.b - beta) < 4 * f.se)
    assert f.s == pytest.approx(0.5, rel=0.15)


def test_overall_f_equals_r2_formula(fit):
    test = fit.overall_f_test()
    k = fit.K - 1
    assert test.F == pytest.approx((fit.r2 / k) / ((1 - fit.r2) / fit.df_resid))
    assert test.df1 == 24 and test.df2 == 57 and test.rejected
    assert test.F == pytest.approx(12.2845, abs=1e-4)


def test_single_restriction_f_is_t_squared(fit):
    k = fit.labels.index("PSA")
    test = fit.zero_test(["PSA"])
    assert test.F == pytest.approx(fit.t[k] ** 2)
    assert test.p == pytest.approx(fit.p_values[k])


def test_f_test_equals_ssr_comparison(data, fit):
    """F для «коэффициенты при группе равны 0» совпадает с формулой через SSR вложенных моделей."""
    drop = ["PSA", "MR"]
    small = fit_ols_data(data, [c for c in data.columns if c not in drop])
    F = ((small.ssr - fit.ssr) / len(drop)) / fit.s2
    assert fit.zero_test(drop).F == pytest.approx(F)


def test_general_linear_hypothesis(fit):
    """H0: β_HBA1 + β_HBA2 = 0 через матрицу R и вектор r."""
    R = np.zeros((1, fit.K))
    R[0, fit.labels.index("HBA1")] = 1.0
    R[0, fit.labels.index("HBA2")] = 1.0
    test = fit.f_test(R, [0.0])
    se = np.sqrt(R @ fit.cov @ R.T)[0, 0]
    assert test.F == pytest.approx(((R @ fit.b)[0] / se) ** 2)


def test_rank_deficiency_is_rejected(data):
    X = np.column_stack([data.design(), data.column("C")])
    with pytest.raises(ValueError, match="ранг"):
        fit_ols(X, data.y)
    with pytest.raises(ValueError, match="больше"):
        fit_ols(data.design()[:20], data.y[:20])


def test_information_criteria(fit):
    assert fit.loglik == pytest.approx(np.sum(stats.norm.logpdf(fit.residuals, scale=np.sqrt(fit.sigma2_ml))))
    assert fit.bic > fit.aic


def test_summary_and_table(fit):
    s = fit.summary()
    assert s["n"] == 82 and s["K"] == 25 and s["t_critical"] == pytest.approx(2.00247, abs=1e-5)
    table = fit.table()
    assert len(table) == 25 and table[0]["label"] == "const"
    assert fit.coefficient("C") == pytest.approx(table[fit.labels.index("C")]["b"])
