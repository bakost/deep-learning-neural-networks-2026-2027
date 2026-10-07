"""Вклад регрессоров и пары молекул."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_ols import (
    contribution_table,
    drop_one_r2,
    fit_ols_data,
    matched_pairs,
    pairwise_correlations,
    standardized_coefficients,
)


def test_standardized_equals_fit_on_zscores(data, fit):
    z = (data.X - data.X.mean(0)) / data.X.std(0, ddof=1)
    zy = (data.y - data.y.mean()) / data.y.std(ddof=1)
    from lipo_ols import fit_ols

    zfit = fit_ols(np.column_stack([np.ones(data.n), z]), zy)
    std = standardized_coefficients(fit, data)
    assert np.allclose(list(std.values()), zfit.b[1:])
    assert abs(zfit.b[0]) < 1e-12


def test_drop_one_matches_refit(data, fit):
    drop = drop_one_r2(fit)
    for name in ("PSA", "C", "RCHO"):
        smaller = fit_ols_data(data, [c for c in data.columns if c != name])
        assert drop[name] == pytest.approx(fit.r2 - smaller.r2)


def test_largest_contributions(data, fit):
    std = standardized_coefficients(fit, data)
    top = sorted(std, key=lambda k: -abs(std[k]))[:5]
    assert set(top) == {"C", "HBA2", "PSA", "HBA1", "MR"}
    assert std["C"] > 0 and std["PSA"] < 0


def test_pairwise_correlations(data):
    r = pairwise_correlations(data)
    assert max(r, key=lambda k: abs(r[k])) == "C"
    assert r["C"] ** 2 == pytest.approx(0.50, abs=0.005)


def test_contribution_table(data, fit):
    rows = contribution_table(fit, data)
    assert len(rows) == 24
    for row in rows:
        assert row["beta_low"] <= row["beta"] <= row["beta_high"]


def test_matched_pairs(data, fit):
    pairs = {p.child: p for p in matched_pairs(fit, data)}
    phenol = pairs["Phenol"]
    assert phenol.delta_exp == pytest.approx(-0.67)
    assert phenol.delta_model == pytest.approx(sum(phenol.terms.values()))
    assert phenol.delta_model == pytest.approx(fit.fitted[data.index("Phenol")] - fit.fitted[data.index("Benzene")])
    assert pairs["Bromobenzene"].exact_fit and not pairs["Chlorobenzene"].exact_fit
    assert set(pairs["Chlorobenzene"].terms) == {"RCl", "MR"}
