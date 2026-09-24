"""Гистограммы и критерии согласия."""

from __future__ import annotations

import numpy as np
import pytest

from level_spacing import (
    chi2_test,
    histogram,
    ks_statistic,
    ks_test,
    l1_distance,
    moments,
    small_s_exponent,
    wigner_cdf,
    wigner_sample,
)
from level_spacing.distributions import poisson_pdf, wigner_pdf
from level_spacing.stats import (
    degenerate_fraction,
    ks_null_distribution,
    ks_pvalue_normalized,
    small_s_exponent_of_pdf,
    uniform_bins,
)


def EXP_CDF(s):
    return 1.0 - np.exp(-np.maximum(s, 0.0))


class TestHistogram:
    def test_density_integrates_to_one(self):
        x = wigner_sample(10_000, rng=0)
        h = histogram(x, uniform_bins(0.1, 6.0))
        assert np.sum(h.density * h.widths) == pytest.approx(1.0)

    def test_counts_outside_range_reduce_mass(self):
        h = histogram(np.array([0.5, 1.5, 10.0]), uniform_bins(1.0, 2.0))
        assert h.total == 3
        np.testing.assert_allclose(h.density, [1 / 3, 1 / 3])

    def test_expected_density_is_bin_average(self):
        h = histogram(np.array([0.1]), uniform_bins(0.5, 1.0))
        np.testing.assert_allclose(h.expected_density(lambda s: np.clip(s, 0, 1)), [1.0, 1.0])

    def test_errors_are_poisson(self):
        h = histogram(np.full(100, 0.5), uniform_bins(1.0, 1.0))
        assert h.errors[0] == pytest.approx(np.sqrt(100) / 100)

    def test_bad_bins(self):
        with pytest.raises(ValueError):
            uniform_bins(0.0)


class TestKolmogorovSmirnov:
    def test_statistic_matches_scipy(self):
        x = wigner_sample(3000, rng=1)
        assert ks_statistic(x, wigner_cdf) == pytest.approx(ks_test(x, wigner_cdf)[0], abs=1e-12)

    def test_true_law_small_distance(self):
        x = wigner_sample(100_000, rng=2)
        assert ks_statistic(x, wigner_cdf) < 1.5 / np.sqrt(x.size)

    def test_wrong_law_detected(self):
        x = wigner_sample(10_000, rng=3)
        assert ks_statistic(x, EXP_CDF) > 0.1

    def test_null_distribution_is_below_kolmogorov(self):
        # Нормировка на выборочное среднее уменьшает D: медиана D√M ≈ 0.66 < 0.83.
        null = ks_null_distribution(wigner_cdf)
        assert 0.6 < np.median(null) < 0.72
        assert np.all(np.diff(null) >= 0)

    def test_normalized_pvalue_uniform_under_null(self):
        rng = np.random.default_rng(4)
        pvalues = []
        for _ in range(60):
            x = wigner_sample(3000, rng)
            x = x / x.mean()
            pvalues.append(ks_pvalue_normalized(ks_statistic(x, wigner_cdf), x.size, wigner_cdf))
        assert 0.3 < np.mean(pvalues) < 0.7

    def test_normalized_pvalue_small_for_wrong_law(self):
        x = np.random.default_rng(5).exponential(size=5000)
        assert ks_pvalue_normalized(ks_statistic(x, wigner_cdf), x.size, wigner_cdf) < 0.002


class TestChiSquare:
    def test_true_law_accepted(self):
        x = wigner_sample(50_000, rng=6)
        chi2, dof, p = chi2_test(x, wigner_cdf, uniform_bins(0.1, 4.0), ddof=0)
        assert p > 0.001 and chi2 == pytest.approx(dof, rel=0.5)

    def test_wrong_law_rejected(self):
        x = wigner_sample(50_000, rng=7)
        assert chi2_test(x, EXP_CDF, uniform_bins(0.1, 4.0))[2] < 1e-10

    def test_bins_merged_in_tail(self):
        x = wigner_sample(1000, rng=8)
        _, dof, _ = chi2_test(x, wigner_cdf, uniform_bins(0.05, 4.0))
        assert dof < 80 - 1


class TestOtherMeasures:
    def test_l1_small_for_true_law(self):
        x = wigner_sample(200_000, rng=9)
        assert l1_distance(histogram(x, uniform_bins(0.1, 4.0)), wigner_cdf) < 0.02

    def test_l1_large_for_disjoint(self):
        h = histogram(np.full(100, 3.5), uniform_bins(0.1, 4.0))
        assert l1_distance(h, lambda s: np.clip(s, 0.0, 1.0)) == pytest.approx(2.0)

    def test_moments(self):
        m = moments(np.array([0.0, 2.0]))
        assert m["mean"] == 1.0 and m["variance"] == 1.0 and m["second_moment"] == 2.0

    def test_degenerate_fraction(self):
        assert degenerate_fraction(np.array([0.0, 1e-12, 0.5, 1.0])) == 0.5


class TestSmallSExponent:
    def test_reference_values(self):
        # Та же оценка для самих законов: не ровно 2 и 1 из-за поправок к степени.
        assert small_s_exponent_of_pdf(wigner_pdf) == pytest.approx(1.984, abs=0.001)
        assert small_s_exponent_of_pdf(poisson_pdf) == pytest.approx(0.952, abs=0.001)

    def test_wigner_gives_two(self):
        assert small_s_exponent(wigner_sample(400_000, rng=10)) == pytest.approx(1.984, abs=0.05)

    def test_poisson_gives_one(self):
        x = np.random.default_rng(11).exponential(size=400_000)
        assert small_s_exponent(x) == pytest.approx(0.952, abs=0.02)

    def test_pure_power_law_exact(self):
        # F(s) = (s/0.2)^3 на [0, 0.2]: оценка Хилла несмещённая.
        x = 0.2 * np.random.default_rng(12).random(200_000) ** (1 / 3)
        assert small_s_exponent(x) == pytest.approx(3.0, abs=0.03)

    def test_zeros_ignored(self):
        x = np.concatenate([np.zeros(500), wigner_sample(100_000, rng=13)])
        assert small_s_exponent(x) == pytest.approx(2.0, abs=0.1)

    def test_none_without_small_values(self):
        assert small_s_exponent(np.full(1000, 1.0)) is None
