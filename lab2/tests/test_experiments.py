"""Главные утверждения отчёта, проверенные автоматически.

Каждый тест — один вывод лабораторной работы. Если после изменения кода
какой-то вывод перестаёт выполняться, это сразу видно.
"""

from __future__ import annotations

import numpy as np
import pytest

from level_spacing import (
    GOE2_RAW_MEAN,
    WIGNER_VARIANCE,
    ks_by_ensemble_size,
    pair_scan,
    pooled_spacings,
    run_study,
    simulate_eigenvalues,
)
from level_spacing.distributions import wigner_cdf
from level_spacing.experiments import ks_noise_constant, study_rng
from level_spacing.stats import ks_statistic

M = 100_000


@pytest.fixture(scope="module")
def goe():
    return {n: run_study("goe", n, M, seed=2026) for n in (2, 4, 16)}


class TestGoe:
    def test_formula_14_mean(self, goe):
        # ⟨s⟩ = 2√π для 2×2 до нормировки.
        s = goe[2]
        assert abs(s.raw_mean - GOE2_RAW_MEAN) < 4 * s.raw.std() / np.sqrt(s.size)

    def test_2x2_is_exactly_wigner(self, goe):
        c = goe[2].compare("wigner")
        assert c["ks_p"] > 0.01 and c["chi2_p"] > 0.001

    def test_16x16_deviates_from_wigner(self, goe):
        # При M = 10⁵ отличие догадки Вигнера от точного закона уже заметно.
        assert goe[16].compare("wigner")["ks_p"] < 0.01

    def test_16x16_matches_exact_goe_limit(self, goe):
        assert goe[16].compare("goe-limit")["ks_p"] > 0.01

    def test_variance_grows_to_goe_limit(self, goe):
        v2, v16 = goe[2].normalized.var(), goe[16].normalized.var()
        assert v2 == pytest.approx(WIGNER_VARIANCE, abs=0.005)
        assert v16 == pytest.approx(0.2855, abs=0.005)

    def test_level_repulsion(self, goe):
        # Около нуля F(s) ∝ s², то есть ρ(s) ∝ s.
        for study in goe.values():
            assert study.summary()["small_s_exponent"] == pytest.approx(2.0, abs=0.15)
            assert np.mean(study.normalized < 0.1) < 0.02   # у Пуассона было бы 0.095

    def test_far_from_poisson(self, goe):
        for study in goe.values():
            assert study.compare("poisson")["ks_d"] > 0.1

    def test_summary_keys(self, goe):
        summary = goe[4].summary()
        for key in ("raw_mean", "variance", "wigner_ks_d", "goe-limit_ks_p", "poisson_l1",
                    "small_s_exponent", "degenerate_fraction"):
            assert key in summary


class TestDiscreteEnsembles:
    def test_pm1_2x2_no_small_spacings(self):
        s = run_study("pm1", 2, 20_000, seed=1).normalized
        assert s.min() == pytest.approx(2 * np.sqrt(2) - 2, abs=0.01)

    def test_pm1_sum_2x2_degenerate_quarter(self):
        study = run_study("pm1-sum", 2, 40_000, seed=1)
        assert np.mean(study.raw <= 1e-9) == pytest.approx(0.25, abs=0.01)

    @pytest.mark.parametrize("name", ["pm1", "pm1-sum"])
    def test_universality_16x16(self, name):
        # Для 16×16 даже матрицы из ±1 дают почти распределение GOE.
        study = run_study(name, 16, 50_000, seed=3)
        assert study.compare("goe-limit")["ks_d"] < 0.01
        assert study.normalized.var() == pytest.approx(0.2855, abs=0.01)

    def test_normal_mirror_2x2_not_wigner(self):
        study = run_study("normal-mirror", 2, 100_000, seed=4)
        assert study.compare("wigner")["ks_p"] < 0.01
        assert study.normalized.var() == pytest.approx(0.2916, abs=0.005)


class TestKsVsSize:
    def test_decreases_for_exact_law(self, goe):
        data = ks_by_ensemble_size(goe[2].raw, [1000, 10_000, 100_000])
        medians = [np.median(data[m]) for m in (1000, 10_000, 100_000)]
        assert medians[0] > medians[1] > medians[2]

    def test_blocks_limited(self, goe):
        assert ks_by_ensemble_size(goe[2].raw, [1000], max_blocks=5)[1000].size == 5

    def test_too_large_block(self, goe):
        with pytest.raises(ValueError):
            ks_by_ensemble_size(goe[2].raw, [10 * M])

    def test_noise_constant(self):
        assert 0.6 < ks_noise_constant(size=2000, repeats=100) < 0.75


class TestPairChoice:
    @pytest.fixture(scope="class")
    @classmethod
    def eigvals(cls):
        return simulate_eigenvalues("goe", 16, 20_000, study_rng(5, "goe", 16))

    def test_mean_spacing_smallest_in_center(self, eigvals):
        scan = pair_scan(eigvals)
        assert np.argmin(scan["mean"]) + 1 in (7, 8, 9)
        assert scan["mean"][0] > 1.5 * scan["mean"][7]

    def test_symmetric_spectrum(self, eigvals):
        scan = pair_scan(eigvals)
        np.testing.assert_allclose(scan["mean"], scan["mean"][::-1], rtol=0.02)

    def test_unfolding_helps(self, eigvals):
        raw = ks_statistic(pooled_spacings(eigvals, unfold=False), wigner_cdf)
        unfolded = ks_statistic(pooled_spacings(eigvals, unfold=True), wigner_cdf)
        assert unfolded < raw

    def test_pooled_sizes(self, eigvals):
        assert pooled_spacings(eigvals).size == eigvals.shape[0] * 15
        assert pooled_spacings(eigvals).mean() == pytest.approx(1.0)


class TestStudyRng:
    def test_independent_streams(self):
        a = study_rng(1, "goe", 4).random()
        b = study_rng(1, "goe", 16).random()
        c = study_rng(1, "pm1", 4).random()
        assert len({a, b, c}) == 3

    def test_reproducible(self):
        assert study_rng(7, "goe", 4).random() == study_rng(7, "goe", 4).random()

    def test_run_study_reproducible(self):
        a = run_study("goe", 4, 100, seed=3).raw
        b = run_study("goe", 4, 100, seed=3).raw
        np.testing.assert_array_equal(a, b)

    def test_bad_compare_reference(self, goe):
        with pytest.raises(ValueError):
            goe[2].compare("gue")
