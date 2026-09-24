"""Теоретические законы: нормировка, моменты, формулы методички, точный закон GOE."""

from __future__ import annotations

import math

import numpy as np
import pytest

from level_spacing import (
    GOE2_RAW_MEAN,
    WIGNER_SECOND_MOMENT,
    DiscreteDistribution,
    goe2_raw_cdf,
    goe2_raw_pdf,
    goe_limit_cdf,
    goe_limit_pdf,
    normal_mirror_2x2_cdf,
    normal_mirror_2x2_pdf,
    pm1_2x2_distribution,
    poisson_pdf,
    wigner_cdf,
    wigner_pdf,
    wigner_sample,
)
from level_spacing.distributions import NORMAL_MIRROR_2X2_RAW_MEAN, gap_probability

S = np.linspace(0.0, 12.0, 240_001)


def integrate(values: np.ndarray, grid: np.ndarray = S) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


class TestWignerSurmise:
    def test_normalized(self):
        assert integrate(wigner_pdf(S)) == pytest.approx(1.0, abs=1e-9)

    def test_mean_is_one(self):
        assert integrate(S * wigner_pdf(S)) == pytest.approx(1.0, abs=1e-9)

    def test_second_moment(self):
        assert integrate(S**2 * wigner_pdf(S)) == pytest.approx(WIGNER_SECOND_MOMENT, abs=1e-9)

    def test_linear_near_zero(self):
        # ρ(s) ≈ (π/2) s: отталкивание уровней.
        assert wigner_pdf(1e-6) / 1e-6 == pytest.approx(math.pi / 2, rel=1e-9)

    def test_cdf_is_integral_of_pdf(self):
        grid = np.linspace(0, 3, 30_001)
        p = wigner_pdf(grid)
        cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))))
        np.testing.assert_allclose(wigner_cdf(grid), cumulative, atol=1e-8)

    def test_zero_for_negative(self):
        assert wigner_pdf(-1.0) == 0.0 and wigner_cdf(-1.0) == 0.0

    def test_sampler(self):
        x = wigner_sample(400_000, rng=0)
        assert x.mean() == pytest.approx(1.0, abs=0.005)
        assert np.mean(x**2) == pytest.approx(WIGNER_SECOND_MOMENT, abs=0.01)


class TestFormula13:
    """Формулы (13)–(15) методички."""

    def test_normalized(self):
        assert integrate(goe2_raw_pdf(S * 3) * 3) == pytest.approx(1.0, abs=1e-7)

    def test_mean_is_two_sqrt_pi(self):
        grid = np.linspace(0, 40, 400_001)
        mean = integrate(grid * goe2_raw_pdf(grid), grid)
        assert mean == pytest.approx(2 * math.sqrt(math.pi), rel=1e-9)
        assert GOE2_RAW_MEAN == pytest.approx(3.5449077, rel=1e-7)

    def test_rescaling_gives_wigner(self):
        # (15) = (13) после замены s → ⟨s⟩ s и умножения на ⟨s⟩.
        np.testing.assert_allclose(GOE2_RAW_MEAN * goe2_raw_pdf(GOE2_RAW_MEAN * S), wigner_pdf(S),
                                   rtol=1e-12, atol=1e-15)

    def test_cdf(self):
        np.testing.assert_allclose(goe2_raw_cdf(GOE2_RAW_MEAN * S), wigner_cdf(S), atol=1e-14)


class TestGoeLimit:
    """Точный закон GOE при n → ∞ через определитель Фредгольма."""

    grid = np.linspace(0.0, 6.0, 60_001)

    def test_gap_probability_bounds(self):
        values = gap_probability(np.linspace(0, 5, 51))
        assert values[0] == 1.0
        assert np.all(np.diff(values) < 0) and np.all(values > 0)

    def test_gap_probability_derivative_at_zero(self):
        # −E'(0) = средняя плотность уровней = 1.
        h = 1e-4
        assert (1.0 - gap_probability(h)) / h == pytest.approx(1.0, rel=1e-3)

    def test_quadrature_converged(self):
        s = np.linspace(0.1, 6.0, 30)
        np.testing.assert_allclose(gap_probability(s, 30), gap_probability(s, 60), atol=1e-13)

    def test_normalized_and_mean(self):
        p = goe_limit_pdf(self.grid)
        assert integrate(p, self.grid) == pytest.approx(1.0, abs=2e-5)
        assert integrate(self.grid * p, self.grid) == pytest.approx(1.0, abs=2e-5)

    def test_variance_known_value(self):
        # Табличное значение дисперсии расстояний GOE: 0.2855 (Мета).
        p = goe_limit_pdf(self.grid)
        variance = integrate(self.grid**2 * p, self.grid) - 1.0
        assert variance == pytest.approx(0.2855, abs=2e-4)

    def test_small_s_slope(self):
        # p(s) ≈ (π²/6) s при s → 0, у догадки Вигнера (π/2) s.
        assert goe_limit_pdf(0.02) / 0.02 == pytest.approx(math.pi**2 / 6, rel=0.01)

    def test_close_to_wigner_but_not_equal(self):
        diff = np.abs(goe_limit_pdf(self.grid) - wigner_pdf(self.grid))
        assert 0.012 < diff.max() < 0.02

    def test_cdf_consistent_with_pdf(self):
        grid = self.grid
        p = goe_limit_pdf(grid)
        cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))))
        np.testing.assert_allclose(goe_limit_cdf(grid), cumulative, atol=2e-5)


class TestNormalMirror2x2:
    def test_raw_normalized(self):
        grid = np.linspace(0, 20, 200_001)
        total = integrate(normal_mirror_2x2_pdf(grid, normalized=False), grid)
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_raw_mean_closed_form(self):
        grid = np.linspace(0, 20, 200_001)
        mean = integrate(grid * normal_mirror_2x2_pdf(grid, normalized=False), grid)
        assert mean == pytest.approx(NORMAL_MIRROR_2X2_RAW_MEAN, rel=1e-8)

    def test_raw_second_moment(self):
        # ⟨s²⟩ = 4(σ_b² + σ_d²) = 4(1 + 1/2) = 6.
        grid = np.linspace(0, 20, 200_001)
        second = integrate(grid**2 * normal_mirror_2x2_pdf(grid, normalized=False), grid)
        assert second == pytest.approx(6.0, rel=1e-8)

    def test_normalized_mean_one_and_wider_than_wigner(self):
        p = normal_mirror_2x2_pdf(S)
        assert integrate(S * p) == pytest.approx(1.0, abs=1e-8)
        variance = integrate(S**2 * p) - 1.0
        assert variance == pytest.approx(6.0 / NORMAL_MIRROR_2X2_RAW_MEAN**2 - 1.0, rel=1e-6)
        assert variance > WIGNER_SECOND_MOMENT - 1.0 + 0.015

    def test_cdf(self):
        assert normal_mirror_2x2_cdf(0.0) == 0.0
        assert normal_mirror_2x2_cdf(8.0) == pytest.approx(1.0)
        assert 0.4 < normal_mirror_2x2_cdf(1.0) < 0.6


class TestPoisson:
    def test_normalized(self):
        assert integrate(poisson_pdf(S * 3) * 3) == pytest.approx(1.0, abs=1e-6)

    def test_maximal_at_zero(self):
        assert poisson_pdf(0.0) == 1.0


class TestDiscrete:
    def test_sorted_and_moments(self):
        d = DiscreteDistribution([3.0, 1.0], [0.25, 0.75])
        np.testing.assert_array_equal(d.values, [1.0, 3.0])
        assert d.mean == pytest.approx(1.5)
        assert d.variance == pytest.approx(0.75)

    def test_cdf_steps(self):
        d = DiscreteDistribution([0.0, 1.0], [0.5, 0.5])
        np.testing.assert_allclose(d.cdf([-1.0, 0.0, 0.5, 1.0, 2.0]), [0.0, 0.5, 0.5, 1.0, 1.0])

    @pytest.mark.parametrize("values, probs", [([1.0], [0.5]), ([1.0, 2.0], [0.5]), ([], []),
                                               ([1.0, 2.0], [1.5, -0.5])])
    def test_invalid(self, values, probs):
        with pytest.raises(ValueError):
            DiscreteDistribution(values, probs)

    def test_sample(self):
        d = DiscreteDistribution([1.0, 2.0], [0.2, 0.8])
        assert np.mean(d.sample(100_000, rng=0) == 2.0) == pytest.approx(0.8, abs=0.01)


class TestPm1Analytic:
    """Пункт 10: аналитические законы для матриц 2×2 из ±1."""

    def test_mirror(self):
        d = pm1_2x2_distribution("mirror")
        np.testing.assert_allclose(d.values, [2.0, 2.0 * math.sqrt(2.0)])
        np.testing.assert_allclose(d.probabilities, [0.5, 0.5])
        assert d.mean == pytest.approx(1 + math.sqrt(2))
        np.testing.assert_allclose(d.normalized().values,
                                   [2 * math.sqrt(2) - 2, 4 - 2 * math.sqrt(2)], rtol=1e-14)

    def test_sum(self):
        d = pm1_2x2_distribution("sum")
        np.testing.assert_allclose(d.values, [0.0, 4.0, 4.0 * math.sqrt(2.0)])
        np.testing.assert_allclose(d.probabilities, [0.25, 0.5, 0.25])
        assert d.mean == pytest.approx(2 + math.sqrt(2))
        np.testing.assert_allclose(d.normalized().values,
                                   [0.0, 4 - 2 * math.sqrt(2), 4 * math.sqrt(2) - 4], atol=1e-14)

    def test_normalized_means(self):
        for construction in ("mirror", "sum"):
            assert pm1_2x2_distribution(construction).normalized().mean == pytest.approx(1.0)

    def test_unknown(self):
        with pytest.raises(ValueError):
            pm1_2x2_distribution("average")
