"""Полный перебор матриц из ±1 и его согласие с аналитикой пункта 10."""

from __future__ import annotations

import math

import numpy as np
import pytest

from level_spacing import (
    enumerate_pm1_matrices,
    exact_pm1_spacing,
    pm1_2x2_distribution,
    run_study,
)
from level_spacing.exact import group_atoms


class TestEnumeration:
    @pytest.mark.parametrize("n, construction, count", [(2, "mirror", 8), (2, "sum", 16),
                                                         (3, "mirror", 64), (4, "mirror", 1024)])
    def test_counts(self, n, construction, count):
        assert enumerate_pm1_matrices(n, construction).shape == (count, n, n)

    @pytest.mark.parametrize("construction", ["mirror", "sum"])
    def test_all_distinct_and_symmetric(self, construction):
        h = enumerate_pm1_matrices(3, construction)
        assert np.array_equal(h, np.swapaxes(h, -1, -2))
        if construction == "mirror":   # у sum разные A могут дать одну и ту же H
            assert np.unique(h.reshape(len(h), -1), axis=0).shape[0] == len(h)

    def test_mirror_values(self):
        assert set(np.unique(enumerate_pm1_matrices(3, "mirror"))) == {-1.0, 1.0}

    def test_too_large(self):
        with pytest.raises(ValueError, match="невозможен"):
            enumerate_pm1_matrices(6, "mirror")     # 2^21 матриц

    def test_unknown_construction(self):
        with pytest.raises(ValueError):
            enumerate_pm1_matrices(2, "average")


class TestGroupAtoms:
    def test_merges_rounding_noise(self):
        law = group_atoms(np.array([2.0, 2.0 + 1e-13, 2.0 - 1e-13, 5.0]))
        np.testing.assert_allclose(law.values, [2.0, 5.0])
        np.testing.assert_allclose(law.probabilities, [0.75, 0.25])

    def test_zero_snapped(self):
        assert group_atoms(np.array([3e-15, 1.0])).values[0] == 0.0

    def test_empty(self):
        with pytest.raises(ValueError):
            group_atoms(np.array([]))


class TestAnalyticVsEnumeration:
    """Аналитический вывод для 2×2 совпадает с полным перебором."""

    @pytest.mark.parametrize("construction", ["mirror", "sum"])
    def test_2x2(self, construction):
        analytic = pm1_2x2_distribution(construction)
        exact = exact_pm1_spacing(2, construction)
        np.testing.assert_allclose(exact.values, analytic.values, atol=1e-12)
        np.testing.assert_allclose(exact.probabilities, analytic.probabilities)

    def test_mirror_2x2_eigenvalues(self):
        # a = c: спектр {a − 1, a + 1}; a ≠ c: спектр {−√2, √2}.
        ev = np.linalg.eigvalsh(enumerate_pm1_matrices(2, "mirror"))
        spectra = {tuple(np.round(row, 12)) for row in ev}
        r2 = round(math.sqrt(2), 12)
        assert spectra == {(0.0, 2.0), (-2.0, 0.0), (-r2, r2)}


class TestLargerExact:
    def test_mirror_4x4_degenerate_probability(self):
        # Центральная пара 4×4 из ±1 вырождена ровно в 9/64 случаев.
        assert exact_pm1_spacing(4, "mirror").probability_of(0.0) == pytest.approx(9 / 64)

    def test_sum_4x4_has_zero_atom(self):
        law = exact_pm1_spacing(4, "sum")
        assert law.probability_of(0.0) == pytest.approx(9168 / 65536)
        assert law.values.size == 40

    @pytest.mark.parametrize("name, construction", [("pm1", "mirror"), ("pm1-sum", "sum")])
    @pytest.mark.parametrize("n", [2, 4])
    def test_monte_carlo_matches_exact(self, name, construction, n):
        # Частоты атомов в Монте-Карло совпадают с точными вероятностями (χ²).
        law = exact_pm1_spacing(n, construction)
        study = run_study(name, n, 40_000, seed=11)
        counts = np.array([np.sum(np.abs(study.raw - v) <= 1e-9) for v in law.values])
        assert counts.sum() == study.size                 # других значений не бывает
        expected = law.probabilities * study.size
        mask = expected >= 5
        chi2 = np.sum((counts[mask] - expected[mask]) ** 2 / expected[mask])
        dof = mask.sum() - 1
        assert chi2 < dof + 5 * math.sqrt(2 * dof)
