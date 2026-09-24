"""Ортогональность ансамбля: распределение Qᵀ H Q совпадает с распределением H."""

from __future__ import annotations

import numpy as np
import pytest

from level_spacing import sample_matrices
from level_spacing.ensembles import random_orthogonal, rotation_2x2
from level_spacing.experiments import orthogonal_invariance, rotate, variance_vs_angle


class TestRandomOrthogonal:
    @pytest.mark.parametrize("n", [1, 2, 4, 16])
    def test_is_orthogonal(self, n):
        q = random_orthogonal(n, rng=0)
        np.testing.assert_allclose(q.T @ q, np.eye(n), atol=1e-13)
        np.testing.assert_allclose(q @ q.T, np.eye(n), atol=1e-13)

    def test_batch(self):
        q = random_orthogonal(5, size=100, rng=1)
        np.testing.assert_allclose(np.swapaxes(q, -1, -2) @ q, np.broadcast_to(np.eye(5), q.shape),
                                   atol=1e-13)

    def test_haar_moments(self):
        # Для меры Хаара на O(n): E[Q_ij²] = 1/n, а det = ±1 равновероятно.
        q = random_orthogonal(4, size=40_000, rng=2)
        np.testing.assert_allclose((q**2).mean(axis=0), np.full((4, 4), 0.25), atol=0.01)
        assert np.mean(np.linalg.det(q) > 0) == pytest.approx(0.5, abs=0.01)

    def test_no_sign_bias(self):
        # Без согласования знаков с R у Q_11 был бы перекос знака.
        q = random_orthogonal(3, size=40_000, rng=3)
        assert abs(q[:, 0, 0].mean()) < 0.01

    def test_rotation_2x2(self):
        q = rotation_2x2(0.3)
        np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-15)
        assert np.linalg.det(q) == pytest.approx(1.0)


class TestRotate:
    def test_spectrum_preserved(self):
        h = sample_matrices("goe", 6, 50, rng=4)
        q = random_orthogonal(6, rng=5)
        np.testing.assert_allclose(np.linalg.eigvalsh(rotate(h, q)), np.linalg.eigvalsh(h), atol=1e-12)

    def test_symmetry_preserved(self):
        h = rotate(sample_matrices("pm1", 4, 10, rng=6), random_orthogonal(4, rng=7))
        np.testing.assert_allclose(h, np.swapaxes(h, -1, -2), atol=1e-13)


class TestInvariance:
    @pytest.mark.parametrize("n", [2, 4, 16])
    def test_goe_is_orthogonal_ensemble(self, n):
        c = orthogonal_invariance("goe", n, 60_000, seed=8)
        assert c["diag_var_after"] == pytest.approx(4.0, rel=0.03)
        assert c["off_var_after"] == pytest.approx(2.0, rel=0.03)
        assert c["ks_diag_p"] > 0.001 and c["ks_off_p"] > 0.001
        assert c["orthogonality_error"] < 1e-13 and c["spectrum_error"] < 1e-12

    @pytest.mark.parametrize("name", ["normal-mirror", "pm1"])
    def test_equal_variances_not_invariant(self, name):
        c = orthogonal_invariance(name, 16, 40_000, seed=9)
        assert c["diag_var_after"] > 1.5 * c["diag_var_before"]
        assert c["ks_diag_p"] < 1e-6

    def test_pm1_sum_keeps_variances_but_not_law(self):
        c = orthogonal_invariance("pm1-sum", 4, 40_000, seed=10)
        assert c["diag_var_after"] == pytest.approx(c["diag_var_before"], rel=0.03)
        assert c["ks_diag_p"] < 1e-6

    def test_angle_formula(self):
        # Var H'_11 / Var H_11 = 1 для A + Aᵀ и 1 + ½ sin² 2φ для одинаковых дисперсий.
        angles = np.radians([0, 22.5, 45, 90])
        goe = variance_vs_angle("goe", angles, 100_000, seed=11)
        mirror = variance_vs_angle("normal-mirror", angles, 100_000, seed=11)
        np.testing.assert_allclose(goe["diag_ratio"], 1.0, atol=0.02)
        np.testing.assert_allclose(goe["off_ratio"], 1.0, atol=0.02)
        np.testing.assert_allclose(mirror["diag_ratio"], 1 + 0.5 * np.sin(2 * angles) ** 2, atol=0.02)
        np.testing.assert_allclose(mirror["off_ratio"], 1 - 0.5 * np.sin(2 * angles) ** 2, atol=0.02)
