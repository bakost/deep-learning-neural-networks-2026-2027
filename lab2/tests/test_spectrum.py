"""Пункты 2–4: собственные числа, разности соседних уровней, нормировка."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from level_spacing import (
    all_spacings,
    central_pair,
    eigenvalues,
    normalize,
    pair_spacings,
    sample_matrices,
    simulate_eigenvalues,
    simulate_spacings,
)

HAS_JACOBI = importlib.util.find_spec("jacobi_eigen") is not None


class TestEigenvalues:
    def test_known_2x2(self):
        # Формула (4): λ = (a + c)/2 ± ½√((c − a)² + 4b²).
        a, b, c = 1.0, 2.0, -3.0
        expected = (a + c) / 2 + np.array([-1, 1]) * 0.5 * np.sqrt((c - a) ** 2 + 4 * b**2)
        np.testing.assert_allclose(eigenvalues(np.array([[a, b], [b, c]])), expected)

    def test_sorted_ascending(self):
        ev = eigenvalues(sample_matrices("goe", 8, 500, rng=0))
        assert np.all(np.diff(ev, axis=1) >= 0.0)

    def test_matches_characteristic_properties(self):
        h = sample_matrices("goe", 5, 200, rng=1)
        ev = eigenvalues(h)
        np.testing.assert_allclose(ev.sum(axis=1), np.trace(h, axis1=1, axis2=2), atol=1e-10)
        np.testing.assert_allclose(ev.prod(axis=1), np.linalg.det(h), rtol=1e-8, atol=1e-8)

    def test_formula_5_for_every_matrix(self):
        # s = 2√(d² + b²), d = (c − a)/2 — формула (5) методички.
        h = sample_matrices("goe", 2, 10_000, rng=2)
        s = pair_spacings(eigenvalues(h))
        b = h[:, 0, 1]
        d = 0.5 * (h[:, 1, 1] - h[:, 0, 0])
        np.testing.assert_allclose(s, 2.0 * np.hypot(b, d), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 3)), np.zeros((4, 2, 3))])
    def test_rejects_non_square(self, bad):
        with pytest.raises(ValueError, match="квадратные"):
            eigenvalues(bad)

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="бэкенд"):
            eigenvalues(np.eye(2), backend="torch")

    @pytest.mark.skipif(not HAS_JACOBI, reason="пакет jacobi_eigen из ЛР1 недоступен")
    @pytest.mark.parametrize("n", [2, 4, 16])
    def test_jacobi_backend_agrees_with_lapack(self, n):
        h = sample_matrices("goe", n, 30, rng=3)
        np.testing.assert_allclose(eigenvalues(h, backend="jacobi"), eigenvalues(h),
                                   atol=1e-10 * n)


class TestSpacings:
    def test_central_pair(self):
        assert [central_pair(n) for n in (2, 4, 16)] == [1, 2, 8]

    def test_central_pair_needs_two_levels(self):
        with pytest.raises(ValueError):
            central_pair(1)

    def test_pair_is_one_based(self):
        ev = np.array([[0.0, 1.0, 3.0, 6.0, 10.0]])
        assert [pair_spacings(ev, k)[0] for k in (1, 2, 3, 4)] == [1.0, 2.0, 3.0, 4.0]

    def test_default_is_central(self):
        ev = np.arange(16.0) ** 2
        assert pair_spacings(ev[None, :])[0] == 8**2 - 7**2

    @pytest.mark.parametrize("k", [0, 4, -1])
    def test_bad_pair(self, k):
        with pytest.raises(ValueError):
            pair_spacings(np.zeros((1, 4)), k)

    def test_all_spacings(self):
        ev = np.array([[0.0, 1.0, 3.0]])
        np.testing.assert_array_equal(all_spacings(ev), [[1.0, 2.0]])

    def test_spacings_non_negative(self, ensemble_name):
        s = simulate_spacings(ensemble_name, 4, 2000, rng=4)
        assert np.all(s >= -1e-12)


class TestNormalize:
    def test_mean_is_one(self):
        x = normalize(np.array([0.5, 1.5, 7.0]))
        assert x.mean() == pytest.approx(1.0)

    def test_scale_invariance(self):
        s = simulate_spacings("goe", 4, 1000, rng=5)
        np.testing.assert_allclose(normalize(s), normalize(17.0 * s))

    @pytest.mark.parametrize("bad", [np.array([]), np.zeros(3), np.array([np.nan, 1.0])])
    def test_rejects_degenerate(self, bad):
        with pytest.raises(ValueError):
            normalize(bad)


class TestSimulation:
    def test_chunking_does_not_change_result(self):
        whole = simulate_spacings("goe", 4, 1000, rng=6, chunk_size=1000)
        chunked = simulate_spacings("goe", 4, 1000, rng=6, chunk_size=7)
        np.testing.assert_array_equal(whole, chunked)

    def test_matches_manual_pipeline(self):
        manual = pair_spacings(eigenvalues(sample_matrices("pm1", 4, 300, rng=7)), k=2)
        np.testing.assert_allclose(simulate_spacings("pm1", 4, 300, rng=7, k=2), manual)

    def test_eigenvalues_shape(self):
        assert simulate_eigenvalues("goe", 6, 11, rng=8, chunk_size=4).shape == (11, 6)

    def test_rejects_bad_chunk(self):
        with pytest.raises(ValueError):
            simulate_spacings("goe", 4, 10, rng=0, chunk_size=0)
