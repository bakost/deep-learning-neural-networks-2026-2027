"""Генерация ансамблей: форма, симметрия, распределение элементов, воспроизводимость."""

from __future__ import annotations

import numpy as np
import pytest

from level_spacing import ENSEMBLES, Ensemble, get_ensemble, sample_matrices


class TestShapeAndSymmetry:
    @pytest.mark.parametrize("n", [1, 2, 4, 16])
    def test_shape(self, ensemble_name, n):
        assert sample_matrices(ensemble_name, n, 7, rng=0).shape == (7, n, n)

    def test_exactly_symmetric(self, ensemble_name):
        h = sample_matrices(ensemble_name, 5, 100, rng=1)
        assert np.array_equal(h, np.swapaxes(h, -1, -2))

    def test_empty_ensemble(self, ensemble_name):
        assert sample_matrices(ensemble_name, 3, 0, rng=0).shape == (0, 3, 3)


class TestEntryDistributions:
    """Распределение элементов H — то, о чём говорит раздел 1.3 методички."""

    def test_goe_variances(self):
        # H = A + Aᵀ: дисперсия диагонали 4, вне диагонали 2 (σ_a = 2, σ_b = √2).
        h = sample_matrices("goe", 3, 200_000, rng=2)
        diag = h[:, 0, 0]
        off = h[:, 0, 1]
        assert diag.var() == pytest.approx(4.0, rel=0.02)
        assert off.var() == pytest.approx(2.0, rel=0.02)
        assert abs(diag.mean()) < 0.02 and abs(off.mean()) < 0.02

    def test_goe_d_has_same_variance_as_b(self):
        # d = (c - a)/2 имеет дисперсию 2 = σ_b²: точка (d, b) изотропна.
        h = sample_matrices("goe", 2, 200_000, rng=3)
        d = 0.5 * (h[:, 1, 1] - h[:, 0, 0])
        assert d.var() == pytest.approx(2.0, rel=0.02)
        assert np.corrcoef(d, h[:, 0, 1])[0, 1] == pytest.approx(0.0, abs=0.01)

    def test_normal_mirror_variances(self):
        h = sample_matrices("normal-mirror", 3, 200_000, rng=4)
        assert h[:, 1, 1].var() == pytest.approx(1.0, rel=0.02)
        assert h[:, 0, 2].var() == pytest.approx(1.0, rel=0.02)

    def test_pm1_values(self):
        h = sample_matrices("pm1", 4, 500, rng=5)
        assert set(np.unique(h)) == {-1.0, 1.0}

    def test_pm1_equal_probabilities(self):
        h = sample_matrices("pm1", 2, 100_000, rng=6)
        assert np.mean(h[:, 0, 1] == 1.0) == pytest.approx(0.5, abs=0.006)

    def test_pm1_sum_values(self):
        h = sample_matrices("pm1-sum", 4, 20_000, rng=7)
        diag = np.diagonal(h, axis1=1, axis2=2)
        off = h[:, 0, 1]
        assert set(np.unique(diag)) == {-2.0, 2.0}
        assert set(np.unique(off)) == {-2.0, 0.0, 2.0}
        assert np.mean(off == 0.0) == pytest.approx(0.5, abs=0.015)

    def test_pm1_upper_triangle_independent(self):
        h = sample_matrices("pm1", 3, 100_000, rng=8)
        assert np.corrcoef(h[:, 0, 1], h[:, 1, 2])[0, 1] == pytest.approx(0.0, abs=0.015)


class TestReproducibility:
    def test_same_seed_same_matrices(self, ensemble_name):
        a = sample_matrices(ensemble_name, 4, 10, rng=42)
        b = sample_matrices(ensemble_name, 4, 10, rng=42)
        assert np.array_equal(a, b)

    def test_different_seed_differs(self):
        assert not np.array_equal(sample_matrices("goe", 4, 10, rng=1),
                                  sample_matrices("goe", 4, 10, rng=2))

    def test_generator_is_consumed_sequentially(self):
        # Две порции подряд из одного генератора == одна большая порция.
        rng = np.random.default_rng(9)
        parts = np.concatenate([sample_matrices("goe", 3, 4, rng), sample_matrices("goe", 3, 6, rng)])
        whole = sample_matrices("goe", 3, 10, rng=9)
        assert np.array_equal(parts, whole)


class TestRegistryAndValidation:
    def test_registry(self):
        assert set(ENSEMBLES) == {"goe", "pm1", "pm1-sum", "normal-mirror"}
        assert get_ensemble("pm1").is_discrete and not get_ensemble("goe").is_discrete

    def test_object_passthrough(self):
        e = ENSEMBLES["goe"]
        assert get_ensemble(e) is e

    def test_unknown_ensemble(self):
        with pytest.raises(ValueError, match="неизвестный ансамбль"):
            get_ensemble("gue")

    @pytest.mark.parametrize("kwargs", [{"entries": "cauchy"}, {"construction": "average"}])
    def test_bad_definition(self, kwargs):
        params = {"name": "x", "entries": "normal", "construction": "sum", "title": "x"}
        params.update(kwargs)
        with pytest.raises(ValueError):
            Ensemble(**params)

    @pytest.mark.parametrize("n, size, error", [(0, 5, ValueError), (2, -1, ValueError),
                                                (2.5, 5, TypeError), (2, "10", TypeError),
                                                (True, 5, TypeError)])
    def test_bad_sizes(self, n, size, error):
        with pytest.raises(error):
            sample_matrices("goe", n, size, rng=0)
