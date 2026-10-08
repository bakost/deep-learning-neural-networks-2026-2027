"""Эволюционные алгоритмы и замер скорости."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_mlp import init_params, loss
from lipo_mlp.evolution import cma_es, evolution_strategy, population_loss


def test_population_loss_matches_loss(std):
    Z, t, _ = std
    ps = [init_params(24, 4, s) for s in range(3)]
    thetas = np.stack([p.flat() for p in ps])
    assert np.allclose(population_loss(thetas, Z, t, 4), [loss(p, Z, t) for p in ps])
    assert population_loss(ps[0].flat(), Z, t, 4).shape == (1,)


def test_own_es_improves_and_is_reproducible(std):
    Z, t, _ = std
    start = loss(init_params(24, 2, np.random.default_rng(5)), Z, t)
    a = evolution_strategy(Z, t, 2, max_evals=3000, seed=5)
    b = evolution_strategy(Z, t, 2, max_evals=3000, seed=5)
    assert a.loss < 0.5 * start and a.loss == b.loss
    assert a.evaluations <= 3000 and a.generations == (a.evaluations - 1) // 60
    assert np.all(np.diff(a.history["best_loss"]) <= 0)              # лучшая J не растёт
    assert a.loss == pytest.approx(loss(a.params, Z, t))


def test_own_es_target_stops_early(std):
    Z, t, _ = std
    res = evolution_strategy(Z, t, 2, max_evals=50_000, seed=0, target=0.3)
    assert res.loss <= 0.3 and res.evaluations < 50_000


def test_cma_es(std):
    pytest.importorskip("cma")
    Z, t, _ = std
    res = cma_es(Z, t, 2, max_evals=2000, seed=1)
    assert res.loss == pytest.approx(loss(res.params, Z, t))
    assert res.history["best_loss"][-1] < res.history["best_loss"][0]
    assert res.evaluations >= 2000


def test_benchmark_keys():
    pytest.importorskip("cma")
    from lipo_mlp.benchmark import timing_benchmark

    t = timing_benchmark(hidden=2, units=100, repeats=1)
    assert set(t) == {"gd_own", "gd_sklearn", "es_own", "cma"}
    assert all(v["us_per_unit"] > 0 for v in t.values())
