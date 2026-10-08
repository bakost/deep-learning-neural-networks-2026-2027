r"""Замер скорости: свой код против пакетов на одной и той же задаче.

Все методы запускаются последовательно в одном процессе (без параллельной
нагрузки), каждый — ``repeats`` раз; берётся наименьшее время. Измеряется
стоимость одной «единицы работы»: итерации градиентного спуска (прямой и
обратный проход по всей выборке) или одного вычисления функции потерь в
эволюционном алгоритме (только прямой проход).

>>> from lipo_mlp.benchmark import timing_benchmark
>>> t = timing_benchmark(hidden=2, units=200, repeats=1)
>>> sorted(t)
['cma', 'es_own', 'gd_own', 'gd_sklearn']
"""

from __future__ import annotations

import time
from typing import Callable, Dict

from .data import Standardizer, load_lipo
from .evolution import cma_es, evolution_strategy
from .network import init_params
from .sklearn_compare import sklearn_trajectory
from .training import gradient_descent

__all__ = ["timing_benchmark"]


def _best_time(fn: Callable[[], object], repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def timing_benchmark(hidden: int = 8, units: int = 20_000, repeats: int = 3, lr: float = 0.1) -> Dict[str, dict]:
    """Время на единицу работы (мкс): свой и пакетный градиентный спуск, своя ЭС и CMA-ES."""
    data = load_lipo()
    Z, t = Standardizer.fit(data.X, data.y).transform(data.X, data.y)
    p0 = init_params(data.k, hidden, 0)
    cases = {
        "gd_own": ("итерация", lambda: gradient_descent(p0, Z, t, lr=lr, delta=1e-300, max_iter=units,
                                                        record_every=10**9)),
        "gd_sklearn": ("итерация", lambda: sklearn_trajectory(p0, Z, t, lr, units)),
        "es_own": ("вычисление J", lambda: evolution_strategy(Z, t, hidden, max_evals=units, seed=0)),
        "cma": ("вычисление J", lambda: cma_es(Z, t, hidden, max_evals=units, seed=0)),
    }
    out = {}
    for name, (unit, fn) in cases.items():
        seconds = _best_time(fn, repeats)
        out[name] = {"unit": unit, "units": units, "seconds": seconds, "us_per_unit": seconds / units * 1e6}
    return out
