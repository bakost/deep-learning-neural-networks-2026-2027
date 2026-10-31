"""Задача 1: поиск отрезка, золотое сечение, метод Брента."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from sgd_lab import GOLDEN_RATIO, bracket_minimum, brent, df_task, exact_minimum, f_task, golden_section

Z_STAR = exact_minimum()


def test_exact_minimum_is_stationary_and_unique():
    assert abs(df_task(Z_STAR)) < 1e-12
    z = np.linspace(-3, 4, 700_001)
    d = df_task(z)
    # f′ меняет знак ровно один раз — минимум единственный, у z ≈ 0.6 только «полка»
    assert np.count_nonzero(np.diff(np.sign(d[np.abs(d) > 1e-12])) != 0) == 1
    assert 0.38 < Z_STAR < 0.381


@pytest.mark.parametrize("start", [(0.0, 1.0), (1.0, 0.0), (0.6, 0.7), (-3.0, -2.0), (2.0, 3.0)])
def test_bracket_is_valid(start):
    (a, b, c), nfev = bracket_minimum(f_task, *start)
    assert a < b < c and f_task(b) <= f_task(a) and f_task(b) < f_task(c)
    assert a < Z_STAR < c and nfev < 30


@pytest.mark.parametrize("method", [golden_section, brent])
def test_methods_find_minimum(method):
    trip, _ = bracket_minimum(f_task, 0.0, 1.0)
    res = method(f_task, trip)
    assert abs(res.x - Z_STAR) < 1e-8
    assert res.fx == pytest.approx(f_task(Z_STAR), abs=1e-15)


def test_brent_needs_fewer_evaluations_and_matches_scipy():
    trip, _ = bracket_minimum(f_task, 0.0, 1.0)
    g, b = golden_section(f_task, trip), brent(f_task, trip)
    assert b.nfev < g.nfev / 2
    sb = minimize_scalar(f_task, bracket=trip, method="brent")
    sg = minimize_scalar(f_task, bracket=trip, method="golden")
    assert abs(b.x - sb.x) < 1e-8 and b.iterations == sb.nit
    assert abs(g.x - sg.x) < 1e-8 and g.iterations == sg.nit


def test_golden_shrinks_by_golden_ratio():
    trip, _ = bracket_minimum(f_task, 0.0, 1.0)
    w = golden_section(f_task, trip).widths
    ratios = w[1:] / w[:-1]
    assert np.allclose(ratios[2:], 1 / GOLDEN_RATIO, atol=1e-6)


def test_brent_uses_parabolic_steps():
    trip, _ = bracket_minimum(f_task, 0.0, 1.0)
    kinds = [h[4] for h in brent(f_task, trip).history]
    assert kinds.count("parabolic") >= kinds.count("golden")


def test_accuracy_floor_sqrt_eps():
    """Точнее ~√ε_маш по значениям функции минимум не найти, сколько ни уменьшай tol."""
    trip, _ = bracket_minimum(f_task, 0.0, 1.0)
    floor = math.sqrt(2 * np.finfo(float).eps * f_task(Z_STAR) / 36.8)
    for tol in (1e-10, 1e-12):
        assert abs(brent(f_task, trip, tol=tol).x - Z_STAR) < 10 * floor


def test_quadratic_exact_for_brent():
    res = brent(lambda x: (x - 1.234) ** 2 + 5, (0.0, 1.0, 3.0))
    assert abs(res.x - 1.234) < 1e-7 and res.nfev < 10


def test_plateau_gives_no_information():
    """Вдали от минимума f ≡ 1 в машинной арифметике: поиск отрезка не может выбрать направление."""
    assert f_task(5.0) == f_task(6.0) == 1.0
    (a, _b, c), _ = bracket_minimum(f_task, 5.0, 6.0)
    assert not a < Z_STAR < c
