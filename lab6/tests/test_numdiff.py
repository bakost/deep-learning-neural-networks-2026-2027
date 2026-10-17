"""Численное дифференцирование (NR 5.7)."""

from __future__ import annotations

import numpy as np

from sgd_lab import Objective, X, Y, central_gradient, init_params, ridders_derivative


def test_gradient_matches_analytic(model):
    obj = Objective(model, X, Y)
    for s in range(10):
        w = init_params(model, s)
        g, ga = obj.grad(w), model.analytic_grad(w, X, Y)
        assert np.max(np.abs(g - ga)) <= 1e-7 * max(1.0, np.max(np.abs(ga)))


def test_gradient_batched_rows_equal_single():
    f = lambda W: np.sum(np.sin(W) * W, axis=-1)  # noqa: E731
    W = np.random.default_rng(0).normal(size=(5, 3))
    batched = central_gradient(f, W)
    single = np.stack([central_gradient(f, w) for w in W])
    assert np.array_equal(batched, single)
    assert np.allclose(batched, np.sin(W) + W * np.cos(W), atol=1e-9)


def test_step_is_exactly_representable():
    # для w = 0.1 шаг (w + h) − w — точное двоичное число, поэтому ошибка округления минимальна
    g = central_gradient(lambda W: np.sum(W ** 3, axis=-1), np.array([0.1, 3.0]))
    assert np.allclose(g, [0.03, 27.0], rtol=1e-9)


def test_ridders_reaches_machine_precision():
    d, err = ridders_derivative(np.exp, 1.0)
    assert abs(d - np.e) < 1e-12 and err < 1e-11
