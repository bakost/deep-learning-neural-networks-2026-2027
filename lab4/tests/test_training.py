"""Градиентный спуск."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_mlp import exponential, gradient_descent, gradients, init_params, interpolation_floor


def test_loss_decreases_and_input_untouched(std):
    Z, t, _ = std
    p0 = init_params(24, 8, 0)
    before = p0.flat().copy()
    res = gradient_descent(p0, Z, t, lr=0.1, delta=1e-12, max_iter=3000, record_every=100)
    assert np.array_equal(p0.flat(), before)
    assert res.status == "max_iter" and res.iterations == 3000
    assert res.history["loss"][-1] < 0.1 * res.history["loss"][0]
    assert res.history["iteration"][0] == 0 and res.history["iteration"][-1] == 3000


def test_converged_status_and_criterion(std):
    Z, t, _ = std
    res = gradient_descent(init_params(24, 8, 0), Z, t, lr=0.1, delta=0.05, max_iter=100_000)
    assert res.converged and res.grad_norm < 0.05
    _, g = gradients(res.params, Z, t)
    assert np.linalg.norm(g.flat()) == pytest.approx(res.grad_norm)
    assert res.history["grad_norm"][-2] >= 0.05 or res.iterations == 0


def test_diverges_with_large_step(std):
    Z, t, _ = std
    res = gradient_descent(init_params(24, 16, 0), Z, t, lr=1.0, delta=1e-4, max_iter=1000)
    assert res.status == "diverged" and res.iterations < 50


def test_reaches_interpolation_floor(data, std):
    """Широкая сеть подгоняет всё, кроме изомеров с одинаковыми X: RMSE → √(SS_pe/n)."""
    Z, t, scaler = std
    res = gradient_descent(init_params(24, 32, 1), Z, t, lr=0.1, delta=1e-6, max_iter=40_000)
    rmse = np.sqrt(2 * res.loss) * scaler.y_std
    floor = interpolation_floor(data)
    assert floor == pytest.approx(0.12374, abs=1e-5)
    assert floor <= rmse < floor + 0.01


def test_schedule_is_recorded(std):
    Z, t, _ = std
    sch = exponential(0.1, 0.99)
    res = gradient_descent(init_params(24, 4, 0), Z, t, lr=sch, delta=1e-12, max_iter=300, record_every=100)
    assert np.allclose(res.history["lr"], [sch(int(j)) for j in res.history["iteration"]])


def test_minibatch_reproducible(std):
    Z, t, _ = std
    a = gradient_descent(init_params(24, 4, 0), Z, t, lr=0.05, max_iter=500, batch_size=16, seed=3, record_every=50)
    b = gradient_descent(init_params(24, 4, 0), Z, t, lr=0.05, max_iter=500, batch_size=16, seed=3, record_every=50)
    full = gradient_descent(init_params(24, 4, 0), Z, t, lr=0.05, max_iter=500, record_every=50)
    assert np.array_equal(a.params.flat(), b.params.flat())
    assert not np.allclose(a.params.flat(), full.params.flat())


def test_validation_history(std):
    Z, t, _ = std
    res = gradient_descent(init_params(24, 4, 0), Z[:60], t[:60], lr=0.1, max_iter=200, record_every=50,
                           X_val=Z[60:], y_val=t[60:])
    assert res.history["val_loss"].shape == res.history["loss"].shape


def test_bad_arguments(std):
    Z, t, _ = std
    with pytest.raises(ValueError):
        gradient_descent(init_params(24, 2, 0), Z, t, delta=0)
