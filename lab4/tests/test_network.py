"""Модель (2) и ручной градиент (лекция 4)."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_mlp import (
    Params,
    forward,
    gradients,
    init_params,
    input_gradients,
    loss,
    max_gradient_error,
    n_params,
    numerical_gradients,
    predict,
    relu,
)


@pytest.mark.parametrize("hidden, count", [(2, 53), (4, 105), (8, 209), (16, 417), (32, 833)])
def test_parameter_count(hidden, count):
    assert n_params(24, hidden) == count
    assert init_params(24, hidden, 0).n_params == count


def test_flat_roundtrip():
    p = init_params(5, 3, 1, "glorot")
    q = Params.from_flat(p.flat(), 5, 3)
    assert np.array_equal(p.flat(), q.flat())
    assert p.flat()[0] == p.b2                  # порядок лекции: [b², w², b¹, w¹]
    with pytest.raises(ValueError):
        Params.from_flat(np.zeros(7), 5, 3)


def test_forward_matches_formula():
    rng = np.random.default_rng(2)
    p = init_params(4, 3, rng)
    p.b1 = rng.normal(size=3)
    p.b2 = 0.7
    X = rng.normal(size=(6, 4))
    expected = np.array([p.b2 + p.w2 @ relu(p.W1 @ x + p.b1) for x in X])
    assert np.allclose(predict(p, X), expected)
    _, Z, A = forward(p, X)
    assert Z.shape == (6, 3) and np.all(A >= 0)


def test_loss_definition():
    rng = np.random.default_rng(3)
    p = init_params(4, 3, rng)
    X, y = rng.normal(size=(9, 4)), rng.normal(size=9)
    assert loss(p, X, y) == pytest.approx(np.sum((y - predict(p, X)) ** 2) / 18)


@pytest.mark.parametrize("hidden", [1, 2, 8, 32])
def test_gradient_matches_finite_differences(std, hidden):
    Z, t, _ = std
    rng = np.random.default_rng(hidden)
    p = init_params(24, hidden, rng)
    p.b1 = rng.normal(scale=0.3, size=hidden)
    assert max_gradient_error(p, Z, t) < 1e-7


def test_gradient_value_returned(std):
    Z, t, _ = std
    p = init_params(24, 4, 0)
    J, _ = gradients(p, Z, t)
    assert J == pytest.approx(loss(p, Z, t))


def test_dead_network_has_only_bias_gradient(std):
    """Если все нейроны неактивны, сеть выдаёт константу b², и меняться может только b²."""
    Z, t, _ = std
    p = init_params(24, 3, 0)
    p.b1[:] = -100.0
    _, g = gradients(p, Z, t)
    assert np.all(g.W1 == 0) and np.all(g.b1 == 0) and np.all(g.w2 == 0)
    assert g.b2 == pytest.approx(-(t - p.b2).mean())


def test_input_gradients_numerical():
    rng = np.random.default_rng(5)
    p = init_params(3, 4, rng)
    p.b1 = rng.normal(size=4)
    X = rng.normal(size=(5, 3))
    G = input_gradients(p, X)
    h = 1e-6
    for j in range(3):
        E = np.zeros(3)
        E[j] = h
        num = (predict(p, X + E) - predict(p, X - E)) / (2 * h)
        assert np.allclose(G[:, j], num, atol=1e-6)


def test_init_schemes():
    he = init_params(400, 300, 0)
    assert he.W1.std() == pytest.approx(np.sqrt(2 / 400), rel=0.02)
    assert np.all(he.b1 == 0) and he.b2 == 0
    gl = init_params(24, 8, 0, "glorot")
    assert np.all(np.abs(gl.W1) <= np.sqrt(6 / 32))
    with pytest.raises(ValueError):
        init_params(3, 3, 0, "zeros")


def test_numerical_gradient_shape():
    rng = np.random.default_rng(6)
    p = init_params(2, 2, rng)
    assert numerical_gradients(p, rng.normal(size=(4, 2)), rng.normal(size=4)).shape == (p.n_params,)
