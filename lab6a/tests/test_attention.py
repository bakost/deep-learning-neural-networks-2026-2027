"""Проверка ориентации матриц, общих смещений и перестановок."""

from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from transformer_lab import init_params, permutation_matrix, self_attention


def test_assignment_permutation():
    p = permutation_matrix([1, 3, 0, 2])
    assert_array_equal(p, [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    x = np.arange(12).reshape(3, 4)
    assert_array_equal(x @ p, x[:, [1, 3, 0, 2]])


@pytest.mark.parametrize("scaled", [False, True])
def test_attention_matches_scalar_definition(scaled):
    x = np.random.default_rng(11).normal(size=(3, 4))
    params = init_params(3, 21)
    actual, a = self_attention(x, params, scaled=scaled)
    expected = np.zeros_like(x)
    for n in range(x.shape[1]):
        q = params.wq @ x[:, n] + params.bq[:, 0]
        scores, values = [], []
        for m in range(x.shape[1]):
            k = params.wk @ x[:, m] + params.bk[:, 0]
            scores.append(k @ q / (np.sqrt(3) if scaled else 1))
            values.append(params.wv @ x[:, m] + params.bv[:, 0])
        weights = np.exp(scores - np.max(scores))
        weights /= weights.sum()
        expected[:, n] = sum(w * v for w, v in zip(weights, values))
    assert_allclose(actual, expected, atol=1e-14)
    assert_allclose(a.sum(axis=0), 1, atol=1e-15)
    assert np.all(a >= 0)


@pytest.mark.parametrize("scaled", [False, True])
def test_equivariance_all_24_permutations_with_biases(scaled):
    x = np.random.default_rng(42).normal(size=(3, 4))
    params = init_params(3, 2026)
    y, a = self_attention(x, params, scaled=scaled)
    for order in permutations(range(4)):
        p = permutation_matrix(order)
        yp, ap = self_attention(x @ p, params, scaled=scaled)
        assert_allclose(yp, y @ p, rtol=2e-14, atol=2e-14)
        assert_allclose(ap, p.T @ a @ p, rtol=2e-14, atol=2e-14)


def test_single_token_and_fixed_positions():
    params = init_params(3)
    x = np.array([[1.0], [2.0], [-1.0]])
    y, a = self_attention(x, params)
    assert_array_equal(a, [[1]])
    assert_allclose(y, params.wv @ x + params.bv)
    rng = np.random.default_rng(10)
    x, positions = rng.normal(size=(2, 3, 4))
    p = permutation_matrix([1, 3, 0, 2])
    baseline, _ = self_attention(x + positions, params)
    fixed, _ = self_attention(x @ p + positions, params)
    permuted, _ = self_attention((x + positions) @ p, params)
    assert not np.allclose(fixed, baseline @ p)
    assert_allclose(permuted, baseline @ p, atol=1e-14)


@pytest.mark.parametrize("order", [[], [0, 0], [1, 2], [0.0, 1.0], [[0, 1]]])
def test_invalid_permutations(order):
    with pytest.raises(ValueError):
        permutation_matrix(order)


@pytest.mark.parametrize("x", [np.zeros(3), np.zeros((3, 0)), np.zeros((2, 3)), np.full((3, 4), np.nan)])
def test_invalid_inputs(x):
    with pytest.raises(ValueError):
        self_attention(x, init_params(3))
