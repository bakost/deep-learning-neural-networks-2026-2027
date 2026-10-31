"""Проверка производных, в том числе сверхмалых, без абсолютного допуска 1e-8."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from transformer_lab import TASK_LOGITS, decimal_reference, softmax, softmax_jacobian


def test_shift_invariance_and_normalization():
    z = np.array([[1000, -1000], [999, -1002], [998, -1004]], dtype=float)
    y = softmax(z)
    assert_allclose(y.sum(axis=0), 1)
    assert_allclose(y, softmax(z + np.array([5000, -5000])))
    assert_allclose(softmax(z, axis=1).sum(axis=1), 1)
    assert_array_equal(softmax([1000]), [1])


def test_jacobian_finite_differences_regular_logits():
    z = np.array([-2, 0, 1, 3], dtype=float)
    step = 1e-5
    fd = np.column_stack([(softmax(z + e * step) - softmax(z - e * step)) / (2 * step)
                          for e in np.eye(len(z))])
    assert_allclose(softmax_jacobian(z), fd, rtol=2e-8, atol=1e-12)


def test_all_25_entries_against_decimal_analytic():
    jac = softmax_jacobian(TASK_LOGITS)
    ref = np.array(decimal_reference()["jacobian"], dtype=float)
    assert np.count_nonzero(jac) == 25
    assert_allclose(jac, ref, rtol=5e-15, atol=0)
    assert softmax(TASK_LOGITS)[2] == 1.0
    assert jac[2, 2] > 5e-42
    assert_allclose(jac, jac.T, rtol=0, atol=0)
    assert np.all(np.diag(jac) > 0)
    assert np.all(jac[~np.eye(5, dtype=bool)] < 0)
    # Допуск соразмерен самим производным, а не единице.
    assert np.max(np.abs(jac.sum(axis=1))) < 1e-56


def test_all_25_entries_against_decimal_finite_differences():
    # Численно дифференцируем саму функцию, независимо от формулы якобиана.
    with localcontext() as ctx:
        ctx.prec = 110
        z = [Decimal(v) for v in TASK_LOGITS]
        step = Decimal("1e-5")

        def evaluate(values):
            exps = [v.exp() for v in values]
            return [v / sum(exps) for v in exps]

        fd = np.empty((5, 5))
        for j in range(5):
            plus, minus = z.copy(), z.copy()
            plus[j] += step
            minus[j] -= step
            for i, (a, b) in enumerate(zip(evaluate(plus), evaluate(minus))):
                fd[i, j] = float((a - b) / (2 * step))
    assert_allclose(softmax_jacobian(TASK_LOGITS), fd, rtol=3e-11, atol=0)


def test_jacobian_psd_nullspace_and_chain_rule():
    z = np.array([-2.0, 0.0, 1.0, 3.0])
    jac = softmax_jacobian(z)
    assert np.linalg.eigvalsh(jac).min() > -1e-16
    assert_allclose(jac @ np.ones(4), 0, atol=1e-16)
    assert_array_equal(softmax_jacobian([3]), [[0]])
    scale, step = 2.3, 1e-5
    fd = np.column_stack([(softmax((z + e * step) / scale) - softmax((z - e * step) / scale))
                          / (2 * step) for e in np.eye(4)])
    assert_allclose(softmax_jacobian(z / scale) / scale, fd, rtol=2e-8, atol=1e-12)


@pytest.mark.parametrize("z", [[], [np.inf, 0], [np.nan], 1.0])
def test_invalid_softmax(z):
    with pytest.raises(ValueError):
        softmax(z)
