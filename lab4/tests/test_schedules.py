"""Законы изменения скорости обучения."""

from __future__ import annotations

import math

import pytest

from lipo_mlp import Schedule, constant, exponential, inverse_time, power, step


def test_values():
    assert constant(0.1)(10**6) == 0.1
    assert inverse_time(0.1, 100)(100) == pytest.approx(0.05)
    assert power(0.1, 1, 0.5)(3) == pytest.approx(0.05)
    assert exponential(0.1, 0.5)(3) == pytest.approx(0.0125)
    s = step(0.1, 0.5, 10)
    assert s(9) == 0.1 and s(10) == 0.05 and s(25) == pytest.approx(0.025)


def test_totals():
    assert math.isinf(constant(0.1).total()) and math.isinf(inverse_time(0.1, 10).total())
    assert exponential(0.1, 0.9999).total() == pytest.approx(1000)
    assert step(0.1, 0.5, 20).total() == pytest.approx(4.0)
    assert math.isinf(power(0.1, 10, 0.7).total())


def test_monotone_decay():
    for s in (inverse_time(0.1, 50), exponential(0.1, 0.99), step(0.1, 0.5, 7), power(0.1, 5, 0.6)):
        values = [s(j) for j in range(200)]
        assert all(a >= b for a, b in zip(values, values[1:])) and values[0] == 0.1


def test_labels_and_validation():
    assert "τ = 1000" in inverse_time(0.1, 1000).label()
    assert "γ" in exponential(0.1, 0.9).label() and "m = 5" in step(0.1, 0.5, 5).label()
    with pytest.raises(ValueError):
        constant(0.0)
    with pytest.raises(ValueError):
        Schedule("cosine", 0.1)
    with pytest.raises(ValueError):
        exponential(0.1, 1.5)
