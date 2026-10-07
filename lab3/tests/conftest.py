"""Общие фикстуры: выборка и модель (2) по всем 24 регрессорам.

Случайность в тестах — только в синтетических проверках и перекрёстной
проверке; все генераторы инициализируются фиксированными зёрнами.
"""

from __future__ import annotations

import numpy as np
import pytest

from lipo_ols import fit_ols, fit_ols_data, load_lipo


@pytest.fixture(scope="session")
def data():
    return load_lipo()


@pytest.fixture(scope="session")
def fit(data):
    return fit_ols_data(data)


@pytest.fixture
def synthetic():
    """Модель с известными β: y = Xβ + ε, ε ~ N(0, 0.5²); X — константа и 3 регрессора."""
    rng = np.random.default_rng(7)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    beta = np.array([1.0, 2.0, -0.5, 0.0])
    y = X @ beta + rng.normal(scale=0.5, size=n)
    return X, y, beta, fit_ols(X, y, ["const", "x1", "x2", "x3"])
