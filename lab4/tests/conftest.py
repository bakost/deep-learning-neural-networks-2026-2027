"""Общие фикстуры. Все генераторы инициализируются фиксированными зёрнами."""

from __future__ import annotations

import pytest

from lipo_mlp import Standardizer, load_lipo


@pytest.fixture(scope="session")
def data():
    return load_lipo()


@pytest.fixture(scope="session")
def std(data):
    """Стандартизованные признаки и отклик по всей выборке."""
    scaler = Standardizer.fit(data.X, data.y)
    Z, t = scaler.transform(data.X, data.y)
    return Z, t, scaler
