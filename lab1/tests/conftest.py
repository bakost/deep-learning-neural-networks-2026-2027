"""Общие фикстуры и вспомогательные функции для тестов."""

from __future__ import annotations

import numpy as np
import pytest

from jacobi_eigen.kernels import available_kernels
from jacobi_eigen.strategies import available_strategies


@pytest.fixture(params=sorted(available_kernels()))
def kernel(request) -> str:
    """Параметризация по всем ядрам вращения."""
    return request.param


@pytest.fixture(params=sorted(available_strategies()))
def strategy(request) -> str:
    """Параметризация по всем стратегиям выбора элемента."""
    return request.param


def random_symmetric(n: int, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Случайная симметричная матрица ``n x n`` с воспроизводимым зерном."""
    generator = np.random.default_rng(seed)
    m = generator.normal(scale=scale, size=(n, n))
    return (m + m.T) / 2.0


def symmetric_with_spectrum(eigenvalues, seed: int = 0) -> np.ndarray:
    """Симметричная матрица с *заранее заданным* спектром.

    Строится как :math:`Q \\Lambda Q^{T}` со случайной ортогональной
    :math:`Q`, полученной QR-разложением. Позволяет тестировать сложные
    случаи: кратные корни, вырожденность, разный масштаб.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    n = eigenvalues.size
    generator = np.random.default_rng(seed)
    q, r = np.linalg.qr(generator.normal(size=(n, n)))
    q = q * np.sign(np.diagonal(r))  # каноническая ортогональная матрица
    a = q @ np.diag(eigenvalues) @ q.T
    return (a + a.T) / 2.0
