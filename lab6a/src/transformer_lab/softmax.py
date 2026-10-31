"""Задача 3: softmax, устойчивый якобиан и эталон Decimal."""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Sequence

import numpy as np

TASK_LOGITS = (-3, 1, 100, 5, -1)


def softmax(z: np.ndarray, axis: int = 0) -> np.ndarray:
    """Нормировать вдоль axis, предварительно вычтя максимум.

    Для X размера D × N в лекции softmax применяется по строкам каждого
    столбца матрицы KᵀQ, поэтому axis=0.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 0 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("softmax ожидает непустой конечный массив")
    exp = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)


def softmax_jacobian(z: np.ndarray) -> np.ndarray:
    """Вернуть J[i,j] = ∂y_i/∂z_j без вычитания 1 − y_i.

    На диагонали y_i * sum_{j != i} y_j. Это математически эквивалентно
    y_i(1-y_i), но сохраняет малую производную при округлении y_i до 1.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("якобиан ожидает одномерный вектор логитов")
    y = softmax(z)
    jac = -np.outer(y, y)
    np.fill_diagonal(jac, 0.0)
    np.fill_diagonal(jac, -jac.sum(axis=1))
    return jac


def decimal_reference(z: Sequence = TASK_LOGITS, precision: int = 110) -> dict:
    """Независимый эталон через y_i(δ_ij − y_j), возвращаемый строками.

    Decimal используется только для проверки и отчёта. Все рабочие операции
    самовнимания реализованы в NumPy float64.
    """
    if precision < 100:
        raise ValueError("для задачи 3 требуется не менее 100 десятичных знаков")
    with localcontext() as ctx:
        ctx.prec = precision
        values = [Decimal(str(v)) for v in z]
        if not values or not all(v.is_finite() for v in values):
            raise ValueError("ожидаются конечные логиты")
        shift = max(values)
        exp = [(v - shift).exp() for v in values]
        total = sum(exp)
        y = [v / total for v in exp]
        jac = [[yi * (Decimal(i == j) - yj) for j, yj in enumerate(y)] for i, yi in enumerate(y)]
        return {
            "precision": precision,
            "probabilities": [str(v) for v in y],
            "jacobian": [[str(v) for v in row] for row in jac],
            "one_minus_y3": str(1 - y[2]) if len(y) == 5 else None,
        }
