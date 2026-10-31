"""Задача 1: число обучаемых параметров и вычисляемых коэффициентов."""

from __future__ import annotations

from numbers import Integral
from typing import Optional


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} должно быть положительным целым числом")
    return int(value)


def parameter_counts(d: int, n: int, hidden: Optional[int] = None) -> dict:
    """Сравнить Q/K/V и MLP DN → H → DN; по умолчанию H = DN.

    Выходная проекция, LayerNorm и позиционные эмбеддинги в условие задачи
    не входят. Элементы A(X) вычисляются, а не обучаются независимо.
    """
    d, n = positive_int(d, "D"), positive_int(n, "N")
    m = d * n
    h = m if hidden is None else positive_int(hidden, "H")
    return {
        "D": d, "N": n, "H": h,
        "qkv_weights": 3 * d * d,
        "qkv_biases": 3 * d,
        "qkv_total": 3 * d * (d + 1),
        "attention_entries": n * n,
        "attention_trainable": 0,
        "mlp_weights": 2 * m * h,
        "mlp_biases": h + m,
        "mlp_total": 2 * m * h + h + m,
    }
