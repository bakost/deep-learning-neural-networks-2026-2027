"""Одноголовое самовнимание в соглашении лекции: токены — столбцы."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .counts import positive_int
from .softmax import softmax


@dataclass
class AttentionParams:
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    bq: np.ndarray
    bk: np.ndarray
    bv: np.ndarray


def init_params(d: int, seed: int = 0) -> AttentionParams:
    """Воспроизводимые ненулевые веса и смещения для проверки тождества."""
    d = positive_int(d, "D")
    rng = np.random.default_rng(seed)
    weights = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3)]
    biases = [rng.normal(size=(d, 1)) for _ in range(3)]
    return AttentionParams(*weights, *biases)


def self_attention(x: np.ndarray, params: AttentionParams, *, scaled: bool = False) -> tuple:
    """Вернуть (Sa(X), A(X)); A[m,n] — внимание n-го запроса к m-му ключу.

    По умолчанию формула лекции 5; scaled=True включает 1/√D из лекции 6.
    Масок, позиционного кодирования и выходной проекции здесь нет.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or 0 in x.shape or not np.all(np.isfinite(x)):
        raise ValueError("X должна быть конечной непустой матрицей D × N")
    d = x.shape[0]
    for w, b in ((params.wq, params.bq), (params.wk, params.bk), (params.wv, params.bv)):
        if w.shape != (d, d) or b.shape != (d, 1):
            raise ValueError("размеры весов и смещений должны быть D × D и D × 1")
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
            raise ValueError("параметры должны быть конечными")
    q, k, v = (w @ x + b for w, b in (
        (params.wq, params.bq), (params.wk, params.bk), (params.wv, params.bv),
    ))
    scores = k.T @ q
    if scaled:
        scores /= np.sqrt(d)
    a = softmax(scores, axis=0)
    return v @ a, a


def permutation_matrix(order: np.ndarray) -> np.ndarray:
    """P с X @ P == X[:, order], в том числе для несамосопряжённых P."""
    order = np.asarray(order)
    if (order.ndim != 1 or order.size == 0 or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(order.size))):
        raise ValueError("order должен содержать перестановку 0, ..., N−1")
    return np.eye(order.size)[:, order]
