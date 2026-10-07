r"""Насколько модель предсказывает, а не только описывает выборку.

:math:`R^2` на обучающей выборке всегда растёт с числом регрессоров. Чтобы
судить об адекватности модели (пункт 7, первый вопрос), её качество
оценивается на веществах, *не участвовавших* в оценивании:

* **скользящий контроль (leave-one-out)**: :math:`n` раз оценить модель без
  одного вещества и предсказать его; сумма квадратов ошибок — PRESS,
  :math:`Q^2 = 1 - PRESS/TSS`;
* **k-кратная перекрёстная проверка** с многократным случайным разбиением.

Если вещество — единственный носитель какого-то регрессора (например,
бромбензол для ``RBr``), без него столбец целиком нулевой и коэффициент не
определён. Тогда используется решение с минимальной нормой
(:func:`numpy.linalg.lstsq`): коэффициент при пустом столбце равен нулю,
то есть модель «не знает» про эту группу — как и было бы на практике.

Здесь же — пошаговое исключение незначимых регрессоров (backward
elimination) по t-тесту.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .data import LipoData
from .ols import OLSResult, fit_ols_data

__all__ = [
    "CVResult",
    "EliminationStep",
    "backward_elimination",
    "final_model",
    "kfold_cv",
    "loo_predictions",
    "press_statistics",
]


def _lstsq_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_test @ coef


def loo_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Прогнозы скользящего контроля: :math:`\hat y_{(i)}` по модели без :math:`i`-го наблюдения.

    Для наблюдений с :math:`h_{ii} < 1` совпадает с известной формулой
    :math:`y_i - \hat y_{(i)} = e_i/(1 - h_{ii})` (проверяется в тестах).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        out[i] = _lstsq_predict(X[mask], y[mask], X[i:i + 1])[0]
        mask[i] = True
    return out


def press_statistics(X: np.ndarray, y: np.ndarray) -> dict:
    """PRESS, :math:`Q^2` и RMSE скользящего контроля."""
    pred = loo_predictions(X, y)
    err = y - pred
    press = float(err @ err)
    tss = float(np.sum((y - y.mean()) ** 2))
    return {"press": press, "q2": 1.0 - press / tss, "rmse": float(np.sqrt(press / len(y))),
            "mae": float(np.mean(np.abs(err))), "predictions": pred}


@dataclass(frozen=True)
class CVResult:
    """Повторная k-кратная перекрёстная проверка.

    ``rmse`` и ``q2`` — по одному значению на каждое повторение разбиения.
    """

    k: int
    rmse: np.ndarray
    q2: np.ndarray

    @property
    def rmse_mean(self) -> float:
        return float(self.rmse.mean())

    @property
    def rmse_std(self) -> float:
        return float(self.rmse.std(ddof=1)) if self.rmse.size > 1 else 0.0

    @property
    def q2_mean(self) -> float:
        return float(self.q2.mean())


def kfold_cv(X: np.ndarray, y: np.ndarray, k: int = 5, repeats: int = 200,
             seed: Optional[int] = 2026) -> CVResult:
    """k-кратная перекрёстная проверка, ``repeats`` случайных разбиений."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if not 2 <= k <= n:
        raise ValueError("число блоков k должно лежать в [2, n]")
    rng = np.random.default_rng(seed)
    tss = float(np.sum((y - y.mean()) ** 2))
    rmse = np.empty(repeats)
    q2 = np.empty(repeats)
    for rep in range(repeats):
        pred = np.empty(n)
        for fold in np.array_split(rng.permutation(n), k):
            train = np.setdiff1d(np.arange(n), fold)
            pred[fold] = _lstsq_predict(X[train], y[train], X[fold])
        sse = float(np.sum((y - pred) ** 2))
        rmse[rep] = np.sqrt(sse / n)
        q2[rep] = 1.0 - sse / tss
    return CVResult(k, rmse, q2)


@dataclass(frozen=True)
class EliminationStep:
    """Шаг исключения: какой регрессор убран, его p-значение и качество модели после шага."""

    removed: Optional[str]
    p_removed: float
    columns: tuple
    r2: float
    r2_adj: float
    s: float
    bic: float


def backward_elimination(data: LipoData, alpha: float = 0.05,
                         start: Optional[Sequence[str]] = None) -> List[EliminationStep]:
    """Последовательно убирать регрессор с наибольшим p-значением t-теста, пока все p < ``alpha``.

    Первый элемент списка — исходная модель (``removed=None``), последний — итоговая.
    """
    columns = list(start if start is not None else data.columns)
    fit = fit_ols_data(data, columns)
    steps = [EliminationStep(None, float("nan"), tuple(columns), fit.r2, fit.r2_adj, fit.s, fit.bic)]
    while columns:
        p = fit.p_values[1:]
        worst = int(np.argmax(p))
        if p[worst] < alpha:
            break
        removed = columns.pop(worst)
        fit = fit_ols_data(data, columns)
        steps.append(EliminationStep(removed, float(p[worst]), tuple(columns),
                                     fit.r2, fit.r2_adj, fit.s, fit.bic))
    return steps


def final_model(data: LipoData, alpha: float = 0.05) -> OLSResult:
    """Модель после :func:`backward_elimination`."""
    return fit_ols_data(data, backward_elimination(data, alpha)[-1].columns)
