r"""Какие регрессоры вносят наибольший вклад в прогноз перцептрона (пункт 5).

У перцептрона нет одного коэффициента на регрессор, как у линейной модели:
вклад :math:`x_j` зависит от того, какие нейроны активны. Используются две
стандартные меры, обе — в стандартизованных единицах, чтобы их можно было
сравнить со стандартизованными коэффициентами МНК :math:`\beta^*_j` из ЛР3:

* **чувствительность** — среднее по веществам значение производной
  :math:`\partial\hat y/\partial x_j` (и её модуля). Для ReLU-сети она
  кусочно-постоянна: :math:`\nabla_x\hat y = (w^1)^T (w^2 \circ H(w^1x + b^1))`.
  Для линейной модели производная постоянна и равна :math:`\beta^*_j`;
* **перестановочная важность** — насколько вырастет среднеквадратичная
  ошибка, если значения :math:`x_j` случайно перемешать между веществами
  (связь :math:`x_j` с откликом разрушается, распределение :math:`x_j` — нет).
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from .network import Params, input_gradients, predict

__all__ = ["ensemble_predict", "ols_standardized", "permutation_importance", "sensitivity"]


def sensitivity(params_list: Sequence[Params], Z: np.ndarray) -> Dict[str, np.ndarray]:
    """Средние по веществам (и по ансамблю сетей) производные прогноза по стандартизованным входам.

    Возвращает ``mean`` — среднюю производную (со знаком) и ``mean_abs`` —
    средний модуль, оба формы ``(d,)``.
    """
    grads = np.stack([input_gradients(p, Z) for p in params_list])     # (сети, n, d)
    return {"mean": grads.mean(axis=(0, 1)), "mean_abs": np.abs(grads).mean(axis=(0, 1))}


def permutation_importance(predict_fn: Callable[[np.ndarray], np.ndarray], Z: np.ndarray, t: np.ndarray,
                           repeats: int = 30, seed: int = 0) -> np.ndarray:
    r"""Прирост MSE (в стандартизованных единицах отклика) при перемешивании каждого признака."""
    rng = np.random.default_rng(seed)
    base = float(np.mean((t - predict_fn(Z)) ** 2))
    out = np.zeros(Z.shape[1])
    for j in range(Z.shape[1]):
        for _ in range(repeats):
            Zp = Z.copy()
            Zp[:, j] = Z[rng.permutation(len(t)), j]
            out[j] += float(np.mean((t - predict_fn(Zp)) ** 2)) - base
    return out / repeats


def ensemble_predict(params_list: Sequence[Params]) -> Callable[[np.ndarray], np.ndarray]:
    """Прогноз ансамбля — среднее прогнозов сетей."""
    return lambda Z: np.mean([predict(p, Z) for p in params_list], axis=0)


def ols_standardized(Z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Стандартизованные коэффициенты МНК :math:`\\beta^*`.

    Регрессия стандартизованного отклика на стандартизованные признаки.
    """
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(t)), Z]), t, rcond=None)
    return coef[1:]
