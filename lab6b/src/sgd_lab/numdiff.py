r"""Численное дифференцирование (Numerical Recipes, разд. 5.7).

Центральная разность

.. math::

    \frac{\partial J}{\partial w_k} \approx \frac{J(w + h_k e_k) - J(w - h_k e_k)}{2 h_k}

имеет погрешность усечения :math:`\tfrac16 J''' h^2` и погрешность
округления :math:`\sim \varepsilon_{маш} |J| / h`. Их сумма минимальна при

.. math::

    h \sim \varepsilon_{маш}^{1/3}\, x_c \approx 6\cdot10^{-6}\, x_c

(формула (5.7.8) NR), где :math:`x_c` — характерный масштаб аргумента; здесь
:math:`x_c = \max(|w_k|, 1)`. Относительная точность производной тогда
:math:`\sim \varepsilon_{маш}^{2/3} \approx 4\cdot10^{-11}`. Шаг делается
точно представимым в двоичной арифметике: ``h = (w + h) - w`` (5.7.4).

Для проверки реализован и метод Риддерса (``dfridr``, тот же раздел):
экстраполяция центральных разностей к :math:`h \to 0` по Ричардсону,
точность близка к машинной.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

__all__ = ["central_gradient", "ridders_derivative", "STEP_FACTOR"]

STEP_FACTOR = np.finfo(float).eps ** (1.0 / 3.0)


def central_gradient(fun: Callable[[np.ndarray], np.ndarray], w: np.ndarray) -> np.ndarray:
    """Градиент центральными разностями за один векторизованный вызов ``fun``.

    ``fun`` принимает массив параметров формы ``(..., K, p)`` и возвращает
    ``(..., K)``. Строки — :math:`w + h_k e_k` и :math:`w - h_k e_k` для всех
    :math:`k`. ``w`` формы ``(p,)`` — одна точка, ``(S, p)`` — ``S`` точек сразу.
    """
    w = np.asarray(w, dtype=float)
    p = w.shape[-1]
    h = STEP_FACTOR * np.maximum(np.abs(w), 1.0)
    h = (w + h) - w                                 # точно представимый шаг
    pts = np.repeat(w[..., None, :], 2 * p, axis=-2)
    idx = np.arange(p)
    pts[..., idx, idx] += h
    pts[..., p + idx, idx] -= h
    vals = fun(pts)
    return (vals[..., :p] - vals[..., p:]) / (2.0 * h)


def ridders_derivative(fun: Callable[[float], float], x: float, h: float = 0.1,
                       ntab: int = 10, con: float = 1.4, safe: float = 2.0) -> Tuple[float, float]:
    """Производная скалярной функции методом Риддерса (``dfridr`` из NR 5.7).

    Возвращает оценку производной и оценку её погрешности.
    """
    con2 = con * con
    a = np.zeros((ntab, ntab))
    a[0, 0] = (fun(x + h) - fun(x - h)) / (2.0 * h)
    err = np.inf
    ans = a[0, 0]
    for i in range(1, ntab):
        h /= con
        a[0, i] = (fun(x + h) - fun(x - h)) / (2.0 * h)
        fac = con2
        for j in range(1, i + 1):
            a[j, i] = (a[j - 1, i] * fac - a[j - 1, i - 1]) / (fac - 1.0)
            fac *= con2
            errt = max(abs(a[j, i] - a[j - 1, i]), abs(a[j, i] - a[j - 1, i - 1]))
            if errt <= err:
                err, ans = errt, a[j, i]
        if abs(a[i, i] - a[i - 1, i - 1]) >= safe * err:
            break
    return float(ans), float(err)
