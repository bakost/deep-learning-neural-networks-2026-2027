r"""Задача 1: одномерная минимизация без производных.

Функция задания

.. math::

    f(z) = 1 - \tfrac12 e^{-10(z - 0.65)^2} - 0.45\, e^{-50(z - 0.35)^2}

и три алгоритма из Numerical Recipes [NR, гл. 10]:

* :func:`bracket_minimum` — начальная локализация минимума тройкой
  :math:`a < b < c`, :math:`f(b) < f(a), f(c)` (``mnbrak``, разд. 10.1);
* :func:`golden_section` — метод золотого сечения (``Golden``, разд. 10.2):
  отрезок, содержащий минимум, на каждом шаге сокращается в
  :math:`\varphi = 1.618\ldots` раз, одно новое вычисление функции на шаг;
* :func:`brent` — метод Брента (``Brent``, разд. 10.3): параболическая
  интерполяция по трём лучшим точкам, а когда парабола ненадёжна — шаг
  золотого сечения. Сходимость сверхлинейная.

Обе процедуры останавливаются, когда длина отрезка неопределённости
становится меньше :math:`2\,\mathrm{tol}\,|x|`. Точнее, чем
:math:`\mathrm{tol} \approx \sqrt{\varepsilon_{маш}} \approx 1.5\cdot10^{-8}`,
минимум по значениям функции найти нельзя: вблизи минимума
:math:`f(x) \approx f(x^*) + \tfrac12 f''(x^*)(x - x^*)^2`, и при
:math:`|x - x^*| \lesssim \sqrt{\varepsilon_{маш}}` разность значений
теряется в округлении (формула (10.1.2) NR).

Записи истории — кортежи ``(a, b, x, f(x), kind)``: текущий отрезок
неопределённости, лучшая точка и тип шага (``"golden"`` или ``"parabolic"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np

__all__ = [
    "GOLDEN_RATIO",
    "LineSearchResult",
    "bracket_minimum",
    "brent",
    "df_task",
    "f_task",
    "golden_section",
    "exact_minimum",
]

GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
_R = GOLDEN_RATIO - 1.0  # 0.618…, доля отрезка, остающаяся после шага
_C = 1.0 - _R            # 0.382…
_TOL = math.sqrt(np.finfo(float).eps)


def f_task(z):
    """Функция задачи 1 (работает и с массивами)."""
    z = np.asarray(z, dtype=float) if not np.isscalar(z) else float(z)
    return 1.0 - 0.5 * np.exp(-10.0 * (z - 0.65) ** 2) - 0.45 * np.exp(-50.0 * (z - 0.35) ** 2)


def df_task(z):
    r"""Производная :math:`f'(z) = 10(z-0.65)e^{-10(z-0.65)^2} + 45(z-0.35)e^{-50(z-0.35)^2}`.

    Методы золотого сечения и Брента её не используют; она нужна только
    для проверки результата (:func:`exact_minimum`).
    """
    z = np.asarray(z, dtype=float) if not np.isscalar(z) else float(z)
    return 10.0 * (z - 0.65) * np.exp(-10.0 * (z - 0.65) ** 2) + 45.0 * (z - 0.35) * np.exp(-50.0 * (z - 0.35) ** 2)


class _Counted:
    """Обёртка, считающая вычисления функции."""

    def __init__(self, fun: Callable[[float], float]):
        self.fun, self.n = fun, 0

    def __call__(self, x: float) -> float:
        self.n += 1
        return float(self.fun(x))


@dataclass
class LineSearchResult:
    x: float
    fx: float
    nfev: int
    iterations: int
    history: List[Tuple[float, float, float, float, str]] = field(default_factory=list)
    method: str = ""

    @property
    def widths(self) -> np.ndarray:
        """Длины отрезка неопределённости по итерациям."""
        return np.array([b - a for a, b, *_ in self.history])


def bracket_minimum(fun: Callable[[float], float], a: float = 0.0, b: float = 1.0,
                    grow_limit: float = 100.0) -> Tuple[Tuple[float, float, float], int]:
    """Найти тройку ``(a, b, c)`` с ``f(b) <= f(a)`` и ``f(b) < f(c)`` (``mnbrak`` из NR 10.1).

    Идём под уклон от ``a`` к ``b``, увеличивая шаг в :math:`\\varphi` раз
    и пробуя экстраполяцию параболой. Возвращает тройку (упорядоченную по
    возрастанию) и число вычислений функции.
    """
    f = _Counted(fun)
    fa, fb = f(a), f(b)
    if fb > fa:
        a, b, fa, fb = b, a, fb, fa
    c = b + GOLDEN_RATIO * (b - a)
    fc = f(c)
    while fb > fc:
        r = (b - a) * (fb - fc)
        q = (b - c) * (fb - fa)
        denom = 2.0 * math.copysign(max(abs(q - r), 1e-20), q - r)
        u = b - ((b - c) * q - (b - a) * r) / denom
        ulim = b + grow_limit * (c - b)
        if (b - u) * (u - c) > 0.0:          # парабола между b и c
            fu = f(u)
            if fu < fc:
                a, b, fa, fb = b, u, fb, fu
                break
            if fu > fb:
                c, fc = u, fu
                break
            u = c + GOLDEN_RATIO * (c - b)
            fu = f(u)
        elif (c - u) * (u - ulim) > 0.0:     # парабола между c и пределом
            fu = f(u)
            if fu < fc:
                b, c, u = c, u, u + GOLDEN_RATIO * (u - c)
                fb, fc, fu = fc, fu, f(u)
        elif (u - ulim) * (ulim - c) >= 0.0:
            u = ulim
            fu = f(u)
        else:
            u = c + GOLDEN_RATIO * (c - b)
            fu = f(u)
        a, b, c = b, c, u
        fa, fb, fc = fb, fc, fu
    lo, hi = (a, c) if a < c else (c, a)
    return (lo, b, hi), f.n


def golden_section(fun: Callable[[float], float], bracket: Tuple[float, float, float],
                   tol: float = _TOL, max_iter: int = 500) -> LineSearchResult:
    r"""Метод золотого сечения (``Golden`` из NR 10.2).

    ``bracket`` — тройка :math:`a < b < c` с :math:`f(b) < f(a), f(c)`.
    Новая точка ставится в долю :math:`0.382` *большего* из отрезков
    :math:`[a, b]`, :math:`[b, c]`; после этого отрезок неопределённости
    сокращается в :math:`1/0.618 = 1.618` раза.
    """
    f = _Counted(fun)
    x0, b, x3 = bracket
    if abs(x3 - b) > abs(b - x0):
        x1, x2 = b, b + _C * (x3 - b)
    else:
        x2, x1 = b, b - _C * (b - x0)
    f1, f2 = f(x1), f(x2)
    history = []
    it = 0
    while abs(x3 - x0) > tol * (abs(x1) + abs(x2)) and it < max_iter:
        if f2 < f1:
            x0, x1, x2 = x1, x2, _R * x2 + _C * x3
            f1, f2 = f2, f(x2)
        else:
            x3, x2, x1 = x2, x1, _R * x1 + _C * x0
            f2, f1 = f1, f(x1)
        it += 1
        best = (x1, f1) if f1 < f2 else (x2, f2)
        history.append((min(x0, x3), max(x0, x3), best[0], best[1], "golden"))
    x, fx = (x1, f1) if f1 < f2 else (x2, f2)
    return LineSearchResult(x, fx, f.n, it, history, "golden")


def brent(fun: Callable[[float], float], bracket: Tuple[float, float, float],
          tol: float = _TOL, max_iter: int = 200) -> LineSearchResult:
    r"""Метод Брента (``Brent`` из NR 10.3).

    Хранит три точки: ``x`` — лучшую, ``w`` — вторую по значению, ``v`` —
    предыдущее значение ``w``. Через них проводится парабола; её вершина
    принимается, если лежит внутри отрезка и шаг меньше половины
    предпоследнего (иначе возможен «цикл» без сокращения отрезка). В
    противном случае делается шаг золотого сечения в больший из отрезков.
    """
    f = _Counted(fun)
    ax, bx, cx = bracket
    a, b = min(ax, cx), max(ax, cx)
    x = w = v = bx
    fx = fw = fv = f(x)
    d = e = 0.0
    zeps = np.finfo(float).eps * 1e-3
    history = []
    for it in range(max_iter):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + zeps
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            return LineSearchResult(x, fx, f.n, it, history, "brent")
        kind = "golden"
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp, e = e, d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = a - x if x >= xm else b - x
                d = _C * e
            else:
                d = p / q
                u = x + d
                kind = "parabolic"
                if u - a < tol2 or b - u < tol2:
                    d = math.copysign(tol1, xm - x)
        else:
            e = a - x if x >= xm else b - x
            d = _C * e
        u = x + d if abs(d) >= tol1 else x + math.copysign(tol1, d)
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
        history.append((a, b, x, fx, kind))
    raise RuntimeError("метод Брента: превышено число итераций")


def exact_minimum(lo: float = 0.3, hi: float = 0.45) -> float:
    """Точка минимума как корень :math:`f'(z) = 0` (метод Брента для корня из SciPy).

    Корень производной находится с точностью до :math:`\\sim 10^{-16}`, а не
    :math:`\\sqrt{\\varepsilon_{маш}}`, потому что :math:`f'` меняет знак
    линейно. Используется как эталон для оценки погрешности поиска по
    значениям функции.
    """
    from scipy.optimize import brentq

    return float(brentq(df_task, lo, hi, xtol=1e-16, rtol=4 * np.finfo(float).eps))
