r"""Распределения Стьюдента и Фишера по формулам лекции 3.

Для доверительных интервалов (пункт 6) нужен квантиль
:math:`t^{n-K}_\alpha`, при котором

.. math::

    \int_{-t^{n-K}_\alpha}^{t^{n-K}_\alpha} \rho^{n-K}_t(x)\,dx = \alpha,

где :math:`\alpha` — доверительная вероятность (в лекции :math:`\alpha = 0.95`).
Здесь плотности записаны «как в лекции» через гамма-функцию, а квантили
находятся численным интегрированием и методом Брента. В расчётах пакета
используются те же величины из :mod:`scipy.stats`; модуль нужен, чтобы
показать, откуда берутся числа, и проверить их (тесты сравнивают оба
способа до :math:`10^{-9}`).

>>> round(student_quantile(0.95, 57), 4)          # n − K = 82 − 25
2.0025
>>> round(fisher_quantile(0.95, 24, 57), 4)
1.7098
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize

__all__ = [
    "fisher_pdf",
    "fisher_quantile",
    "student_central_probability",
    "student_pdf",
    "student_quantile",
]


def student_pdf(x, nu: float):
    r"""Плотность распределения Стьюдента с :math:`\nu` степенями свободы.

    .. math::

        \rho_t^{\nu}(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}
        {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}}
        \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}

    Гамма-функция берётся в логарифмах (:func:`math.lgamma`), иначе при
    больших :math:`\nu` она переполняется.

    >>> round(float(student_pdf(0.0, 1)), 6)          # Коши: 1/π
    0.31831
    """
    if nu <= 0:
        raise ValueError("число степеней свободы должно быть положительным")
    x = np.asarray(x, dtype=float)
    log_norm = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    return np.exp(log_norm - (nu + 1) / 2 * np.log1p(x * x / nu))


def student_central_probability(t: float, nu: float) -> float:
    r""":math:`\int_{-t}^{t} \rho_t^{\nu}(x)\,dx` — вероятность попасть в :math:`[-t, t]`."""
    if t <= 0:
        return 0.0
    value, _ = integrate.quad(lambda x: float(student_pdf(x, nu)), 0.0, t, epsabs=1e-13, epsrel=1e-12)
    return 2.0 * value


def student_quantile(alpha: float, nu: float) -> float:
    r"""Двусторонний квантиль :math:`t^{\nu}_\alpha`: :math:`P(|T| \le t) = \alpha`."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("доверительная вероятность α должна лежать в (0, 1)")
    upper = 1.0
    while student_central_probability(upper, nu) < alpha:
        upper *= 2.0
    return optimize.brentq(lambda t: student_central_probability(t, nu) - alpha, 0.0, upper,
                           xtol=1e-14, rtol=1e-14)


def fisher_pdf(x, d1: float, d2: float):
    r"""Плотность распределения Фишера :math:`F(d_1, d_2)` (лекция 3, «F-отношение»).

    .. math::

        \rho_F(x, d_1, d_2) = \frac{\Gamma\left(\frac{d_1}{2} + \frac{d_2}{2}\right)}
        {x\,\Gamma\left(\frac{d_1}{2}\right)\Gamma\left(\frac{d_2}{2}\right)}
        \sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}},
        \qquad x > 0.

    Здесь :math:`d_1 = \#r` — число ограничений, :math:`d_2 = n - K`.
    """
    if d1 <= 0 or d2 <= 0:
        raise ValueError("числа степеней свободы должны быть положительными")
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    xp = x[pos]
    log_norm = math.lgamma((d1 + d2) / 2) - math.lgamma(d1 / 2) - math.lgamma(d2 / 2)
    log_root = 0.5 * (d1 * np.log(d1 * xp) + d2 * math.log(d2) - (d1 + d2) * np.log(d1 * xp + d2))
    out[pos] = np.exp(log_norm + log_root - np.log(xp))
    return out if out.ndim else float(out)


def fisher_quantile(alpha: float, d1: float, d2: float) -> float:
    r""":math:`F_\alpha(d_1, d_2)`: :math:`\int_0^{F_\alpha} \rho_F(x)\,dx = \alpha`."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("доверительная вероятность α должна лежать в (0, 1)")

    def cdf(f: float) -> float:
        value, _ = integrate.quad(lambda x: float(fisher_pdf(np.array(x), d1, d2)), 0.0, f,
                                  epsabs=1e-13, epsrel=1e-12, limit=200)
        return value

    upper = 1.0
    while cdf(upper) < alpha:
        upper *= 2.0
    return optimize.brentq(lambda f: cdf(f) - alpha, 0.0, upper, xtol=1e-13, rtol=1e-13)
