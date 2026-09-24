r"""Теоретические распределения расстояния между соседними уровнями.

Непрерывные законы (все, кроме :math:`\rho_{2\times2}`, нормированы на
среднее 1):

=========================  =====================================================
Функция                    Закон
=========================  =====================================================
:func:`wigner_pdf`         догадка Вигнера (15): :math:`\frac{\pi s}{2} e^{-\pi s^2/4}`
:func:`goe2_raw_pdf`       формула (13) до нормировки: :math:`\frac{s}{8} e^{-s^2/16}`
:func:`poisson_pdf`        некоррелированные уровни: :math:`e^{-s}`
:func:`goe_limit_pdf`      точный закон GOE при :math:`n \to \infty` (Годен–Мета)
:func:`normal_mirror_2x2_pdf`  матрица 2×2 с одинаковыми дисперсиями всех элементов
=========================  =====================================================

Дискретные законы для матриц 2×2 из ±1 (пункт 10 задания) строит
:func:`pm1_2x2_distribution`; вывод приведён в её документации и в отчёте.

Точный закон GOE
----------------
Для :math:`n \to \infty` плотность расстояний выражается через вероятность
:math:`E(0; s)` того, что на отрезке длины :math:`s` нет ни одного уровня:
:math:`p(s) = E''(s)`. Для GOE (Мета [2], гл. 6–7)

.. math::
    E(0; s) = \det\bigl(I - K_s\bigr), \qquad
    K_s(x, y) = \operatorname{sinc}(x - y) + \operatorname{sinc}(x + y),
    \quad x, y \in (0, s/2),

где :math:`\operatorname{sinc} u = \sin(\pi u) / (\pi u)`. Определитель
Фредгольма считается по методу Борнемана: интегральный оператор заменяется
матрицей на узлах квадратуры Гаусса–Лежандра, и ошибка убывает
экспоненциально с числом узлов (F. Bornemann, Math. Comp. 79 (2010), 871–915).
Это эталон, с которым полезно сравнивать ансамбль 16×16: догадка Вигнера
точна только для 2×2.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import special

from .ensembles import RandomState, make_rng

__all__ = [
    "DiscreteDistribution",
    "GOE2_RAW_MEAN",
    "NORMAL_MIRROR_2X2_RAW_MEAN",
    "WIGNER_SECOND_MOMENT",
    "WIGNER_VARIANCE",
    "gap_probability",
    "goe2_raw_cdf",
    "goe2_raw_pdf",
    "goe_limit_cdf",
    "goe_limit_pdf",
    "goe_limit_table",
    "normal_mirror_2x2_cdf",
    "normal_mirror_2x2_pdf",
    "pm1_2x2_distribution",
    "poisson_cdf",
    "poisson_pdf",
    "wigner_cdf",
    "wigner_pdf",
    "wigner_sample",
]

#: :math:`\langle s^2 \rangle = 4/\pi` для догадки Вигнера.
WIGNER_SECOND_MOMENT = 4.0 / math.pi
#: Дисперсия догадки Вигнера :math:`4/\pi - 1 \approx 0.2732`.
WIGNER_VARIANCE = WIGNER_SECOND_MOMENT - 1.0
#: Среднее (14) расстояния для GOE 2×2 при σ = 1 до нормировки: :math:`2\sqrt{\pi}`.
GOE2_RAW_MEAN = 2.0 * math.sqrt(math.pi)
#: Среднее расстояние для ``normal-mirror`` 2×2: :math:`2\sqrt{2/\pi}\,E(1/2)`.
NORMAL_MIRROR_2X2_RAW_MEAN = 2.0 * math.sqrt(2.0 / math.pi) * float(special.ellipe(0.5))


def _as_float_array(s) -> np.ndarray:
    return np.asarray(s, dtype=float)


def wigner_pdf(s):
    r"""Догадка Вигнера (15): :math:`\rho(s) = \frac{\pi s}{2} e^{-\pi s^2 / 4}`, :math:`s \ge 0`.

    >>> round(float(wigner_pdf(1.0)), 6)
    0.716186
    """
    s = _as_float_array(s)
    return np.where(s >= 0.0, 0.5 * np.pi * s * np.exp(-0.25 * np.pi * s * s), 0.0)


def wigner_cdf(s):
    r"""Функция распределения догадки Вигнера: :math:`F(s) = 1 - e^{-\pi s^2 / 4}`."""
    s = np.maximum(_as_float_array(s), 0.0)
    return -np.expm1(-0.25 * np.pi * s * s)


def wigner_sample(size: int, rng: RandomState = None) -> np.ndarray:
    r"""Выборка из догадки Вигнера методом обратной функции: :math:`s = \sqrt{-4 \ln U / \pi}`.

    Нужна как «идеальный» эталон: так выглядела бы выборка, если бы догадка
    Вигнера была точной.

    >>> float(abs(wigner_sample(200_000, rng=0).mean() - 1.0)) < 0.01
    True
    """
    u = make_rng(rng).random(size)
    return np.sqrt(-4.0 * np.log1p(-u) / np.pi)


def goe2_raw_pdf(s):
    r"""Формула (13): плотность :math:`s = \lambda_2 - \lambda_1` для :math:`A + A^T`, σ = 1.

    Это распределение Рэлея: :math:`b` и :math:`d = (c - a)/2` независимы и
    имеют дисперсию 2, а :math:`s = 2\sqrt{b^2 + d^2}`.
    """
    s = _as_float_array(s)
    return np.where(s >= 0.0, s / 8.0 * np.exp(-s * s / 16.0), 0.0)


def goe2_raw_cdf(s):
    r""":math:`F(s) = 1 - e^{-s^2/16}` — функция распределения для (13)."""
    s = np.maximum(_as_float_array(s), 0.0)
    return -np.expm1(-s * s / 16.0)


def poisson_pdf(s):
    r"""Закон Пуассона :math:`e^{-s}` — расстояния между *независимыми* уровнями."""
    s = _as_float_array(s)
    return np.where(s >= 0.0, np.exp(-np.maximum(s, 0.0)), 0.0)


def poisson_cdf(s):
    r""":math:`F(s) = 1 - e^{-s}`."""
    return -np.expm1(-np.maximum(_as_float_array(s), 0.0))


# --------------------------------------------------------------------------
# Точный закон GOE при n → ∞
# --------------------------------------------------------------------------


def gap_probability(s, nodes: int = 40):
    r"""Вероятность :math:`E(0; s)` отсутствия уровней GOE на отрезке длины ``s``.

    Средняя плотность уровней равна 1. ``nodes`` — число узлов квадратуры
    Гаусса–Лежандра; 40 узлов дают машинную точность при :math:`s \le 6`.
    Принимает число или массив; определители для массива считаются одной
    пачкой.

    >>> float(gap_probability(0.0))
    1.0
    >>> 0.0 < float(gap_probability(1.0)) < 1.0
    True
    """
    s = _as_float_array(s)
    if np.any(s < 0.0):
        raise ValueError("длина отрезка должна быть неотрицательной")
    x, w = np.polynomial.legendre.leggauss(nodes)
    half = 0.5 * s.reshape(-1, 1)
    x = 0.5 * half * (x + 1.0)          # узлы на (0, s/2), форма (len(s), nodes)
    root_w = np.sqrt(0.5 * half * w)
    kernel = np.sinc(x[:, :, None] - x[:, None, :]) + np.sinc(x[:, :, None] + x[:, None, :])
    matrices = np.eye(nodes) - root_w[:, :, None] * kernel * root_w[:, None, :]
    return np.linalg.det(matrices).reshape(s.shape)


@functools.lru_cache(maxsize=4)
def goe_limit_table(
    s_max: float = 6.0, step: float = 0.005, nodes: int = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Таблица точного закона GOE: узлы :math:`s`, плотность :math:`p(s)`, функция :math:`F(s)`.

    Плотность :math:`p(s) = E''(s)` находится центральной разностью второго
    порядка, в нуле берётся точное :math:`p(0) = 0` (отталкивание уровней).
    Функция распределения — интеграл плотности по формуле трапеций. За
    пределами :math:`s_{max} = 6` масса закона меньше :math:`10^{-9}`.
    """
    grid = np.arange(0.0, s_max + 0.5 * step, step)
    gap = gap_probability(grid, nodes)
    density = np.zeros_like(gap)
    density[1:-1] = (gap[2:] - 2.0 * gap[1:-1] + gap[:-2]) / step**2
    density = np.maximum(density, 0.0)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * step * (density[1:] + density[:-1]))))
    density /= cdf[-1]
    cdf /= cdf[-1]
    for array in (grid, density, cdf):
        array.setflags(write=False)
    return grid, density, cdf


def goe_limit_pdf(s):
    r"""Плотность расстояний GOE при :math:`n \to \infty` (среднее 1).

    >>> round(float(goe_limit_pdf(1.0)), 3)   # догадка Вигнера даёт 0.716
    0.702
    """
    grid, density, _ = goe_limit_table()
    return np.interp(_as_float_array(s), grid, density, left=0.0, right=0.0)


def goe_limit_cdf(s):
    r"""Функция распределения расстояний GOE при :math:`n \to \infty`."""
    grid, _, cdf = goe_limit_table()
    return np.interp(_as_float_array(s), grid, cdf, left=0.0, right=1.0)


# --------------------------------------------------------------------------
# 2×2 с одинаковыми дисперсиями элементов (ансамбль normal-mirror)
# --------------------------------------------------------------------------


def _normal_mirror_2x2_raw_pdf(s: np.ndarray) -> np.ndarray:
    # b ~ N(0, 1), d = (c - a)/2 ~ N(0, 1/2); r = sqrt(b^2 + d^2), s = 2r.
    # p_r(r) = sqrt(2) r exp(-3r^2/4) I0(r^2/4); i0e(x) = I0(x) e^{-x}.
    r = 0.5 * np.maximum(s, 0.0)
    p_r = math.sqrt(2.0) * r * special.i0e(0.25 * r * r) * np.exp(-0.5 * r * r)
    return np.where(s >= 0.0, 0.5 * p_r, 0.0)


def normal_mirror_2x2_pdf(s, normalized: bool = True):
    r"""Плотность :math:`s = \lambda_2 - \lambda_1` для симметричной 2×2 с элементами N(0, 1).

    Здесь :math:`a, b, c \sim \mathcal{N}(0, 1)`, поэтому
    :math:`d = (c - a)/2 \sim \mathcal{N}(0, 1/2)`, а :math:`b \sim \mathcal{N}(0, 1)`:
    точка :math:`(d, b)` распределена *анизотропно*, и интегрирование по углу
    даёт функцию Бесселя вместо константы:

    .. math::
        \rho(s) = \frac{s}{2\sqrt{2}}\, e^{-3 s^2/16}\, I_0\!\left(\frac{s^2}{16}\right),
        \qquad \langle s \rangle = 2\sqrt{2/\pi}\, E(1/2) \approx 2.155,

    где :math:`E` — полный эллиптический интеграл второго рода. При
    ``normalized=True`` закон перемасштабирован на среднее 1.
    """
    s = _as_float_array(s)
    if not normalized:
        return _normal_mirror_2x2_raw_pdf(s)
    mu = NORMAL_MIRROR_2X2_RAW_MEAN
    return mu * _normal_mirror_2x2_raw_pdf(mu * s)


@functools.lru_cache(maxsize=1)
def _normal_mirror_2x2_cdf_table() -> Tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 8.0, 16001)
    density = normal_mirror_2x2_pdf(grid)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))))
    return grid, cdf / cdf[-1]


def normal_mirror_2x2_cdf(s):
    """Функция распределения для :func:`normal_mirror_2x2_pdf` (среднее 1)."""
    grid, cdf = _normal_mirror_2x2_cdf_table()
    return np.interp(_as_float_array(s), grid, cdf, left=0.0, right=1.0)


# --------------------------------------------------------------------------
# Дискретные законы
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscreteDistribution:
    """Дискретное распределение: атомы ``values`` с вероятностями ``probabilities``.

    >>> d = DiscreteDistribution([2.0, 1.0], [0.5, 0.5])
    >>> d.values, d.mean
    (array([1., 2.]), 1.5)
    """

    values: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).ravel()
        probabilities = np.asarray(self.probabilities, dtype=float).ravel()
        if values.shape != probabilities.shape or values.size == 0:
            raise ValueError("нужно одинаковое ненулевое число атомов и вероятностей")
        if np.any(probabilities < 0.0) or not math.isclose(probabilities.sum(), 1.0, abs_tol=1e-9):
            raise ValueError("вероятности должны быть неотрицательны и давать в сумме 1")
        order = np.argsort(values, kind="stable")
        object.__setattr__(self, "values", values[order])
        object.__setattr__(self, "probabilities", probabilities[order])

    @property
    def mean(self) -> float:
        return float(np.dot(self.values, self.probabilities))

    @property
    def second_moment(self) -> float:
        return float(np.dot(self.values**2, self.probabilities))

    @property
    def variance(self) -> float:
        return self.second_moment - self.mean**2

    def normalized(self) -> "DiscreteDistribution":
        """Тот же закон, перемасштабированный на среднее 1 (пункт 4)."""
        if self.mean <= 0.0:
            raise ValueError("нормировка невозможна: среднее не положительно")
        return DiscreteDistribution(self.values / self.mean, self.probabilities)

    def cdf(self, s):
        """:math:`F(s) = P(X \\le s)` — ступенчатая функция."""
        cumulative = np.concatenate(([0.0], np.cumsum(self.probabilities)))
        index = np.searchsorted(self.values, _as_float_array(s), side="right")
        return np.minimum(cumulative[index], 1.0)

    def probability_of(self, value: float, tol: float = 1e-9) -> float:
        """Вероятность атома ``value`` (0, если такого атома нет)."""
        return float(self.probabilities[np.abs(self.values - value) <= tol].sum())

    def sample(self, size: int, rng: RandomState = None) -> np.ndarray:
        return make_rng(rng).choice(self.values, size=size, p=self.probabilities)


def pm1_2x2_distribution(construction: str = "mirror") -> DiscreteDistribution:
    r"""Точный закон :math:`s = \lambda_2 - \lambda_1` для матриц 2×2 из ±1 (пункт 10), до нормировки.

    По формуле (5) :math:`s = 2\sqrt{d^2 + b^2}`, :math:`d = (c - a)/2`.

    **mirror** — :math:`H = \begin{bmatrix} a & b \\ b & c \end{bmatrix}`,
    :math:`a, b, c = \pm 1` независимо. Тогда :math:`b^2 = 1` всегда, а
    :math:`d = 0` при :math:`a = c` (вероятность 1/2) и :math:`d = \pm 1`
    иначе. Значит :math:`s = 2` или :math:`s = 2\sqrt{2}` с вероятностями
    1/2; :math:`\langle s \rangle = 1 + \sqrt{2}`, после нормировки атомы
    :math:`2\sqrt{2} - 2 \approx 0.828` и :math:`4 - 2\sqrt{2} \approx 1.172`.

    **sum** — :math:`H = A + A^T`: :math:`a = 2A_{11}`, :math:`c = 2A_{22}`,
    :math:`b = A_{12} + A_{21}`. Теперь :math:`b` и :math:`d` независимы и
    принимают значения :math:`0` (1/2) и :math:`\pm 2` (1/2). Значит
    :math:`s = 0` (1/4), :math:`4` (1/2), :math:`4\sqrt{2}` (1/4);
    :math:`\langle s \rangle = 2 + \sqrt{2}`, после нормировки атомы
    :math:`0`, :math:`4 - 2\sqrt{2} \approx 1.172`,
    :math:`4\sqrt{2} - 4 \approx 1.657`.

    >>> pm1_2x2_distribution("mirror").normalized().values.round(4)
    array([0.8284, 1.1716])
    >>> pm1_2x2_distribution("sum").normalized().values.round(4)
    array([0.    , 1.1716, 1.6569])
    """
    root2 = math.sqrt(2.0)
    if construction == "mirror":
        return DiscreteDistribution([2.0, 2.0 * root2], [0.5, 0.5])
    if construction == "sum":
        return DiscreteDistribution([0.0, 4.0, 4.0 * root2], [0.25, 0.5, 0.25])
    raise ValueError(f"неизвестный способ симметризации {construction!r}; доступны: sum, mirror")
