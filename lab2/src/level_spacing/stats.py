r"""Гистограммы и количественные меры близости выборки к теоретическому закону.

Пункт 8 задания просит описать, насколько график (15) близок к гистограмме.
«На глаз» этого мало, поэтому используются четыре меры:

* **расстояние Колмогорова–Смирнова** :math:`D = \sup_s |F_M(s) - F(s)|` —
  не зависит от выбора бинов; при верной гипотезе и нормировке выборки на
  своё среднее типичное значение :math:`D \approx 0.66/\sqrt{M}`;
* **критерий** :math:`\chi^2` по бинам гистограммы;
* **площадь расхождения** :math:`L_1 = \int |h(s) - \rho(s)|\,ds` между
  гистограммой и кривой (0 — полное совпадение, 2 — непересекающиеся законы);
* **моменты**: при нормировке на среднее 1 у догадки Вигнера
  :math:`\langle s^2 \rangle = 4/\pi`, дисперсия :math:`4/\pi - 1 \approx 0.273`.

Про p-значения. Выборка нормирована на *своё же* среднее, то есть один
параметр оценён по данным. Для критерия :math:`\chi^2` это учитывается
уменьшением числа степеней свободы. Табличное распределение Колмогорова для
такой выборки не годится (p-значение получается сильно завышенным — поправка
Лиллиефорса), поэтому нулевое распределение :math:`D\sqrt{M}` строится
моделированием: :func:`ks_null_distribution` генерирует выборки *точно* из
проверяемого закона, нормирует их на своё среднее и считает :math:`D`.
При :math:`M \sim 10^5\text{–}10^6` оба критерия замечают отклонения порядка
процента, поэтому главное — сама величина расхождения, а не факт
«отвергается / не отвергается».
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats as _scipy_stats

__all__ = [
    "Histogram",
    "chi2_test",
    "degenerate_fraction",
    "histogram",
    "ks_null_distribution",
    "ks_pvalue_normalized",
    "ks_statistic",
    "ks_test",
    "l1_distance",
    "moments",
    "small_s_exponent",
    "small_s_exponent_of_pdf",
    "uniform_bins",
]

Cdf = Callable[[np.ndarray], np.ndarray]


def uniform_bins(width: float, upper: float = 4.0) -> np.ndarray:
    """Равномерные границы бинов на :math:`[0, \\text{upper}]`.

    >>> uniform_bins(0.5, 2.0)
    array([0. , 0.5, 1. , 1.5, 2. ])
    """
    if width <= 0.0 or upper <= 0.0:
        raise ValueError("ширина бина и правая граница должны быть положительными")
    count = round(upper / width)
    return np.linspace(0.0, count * width, count + 1)


@dataclass(frozen=True)
class Histogram:
    """Гистограмма, нормированная как плотность (пункт 5 задания).

    Attributes
    ----------
    edges:
        границы бинов;
    counts:
        число значений в каждом бине;
    total:
        объём выборки :math:`M` (включая значения вне ``edges``).
    """

    edges: np.ndarray
    counts: np.ndarray
    total: int

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def density(self) -> np.ndarray:
        """Высоты столбцов: доля выборки в бине, делённая на ширину бина."""
        return self.counts / (self.total * self.widths)

    @property
    def errors(self) -> np.ndarray:
        """Статистическая погрешность высоты (одна сигма, пуассоновская оценка)."""
        return np.sqrt(self.counts) / (self.total * self.widths)

    def expected_density(self, cdf: Cdf) -> np.ndarray:
        """Средняя по бину теоретическая плотность :math:`(F(b_{i+1}) - F(b_i)) / \\Delta_i`.

        Сравнивать гистограмму нужно именно с ней, а не со значением плотности
        в центре бина: иначе на крутых участках появляется систематическая
        ошибка порядка :math:`\\Delta^2 \\rho''`.
        """
        return np.diff(cdf(self.edges)) / self.widths


def histogram(samples: np.ndarray, edges: np.ndarray) -> Histogram:
    """Построить гистограмму выборки с заданными границами бинов."""
    samples = np.asarray(samples, dtype=float).ravel()
    counts, edges = np.histogram(samples, bins=edges)
    return Histogram(edges=edges, counts=counts, total=samples.size)


def ks_test(samples: np.ndarray, cdf: Cdf) -> Tuple[float, float]:
    """Статистика Колмогорова–Смирнова :math:`D` и p-значение для *полностью заданного* закона.

    Если выборка нормирована на своё среднее, p-значение нужно брать из
    :func:`ks_pvalue_normalized`.

    >>> rng = np.random.default_rng(0)
    >>> d, p = ks_test(rng.exponential(size=2000), lambda s: 1 - np.exp(-s))
    >>> d < 0.05 and p > 0.01
    True
    """
    result = _scipy_stats.kstest(np.asarray(samples, dtype=float).ravel(), cdf)
    return float(result.statistic), float(result.pvalue)


def ks_statistic(samples: np.ndarray, cdf: Cdf) -> float:
    r"""Статистика Колмогорова–Смирнова :math:`D = \sup_s |F_M(s) - F(s)|`.

    >>> ks_statistic(np.array([0.5]), lambda s: s)   # F_M прыгает с 0 до 1 в точке 0.5
    0.5
    """
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    f = cdf(x)
    m = x.size
    upper = np.max(np.arange(1, m + 1) / m - f)
    lower = np.max(f - np.arange(0, m) / m)
    return float(max(upper, lower))


def _inverse_cdf_sampler(cdf: Cdf, upper: float = 8.0, step: float = 5e-4):
    grid = np.arange(0.0, upper + step, step)
    values = np.maximum.accumulate(cdf(grid))
    keep = np.concatenate(([True], np.diff(values) > 0))   # строго возрастающая таблица
    grid, values = grid[keep], values[keep]
    return lambda u: np.interp(u, values, grid)


@functools.lru_cache(maxsize=8)
def ks_null_distribution(cdf: Cdf, size: int = 2000, repeats: int = 1000, seed: int = 0) -> np.ndarray:
    r"""Нулевое распределение :math:`D\sqrt{M}` для выборки, нормированной на своё среднее.

    Выборки объёма ``size`` берутся *точно* из закона ``cdf`` (методом
    обратной функции), нормируются на выборочное среднее и сравниваются с
    ``cdf``. При больших :math:`M` распределение :math:`D\sqrt{M}` от
    :math:`M` не зависит, поэтому один расчёт годится для любых объёмов.
    Возвращается отсортированный массив ``repeats`` значений.
    """
    rng = np.random.default_rng(seed)
    inverse = _inverse_cdf_sampler(cdf)
    root = np.sqrt(size)
    values = np.empty(repeats)
    for i in range(repeats):
        sample = inverse(rng.random(size))
        values[i] = ks_statistic(sample / sample.mean(), cdf) * root
    values.sort()
    values.setflags(write=False)
    return values


def ks_pvalue_normalized(d: float, size: int, cdf: Cdf) -> float:
    r"""p-значение критерия Колмогорова–Смирнова с поправкой на нормировку по среднему.

    Доля нулевого распределения :func:`ks_null_distribution`, не меньшая
    наблюдаемого :math:`D\sqrt{M}`; наименьшее возможное значение —
    :math:`1/(\text{repeats} + 1)`.
    """
    null = ks_null_distribution(cdf)
    exceed = null.size - np.searchsorted(null, d * np.sqrt(size), side="left")
    return float((exceed + 1) / (null.size + 1))


def chi2_test(
    samples: np.ndarray, cdf: Cdf, edges: np.ndarray, *, ddof: int = 1, min_expected: float = 5.0
) -> Tuple[float, int, float]:
    r"""Критерий :math:`\chi^2` согласия по бинам ``edges`` плюс хвост :math:`[b_{last}, \infty)`.

    Соседние бины объединяются, пока ожидаемое число попаданий меньше
    ``min_expected`` (стандартное условие применимости критерия). ``ddof`` —
    число параметров, оценённых по выборке (среднее при нормировке).

    Returns
    -------
    (chi2, dof, p_value)
    """
    samples = np.asarray(samples, dtype=float).ravel()
    edges = np.asarray(edges, dtype=float)
    observed = np.append(np.histogram(samples, bins=edges)[0], np.sum(samples >= edges[-1]))
    observed[0] += np.sum(samples < edges[0])
    probabilities = np.append(np.diff(cdf(edges)), 1.0 - cdf(edges[-1:]))
    probabilities[0] += cdf(edges[:1])[0]
    expected = probabilities * samples.size

    merged_obs, merged_exp = [], []
    acc_obs = acc_exp = 0.0
    for o, e in zip(observed, expected):
        acc_obs += o
        acc_exp += e
        if acc_exp >= min_expected:
            merged_obs.append(acc_obs)
            merged_exp.append(acc_exp)
            acc_obs = acc_exp = 0.0
    if acc_exp > 0.0 or acc_obs > 0:
        if merged_exp:
            merged_obs[-1] += acc_obs
            merged_exp[-1] += acc_exp
        else:
            merged_obs.append(acc_obs)
            merged_exp.append(acc_exp)
    obs = np.asarray(merged_obs)
    exp = np.asarray(merged_exp)
    if np.any(exp <= 0.0):
        raise ValueError("теоретический закон не даёт массы ни в одном бине")
    dof = max(obs.size - 1 - ddof, 1)
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    return chi2, dof, float(_scipy_stats.chi2.sf(chi2, dof))


def l1_distance(hist: Histogram, cdf: Cdf) -> float:
    r"""Площадь между гистограммой и теоретической кривой :math:`\sum_i |h_i - \bar\rho_i| \Delta_i`.

    Масса за правой границей гистограммы учитывается отдельно.
    """
    inside = float(np.sum(np.abs(hist.density - hist.expected_density(cdf)) * hist.widths))
    tail_observed = 1.0 - hist.counts.sum() / hist.total
    tail_expected = float(1.0 - cdf(hist.edges[-1:])[0] + cdf(hist.edges[:1])[0])
    return inside + abs(tail_observed - tail_expected)


def moments(samples: np.ndarray) -> Dict[str, float]:
    """Среднее, второй момент и дисперсия выборки с погрешностями (одна сигма).

    >>> m = moments(np.array([1.0, 1.0, 1.0]))
    >>> m["mean"], m["variance"]
    (1.0, 0.0)
    """
    samples = np.asarray(samples, dtype=float).ravel()
    size = samples.size
    mean = float(samples.mean())
    second = float(np.mean(samples**2))
    variance = float(samples.var())
    centered = samples - mean
    # Погрешность выборочной дисперсии: sqrt((mu4 - sigma^4) / M).
    variance_error = float(np.sqrt(max(np.mean(centered**4) - variance**2, 0.0) / size))
    return {
        "mean": mean,
        "mean_error": float(samples.std() / np.sqrt(size)),
        "second_moment": second,
        "variance": variance,
        "variance_error": variance_error,
    }


def degenerate_fraction(samples: np.ndarray, tol: float = 1e-9) -> float:
    """Доля вырожденных пар (:math:`s \\le` ``tol``): для дискретных ансамблей она бывает > 0."""
    samples = np.asarray(samples, dtype=float).ravel()
    return float(np.mean(samples <= tol))


def small_s_exponent(
    samples: np.ndarray, s_max: float = 0.2, min_count: int = 20, tol: float = 1e-9
) -> Optional[float]:
    r"""Показатель :math:`\gamma` в :math:`F(s) \propto s^{\gamma}` при малых :math:`s`.

    Если плотность ведёт себя как :math:`\rho(s) \propto s^{\beta}`, то
    :math:`F(s) \propto s^{\beta + 1}`, то есть :math:`\gamma = \beta + 1`:
    :math:`\gamma = 2` — линейное отталкивание (GOE), :math:`\gamma = 1` —
    отталкивания нет (закон Пуассона).

    Оценка максимального правдоподобия (оценка Хилла): если на
    :math:`[0, s_{max}]` закон степенной, :math:`F(s) = (s/s_{max})^{\gamma}`,
    то по :math:`N` значениям выборки из этого отрезка

    .. math::
        \hat\gamma = -N \Big/ \sum_{s_i < s_{max}} \ln (s_i / s_{max}),
        \qquad \delta\hat\gamma \approx \hat\gamma / \sqrt{N}.

    Закон на :math:`[0, 0.2]` степенной лишь приближённо, поэтому та же
    оценка для самих теоретических законов даёт не ровно 2 и 1, а 1.98 для
    догадки Вигнера и 0.95 для Пуассона (:func:`small_s_exponent_of_pdf`).

    Точные нули (вырожденные пары у дискретных ансамблей) исключаются —
    их доля считается отдельно, :func:`degenerate_fraction`. Возвращает
    ``None``, если на отрезке меньше ``min_count`` значений.

    >>> rng = np.random.default_rng(1)
    >>> round(small_s_exponent(rng.exponential(size=100_000)), 2)
    0.95
    """
    samples = np.asarray(samples, dtype=float).ravel()
    small = samples[(samples > tol) & (samples < s_max)]
    if small.size < min_count:
        return None
    return float(-small.size / np.sum(np.log(small / s_max)))


def small_s_exponent_of_pdf(pdf: Callable[[np.ndarray], np.ndarray], s_max: float = 0.2) -> float:
    """Значение оценки :func:`small_s_exponent` для теоретического закона (бесконечная выборка)."""
    grid = np.linspace(s_max * 1e-7, s_max, 400_001)
    weights = pdf(grid)
    weights = weights / np.sum(weights)
    return float(-1.0 / np.sum(weights * np.log(grid / s_max)))
