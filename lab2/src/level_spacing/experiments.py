r"""Сценарии эксперимента: пункты 1–8 задания для одного ансамбля и одного размера.

Основной объект — :class:`SpacingStudy`: выборка расстояний между
фиксированной парой соседних уровней плюс все меры её близости к
теоретическим законам.

>>> study = run_study("goe", n=2, size=20_000, seed=1)
>>> abs(study.raw_mean / GOE2_RAW_MEAN - 1) < 0.01     # формула (14): 2√π
True
>>> bool(abs(study.normalized.mean() - 1) < 1e-12)     # пункт 4
True
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from .distributions import (
    GOE2_RAW_MEAN,  # noqa: F401 (нужна доктесту модуля)
    goe_limit_cdf,
    poisson_cdf,
    wigner_cdf,
    wigner_sample,
)
from .ensembles import Ensemble, check_int, get_ensemble
from .spectrum import (
    DEFAULT_CHUNK_SIZE,
    all_spacings,
    central_pair,
    normalize,
    simulate_spacings,
)
from .stats import (
    chi2_test,
    degenerate_fraction,
    histogram,
    ks_pvalue_normalized,
    ks_statistic,
    l1_distance,
    moments,
    small_s_exponent,
    uniform_bins,
)

__all__ = [
    "DEFAULT_SEED",
    "REFERENCE_CDFS",
    "SpacingStudy",
    "ks_by_ensemble_size",
    "ks_noise_constant",
    "pair_scan",
    "pooled_spacings",
    "run_study",
    "study_rng",
]

#: Зерно по умолчанию — все числа отчёта получены с ним.
DEFAULT_SEED = 2026

#: Теоретические законы, с которыми сравнивается выборка.
REFERENCE_CDFS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "wigner": wigner_cdf,
    "goe-limit": goe_limit_cdf,
    "poisson": poisson_cdf,
}

#: Бины для χ² и L1: ширина 0.1 на [0, 4] (за 4 у Вигнера лишь 3·10⁻⁶ массы).
DEFAULT_EDGES = uniform_bins(0.1, 4.0)


def study_rng(seed: Optional[int], ensemble: Union[str, Ensemble], n: int) -> np.random.Generator:
    """Независимый воспроизводимый генератор для пары (ансамбль, размер).

    Из одного зерна пользователя получаются разные потоки для разных
    ансамблей и размеров, и результат для одного ансамбля не зависит от того,
    какие ещё ансамбли считаются рядом.
    """
    if seed is None:
        return np.random.default_rng()
    name = get_ensemble(ensemble).name
    return np.random.default_rng([check_int("seed", seed, 0), n, zlib.crc32(name.encode())])


@dataclass
class SpacingStudy:
    """Результат пунктов 1–4 для одного ансамбля: выборка расстояний и её анализ.

    Attributes
    ----------
    ensemble, n, k:
        ансамбль, размер матриц и номер пары :math:`(\\lambda_k, \\lambda_{k+1})`;
    raw:
        расстояния до нормировки, по одному на матрицу;
    seed:
        зерно генератора (``None`` — случайное).
    """

    ensemble: Ensemble
    n: int
    k: int
    raw: np.ndarray
    seed: Optional[int] = None
    _normalized: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def size(self) -> int:
        """Число матриц в ансамбле :math:`M`."""
        return int(self.raw.size)

    @property
    def raw_mean(self) -> float:
        """Среднее расстояние до нормировки."""
        return float(self.raw.mean())

    @property
    def normalized(self) -> np.ndarray:
        """Расстояния, нормированные на среднее 1 (пункт 4)."""
        if self._normalized is None:
            self._normalized = normalize(self.raw)
        return self._normalized

    def histogram(self, edges: Optional[np.ndarray] = None):
        """Гистограмма нормированных расстояний (пункт 5)."""
        return histogram(self.normalized, DEFAULT_EDGES if edges is None else edges)

    def compare(self, reference: str = "wigner", edges: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Меры близости к теоретическому закону ``reference`` (пункт 8)."""
        try:
            cdf = REFERENCE_CDFS[reference]
        except KeyError:
            raise ValueError(
                f"неизвестный закон {reference!r}; доступны: {', '.join(REFERENCE_CDFS)}"
            ) from None
        edges = DEFAULT_EDGES if edges is None else edges
        d = ks_statistic(self.normalized, cdf)
        p = ks_pvalue_normalized(d, self.size, cdf)
        chi2, dof, chi2_p = chi2_test(self.normalized, cdf, edges)
        return {
            "ks_d": d,
            "ks_p": p,
            "chi2": chi2,
            "chi2_dof": dof,
            "chi2_p": chi2_p,
            "l1": l1_distance(histogram(self.normalized, edges), cdf),
        }

    def summary(self) -> Dict[str, object]:
        """Сводка для таблиц отчёта: моменты, поведение около нуля, сравнение с законами."""
        m = moments(self.normalized)
        result: Dict[str, object] = {
            "ensemble": self.ensemble.name,
            "n": self.n,
            "k": self.k,
            "size": self.size,
            "raw_mean": self.raw_mean,
            "raw_mean_error": float(self.raw.std() / np.sqrt(self.size)),
            "variance": m["variance"],
            "variance_error": m["variance_error"],
            "second_moment": m["second_moment"],
            "degenerate_fraction": degenerate_fraction(self.raw),
            "small_s_exponent": small_s_exponent(self.normalized),
        }
        for reference in ("wigner", "goe-limit", "poisson"):
            for key, value in self.compare(reference).items():
                result[f"{reference}_{key}"] = value
        return result


def run_study(
    ensemble: Union[str, Ensemble],
    n: int,
    size: int,
    seed: Optional[int] = DEFAULT_SEED,
    *,
    k: Optional[int] = None,
    backend: str = "numpy",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> SpacingStudy:
    """Пункты 1–4: сгенерировать ``size`` матриц ``n x n`` и найти расстояния пары ``k``."""
    ensemble = get_ensemble(ensemble)
    k = central_pair(n) if k is None else k
    raw = simulate_spacings(
        ensemble, n, size, study_rng(seed, ensemble, n), k=k, chunk_size=chunk_size, backend=backend
    )
    return SpacingStudy(ensemble=ensemble, n=n, k=k, raw=raw, seed=seed)


def ks_by_ensemble_size(
    raw: np.ndarray,
    sizes: Sequence[int],
    cdf: Callable[[np.ndarray], np.ndarray] = wigner_cdf,
    max_blocks: int = 30,
) -> Dict[int, np.ndarray]:
    r"""Как расстояние Колмогорова–Смирнова зависит от объёма ансамбля :math:`M`.

    Большая выборка режется на непересекающиеся блоки по :math:`M` значений;
    каждый блок нормируется на *своё* среднее — ровно так, как если бы
    ансамбль состоял из :math:`M` матриц. Возвращаются значения :math:`D`
    по блокам (не больше ``max_blocks`` на размер).

    Если закон верен, :math:`D \propto 1/\sqrt{M}` неограниченно убывает. Если
    между законом и истинным распределением есть систематическое расхождение
    :math:`\delta`, то :math:`D` выходит на полку :math:`\approx \delta`.
    """
    raw = np.asarray(raw, dtype=float).ravel()
    result: Dict[int, np.ndarray] = {}
    for m in sizes:
        m = check_int("size", m, 2)
        blocks = min(raw.size // m, max_blocks)
        if blocks == 0:
            raise ValueError(f"выборки из {raw.size} значений не хватает на блок размера {m}")
        result[m] = np.array(
            [ks_statistic(normalize(raw[i * m:(i + 1) * m]), cdf) for i in range(blocks)]
        )
    return result


def ks_noise_constant(size: int = 10_000, repeats: int = 300, seed: int = DEFAULT_SEED) -> float:
    r"""Типичное (медианное) :math:`D\sqrt{M}` для выборки *точно* из догадки Вигнера.

    Выборка нормируется на своё среднее, как и в эксперименте, поэтому
    константа меньше табличной 0.87 для полностью заданного закона: оценка
    масштаба по той же выборке «подгоняет» закон к данным. Уровень
    статистического шума для ансамбля из :math:`M` матриц —
    :math:`c/\sqrt{M}`.
    """
    rng = np.random.default_rng([check_int("seed", seed, 0), size, repeats])
    values = [
        ks_statistic(normalize(wigner_sample(size, rng)), wigner_cdf) * np.sqrt(size)
        for _ in range(check_int("repeats", repeats, 1))
    ]
    return float(np.median(values))


def pair_scan(eigvals: np.ndarray) -> Dict[str, np.ndarray]:
    r"""Статистика расстояний для *каждой* пары :math:`k = 1, \dots, n-1` (сноска 2 методички).

    Returns
    -------
    dict
        ``k`` — номера пар; ``mean`` — средние расстояния до нормировки;
        ``variance`` — дисперсия после нормировки; ``ks_d`` — расстояние
        Колмогорова–Смирнова до догадки Вигнера.
    """
    spacings = all_spacings(eigvals)
    normalized = spacings / spacings.mean(axis=0)
    return {
        "k": np.arange(1, spacings.shape[1] + 1),
        "mean": spacings.mean(axis=0),
        "variance": normalized.var(axis=0),
        "ks_d": np.array([ks_statistic(column, wigner_cdf) for column in normalized.T]),
    }


def pooled_spacings(eigvals: np.ndarray, unfold: bool = False) -> np.ndarray:
    r"""Все расстояния всех матриц одной выборкой.

    * ``unfold=False`` — общая нормировка на глобальное среднее. Так делать
      *не стоит*: в центре спектра уровни гуще, чем на краях (закон
      полукруга), и смешиваются распределения с разными масштабами.
    * ``unfold=True`` — каждая пара :math:`k` нормируется на своё среднее
      :math:`\langle \lambda_{k+1} - \lambda_k \rangle` (простейшее
      «развёртывание» спектра по ансамблю), и только потом всё объединяется.
    """
    spacings = all_spacings(eigvals)
    if unfold:
        return (spacings / spacings.mean(axis=0)).ravel()
    return normalize(spacings.ravel())
