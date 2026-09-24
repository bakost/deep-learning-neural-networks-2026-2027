r"""Точные распределения для матриц из ±1 полным перебором.

Матрица из ±1 размера :math:`n \times n` определяется конечным набором
«битов»: :math:`n(n+1)/2` элементами верхнего треугольника (``mirror``) или
всеми :math:`n^2` элементами :math:`A` (``sum``). Все наборы равновероятны,
поэтому точный закон расстояния — это просто частоты по *всем* матрицам:

====  ===================  ===================
n     mirror               sum
====  ===================  ===================
2     :math:`2^3 = 8`      :math:`2^4 = 16`
3     :math:`2^6 = 64`     :math:`2^9 = 512`
4     :math:`2^{10}=1024`  :math:`2^{16}=65536`
16    :math:`2^{136}`      :math:`2^{256}`
====  ===================  ===================

Для :math:`n \le 4` перебор занимает доли секунды и даёт эталон, с которым
сравнивается метод Монте-Карло; для 16×16 он невозможен.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .distributions import DiscreteDistribution
from .ensembles import check_int
from .spectrum import eigenvalues, pair_spacings

__all__ = ["MAX_ENUMERATION_BITS", "enumerate_pm1_matrices", "exact_pm1_spacing", "group_atoms"]

#: Предел перебора: :math:`2^{20} \approx 10^6` матриц.
MAX_ENUMERATION_BITS = 20


def _bit_count(n: int, construction: str) -> int:
    if construction == "mirror":
        return n * (n + 1) // 2
    if construction == "sum":
        return n * n
    raise ValueError(f"неизвестный способ симметризации {construction!r}; доступны: sum, mirror")


def enumerate_pm1_matrices(n: int, construction: str = "mirror") -> np.ndarray:
    """Все различные наборы ±1 и соответствующие симметричные матрицы, форма ``(2^m, n, n)``.

    >>> H = enumerate_pm1_matrices(2, "mirror")
    >>> H.shape
    (8, 2, 2)
    >>> enumerate_pm1_matrices(2, "sum")[0]
    array([[-2., -2.],
           [-2., -2.]])
    """
    n = check_int("n", n, 1)
    bits = _bit_count(n, construction)
    if bits > MAX_ENUMERATION_BITS:
        raise ValueError(
            f"перебор 2^{bits} матриц {n}x{n} ({construction}) невозможен: "
            f"предел 2^{MAX_ENUMERATION_BITS}"
        )
    codes = np.arange(2**bits, dtype=np.int64)[:, None]
    signs = 2.0 * ((codes >> np.arange(bits)) & 1) - 1.0  # (2^m, m), все комбинации ±1
    if construction == "sum":
        a = signs.reshape(-1, n, n)
        return a + np.swapaxes(a, -1, -2)
    rows, cols = np.triu_indices(n)
    h = np.zeros((signs.shape[0], n, n))
    h[:, rows, cols] = signs
    h[:, cols, rows] = signs
    return h


def group_atoms(values: np.ndarray, tol: float = 1e-9) -> DiscreteDistribution:
    """Равновероятные значения ``values`` → дискретный закон с объединёнными атомами.

    Значения, отличающиеся не больше чем на ``tol`` (с учётом масштаба),
    считаются одним атомом: собственные числа вычислены с ошибкой округления,
    и «одинаковые» расстояния совпадают лишь до :math:`10^{-15}`.

    >>> group_atoms(np.array([1.0, 1.0 + 1e-14, 2.0, 0.0])).probabilities
    array([0.25, 0.5 , 0.25])
    """
    values = np.sort(np.asarray(values, dtype=float).ravel())
    if values.size == 0:
        raise ValueError("нет значений для группировки")
    scale = max(1.0, float(np.abs(values).max()))
    breaks = np.flatnonzero(np.diff(values) > tol * scale) + 1
    groups = np.split(values, breaks)
    atoms = np.array([g.mean() for g in groups])
    atoms[np.abs(atoms) <= tol * scale] = 0.0   # ошибки округления около нуля
    weights = np.array([g.size for g in groups], dtype=float) / values.size
    return DiscreteDistribution(atoms, weights)


def exact_pm1_spacing(
    n: int, construction: str = "mirror", k: Optional[int] = None, tol: float = 1e-9
) -> DiscreteDistribution:
    r"""Точный закон :math:`\lambda_{k+1} - \lambda_k` для матриц из ±1 (до нормировки).

    >>> d = exact_pm1_spacing(2, "mirror")
    >>> d.values.round(6), d.probabilities
    (array([2.      , 2.828427]), array([0.5, 0.5]))
    """
    spectra = eigenvalues(enumerate_pm1_matrices(n, construction))
    return group_atoms(pair_spacings(spectra, k), tol=tol)
