r"""Ядра вращения — три способа выполнить один шаг метода Якоби.

Шаг метода Якоби математически записывается как

.. math::
    A_{i+1} = P_i^{T} A_i P_i, \qquad V_{i+1} = V_i P_i,

где :math:`P_i` — матрица поворота Гивенса. Но *вычислять* этот шаг можно
по-разному, и в лабораторной работе прямо предлагается выбор между
поэлементным пересчётом и честным умножением матриц. Здесь реализованы оба
варианта плюс промежуточный (векторизованный), что позволяет честно
сравнить их скорость на одинаковом алгоритме.

===================  ========================  ==================================
Ядро                 Сложность одного шага     Комментарий
===================  ========================  ==================================
:class:`ScalarKernel`      :math:`O(n)`        Формулы (12)–(15), (17)–(18)
                                               в чистом Python — буквальный
                                               перевод алгоритма из методички
:class:`VectorizedKernel`  :math:`O(n)`        Те же формулы срезами NumPy
:class:`MatrixKernel`      :math:`O(n^3)`      Явная матрица Гивенса и два
                                               матричных умножения
===================  ========================  ==================================

Все ядра модифицируют матрицы ``A`` и ``V`` **на месте** и обязаны давать
численно эквивалентный результат.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

from .exceptions import InvalidParameterError

__all__ = [
    "RotationKernel",
    "ScalarKernel",
    "VectorizedKernel",
    "MatrixKernel",
    "givens_matrix",
    "get_kernel",
    "available_kernels",
]


def givens_matrix(n: int, p: int, q: int, cos_phi: float, sin_phi: float) -> np.ndarray:
    r"""Построить матрицу поворота Гивенса :math:`P_{pq}` размера ``n x n``.

    Матрица совпадает с единичной всюду, кроме четырёх элементов:

    .. math::
        P_{pp} = P_{qq} = \cos\varphi, \quad
        P_{pq} = \sin\varphi, \quad
        P_{qp} = -\sin\varphi.

    Examples
    --------
    >>> givens_matrix(3, 0, 2, 0.6, 0.8)
    array([[ 0.6,  0. ,  0.8],
           [ 0. ,  1. ,  0. ],
           [-0.8,  0. ,  0.6]])
    """
    rotation = np.eye(n, dtype=np.float64)
    rotation[p, p] = cos_phi
    rotation[q, q] = cos_phi
    rotation[p, q] = sin_phi
    rotation[q, p] = -sin_phi
    return rotation


class RotationKernel(ABC):
    """Абстрактное ядро: применяет одно вращение Якоби к ``A`` и ``V``."""

    #: Короткое имя ядра, используется в CLI и в фабрике :func:`get_kernel`.
    name: str = "abstract"

    @abstractmethod
    def apply(
        self,
        a: np.ndarray,
        v: Optional[np.ndarray],
        p: int,
        q: int,
        cos_phi: float,
        sin_phi: float,
        tan_phi: float,
    ) -> None:
        """Выполнить вращение на месте.

        Parameters
        ----------
        a:
            Текущая матрица :math:`A_i` (изменяется на месте).
        v:
            Текущая матрица собственных векторов :math:`V_i` (изменяется на
            месте) или ``None``, если векторы не нужны.
        p, q:
            Индексы зануляемого элемента, ``p < q``.
        cos_phi, sin_phi, tan_phi:
            Косинус, синус и тангенс угла поворота.
        """

    def __repr__(self) -> str:  # pragma: no cover - косметика
        return f"{type(self).__name__}()"


class ScalarKernel(RotationKernel):
    r"""Поэлементный пересчёт «как в методичке», чистый Python.

    Реализует формулы (12)–(15) для матрицы :math:`A` и (17)–(18) для
    матрицы собственных векторов :math:`V`:

    .. math::
        A^{i+1}_{pp} = A^i_{pp} - A^i_{pq}\,\mathrm{tg}\,\varphi, \qquad
        A^{i+1}_{qq} = A^i_{qq} + A^i_{pq}\,\mathrm{tg}\,\varphi,

    .. math::
        A^{i+1}_{rp} = A^i_{rp} - \sin\varphi
            \left(A^i_{rq} + \frac{\sin\varphi}{1 + \cos\varphi} A^i_{rp}\right),

    .. math::
        A^{i+1}_{rq} = A^i_{rq} + \sin\varphi
            \left(A^i_{rp} - \frac{\sin\varphi}{1 + \cos\varphi} A^i_{rq}\right).

    Это самая медленная, но самая наглядная реализация: её удобно сверять
    с текстом задания.
    """

    name = "scalar"

    def apply(
        self,
        a: np.ndarray,
        v: Optional[np.ndarray],
        p: int,
        q: int,
        cos_phi: float,
        sin_phi: float,
        tan_phi: float,
    ) -> None:
        n = a.shape[0]
        # tau = sin / (1 + cos) = tg(phi / 2) — устойчивая форма записи
        tau = sin_phi / (1.0 + cos_phi)

        a_pq = a[p, q]
        a[p, p] -= a_pq * tan_phi
        a[q, q] += a_pq * tan_phi
        a[p, q] = 0.0
        a[q, p] = 0.0

        for r in range(n):
            if r == p or r == q:
                continue
            a_rp = a[r, p]
            a_rq = a[r, q]
            new_rp = a_rp - sin_phi * (a_rq + tau * a_rp)
            new_rq = a_rq + sin_phi * (a_rp - tau * a_rq)
            a[r, p] = new_rp
            a[p, r] = new_rp
            a[r, q] = new_rq
            a[q, r] = new_rq

        if v is not None:
            for r in range(n):
                v_rp = v[r, p]
                v_rq = v[r, q]
                v[r, p] = v_rp * cos_phi - v_rq * sin_phi
                v[r, q] = v_rp * sin_phi + v_rq * cos_phi


class VectorizedKernel(RotationKernel):
    """Те же формулы, но одной операцией NumPy над срезами.

    Математически идентично :class:`ScalarKernel`, однако цикл по ``r``
    выполняется внутри NumPy, а не в интерпретаторе Python. На матрицах
    среднего и большого размера это на порядок-два быстрее.
    """

    name = "vectorized"

    def apply(
        self,
        a: np.ndarray,
        v: Optional[np.ndarray],
        p: int,
        q: int,
        cos_phi: float,
        sin_phi: float,
        tan_phi: float,
    ) -> None:
        n = a.shape[0]
        tau = sin_phi / (1.0 + cos_phi)

        a_pq = a[p, q]
        a_pp = a[p, p] - a_pq * tan_phi
        a_qq = a[q, q] + a_pq * tan_phi

        # Индексы всех строк, кроме p и q.
        others = np.ones(n, dtype=bool)
        others[p] = False
        others[q] = False

        col_p = a[others, p]
        col_q = a[others, q]
        new_p = col_p - sin_phi * (col_q + tau * col_p)
        new_q = col_q + sin_phi * (col_p - tau * col_q)

        a[others, p] = new_p
        a[p, others] = new_p
        a[others, q] = new_q
        a[q, others] = new_q

        a[p, p] = a_pp
        a[q, q] = a_qq
        a[p, q] = 0.0
        a[q, p] = 0.0

        if v is not None:
            v_p = v[:, p].copy()
            v_q = v[:, q]
            v[:, p] = v_p * cos_phi - v_q * sin_phi
            v[:, q] = v_p * sin_phi + v_q * cos_phi


class MatrixKernel(RotationKernel):
    r"""Честное матричное умножение :math:`A_{i+1} = P^{T} A_i P`.

    Ядро строит полную матрицу Гивенса :math:`P_{pq}` и выполняет два
    матричных умножения. Это прямая запись формулы (1) из задания; вариант
    заведомо неэффективный (матрица поворота разрежена, почти все умножения
    выполняются на нулях), но он служит эталоном корректности для
    «быстрых» ядер и наглядно показывает в бенчмарке цену наивного подхода.
    """

    name = "matrix"

    def apply(
        self,
        a: np.ndarray,
        v: Optional[np.ndarray],
        p: int,
        q: int,
        cos_phi: float,
        sin_phi: float,
        tan_phi: float,
    ) -> None:
        n = a.shape[0]
        rotation = givens_matrix(n, p, q, cos_phi, sin_phi)
        a[...] = rotation.T @ a @ rotation
        # Симметризация подавляет накопление ошибок округления, из-за которых
        # A может «уплыть» от симметричности после многих умножений.
        a[...] = a / 2.0 + a.T / 2.0
        a[p, q] = 0.0
        a[q, p] = 0.0
        if v is not None:
            v[...] = v @ rotation


_KERNELS: Dict[str, Type[RotationKernel]] = {
    ScalarKernel.name: ScalarKernel,
    VectorizedKernel.name: VectorizedKernel,
    MatrixKernel.name: MatrixKernel,
}


def available_kernels() -> tuple:
    """Имена всех доступных ядер."""
    return tuple(_KERNELS)


def get_kernel(kernel: "str | RotationKernel") -> RotationKernel:
    """Получить экземпляр ядра по имени или вернуть переданный объект.

    Parameters
    ----------
    kernel:
        Строковое имя (``"scalar"``, ``"vectorized"``, ``"matrix"``) либо
        готовый экземпляр :class:`RotationKernel` (можно передать своё ядро).

    Raises
    ------
    InvalidParameterError
        Если имя неизвестно или объект не является ядром.
    """
    if isinstance(kernel, RotationKernel):
        return kernel
    if isinstance(kernel, str):
        try:
            return _KERNELS[kernel]()
        except KeyError:
            raise InvalidParameterError(
                f"Неизвестное ядро {kernel!r}. Доступны: {', '.join(sorted(_KERNELS))}."
            ) from None
    raise InvalidParameterError(
        f"kernel должен быть строкой или экземпляром RotationKernel, "
        f"получен {type(kernel).__name__}."
    )
