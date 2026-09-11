"""Иерархия исключений пакета :mod:`jacobi_eigen`.

Все исключения, которые пакет может выбросить наружу, наследуются от
:class:`JacobiError`. Это прямое требование лабораторной работы:

    «Недопустимо, чтобы реализация возвращала иные ошибки, кроме
    предусмотренных при разработке».

Поэтому клиентский код может ограничиться единственным обработчиком::

    try:
        result = jacobi_eigh(A)
    except JacobiError as exc:
        print("Ошибка:", exc)

Дополнительно каждое исключение наследуется от подходящего встроенного типа
(:class:`ValueError`, :class:`TypeError`, :class:`RuntimeError`), чтобы не
ломать привычные идиомы Python.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

__all__ = [
    "JacobiError",
    "InvalidMatrixError",
    "NotArrayLikeError",
    "NotTwoDimensionalError",
    "EmptyMatrixError",
    "NotSquareError",
    "NotRealError",
    "NotFiniteError",
    "NotSymmetricError",
    "InvalidParameterError",
    "ConvergenceError",
    "ContractViolation",
    "PreconditionViolation",
    "PostconditionViolation",
    "InvariantViolation",
]


class JacobiError(Exception):
    """Базовый класс всех ошибок пакета.

    Ловля этого класса гарантированно перехватывает любую *предусмотренную*
    ошибку библиотеки.
    """


# --------------------------------------------------------------------------
# Ошибки входной матрицы
# --------------------------------------------------------------------------


class InvalidMatrixError(JacobiError, ValueError):
    """Входные данные не являются допустимой матрицей для метода Якоби."""


class NotArrayLikeError(InvalidMatrixError, TypeError):
    """Объект невозможно интерпретировать как числовой массив NumPy.

    Например, передан словарь, строка, «рваный» вложенный список или список
    объектов, не приводимых к числам.
    """


class NotTwoDimensionalError(InvalidMatrixError):
    """Массив имеет размерность, отличную от двух.

    Parameters
    ----------
    shape:
        Фактическая форма переданного массива.
    """

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        super().__init__(
            f"Ожидалась двумерная матрица, получен массив размерности "
            f"{len(self.shape)} с формой {self.shape}."
        )


class EmptyMatrixError(InvalidMatrixError):
    """Матрица пуста (содержит нулевое число строк или столбцов)."""

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        super().__init__(
            f"Матрица пуста (форма {self.shape}); собственные числа не определены."
        )


class NotSquareError(InvalidMatrixError):
    """Матрица не квадратная."""

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        super().__init__(
            f"Матрица должна быть квадратной, получена матрица {self.shape[0]}x{self.shape[1]}."
        )


class NotRealError(InvalidMatrixError):
    """Матрица не является вещественной.

    Метод Якоби в реализованном виде определён только для вещественных
    симметричных матриц. Для комплексной эрмитовой матрицы нужен вариант
    метода с унитарными вращениями, который здесь не реализован.

    Parameters
    ----------
    index:
        Индексы первого элемента с ненулевой мнимой частью.
    value:
        Значение этого элемента.
    """

    def __init__(
        self,
        index: Optional[Tuple[int, int]] = None,
        value: Optional[complex] = None,
        detail: Optional[str] = None,
    ) -> None:
        self.index = index
        self.value = value
        if detail is not None:
            message = f"Матрица не вещественная: {detail}"
        elif index is not None:
            message = (
                f"Матрица не вещественная: элемент A[{index[0]}, {index[1]}] = {value!r} "
                f"имеет ненулевую мнимую часть."
            )
        else:
            message = "Матрица не вещественная."
        super().__init__(message)


class NotFiniteError(InvalidMatrixError):
    """Матрица содержит NaN или бесконечность.

    Parameters
    ----------
    index:
        Индексы первого «плохого» элемента.
    value:
        Значение этого элемента.
    count:
        Общее число неконечных элементов.
    """

    def __init__(
        self,
        index: Tuple[int, int],
        value: float,
        count: int = 1,
    ) -> None:
        self.index = index
        self.value = value
        self.count = count
        super().__init__(
            f"Матрица содержит неконечные значения ({count} шт.): "
            f"A[{index[0]}, {index[1]}] = {value}. Допустимы только конечные числа."
        )


class NotSymmetricError(InvalidMatrixError):
    """Матрица не симметрична с заданной точностью.

    Parameters
    ----------
    index:
        Индексы пары ``(i, j)`` с максимальным расхождением.
    values:
        Пара значений ``(A[i, j], A[j, i])``.
    discrepancy:
        Модуль разности ``|A[i, j] - A[j, i]|``.
    tolerance:
        Допуск, который был использован при проверке.
    """

    def __init__(
        self,
        index: Tuple[int, int],
        values: Sequence[float],
        discrepancy: float,
        tolerance: float,
    ) -> None:
        self.index = index
        self.values = tuple(values)
        self.discrepancy = discrepancy
        self.tolerance = tolerance
        i, j = index
        super().__init__(
            f"Матрица не симметрична: A[{i}, {j}] = {self.values[0]!r}, "
            f"A[{j}, {i}] = {self.values[1]!r}, расхождение {discrepancy:.6g} "
            f"превышает допуск {tolerance:.6g}. Метод Якоби применим только к "
            f"симметричным матрицам; при необходимости симметризуйте вход "
            f"(A + A.T) / 2 или увеличьте допуск symmetry_atol/symmetry_rtol."
        )


# --------------------------------------------------------------------------
# Ошибки параметров и сходимости
# --------------------------------------------------------------------------


class InvalidParameterError(JacobiError, ValueError):
    """Недопустимое значение параметра решателя (tol, max_sweeps, ядро и т. п.)."""


class ConvergenceError(JacobiError, RuntimeError):
    """Метод не сошёлся за отведённое число итераций.

    Возникает только при исчерпании лимита ``max_sweeps``/``max_rotations``.
    Наличие такого лимита гарантирует, что реализация не зациклится —
    это отдельное требование задания.

    Parameters
    ----------
    sweeps, rotations:
        Сколько свипов и вращений было выполнено.
    off_norm:
        Достигнутая норма недиагональной части.
    threshold:
        Норма, до которой требовалось дойти.
    partial:
        Частичный результат (объект :class:`~jacobi_eigen.result.JacobiResult`
        с ``converged=False``) — может быть полезен для диагностики.
    """

    def __init__(
        self,
        sweeps: int,
        rotations: int,
        off_norm: float,
        threshold: float,
        partial: object = None,
    ) -> None:
        self.sweeps = sweeps
        self.rotations = rotations
        self.off_norm = off_norm
        self.threshold = threshold
        self.partial = partial
        super().__init__(
            f"Метод Якоби не сошёлся за {sweeps} свип(ов) ({rotations} вращений): "
            f"норма недиагональной части {off_norm:.6g} не опустилась ниже порога "
            f"{threshold:.6g}. Увеличьте max_sweeps или ослабьте tol."
        )


# --------------------------------------------------------------------------
# Нарушения контрактов
# --------------------------------------------------------------------------


class ContractViolation(JacobiError, AssertionError):
    """Нарушен контракт (см. :mod:`jacobi_eigen.contracts`).

    Нарушение контракта — это всегда ошибка *в коде*, а не в данных
    пользователя: некорректные данные отсекаются валидацией и приводят к
    :class:`InvalidMatrixError`.
    """


class PreconditionViolation(ContractViolation):
    """Нарушено предусловие: вызывающая сторона нарушила договор."""


class PostconditionViolation(ContractViolation):
    """Нарушено постусловие: функция не выполнила обещанное."""


class InvariantViolation(ContractViolation):
    """Нарушен инвариант класса."""
