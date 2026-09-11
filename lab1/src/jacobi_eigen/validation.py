"""Проверка входных данных перед запуском метода Якоби.

Модуль реализует требование задания: любые некорректные данные должны
приводить к *осмысленной* ошибке из иерархии :class:`~jacobi_eigen.exceptions.JacobiError`,
а не к падению внутри вычислительного ядра.

Проверки выполняются в порядке от «грубых» к «тонким», чтобы сообщение
указывало на первопричину:

1. объект приводится к числовому массиву NumPy;
2. массив двумерный;
3. массив непустой;
4. матрица квадратная;
5. матрица вещественная;
6. все элементы конечны (нет ``NaN``/``inf``);
7. матрица симметрична с заданной точностью.
"""

from __future__ import annotations

import numbers
from typing import Any, Tuple

import numpy as np

from .contracts import ensure
from .exceptions import (
    EmptyMatrixError,
    InvalidParameterError,
    NotArrayLikeError,
    NotFiniteError,
    NotRealError,
    NotSquareError,
    NotSymmetricError,
    NotTwoDimensionalError,
)

__all__ = [
    "validate_matrix",
    "validate_tolerance",
    "validate_positive_int",
    "off_diagonal_sq",
    "off_diagonal_norm",
]

#: Допуски по умолчанию для проверки симметричности.
DEFAULT_SYMMETRY_ATOL = 1e-10
DEFAULT_SYMMETRY_RTOL = 1e-10

_NUMERIC_KINDS = "biuf"  # bool, int, uint, float


def _as_numeric_array(matrix: Any) -> np.ndarray:
    """Привести объект к числовому массиву NumPy.

    Raises
    ------
    NotArrayLikeError
        Если объект не является числовым массивом или «рваной» структурой.
    """
    if isinstance(matrix, (str, bytes)):
        raise NotArrayLikeError(
            "Ожидалась матрица (список списков или numpy.ndarray), получена строка."
        )
    try:
        array = np.asarray(matrix)
    except Exception as exc:  # np.asarray практически всё принимает, но подстрахуемся
        raise NotArrayLikeError(
            f"Объект типа {type(matrix).__name__} невозможно интерпретировать "
            f"как матрицу: {exc}"
        ) from exc

    kind = array.dtype.kind
    if kind in _NUMERIC_KINDS or kind == "c":
        return array
    if kind == "O":
        # Массив объектов: Decimal, Fraction, комплексные, а также None и
        # прочий мусор. Проверяем поэлементно, чтобы `None` был опознан как
        # «не число», а не превратился молча в `nan` (что дало бы совершенно
        # неверную диагностику).
        for item in array.ravel():
            if isinstance(item, (str, bytes)):
                raise NotArrayLikeError(
                    f"Матрица содержит строку {item!r} вместо числа."
                )
            if isinstance(item, numbers.Complex):
                continue
            try:
                complex(item)
            except (TypeError, ValueError) as exc:
                raise NotArrayLikeError(
                    f"Матрица содержит элемент {item!r} типа "
                    f"{type(item).__name__}, который не является числом ({exc})."
                ) from exc
        try:
            return array.astype(np.complex128)
        except (TypeError, ValueError) as exc:  # pragma: no cover - подстраховка
            raise NotArrayLikeError(
                f"Объект типа {type(matrix).__name__} не является числовой матрицей: "
                f"элементы (dtype=object) не приводятся к числам ({exc})."
            ) from exc
    raise NotArrayLikeError(
        f"Неподдерживаемый тип элементов матрицы: dtype={array.dtype}. "
        "Ожидались вещественные числа."
    )


def _ensure_real(array: np.ndarray) -> np.ndarray:
    """Проверить вещественность и вернуть вещественный массив ``float64``.

    Комплексный массив с *тождественно нулевой* мнимой частью считается
    вещественным и молча приводится к ``float64``: математически это та же
    самая вещественная матрица.
    """
    if array.dtype.kind == "c":
        imaginary = array.imag
        nonzero = np.argwhere(imaginary != 0)
        if nonzero.size:
            i, j = (int(nonzero[0][0]), int(nonzero[0][1])) if array.ndim == 2 else (0, 0)
            value = complex(array[tuple(nonzero[0])])
            if array.ndim == 2:
                raise NotRealError(index=(i, j), value=value)
            raise NotRealError(detail=f"найден элемент с ненулевой мнимой частью {value!r}")
        array = array.real
    return np.array(array, dtype=np.float64, copy=True)


def _ensure_finite(array: np.ndarray) -> None:
    """Проверить, что все элементы конечны."""
    finite_mask = np.isfinite(array)
    if not finite_mask.all():
        bad = np.argwhere(~finite_mask)
        i, j = int(bad[0][0]), int(bad[0][1])
        raise NotFiniteError(index=(i, j), value=float(array[i, j]), count=int(bad.shape[0]))


def _ensure_symmetric(array: np.ndarray, atol: float, rtol: float) -> None:
    """Проверить симметричность с допуском ``atol + rtol * |A_ji|``."""
    difference = np.abs(array - array.T)
    if not difference.any():
        return
    allowed = atol + rtol * np.abs(array.T)
    violations = difference > allowed
    if violations.any():
        flat = int(np.argmax(np.where(violations, difference, -np.inf)))
        i, j = divmod(flat, array.shape[1])
        raise NotSymmetricError(
            index=(i, j),
            values=(float(array[i, j]), float(array[j, i])),
            discrepancy=float(difference[i, j]),
            tolerance=float(allowed[i, j]),
        )


@ensure(lambda result: result.ndim == 2 and result.shape[0] == result.shape[1],
        "результат — квадратная двумерная матрица")
@ensure(lambda result: result.dtype == np.float64,
        "результат имеет вещественный тип float64")
@ensure(lambda result: bool(np.isfinite(result).all()),
        "все элементы результата конечны")
def validate_matrix(
    matrix: Any,
    *,
    symmetry_atol: float = DEFAULT_SYMMETRY_ATOL,
    symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
    symmetrize: bool = False,
) -> np.ndarray:
    """Проверить матрицу и вернуть её копию как ``float64``-массив.

    Parameters
    ----------
    matrix:
        Что угодно, что может быть матрицей: ``numpy.ndarray``, список
        списков, кортеж кортежей и т. п.
    symmetry_atol, symmetry_rtol:
        Абсолютный и относительный допуски проверки симметричности.
        Элемент считается симметричным, если
        ``|A[i, j] - A[j, i]| <= symmetry_atol + symmetry_rtol * |A[j, i]|``.
    symmetrize:
        Если ``True``, после успешной проверки матрица дополнительно
        приводится к строго симметричной по формуле ``(A + A.T) / 2``.
        Это убирает расхождения в последних битах и делает алгоритм
        детерминированным.

    Returns
    -------
    numpy.ndarray
        Новый двумерный вещественный массив; входные данные не изменяются.

    Raises
    ------
    NotArrayLikeError
        Объект не является числовой матрицей.
    NotTwoDimensionalError
        Массив не двумерный.
    EmptyMatrixError
        Матрица пуста.
    NotSquareError
        Матрица не квадратная.
    NotRealError
        Матрица содержит комплексные элементы.
    NotFiniteError
        Матрица содержит ``NaN`` или бесконечность.
    NotSymmetricError
        Матрица не симметрична с заданной точностью.
    InvalidParameterError
        Недопустимые значения допусков.

    Examples
    --------
    >>> validate_matrix([[2, 1], [1, 2]])
    array([[2., 1.],
           [1., 2.]])
    >>> validate_matrix([[0, 1], [2, 0]])
    Traceback (most recent call last):
        ...
    jacobi_eigen.exceptions.NotSymmetricError: ...
    """
    if not np.isfinite(symmetry_atol) or symmetry_atol < 0:
        raise InvalidParameterError(
            f"symmetry_atol должен быть конечным неотрицательным числом, получено {symmetry_atol!r}."
        )
    if not np.isfinite(symmetry_rtol) or symmetry_rtol < 0:
        raise InvalidParameterError(
            f"symmetry_rtol должен быть конечным неотрицательным числом, получено {symmetry_rtol!r}."
        )

    array = _as_numeric_array(matrix)

    if array.ndim != 2:
        raise NotTwoDimensionalError(array.shape)
    if array.size == 0:
        raise EmptyMatrixError(array.shape)
    if array.shape[0] != array.shape[1]:
        raise NotSquareError(array.shape)

    array = _ensure_real(array)
    _ensure_finite(array)
    _ensure_symmetric(array, symmetry_atol, symmetry_rtol)

    if symmetrize:
        # Записано как A/2 + A^T/2, а не (A + A^T)/2: второй вариант
        # переполняется на матрицах с элементами порядка 1e308.
        array = array / 2.0 + array.T / 2.0
    return array


def validate_tolerance(tol: Any, name: str = "tol") -> float:
    """Проверить и вернуть положительный конечный допуск.

    Raises
    ------
    InvalidParameterError
        Если значение не является положительным конечным числом.
    """
    try:
        value = float(tol)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(
            f"{name} должен быть числом, получено {tol!r} ({type(tol).__name__})."
        ) from exc
    if not np.isfinite(value):
        raise InvalidParameterError(f"{name} должен быть конечным числом, получено {value!r}.")
    if value <= 0:
        raise InvalidParameterError(f"{name} должен быть строго положительным, получено {value!r}.")
    return value


def validate_positive_int(value: Any, name: str) -> int:
    """Проверить и вернуть строго положительное целое число.

    Raises
    ------
    InvalidParameterError
        Если значение не является положительным целым.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidParameterError(
            f"{name} должен быть целым числом, получено {value!r} ({type(value).__name__})."
        )
    value = int(value)
    if value <= 0:
        raise InvalidParameterError(f"{name} должен быть строго положительным, получено {value}.")
    return value


def off_diagonal_sq(matrix: np.ndarray) -> float:
    r"""Сумма квадратов всех недиагональных элементов.

    Вычисляет величину :math:`S = \sum_{r \neq s} A_{rs}^2` из описания
    алгоритма — именно она служит мерой близости матрицы к диагональной.

    Notes
    -----
    * Сумма берётся по *обоим* симметричным элементам, поэтому она вдвое
      больше суммы по верхнему треугольнику.
    * Диагональ обнуляется **до** суммирования. Заманчивый вариант
      ``sum(A * A) - sum(diag(A) ** 2)`` вычислительно непригоден: у почти
      диагональной матрицы уменьшаемое и вычитаемое почти равны, и разность
      теряет все значащие цифры. Например, для матрицы ``10 * I`` с
      недиагональными элементами ``1e-11`` такая формула даёт ровно ``0.0``
      вместо ``8.7e-20``, что приводит к ложной остановке метода.
    """
    squared = matrix * matrix
    np.fill_diagonal(squared, 0.0)
    return float(np.sum(squared))


def off_diagonal_norm(matrix: np.ndarray) -> float:
    r"""Норма недиагональной части :math:`\mathrm{off}(A) = \sqrt{S}`."""
    return float(np.sqrt(off_diagonal_sq(matrix)))


def frobenius_norm_sq(matrix: np.ndarray) -> float:
    r"""Квадрат нормы Фробениуса :math:`\|A\|_F^2`."""
    return float(np.sum(matrix * matrix))


def matrix_shape(matrix: np.ndarray) -> Tuple[int, int]:
    """Форма матрицы в виде кортежа из двух чисел."""
    return int(matrix.shape[0]), int(matrix.shape[1])
