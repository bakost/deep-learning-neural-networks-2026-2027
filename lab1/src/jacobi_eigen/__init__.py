r"""jacobi_eigen — метод Якоби для симметричной проблемы собственных значений.

Пакет вычисляет **все** собственные числа и собственные векторы вещественной
симметричной матрицы методом циклических вращений Якоби (лабораторная
работа 1).

Быстрый старт
-------------
>>> import numpy as np
>>> from jacobi_eigen import jacobi_eigh
>>> A = np.array([[4.0, 1.0, 0.0],
...               [1.0, 3.0, 1.0],
...               [0.0, 1.0, 2.0]])
>>> result = jacobi_eigh(A)
>>> np.round(result.eigenvalues, 6)
array([1.267949, 3.      , 4.732051])
>>> result.residual_norm(A) < 1e-12   # A V = V L
True

Объектный интерфейс позволяет один раз настроить решатель:

>>> from jacobi_eigen import JacobiEigensolver
>>> solver = JacobiEigensolver(tol=1e-14, kernel="scalar", strategy="max")
>>> solver.solve(A).converged
True

Некорректные данные приводят к понятной ошибке из общей иерархии:

>>> from jacobi_eigen import JacobiError
>>> try:
...     jacobi_eigh([[1.0, 2.0], [3.0, 4.0]])
... except JacobiError as exc:
...     print(type(exc).__name__)
NotSymmetricError

Состав пакета
-------------
=================================  ==========================================
Модуль                             Назначение
=================================  ==========================================
:mod:`jacobi_eigen.solver`         Алгоритм и публичный интерфейс
:mod:`jacobi_eigen.kernels`        Три реализации шага вращения
:mod:`jacobi_eigen.strategies`     Порядок выбора зануляемого элемента
:mod:`jacobi_eigen.result`         Результат и проверки его качества
:mod:`jacobi_eigen.validation`     Проверка входных данных
:mod:`jacobi_eigen.exceptions`     Иерархия ошибок
:mod:`jacobi_eigen.contracts`      Контрактное программирование
=================================  ==========================================
"""

from __future__ import annotations

from .contracts import contracts_disabled, disable, enable, enabled, ensure, invariant, require
from .exceptions import (
    ContractViolation,
    ConvergenceError,
    EmptyMatrixError,
    InvalidMatrixError,
    InvalidParameterError,
    InvariantViolation,
    JacobiError,
    NotArrayLikeError,
    NotFiniteError,
    NotRealError,
    NotSquareError,
    NotSymmetricError,
    NotTwoDimensionalError,
    PostconditionViolation,
    PreconditionViolation,
)
from .kernels import (
    MatrixKernel,
    RotationKernel,
    ScalarKernel,
    VectorizedKernel,
    available_kernels,
    get_kernel,
    givens_matrix,
)
from .result import JacobiResult
from .solver import JacobiEigensolver, jacobi_eigh, rotation_parameters
from .strategies import (
    CyclicStrategy,
    MaxElementStrategy,
    PivotStrategy,
    RoundRobinStrategy,
    available_strategies,
    get_strategy,
)
from .validation import off_diagonal_norm, off_diagonal_sq, validate_matrix

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # Основной интерфейс
    "jacobi_eigh",
    "JacobiEigensolver",
    "JacobiResult",
    "rotation_parameters",
    # Ядра вращения
    "RotationKernel",
    "ScalarKernel",
    "VectorizedKernel",
    "MatrixKernel",
    "givens_matrix",
    "get_kernel",
    "available_kernels",
    # Стратегии выбора элемента
    "PivotStrategy",
    "CyclicStrategy",
    "MaxElementStrategy",
    "RoundRobinStrategy",
    "get_strategy",
    "available_strategies",
    # Валидация
    "validate_matrix",
    "off_diagonal_sq",
    "off_diagonal_norm",
    # Исключения
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
    # Контракты
    "require",
    "ensure",
    "invariant",
    "enabled",
    "enable",
    "disable",
    "contracts_disabled",
]
