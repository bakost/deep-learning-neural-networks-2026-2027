"""Тесты валидации входных данных.

Проверяется главное требование задания: на любых некорректных данных
программа не падает, а сообщает *предусмотренную* ошибку, по тексту которой
понятно, что именно не так.
"""

from __future__ import annotations

import numpy as np
import pytest

from jacobi_eigen import (
    EmptyMatrixError,
    InvalidParameterError,
    JacobiError,
    JacobiEigensolver,
    NotArrayLikeError,
    NotFiniteError,
    NotRealError,
    NotSquareError,
    NotSymmetricError,
    NotTwoDimensionalError,
    jacobi_eigh,
    validate_matrix,
)


class TestAcceptedInput:
    """Данные, которые обязаны приниматься."""

    @pytest.mark.parametrize(
        "matrix",
        [
            [[2, 1], [1, 2]],                                   # список целых
            ((2.0, 1.0), (1.0, 2.0)),                           # кортеж кортежей
            np.array([[2, 1], [1, 2]], dtype=np.int32),         # int32
            np.array([[True, False], [False, True]]),           # bool
            np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32),
            np.array([[1.0]]),                                  # 1x1
        ],
    )
    def test_various_container_types(self, matrix):
        result = validate_matrix(matrix)
        assert result.dtype == np.float64
        assert result.ndim == 2

    def test_complex_with_zero_imaginary_part_is_real(self):
        """Комплексный тип с нулевой мнимой частью — это вещественная матрица."""
        matrix = np.array([[2 + 0j, 1 + 0j], [1 + 0j, 2 + 0j]])
        result = validate_matrix(matrix)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [[2.0, 1.0], [1.0, 2.0]])

    def test_input_is_not_modified(self):
        """Валидация не должна изменять переданную матрицу."""
        original = np.array([[3.0, 1.0], [1.0, 3.0]])
        backup = original.copy()
        validate_matrix(original)
        jacobi_eigh(original)
        np.testing.assert_array_equal(original, backup)

    def test_symmetry_tolerance_allows_tiny_asymmetry(self):
        """Расхождение в последних битах не должно считаться асимметрией."""
        matrix = np.array([[1.0, 2.0], [2.0 + 1e-16, 1.0]])
        validate_matrix(matrix)  # не должно выбросить исключение


class TestRejectedInput:
    """Данные, которые обязаны отвергаться конкретной ошибкой."""

    def test_not_square(self):
        with pytest.raises(NotSquareError, match="квадратной"):
            validate_matrix([[1, 2, 3], [4, 5, 6]])

    @pytest.mark.parametrize("matrix", [[1, 2, 3], np.zeros((2, 2, 2)), 5.0])
    def test_wrong_dimensionality(self, matrix):
        with pytest.raises(NotTwoDimensionalError):
            validate_matrix(matrix)

    @pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
    def test_empty(self, shape):
        with pytest.raises((EmptyMatrixError, NotSquareError)):
            validate_matrix(np.zeros(shape))

    def test_not_symmetric(self):
        with pytest.raises(NotSymmetricError) as info:
            validate_matrix([[1.0, 2.0], [3.0, 4.0]])
        error = info.value
        assert error.index in {(0, 1), (1, 0)}
        assert error.discrepancy == pytest.approx(1.0)
        assert "не симметрична" in str(error)

    def test_not_symmetric_reports_worst_pair(self):
        """В сообщении указывается пара с максимальным расхождением."""
        matrix = np.array([[1.0, 2.0, 0.0], [2.0 + 1e-3, 1.0, 5.0], [0.0, 100.0, 1.0]])
        with pytest.raises(NotSymmetricError) as info:
            validate_matrix(matrix)
        assert set(info.value.index) == {1, 2}

    def test_complex(self):
        with pytest.raises(NotRealError, match="не вещественная"):
            validate_matrix([[1.0, 2j], [-2j, 1.0]])

    def test_hermitian_complex_is_still_rejected(self):
        """Эрмитова матрица корректна математически, но метод её не поддерживает."""
        matrix = np.array([[2.0, 1j], [-1j, 2.0]])
        with pytest.raises(NotRealError):
            validate_matrix(matrix)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite(self, bad):
        matrix = np.array([[1.0, bad], [bad, 1.0]])
        with pytest.raises(NotFiniteError) as info:
            validate_matrix(matrix)
        assert info.value.count == 2

    def test_nan_reported_as_non_finite_not_asymmetric(self):
        """NaN != NaN, но диагностика должна указывать на NaN, а не на асимметрию."""
        matrix = np.array([[np.nan, 1.0], [1.0, 2.0]])
        with pytest.raises(NotFiniteError):
            validate_matrix(matrix)

    @pytest.mark.parametrize(
        "garbage",
        [
            [["a", "b"], ["c", "d"]],
            "не матрица",
            {"a": 1},
            [[1, 2], [3]],
            None,
            object(),
            [[None, 1], [1, None]],
        ],
    )
    def test_garbage_input(self, garbage):
        with pytest.raises(NotArrayLikeError):
            validate_matrix(garbage)


class TestParameterValidation:
    """Проверка параметров решателя."""

    @pytest.mark.parametrize("tol", [0.0, -1e-9, np.nan, np.inf, "мало", None])
    def test_bad_tolerance(self, tol):
        with pytest.raises(InvalidParameterError):
            JacobiEigensolver(tol=tol)

    @pytest.mark.parametrize("sweeps", [0, -5, 2.5, "много", True])
    def test_bad_max_sweeps(self, sweeps):
        with pytest.raises(InvalidParameterError):
            JacobiEigensolver(max_sweeps=sweeps)

    @pytest.mark.parametrize("name", ["quantum", "", 42, None])
    def test_bad_kernel(self, name):
        with pytest.raises(InvalidParameterError):
            JacobiEigensolver(kernel=name)

    @pytest.mark.parametrize("name", ["greedy", "", 42, None])
    def test_bad_strategy(self, name):
        with pytest.raises(InvalidParameterError):
            JacobiEigensolver(strategy=name)

    @pytest.mark.parametrize("mode", ["exact", "", None])
    def test_bad_tol_mode(self, mode):
        with pytest.raises(InvalidParameterError):
            JacobiEigensolver(tol_mode=mode)

    @pytest.mark.parametrize("order", ["random", "", 1])
    def test_bad_sort(self, order):
        with pytest.raises(InvalidParameterError):
            JacobiEigensolver(sort=order)

    def test_bad_symmetry_tolerance(self):
        with pytest.raises(InvalidParameterError):
            validate_matrix([[1.0, 0.0], [0.0, 1.0]], symmetry_atol=-1.0)


class TestErrorHierarchy:
    """Все ошибки должны ловиться одним `except JacobiError`."""

    @pytest.mark.parametrize(
        "bad_input",
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2j], [-2j, 1.0]],
            [[np.nan, 0.0], [0.0, 1.0]],
            "мусор",
            np.zeros((0, 0)),
            [1, 2, 3],
            {"a": 1},
            None,
        ],
    )
    def test_single_handler_catches_everything(self, bad_input):
        with pytest.raises(JacobiError):
            jacobi_eigh(bad_input)

    def test_errors_are_also_standard_python_errors(self):
        """Наследование от ValueError/TypeError сохраняет привычные идиомы."""
        with pytest.raises(ValueError):
            jacobi_eigh([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(TypeError):
            jacobi_eigh("мусор")
