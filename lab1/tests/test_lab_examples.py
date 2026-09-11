"""Сверка реализации с числовыми примерами из методички.

Эти тесты — прямая проверка того, что код считает *ровно то*, что описано в
задании: значения угла поворота, матрица Гивенса и результат одного шага
метода сверяются с числами из разделов 1.2, 1.3 и 1.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from jacobi_eigen import givens_matrix, jacobi_eigh, rotation_parameters
from jacobi_eigen.kernels import MatrixKernel, ScalarKernel, VectorizedKernel
from jacobi_eigen.validation import off_diagonal_sq

#: Матрица 5x5 из примера раздела 1.2: ``A[i][j] = 1 + 3 * (i + j)``.
LAB_MATRIX = np.array(
    [
        [1.0, 4.0, 7.0, 10.0, 13.0],
        [4.0, 7.0, 10.0, 13.0, 16.0],
        [7.0, 10.0, 13.0, 16.0, 19.0],
        [10.0, 13.0, 16.0, 19.0, 22.0],
        [13.0, 16.0, 19.0, 22.0, 25.0],
    ]
)

#: В методичке зануляется элемент ``A_24`` в нумерации с единицы.
P, Q = 1, 3


class TestRotationAngleExample:
    """Раздел 1.2: вычисление угла поворота."""

    def test_c_value(self):
        r""":math:`C = \frac{A_{44} - A_{22}}{2 A_{24}} = \frac{19 - 7}{2 \cdot 13} = \frac{6}{13}`."""
        c_value = (LAB_MATRIX[Q, Q] - LAB_MATRIX[P, P]) / (2 * LAB_MATRIX[P, Q])
        assert c_value == pytest.approx(6 / 13)

    def test_tangent_cosine_sine(self):
        """Методичка приводит ``tg = 0.64``, ``cos = 0.842``, ``sin = 0.539``."""
        cos_phi, sin_phi, tan_phi = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        assert tan_phi == pytest.approx(0.64, abs=5e-3)
        assert cos_phi == pytest.approx(0.842, abs=5e-4)
        assert sin_phi == pytest.approx(0.539, abs=5e-4)

    def test_angle_does_not_exceed_45_degrees(self):
        """Выбор меньшего корня означает :math:`|\\varphi| \\le 45^\\circ`."""
        _, _, tan_phi = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        assert abs(math.degrees(math.atan(tan_phi))) <= 45.0


class TestGivensMatrixExample:
    """Раздел 1.2: вид матрицы поворота :math:`P_{24}`."""

    def test_structure(self):
        expected = np.array(
            [
                [1.0, 0.000, 0.0, 0.000, 0.0],
                [0.0, 0.842, 0.0, 0.539, 0.0],
                [0.0, 0.000, 1.0, 0.000, 0.0],
                [0.0, -0.539, 0.0, 0.842, 0.0],
                [0.0, 0.000, 0.0, 0.000, 1.0],
            ]
        )
        cos_phi, sin_phi, _ = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        rotation = givens_matrix(5, P, Q, cos_phi, sin_phi)
        np.testing.assert_allclose(rotation, expected, atol=1e-3)

    def test_is_orthogonal(self):
        """Матрица Гивенса ортогональна: именно поэтому шаг сохраняет спектр."""
        cos_phi, sin_phi, _ = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        rotation = givens_matrix(5, P, Q, cos_phi, sin_phi)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(5), atol=1e-14)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)


class TestSingleStepExample:
    """Раздел 1.3: результат одного шага метода Якоби."""

    #: Матрица A^{i+1} из методички (значения округлены до двух знаков).
    EXPECTED = np.array(
        [
            [1.00, -2.02, 7.00, 10.58, 13.00],
            [-2.02, -1.32, -0.20, 0.00, 1.62],
            [7.00, -0.20, 13.00, 18.86, 19.00],
            [10.58, 0.00, 18.86, 27.32, 27.15],
            [13.00, 1.62, 19.00, 27.15, 25.00],
        ]
    )

    @pytest.mark.parametrize("kernel_class", [ScalarKernel, VectorizedKernel, MatrixKernel])
    def test_one_rotation_matches_methodichka(self, kernel_class):
        cos_phi, sin_phi, tan_phi = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        a = LAB_MATRIX.copy()
        kernel_class().apply(a, None, P, Q, cos_phi, sin_phi, tan_phi)
        np.testing.assert_allclose(a, self.EXPECTED, atol=1e-2)

    def test_target_element_is_zeroed(self):
        cos_phi, sin_phi, tan_phi = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        a = LAB_MATRIX.copy()
        VectorizedKernel().apply(a, None, P, Q, cos_phi, sin_phi, tan_phi)
        assert a[P, Q] == 0.0 and a[Q, P] == 0.0

    def test_diagonal_elements_from_example(self):
        r""":math:`A_{22} = 7 - 13 \cdot 0.64 = -1.32`, :math:`A_{44} = 19 + 13 \cdot 0.64 = 27.32`."""
        cos_phi, sin_phi, tan_phi = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        a = LAB_MATRIX.copy()
        ScalarKernel().apply(a, None, P, Q, cos_phi, sin_phi, tan_phi)
        assert a[P, P] == pytest.approx(-1.32, abs=5e-3)
        assert a[Q, Q] == pytest.approx(27.32, abs=5e-3)

    def test_step_preserves_symmetry_and_trace(self):
        cos_phi, sin_phi, tan_phi = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        a = LAB_MATRIX.copy()
        VectorizedKernel().apply(a, None, P, Q, cos_phi, sin_phi, tan_phi)
        np.testing.assert_allclose(a, a.T, atol=1e-13)
        assert np.trace(a) == pytest.approx(np.trace(LAB_MATRIX))

    def test_step_is_similarity_transform(self):
        """Шаг — преобразование подобия, поэтому спектр не меняется."""
        cos_phi, sin_phi, tan_phi = rotation_parameters(
            LAB_MATRIX[P, P], LAB_MATRIX[Q, Q], LAB_MATRIX[P, Q]
        )
        a = LAB_MATRIX.copy()
        VectorizedKernel().apply(a, None, P, Q, cos_phi, sin_phi, tan_phi)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(a), np.linalg.eigvalsh(LAB_MATRIX), atol=1e-10
        )


class TestStoppingCriterion:
    r"""Раздел 1.5: убывание суммы квадратов недиагональных элементов."""

    def test_sum_decreases_by_twice_the_squared_element(self):
        r"""Одно вращение зануляет пару :math:`A_{pq}, A_{qp}`, поэтому
        :math:`S_{i+1} = S_i - 2 A_{pq}^2`.

        В формуле (19) методички множитель 2 опущен, хотя сумма :math:`S`
        там определена по всем недиагональным элементам.
        """
        a = LAB_MATRIX.copy()
        s_before = off_diagonal_sq(a)
        a_pq = a[P, Q]

        cos_phi, sin_phi, tan_phi = rotation_parameters(a[P, P], a[Q, Q], a_pq)
        VectorizedKernel().apply(a, None, P, Q, cos_phi, sin_phi, tan_phi)

        s_after = off_diagonal_sq(a)
        assert s_after == pytest.approx(s_before - 2 * a_pq**2, rel=1e-12)

    def test_sum_is_monotonically_decreasing(self):
        """Сумма ограничена снизу нулём и убывает — метод обязан сойтись."""
        a = LAB_MATRIX.copy()
        previous = off_diagonal_sq(a)
        for p in range(5):
            for q in range(p + 1, 5):
                if a[p, q] == 0.0:
                    continue
                cos_phi, sin_phi, tan_phi = rotation_parameters(a[p, p], a[q, q], a[p, q])
                VectorizedKernel().apply(a, None, p, q, cos_phi, sin_phi, tan_phi)
                current = off_diagonal_sq(a)
                assert current <= previous + 1e-12, "сумма квадратов не должна расти"
                previous = current


class TestFullSolutionOfLabMatrix:
    """Полное решение примера из методички."""

    def test_spectrum(self):
        result = jacobi_eigh(LAB_MATRIX)
        np.testing.assert_allclose(
            result.eigenvalues, np.linalg.eigvalsh(LAB_MATRIX), atol=1e-11
        )

    def test_rank_two_matrix_has_three_zero_eigenvalues(self):
        r"""``A[i][j] = 1 + 3(i + j)`` — матрица ранга 2, значит три нуля в спектре."""
        result = jacobi_eigh(LAB_MATRIX)
        assert np.sum(np.abs(result.eigenvalues) < 1e-10) == 3
        assert np.linalg.matrix_rank(LAB_MATRIX) == 2

    def test_eigen_equation_holds(self):
        result = jacobi_eigh(LAB_MATRIX)
        assert result.residual_norm(LAB_MATRIX) < 1e-11
        assert result.orthogonality_error() < 1e-14
