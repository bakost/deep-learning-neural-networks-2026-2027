"""Тесты корректности метода Якоби.

Эталон — :func:`numpy.linalg.eigh` (LAPACK). Дополнительно проверяются
свойства, которые обязаны выполняться независимо от эталона:
:math:`AV = V\\Lambda`, ортогональность :math:`V`, сохранение следа и
определителя.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import random_symmetric, symmetric_with_spectrum
from jacobi_eigen import (
    ConvergenceError,
    JacobiEigensolver,
    MaxElementStrategy,
    jacobi_eigh,
)


class TestKnownMatrices:
    """Матрицы, для которых ответ известен аналитически."""

    def test_two_by_two(self):
        r"""Для ``[[a, b], [b, a]]`` собственные числа равны ``a ± b``."""
        result = jacobi_eigh([[2.0, 1.0], [1.0, 2.0]])
        np.testing.assert_allclose(result.eigenvalues, [1.0, 3.0], atol=1e-14)
        assert result.rotations == 1, "матрица 2x2 диагонализуется одним вращением"

    def test_diagonal_matrix_needs_no_rotations(self):
        result = jacobi_eigh(np.diag([5.0, -2.0, 3.0]))
        assert result.rotations == 0
        assert result.sweeps == 0
        np.testing.assert_allclose(result.eigenvalues, [-2.0, 3.0, 5.0])
        # Сортировка переставляет столбцы, поэтому V — матрица перестановки
        # (в каждом столбце ровно одна единица), а не тождественная.
        np.testing.assert_allclose(np.sort(np.abs(result.eigenvectors), axis=0)[-1], np.ones(3))
        np.testing.assert_allclose(
            result.eigenvectors @ result.eigenvectors.T, np.eye(3), atol=1e-15
        )

    def test_identity(self):
        result = jacobi_eigh(np.eye(4))
        np.testing.assert_allclose(result.eigenvalues, np.ones(4))
        assert result.rotations == 0

    def test_zero_matrix(self):
        result = jacobi_eigh(np.zeros((3, 3)))
        np.testing.assert_allclose(result.eigenvalues, np.zeros(3))
        np.testing.assert_allclose(result.eigenvectors, np.eye(3))

    def test_one_by_one(self):
        result = jacobi_eigh([[42.0]])
        np.testing.assert_allclose(result.eigenvalues, [42.0])
        np.testing.assert_allclose(result.eigenvectors, [[1.0]])

    def test_all_ones_matrix(self):
        r"""Матрица из одних единиц имеет спектр ``{n, 0, ..., 0}``."""
        n = 5
        result = jacobi_eigh(np.ones((n, n)))
        expected = np.array([0.0] * (n - 1) + [float(n)])
        np.testing.assert_allclose(np.sort(result.eigenvalues), expected, atol=1e-13)

    def test_tridiagonal_analytic_spectrum(self):
        r"""Матрица ``2`` на диагонали и ``-1`` рядом: :math:`\\lambda_k = 2 - 2\\cos\\frac{k\\pi}{n+1}`."""
        n = 8
        a = np.diag([2.0] * n) + np.diag([-1.0] * (n - 1), 1) + np.diag([-1.0] * (n - 1), -1)
        expected = np.sort([2 - 2 * np.cos(k * np.pi / (n + 1)) for k in range(1, n + 1)])
        result = jacobi_eigh(a)
        np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-12)


class TestAgainstNumpy:
    """Сверка с эталонной реализацией LAPACK."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13, 25, 40])
    def test_random_matrices(self, n, kernel):
        a = random_symmetric(n, seed=n)
        result = jacobi_eigh(a, kernel=kernel)
        np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a), atol=1e-11)

    def test_all_strategies_agree(self, strategy):
        a = random_symmetric(12, seed=7)
        result = jacobi_eigh(a, strategy=strategy)
        np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a), atol=1e-11)

    def test_kernels_produce_identical_results(self):
        """Три ядра реализуют одну математику и обязаны совпадать."""
        a = random_symmetric(15, seed=3)
        results = {k: jacobi_eigh(a, kernel=k) for k in ("scalar", "vectorized", "matrix")}
        base = results["scalar"]
        for name, result in results.items():
            np.testing.assert_allclose(
                result.eigenvalues, base.eigenvalues, atol=1e-11,
                err_msg=f"ядро {name} разошлось со скалярным",
            )
        # Скалярное и векторизованное ядра считают по одинаковым формулам,
        # поэтому обязаны совпадать бит в бит.
        np.testing.assert_array_equal(
            results["scalar"].eigenvalues, results["vectorized"].eigenvalues
        )

    def test_eigenvectors_match_numpy_up_to_sign(self):
        a = random_symmetric(10, seed=11)
        result = jacobi_eigh(a)
        _, reference = np.linalg.eigh(a)
        for k in range(a.shape[0]):
            ours, theirs = result.eigenvectors[:, k], reference[:, k]
            if np.dot(ours, theirs) < 0:
                ours = -ours
            np.testing.assert_allclose(ours, theirs, atol=1e-7)


class TestMathematicalProperties:
    """Свойства, проверяемые без эталона."""

    @pytest.mark.parametrize("n", [3, 7, 20])
    def test_eigen_equation(self, n, kernel):
        r"""Определение собственного вектора: :math:`AV = V\\Lambda`."""
        a = random_symmetric(n, seed=n + 100)
        result = jacobi_eigh(a, kernel=kernel)
        assert result.residual_norm(a) < 1e-10

    @pytest.mark.parametrize("n", [3, 7, 20])
    def test_eigenvectors_orthonormal(self, n):
        r"""Произведение вращений ортогонально: :math:`V^{T}V = I`."""
        result = jacobi_eigh(random_symmetric(n, seed=n + 200))
        assert result.orthogonality_error() < 1e-12

    def test_reconstruction(self):
        r"""Спектральное разложение: :math:`A = V \\Lambda V^{T}`."""
        a = random_symmetric(12, seed=5)
        result = jacobi_eigh(a)
        assert result.reconstruction_error(a) < 1e-11
        np.testing.assert_allclose(result.reconstruct(), a, atol=1e-11)

    def test_trace_is_preserved(self):
        r"""След — инвариант подобия: :math:`\\sum \\lambda_k = \\mathrm{tr}\\,A`."""
        a = random_symmetric(10, seed=9)
        result = jacobi_eigh(a)
        assert result.trace_error(a) < 1e-11

    def test_determinant_is_preserved(self):
        r"""Определитель равен произведению собственных чисел."""
        a = random_symmetric(6, seed=13)
        result = jacobi_eigh(a)
        assert np.prod(result.eigenvalues) == pytest.approx(np.linalg.det(a), rel=1e-9)

    def test_off_norm_below_threshold(self):
        result = jacobi_eigh(random_symmetric(10, seed=21))
        assert result.converged
        assert result.off_norm <= result.threshold


class TestHardSpectra:
    """Численно неприятные случаи."""

    def test_multiple_eigenvalues(self):
        """Кратные собственные числа: угол поворота при ``C = 0`` не определён формулой (9)."""
        a = symmetric_with_spectrum([2.0, 2.0, 2.0, 5.0, 5.0], seed=1)
        result = jacobi_eigh(a)
        np.testing.assert_allclose(result.eigenvalues, [2.0, 2.0, 2.0, 5.0, 5.0], atol=1e-11)
        assert result.orthogonality_error() < 1e-12

    def test_equal_diagonal_elements(self):
        r"""``C = 0`` в чистом виде: поворот ровно на 45 градусов."""
        a = np.array([[3.0, 4.0], [4.0, 3.0]])
        result = jacobi_eigh(a)
        np.testing.assert_allclose(result.eigenvalues, [-1.0, 7.0], atol=1e-13)

    def test_singular_matrix(self):
        a = symmetric_with_spectrum([0.0, 0.0, 1.0, 3.0], seed=2)
        result = jacobi_eigh(a)
        np.testing.assert_allclose(result.eigenvalues, [0.0, 0.0, 1.0, 3.0], atol=1e-12)

    def test_negative_definite(self):
        a = symmetric_with_spectrum([-7.0, -3.0, -1.0], seed=4)
        result = jacobi_eigh(a)
        assert np.all(result.eigenvalues < 0)
        np.testing.assert_allclose(result.eigenvalues, [-7.0, -3.0, -1.0], atol=1e-12)

    def test_widely_separated_eigenvalues(self):
        """Спектр, растянутый на 12 порядков.

        Требовать здесь высокой *относительной* точности для наименьшего
        собственного числа нельзя, и виноват в этом не метод: сама матрица
        хранится с погрешностью ``||A||_F * eps ~ 2e-10``, что на четыре
        порядка больше, чем ``lambda_min = 1e-6``. Поэтому проверяется
        точность, нормированная на ``||A||_F``, и сравнение с LAPACK.
        """
        spectrum = np.array([1e-6, 1.0, 1e6])
        a = symmetric_with_spectrum(spectrum, seed=6)
        result = jacobi_eigh(a)
        norm = np.linalg.norm(a)

        np.testing.assert_allclose(result.eigenvalues, spectrum, atol=1e-9 * norm)
        # Наибольшие собственные числа обязаны получаться с полной точностью.
        np.testing.assert_allclose(result.eigenvalues[1:], spectrum[1:], rtol=1e-11)
        # На этой задаче метод Якоби не уступает LAPACK.
        reference = np.linalg.eigvalsh(a)
        assert abs(result.eigenvalues[0] - spectrum[0]) <= abs(reference[0] - spectrum[0])

    @pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
    def test_scale_invariance(self, scale):
        r"""Масштабирование матрицы масштабирует спектр: :math:`\\mathrm{spec}(cA) = c\\,\\mathrm{spec}(A)`."""
        a = random_symmetric(6, seed=8)
        base = jacobi_eigh(a).eigenvalues
        scaled = jacobi_eigh(a * scale).eigenvalues
        np.testing.assert_allclose(scaled, base * scale, rtol=1e-10)

    def test_hilbert_matrix(self):
        """Матрица Гильберта — классический пример плохой обусловленности."""
        n = 7
        a = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        result = jacobi_eigh(a)
        np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a), atol=1e-14)
        # Метод Якоби вычисляет даже мельчайшие собственные числа с высокой
        # относительной точностью — известное преимущество перед QR-алгоритмом.
        assert result.eigenvalues[0] > 0, "матрица Гильберта положительно определена"

    def test_nearly_diagonal_matrix(self):
        """Почти диагональная матрица: критерий остановки не должен сработать ложно."""
        a = np.diag([10.0, 20.0, 30.0]) + 1e-11
        a = (a + a.T) / 2
        result = jacobi_eigh(a)
        np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a), atol=1e-15)


class TestSolverOptions:
    """Параметры решателя."""

    def test_sort_ascending_is_default(self):
        values = jacobi_eigh(random_symmetric(8, seed=31)).eigenvalues
        assert np.all(np.diff(values) >= 0)

    def test_sort_descending(self):
        values = jacobi_eigh(random_symmetric(8, seed=31), sort="desc").eigenvalues
        assert np.all(np.diff(values) <= 0)

    def test_sort_none_keeps_algorithmic_order(self):
        a = random_symmetric(8, seed=31)
        values = jacobi_eigh(a, sort=None).eigenvalues
        np.testing.assert_allclose(np.sort(values), np.linalg.eigvalsh(a), atol=1e-11)

    def test_sorting_keeps_vectors_aligned(self):
        """При сортировке столбцы V должны переставляться вместе с числами."""
        a = random_symmetric(9, seed=41)
        for order in ("asc", "desc", None):
            result = jacobi_eigh(a, sort=order)
            assert result.residual_norm(a) < 1e-10, f"sort={order} нарушил соответствие"

    def test_values_only_mode(self):
        a = random_symmetric(10, seed=51)
        solver = JacobiEigensolver(compute_eigenvectors=False)
        values = solver.eigenvalues(a)
        np.testing.assert_allclose(values, np.linalg.eigvalsh(a), atol=1e-11)
        assert solver.compute_eigenvectors is False, "исходный флаг должен восстанавливаться"

        # А для решателя «с векторами» вызов eigenvalues() тоже не должен
        # оставлять после себя изменённых настроек.
        full = JacobiEigensolver()
        full.eigenvalues(a)
        assert full.compute_eigenvectors is True
        assert full.solve(a).eigenvectors.shape == (10, 10)

    def test_absolute_tolerance_mode(self):
        a = random_symmetric(5, seed=61)
        result = jacobi_eigh(a, tol=1e-10, tol_mode="absolute")
        assert result.off_norm <= 1e-10

    def test_looser_tolerance_needs_fewer_rotations(self):
        a = random_symmetric(20, seed=71)
        loose = jacobi_eigh(a, tol=1e-4)
        tight = jacobi_eigh(a, tol=1e-14)
        assert loose.rotations <= tight.rotations
        assert loose.off_norm >= tight.off_norm

    def test_solver_is_reusable(self):
        """Один решатель применяется к разным матрицам без побочных эффектов."""
        solver = JacobiEigensolver()
        for n in (3, 5, 7):
            a = random_symmetric(n, seed=n)
            np.testing.assert_allclose(
                solver.solve(a).eigenvalues, np.linalg.eigvalsh(a), atol=1e-11
            )

    def test_callable_syntax(self):
        solver = JacobiEigensolver()
        a = random_symmetric(4, seed=81)
        np.testing.assert_array_equal(solver(a).eigenvalues, solver.solve(a).eigenvalues)

    def test_custom_kernel_instance_is_accepted(self):
        from jacobi_eigen import ScalarKernel

        result = jacobi_eigh(random_symmetric(5, seed=91), kernel=ScalarKernel())
        assert result.kernel == "scalar"

    def test_custom_strategy_instance_is_accepted(self):
        result = jacobi_eigh(random_symmetric(5, seed=92), strategy=MaxElementStrategy())
        assert result.strategy == "max"


class TestTermination:
    """Гарантия конечности работы — программа не должна зацикливаться."""

    def test_convergence_error_when_budget_exhausted(self):
        """При искусственно заниженном лимите возбуждается предусмотренная ошибка."""
        a = random_symmetric(20, seed=99)
        with pytest.raises(ConvergenceError) as info:
            jacobi_eigh(a, max_sweeps=1, tol=1e-15)
        error = info.value
        assert error.sweeps == 1
        assert error.off_norm > error.threshold
        assert "не сошёлся" in str(error)

    def test_convergence_error_carries_partial_result(self):
        """Частичный результат доступен для диагностики."""
        a = random_symmetric(20, seed=98)
        with pytest.raises(ConvergenceError) as info:
            jacobi_eigh(a, max_sweeps=1, tol=1e-15)
        partial = info.value.partial
        assert partial is not None
        assert partial.converged is False
        assert partial.eigenvalues.shape == (20,)

    @pytest.mark.parametrize("n", [2, 5, 10, 20, 30])
    def test_converges_well_within_default_budget(self, n):
        """Метод Якоби сходится за единицы свипов — запас по умолчанию огромен."""
        result = jacobi_eigh(random_symmetric(n, seed=n * 3))
        assert result.converged
        assert result.sweeps <= 12, "неожиданно медленная сходимость"

    def test_rotation_count_is_bounded_by_sweeps(self):
        n = 15
        result = jacobi_eigh(random_symmetric(n, seed=17))
        assert result.rotations <= result.sweeps * n * (n - 1) // 2

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [60, 100])
    def test_larger_matrices(self, n):
        a = random_symmetric(n, seed=n)
        result = jacobi_eigh(a)
        np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a), atol=1e-10)
