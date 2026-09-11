"""Тесты устойчивости: программа не должна падать ни при каких входных данных.

Задание формулирует требование так:

    «Для корректных входных данных реализация должна либо возвращать
    результат, либо останавливаться с ошибкой. Зацикливание недопустимо.
    Недопустимо, чтобы реализация возвращала иные ошибки, кроме
    предусмотренных при разработке».

Ниже по решателю «стреляют» случайными и патологическими данными и
проверяют, что итог всегда один из двух: корректный результат либо
исключение из иерархии :class:`~jacobi_eigen.exceptions.JacobiError`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from conftest import random_symmetric, symmetric_with_spectrum
from jacobi_eigen import JacobiEigensolver, JacobiError, JacobiResult, jacobi_eigh


def run_safely(matrix, **kwargs):
    """Запустить решатель и вернуть ``("ok", result)`` либо ``("error", exc)``.

    Любое исключение *вне* иерархии пакета немедленно проваливает тест —
    именно это и есть проверяемое требование.
    """
    try:
        return "ok", jacobi_eigh(matrix, **kwargs)
    except JacobiError as exc:
        return "error", exc
    except Exception as exc:  # noqa: BLE001 - намеренно широкий перехват
        pytest.fail(
            f"Непредусмотренное исключение {type(exc).__name__}: {exc}\n"
            f"на входе {matrix!r} с параметрами {kwargs}"
        )


class TestFuzzRandomInput:
    """Случайный перебор всевозможных входов."""

    def test_random_valid_matrices_always_succeed(self):
        """1000 случайных корректных матриц — все обязаны решиться."""
        generator = np.random.default_rng(2024)
        for _ in range(1000):
            n = int(generator.integers(1, 12))
            scale = 10.0 ** float(generator.integers(-6, 7))
            m = generator.normal(scale=scale, size=(n, n))
            a = (m + m.T) / 2
            status, result = run_safely(a)
            assert status == "ok", f"корректная матрица {n}x{n} отвергнута: {result}"
            assert result.converged
            np.testing.assert_allclose(
                result.eigenvalues, np.linalg.eigvalsh(a),
                atol=1e-9 * max(np.linalg.norm(a), 1.0),
            )

    def test_random_garbage_never_crashes(self):
        """Произвольный мусор приводит либо к результату, либо к JacobiError."""
        generator = np.random.default_rng(7)
        garbage_pool = [
            None, "строка", b"bytes", {"k": "v"}, {1, 2}, 3.5, [],
            [[]], [[1, 2], [3]], [[1, 2, 3]], np.zeros((0, 0)), np.zeros((2, 3)),
            np.zeros((2, 2, 2)), [[np.nan, 1], [1, 2]], [[np.inf, 0], [0, 1]],
            [[1, 2], [3, 4]], [[1j, 0], [0, 1j]], [["a", "b"], ["c", "d"]],
            [[None, None], [None, None]], np.array([[1e308, 1e308], [1e308, 1e308]]),
        ]
        for _ in range(200):
            candidate = garbage_pool[int(generator.integers(len(garbage_pool)))]
            run_safely(candidate)  # достаточно, что не упало непредусмотренно

    def test_all_option_combinations(self):
        """Декартово произведение настроек на одной матрице."""
        a = random_symmetric(6, seed=17)
        kernels = ("scalar", "vectorized", "matrix")
        strategies = ("cyclic", "max", "round-robin")
        sorts = ("asc", "desc", None)
        modes = ("relative", "absolute")
        for kernel, strategy, sort, mode in itertools.product(
            kernels, strategies, sorts, modes
        ):
            tol = 1e-12 if mode == "relative" else 1e-10
            status, result = run_safely(
                a, kernel=kernel, strategy=strategy, sort=sort, tol=tol, tol_mode=mode
            )
            assert status == "ok", f"{kernel}/{strategy}/{sort}/{mode}: {result}"
            np.testing.assert_allclose(
                np.sort(result.eigenvalues), np.linalg.eigvalsh(a), atol=1e-9
            )


class TestExtremeValues:
    """Экстремальные, но формально корректные данные."""

    @pytest.mark.parametrize("scale", [1e-300, 1e-150, 1e-15, 1e15, 1e150, 1e300])
    def test_extreme_scales(self, scale):
        """Матрицы на грани диапазона float64 не должны ломать алгоритм."""
        a = np.array([[1.0, 0.5], [0.5, 2.0]]) * scale
        status, result = run_safely(a)
        assert status == "ok"
        np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a), rtol=1e-10)

    def test_huge_values_near_overflow(self):
        """Элементы порядка 1e308: возведение в квадрат переполняется."""
        a = np.array([[1e308, 1e307], [1e307, -1e308]])
        status, result = run_safely(a)
        # Допустимы оба исхода, но не аварийное завершение.
        assert status in ("ok", "error")
        if status == "ok":
            assert np.all(np.isfinite(result.eigenvalues))

    def test_denormal_off_diagonal(self):
        """Субнормальные внедиагональные элементы: деление почти на ноль."""
        a = np.array([[1.0, 5e-324], [5e-324, 2.0]])
        status, result = run_safely(a)
        assert status == "ok"
        np.testing.assert_allclose(result.eigenvalues, [1.0, 2.0], atol=1e-15)

    def test_enormous_diagonal_gap(self):
        r"""Огромная разность диагональных элементов: :math:`C \to \infty`."""
        a = np.array([[1e-200, 1.0], [1.0, 1e200]])
        status, result = run_safely(a)
        assert status == "ok"
        assert np.all(np.isfinite(result.eigenvalues))

    def test_zero_matrix_with_tiny_perturbation(self):
        a = np.full((4, 4), 1e-320)
        a = (a + a.T) / 2
        status, result = run_safely(a)
        assert status == "ok"
        assert np.all(np.isfinite(result.eigenvalues))


class TestTerminationGuarantee:
    """Зацикливание невозможно ни при каких настройках."""

    @pytest.mark.parametrize("tol", [1e-323, 1e-300, 1e-16])
    def test_unreachable_tolerance_terminates(self, tol):
        """Недостижимая точность приводит к ошибке, а не к вечному циклу."""
        a = random_symmetric(10, seed=3)
        status, outcome = run_safely(a, tol=tol, tol_mode="absolute", max_sweeps=5)
        assert status in ("ok", "error")
        if status == "error":
            assert "не сошёлся" in str(outcome)

    @pytest.mark.parametrize("n", [1, 2, 3, 8])
    def test_every_size_terminates(self, n):
        for seed in range(20):
            status, result = run_safely(random_symmetric(n, seed=seed))
            assert status == "ok"
            assert result.sweeps <= 100

    def test_pathological_spectra_terminate(self):
        spectra = [
            [0.0] * 5,
            [1.0] * 6,
            [1e-15, 1e15],
            [-1.0, 1.0, -1.0, 1.0],
            list(range(10)),
            [1.0, 1.0 + 1e-15, 1.0 - 1e-15],
        ]
        for spectrum in spectra:
            a = symmetric_with_spectrum(spectrum, seed=1)
            status, result = run_safely(a)
            assert status == "ok", f"спектр {spectrum}: {result}"
            np.testing.assert_allclose(
                result.eigenvalues, np.sort(spectrum),
                atol=1e-9 * max(np.max(np.abs(spectrum)), 1.0),
            )


class TestPurity:
    """Отсутствие побочных эффектов."""

    def test_input_matrix_is_never_modified(self, kernel):
        a = random_symmetric(7, seed=5)
        backup = a.copy()
        jacobi_eigh(a, kernel=kernel)
        np.testing.assert_array_equal(a, backup)

    def test_repeated_calls_are_deterministic(self):
        a = random_symmetric(9, seed=6)
        first = jacobi_eigh(a)
        second = jacobi_eigh(a)
        np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
        np.testing.assert_array_equal(first.eigenvectors, second.eigenvectors)

    def test_result_is_immutable_dataclass(self):
        result = jacobi_eigh(np.eye(3))
        assert isinstance(result, JacobiResult)
        with pytest.raises(Exception):
            result.eigenvalues = np.zeros(3)

    def test_non_contiguous_input(self):
        """Вход-срез (не непрерывный в памяти) обрабатывается корректно."""
        big = random_symmetric(12, seed=8)
        view = big[::2, ::2]
        status, result = run_safely(view)
        assert status == "ok"
        np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(view), atol=1e-11)

    def test_readonly_input(self):
        a = random_symmetric(5, seed=9)
        a.flags.writeable = False
        status, _ = run_safely(a)
        assert status == "ok"


class TestResultApi:
    """Методы результата тоже не должны давать непредусмотренных ошибок."""

    def test_methods_without_eigenvectors_raise_jacobi_error(self):
        a = random_symmetric(5, seed=10)
        result = JacobiEigensolver(compute_eigenvectors=False).solve(a)
        assert result.has_eigenvectors is False
        for call in (
            lambda: result.orthogonality_error(),
            lambda: result.reconstruct(),
            lambda: result.residual_norm(a),
            lambda: result.reconstruction_error(a),
        ):
            with pytest.raises(JacobiError, match="compute_eigenvectors"):
                call()
        # А то, что не требует векторов, работает.
        assert result.trace_error(a) < 1e-12
        assert "не вычислялись" in result.summary(a)

    def test_residual_with_mismatched_matrix(self):
        result = jacobi_eigh(random_symmetric(4, seed=11))
        with pytest.raises(JacobiError, match="не совпадает"):
            result.residual_norm(random_symmetric(6, seed=12))

    def test_residual_with_garbage_matrix(self):
        result = jacobi_eigh(random_symmetric(4, seed=13))
        with pytest.raises(JacobiError):
            result.residual_norm("не матрица")

    def test_sorted_and_to_dict(self):
        a = random_symmetric(5, seed=14)
        result = jacobi_eigh(a, sort=None)
        ascending = result.sorted("asc")
        assert np.all(np.diff(ascending.eigenvalues) >= 0)
        assert ascending.residual_norm(a) < 1e-11
        assert np.all(np.diff(result.sorted("desc").eigenvalues) <= 0)
        with pytest.raises(JacobiError):
            result.sorted("случайно")
        payload = result.to_dict()
        assert set(payload) >= {"eigenvalues", "eigenvectors", "converged", "kernel"}

    def test_repr_and_summary_do_not_fail(self):
        a = random_symmetric(4, seed=15)
        result = jacobi_eigh(a)
        assert "JacobiResult" in repr(result)
        assert "Метод Якоби" in result.summary(a)
        assert "Метод Якоби" in result.summary()
