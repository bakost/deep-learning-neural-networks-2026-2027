"""Тесты интерфейса командной строки.

Главное проверяемое свойство: при любых входных данных программа
завершается штатно (код 0 или 2) и печатает понятное сообщение — трассировка
стека наружу не попадает.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from jacobi_eigen.__main__ import EXIT_ERROR, EXIT_OK, load_matrix, main, random_symmetric


@pytest.fixture
def matrix_file(tmp_path):
    """Файл с корректной симметричной матрицей."""
    path = tmp_path / "matrix.txt"
    path.write_text("# комментарий\n2 1 0\n1 3 1\n\n0 1 4\n", encoding="utf-8")
    return path


class TestFileLoading:
    def test_reads_matrix_with_comments_and_blank_lines(self, matrix_file):
        matrix = load_matrix(str(matrix_file))
        np.testing.assert_allclose(matrix, [[2, 1, 0], [1, 3, 1], [0, 1, 4]])

    def test_accepts_comma_separated_values(self, tmp_path):
        path = tmp_path / "csv.txt"
        path.write_text("1.0, 2.0\n2.0, 1.0\n", encoding="utf-8")
        np.testing.assert_allclose(load_matrix(str(path)), [[1, 2], [2, 1]])

    def test_missing_file_is_reported(self):
        from jacobi_eigen import JacobiError

        with pytest.raises(JacobiError, match="Не удалось прочитать файл"):
            load_matrix("/no/such/file.txt")

    def test_ragged_file_is_reported(self, tmp_path):
        from jacobi_eigen import NotArrayLikeError

        path = tmp_path / "ragged.txt"
        path.write_text("1 2 3\n4 5\n", encoding="utf-8")
        with pytest.raises(NotArrayLikeError, match="разную длину"):
            load_matrix(str(path))

    def test_non_numeric_file_is_reported(self, tmp_path):
        from jacobi_eigen import NotArrayLikeError

        path = tmp_path / "text.txt"
        path.write_text("привет мир\n", encoding="utf-8")
        with pytest.raises(NotArrayLikeError, match="строка 1"):
            load_matrix(str(path))

    def test_empty_file_is_reported(self, tmp_path):
        from jacobi_eigen import NotArrayLikeError

        path = tmp_path / "empty.txt"
        path.write_text("\n# только комментарий\n", encoding="utf-8")
        with pytest.raises(NotArrayLikeError, match="не содержит"):
            load_matrix(str(path))


class TestSuccessfulRuns:
    def test_demo(self, capsys):
        assert main(["--demo"]) == EXIT_OK
        output = capsys.readouterr().out
        assert "Собственные числа" in output
        assert "Сходимость:  достигнута" in output

    def test_random_with_seed_is_reproducible(self, capsys):
        assert main(["--random", "5", "--seed", "7", "--json"]) == EXIT_OK
        first = json.loads(capsys.readouterr().out)
        assert main(["--random", "5", "--seed", "7", "--json"]) == EXIT_OK
        second = json.loads(capsys.readouterr().out)
        assert first["eigenvalues"] == second["eigenvalues"]

    def test_file_input(self, matrix_file, capsys):
        assert main(["--input", str(matrix_file), "--check"]) == EXIT_OK
        assert "Сверка с numpy" in capsys.readouterr().out

    def test_json_output_is_valid(self, capsys):
        assert main(["--random", "4", "--seed", "1", "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["converged"] is True
        assert len(payload["eigenvalues"]) == 4
        assert len(payload["eigenvectors"]) == 4

    @pytest.mark.parametrize("kernel", ["scalar", "vectorized", "matrix"])
    def test_all_kernels(self, kernel, capsys):
        assert main(["--random", "5", "--seed", "2", "--kernel", kernel]) == EXIT_OK
        assert f"ядро = {kernel}" in capsys.readouterr().out

    @pytest.mark.parametrize("strategy", ["cyclic", "max", "round-robin"])
    def test_all_strategies(self, strategy, capsys):
        assert main(["--random", "5", "--seed", "2", "--strategy", strategy]) == EXIT_OK
        assert f"стратегия = {strategy}" in capsys.readouterr().out

    def test_values_only(self, capsys):
        assert main(["--random", "4", "--seed", "3", "--values-only"]) == EXIT_OK
        assert "Собственные числа" in capsys.readouterr().out

    def test_compare_mode(self, capsys):
        assert main(["--random", "12", "--seed", "5", "--compare"]) == EXIT_OK
        output = capsys.readouterr().out
        assert "numpy.linalg.eigh" in output
        for kernel in ("scalar", "vectorized", "matrix"):
            assert f"jacobi/{kernel}" in output

    def test_sort_options(self, capsys):
        for order in ("asc", "desc", "none"):
            assert main(["--random", "4", "--seed", "6", "--sort", order, "--json"]) == EXIT_OK
            values = json.loads(capsys.readouterr().out)["eigenvalues"]
            if order == "asc":
                assert values == sorted(values)
            elif order == "desc":
                assert values == sorted(values, reverse=True)


class TestErrorHandling:
    """Ошибки печатаются одной строкой в stderr, код возврата — 2."""

    def test_asymmetric_matrix_file(self, tmp_path, capsys):
        path = tmp_path / "asym.txt"
        path.write_text("1 2\n3 4\n", encoding="utf-8")
        assert main(["--input", str(path)]) == EXIT_ERROR
        captured = capsys.readouterr()
        assert "не симметрична" in captured.err
        assert "Traceback" not in captured.err

    def test_missing_file(self, capsys):
        assert main(["--input", "/no/such/file.txt"]) == EXIT_ERROR
        assert "Ошибка:" in capsys.readouterr().err

    def test_non_square_matrix_file(self, tmp_path, capsys):
        path = tmp_path / "rect.txt"
        path.write_text("1 2 3\n4 5 6\n7 8 9\n10 11 12\n", encoding="utf-8")
        assert main(["--input", str(path)]) == EXIT_ERROR
        assert "квадратной" in capsys.readouterr().err

    def test_zero_size_random(self, capsys):
        assert main(["--random", "0"]) == EXIT_ERROR
        assert "положительным" in capsys.readouterr().err

    def test_bad_tolerance(self, capsys):
        assert main(["--random", "3", "--tol", "0"]) == EXIT_ERROR
        assert "положительным" in capsys.readouterr().err

    def test_convergence_failure_is_reported(self, capsys):
        assert main(["--random", "25", "--seed", "1", "--max-sweeps", "1",
                     "--tol", "1e-15"]) == EXIT_ERROR
        captured = capsys.readouterr()
        assert "не сошёлся" in captured.err
        assert "Traceback" not in captured.err

    def test_unknown_kernel_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit) as info:
            main(["--random", "3", "--kernel", "quantum"])
        assert info.value.code != 0

    def test_requires_an_input_source(self):
        with pytest.raises(SystemExit):
            main([])


class TestHelpers:
    def test_random_symmetric_is_symmetric(self):
        matrix = random_symmetric(7, seed=1)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_random_symmetric_rejects_bad_size(self):
        from jacobi_eigen import InvalidParameterError

        with pytest.raises(InvalidParameterError):
            random_symmetric(-3)
