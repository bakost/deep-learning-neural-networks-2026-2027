"""Интерфейс командной строки: ``python -m jacobi_eigen``.

Примеры использования::

    # матрица из текстового файла (строки — строки матрицы)
    python -m jacobi_eigen --input matrix.txt

    # случайная симметричная матрица 6x6 с фиксированным зерном
    python -m jacobi_eigen --random 6 --seed 42 --check

    # разбор примера из методички, шаг за шагом
    python -m jacobi_eigen --demo

    # сравнение трёх ядер и numpy на матрице 100x100
    python -m jacobi_eigen --random 100 --compare

    # машиночитаемый вывод
    python -m jacobi_eigen --random 4 --json

Любая предусмотренная ошибка печатается одной понятной строкой, а программа
завершается с кодом 2 — трассировка стека наружу не выводится.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, List, Optional, Sequence

import numpy as np

from . import __version__
from .exceptions import ConvergenceError, JacobiError
from .kernels import available_kernels
from .solver import JacobiEigensolver
from .strategies import available_strategies

EXIT_OK = 0
EXIT_ERROR = 2

#: Матрица из примера раздела 1.2 методички.
DEMO_MATRIX = np.array([[1.0 + 3.0 * (i + j) for j in range(5)] for i in range(5)])


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="jacobi-eigen",
        description="Метод Якоби: собственные числа и векторы вещественной симметричной матрицы.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", metavar="FILE",
                        help="текстовый файл с матрицей (числа через пробел/запятую)")
    source.add_argument("-r", "--random", type=int, metavar="N",
                        help="сгенерировать случайную симметричную матрицу N x N")
    source.add_argument("-d", "--demo", action="store_true",
                        help="разобрать пример 5x5 из методички")

    parser.add_argument("--seed", type=int, default=None, help="зерно генератора случайных чисел")
    parser.add_argument("-t", "--tol", type=float, default=1e-12, help="точность (по умолчанию 1e-12)")
    parser.add_argument("--tol-mode", choices=("relative", "absolute"), default="relative",
                        help="порог относительно ||A||_F или абсолютный")
    parser.add_argument("-k", "--kernel", choices=sorted(available_kernels()), default="vectorized",
                        help="ядро вращения")
    parser.add_argument("-s", "--strategy", choices=sorted(available_strategies()), default="cyclic",
                        help="стратегия выбора зануляемого элемента")
    parser.add_argument("--max-sweeps", type=int, default=100, help="лимит свипов (защита от зацикливания)")
    parser.add_argument("--sort", choices=("asc", "desc", "none"), default="asc",
                        help="порядок собственных чисел")
    parser.add_argument("--values-only", action="store_true", help="не вычислять собственные векторы")
    parser.add_argument("-c", "--check", action="store_true",
                        help="сверить результат с numpy.linalg.eigh и напечатать невязки")
    parser.add_argument("--compare", action="store_true",
                        help="сравнить время работы всех ядер и numpy")
    parser.add_argument("--json", action="store_true", help="вывод в формате JSON")
    parser.add_argument("-p", "--precision", type=int, default=6, help="знаков после запятой")
    parser.add_argument("-V", "--version", action="version", version=f"jacobi-eigen {__version__}")
    return parser


def load_matrix(path: str) -> np.ndarray:
    """Прочитать матрицу из текстового файла.

    Поддерживаются пробелы и запятые в качестве разделителей, пустые строки
    и комментарии, начинающиеся с ``#``.

    Raises
    ------
    JacobiError
        Если файл не читается или не содержит корректной числовой таблицы.
        Ошибка ввода-вывода намеренно оборачивается в тип пакета, чтобы
        вызывающая сторона могла обойтись одним обработчиком.
    """
    from .exceptions import InvalidParameterError, NotArrayLikeError

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as exc:
        raise InvalidParameterError(f"Не удалось прочитать файл {path!r}: {exc}") from exc

    rows: List[List[float]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.replace(",", " ").replace(";", " ").split()
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as exc:
            raise NotArrayLikeError(
                f"Файл {path!r}, строка {number}: не удалось разобрать число ({exc})."
            ) from exc
    if not rows:
        raise NotArrayLikeError(f"Файл {path!r} не содержит числовых данных.")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise NotArrayLikeError(
            f"Файл {path!r}: строки матрицы имеют разную длину {sorted(widths)}."
        )
    return np.array(rows, dtype=np.float64)


def random_symmetric(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Сгенерировать случайную симметричную матрицу ``n x n``."""
    from .exceptions import InvalidParameterError

    if n < 1:
        raise InvalidParameterError(f"Размер матрицы должен быть положительным, получено {n}.")
    generator = np.random.default_rng(seed)
    m = generator.normal(size=(n, n))
    return (m + m.T) / 2.0


def run_comparison(matrix: np.ndarray, solver_kwargs: dict) -> List[dict]:
    """Замерить время всех ядер и :func:`numpy.linalg.eigh` на одной матрице."""
    reference = np.linalg.eigvalsh(matrix)
    rows: List[dict] = []
    for kernel_name in available_kernels():
        options = dict(solver_kwargs, kernel=kernel_name)
        try:
            result = JacobiEigensolver(**options).solve(matrix)
            rows.append({
                "method": f"jacobi/{kernel_name}",
                "time": result.elapsed,
                "sweeps": result.sweeps,
                "rotations": result.rotations,
                "max_error": float(np.max(np.abs(result.eigenvalues - reference))),
            })
        except ConvergenceError as exc:
            rows.append({"method": f"jacobi/{kernel_name}", "time": float("nan"),
                         "sweeps": exc.sweeps, "rotations": exc.rotations,
                         "max_error": float("nan")})
    started = time.perf_counter()
    np.linalg.eigh(matrix)
    rows.append({"method": "numpy.linalg.eigh", "time": time.perf_counter() - started,
                 "sweeps": 0, "rotations": 0, "max_error": 0.0})
    return rows


def format_comparison(rows: Sequence[dict]) -> str:
    """Отформатировать таблицу сравнения."""
    header = f"{'метод':22s} {'время, мс':>12s} {'свипы':>7s} {'вращения':>10s} {'макс. ошибка':>14s}"
    lines = [header, "-" * len(header)]
    baseline = next((r["time"] for r in rows if r["method"] == "numpy.linalg.eigh"), None)
    for row in rows:
        lines.append(
            f"{row['method']:22s} {row['time'] * 1000:12.3f} {row['sweeps']:7d} "
            f"{row['rotations']:10d} {row['max_error']:14.3e}"
        )
    if baseline:
        lines.append("")
        lines.append("Замедление относительно numpy.linalg.eigh:")
        for row in rows:
            if row["method"] != "numpy.linalg.eigh" and row["time"] == row["time"]:
                lines.append(f"  {row['method']:22s} x{row['time'] / baseline:8.1f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа CLI.

    Returns
    -------
    int
        ``0`` при успехе, ``2`` при любой предусмотренной ошибке.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.demo:
            matrix = DEMO_MATRIX
        elif args.input:
            matrix = load_matrix(args.input)
        else:
            matrix = random_symmetric(args.random, args.seed)

        solver_kwargs: dict = {
            "tol": args.tol,
            "tol_mode": args.tol_mode,
            "strategy": args.strategy,
            "max_sweeps": args.max_sweeps,
            "sort": None if args.sort == "none" else args.sort,
            "compute_eigenvectors": not args.values_only,
        }

        if args.compare:
            rows = run_comparison(matrix, solver_kwargs)
            print(json.dumps(rows, indent=2) if args.json else format_comparison(rows))
            return EXIT_OK

        solver = JacobiEigensolver(kernel=args.kernel, **solver_kwargs)
        result = solver.solve(matrix)

        if args.json:
            payload: dict = result.to_dict()
            if args.check:
                payload["check"] = _check_payload(result, matrix)
            print(json.dumps(payload, indent=2))
        else:
            if args.demo:
                print("Матрица из примера методички:")
                with np.printoptions(precision=args.precision, suppress=True):
                    print(matrix)
                print()
            print(result.summary(matrix, precision=args.precision))
            if args.check:
                check = _check_payload(result, matrix)
                print()
                print("Сверка с numpy.linalg.eigh:")
                print(f"  макс. расхождение собственных чисел: {check['max_eigenvalue_error']:.3e}")
                print(f"  время jacobi: {check['jacobi_time'] * 1000:.3f} мс, "
                      f"numpy: {check['numpy_time'] * 1000:.3f} мс "
                      f"(x{check['slowdown']:.1f})")
        return EXIT_OK

    except JacobiError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - интерактивное прерывание
        print("Прервано пользователем.", file=sys.stderr)
        return EXIT_ERROR


def _check_payload(result: Any, matrix: np.ndarray) -> dict:
    """Сверка результата с NumPy: расхождения и относительная скорость."""
    started = time.perf_counter()
    reference = np.linalg.eigvalsh(matrix)
    numpy_time = time.perf_counter() - started
    return {
        "max_eigenvalue_error": float(np.max(np.abs(np.sort(result.eigenvalues) - reference))),
        "orthogonality_error": result.orthogonality_error() if result.eigenvectors.size else None,
        "jacobi_time": result.elapsed,
        "numpy_time": numpy_time,
        "slowdown": result.elapsed / numpy_time if numpy_time > 0 else float("inf"),
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
