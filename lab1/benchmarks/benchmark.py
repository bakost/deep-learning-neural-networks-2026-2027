#!/usr/bin/env python3
"""Сравнение производительности реализаций метода Якоби и NumPy.

Скрипт измеряет время диагонализации случайных симметричных матриц:

* тремя ядрами вращения (``scalar``, ``vectorized``, ``matrix``);
* тремя стратегиями выбора элемента (``cyclic``, ``max``, ``round-robin``);
* эталонной реализацией :func:`numpy.linalg.eigh` (LAPACK, ``dsyevd``).

Запуск::

    python benchmarks/benchmark.py                    # быстрый прогон
    python benchmarks/benchmark.py --sizes 10 20 40 80 --repeats 5
    python benchmarks/benchmark.py --markdown results.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jacobi_eigen import (  # noqa: E402
    ConvergenceError,
    JacobiEigensolver,
    contracts_disabled,
)

#: Размеры матриц, на которых ядро `scalar` и особенно `matrix` ещё разумны.
DEFAULT_SIZES = (10, 20, 40, 80, 160)


def random_symmetric(n: int, seed: int = 0) -> np.ndarray:
    """Случайная симметричная матрица ``n x n``."""
    generator = np.random.default_rng(seed)
    m = generator.normal(size=(n, n))
    return (m + m.T) / 2.0


def timeit(func: Callable[[], object], repeats: int) -> float:
    """Минимальное время из ``repeats`` запусков (устойчивее среднего)."""
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - started)
    return best


def benchmark_kernels(
    sizes: Sequence[int], repeats: int, skip_slow_above: int
) -> List[Dict[str, object]]:
    """Замерить время всех ядер и NumPy на матрицах заданных размеров."""
    rows: List[Dict[str, object]] = []
    for n in sizes:
        a = random_symmetric(n, seed=n)
        reference = np.linalg.eigvalsh(a)
        norm = float(np.linalg.norm(a))

        for kernel in ("scalar", "vectorized", "matrix"):
            # Кубическое ядро на больших матрицах считается минутами — пропускаем.
            if kernel in ("scalar", "matrix") and n > skip_slow_above:
                rows.append({"n": n, "method": f"jacobi/{kernel}", "time": None,
                             "rotations": None, "error": None})
                continue
            solver = JacobiEigensolver(kernel=kernel)
            try:
                result = solver.solve(a)
            except ConvergenceError:
                rows.append({"n": n, "method": f"jacobi/{kernel}", "time": None,
                             "rotations": None, "error": None})
                continue
            elapsed = timeit(lambda: solver.solve(a), repeats)
            rows.append({
                "n": n,
                "method": f"jacobi/{kernel}",
                "time": elapsed,
                "rotations": result.rotations,
                "error": float(np.max(np.abs(result.eigenvalues - reference))) / norm,
            })

        rows.append({
            "n": n,
            "method": "numpy.linalg.eigh",
            "time": timeit(lambda: np.linalg.eigh(a), repeats),
            "rotations": 0,
            "error": 0.0,
        })
    return rows


def benchmark_strategies(sizes: Sequence[int], repeats: int) -> List[Dict[str, object]]:
    """Сравнить стратегии выбора зануляемого элемента (ядро — векторизованное)."""
    rows: List[Dict[str, object]] = []
    for n in sizes:
        a = random_symmetric(n, seed=n)
        for strategy in ("cyclic", "max", "round-robin"):
            solver = JacobiEigensolver(kernel="vectorized", strategy=strategy)
            try:
                result = solver.solve(a)
            except ConvergenceError:
                continue
            rows.append({
                "n": n,
                "method": strategy,
                "time": timeit(lambda: solver.solve(a), repeats),
                "rotations": result.rotations,
                "sweeps": result.sweeps,
            })
    return rows


def benchmark_contracts(n: int, repeats: int) -> Dict[str, float]:
    """Оценить накладные расходы контрактов."""
    a = random_symmetric(n, seed=n)
    solver = JacobiEigensolver()
    with_contracts = timeit(lambda: solver.solve(a), repeats)
    with contracts_disabled():
        without_contracts = timeit(lambda: solver.solve(a), repeats)
    return {"with": with_contracts, "without": without_contracts}


def format_table(rows: Sequence[Dict[str, object]], markdown: bool = False) -> str:
    """Отформатировать результаты замеров в таблицу."""
    sizes = sorted({int(row["n"]) for row in rows})
    methods: List[str] = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(str(row["method"]))

    lookup = {(int(row["n"]), str(row["method"])): row for row in rows}
    separator = " | " if markdown else "  "

    header = ["n"] + methods
    lines = []
    if markdown:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
    else:
        lines.append(separator.join(f"{title:>20s}" for title in header))
        lines.append("-" * (22 * len(header)))

    for n in sizes:
        cells = [f"{n}"]
        for method in methods:
            row = lookup.get((n, method))
            if row is None or row.get("time") is None:
                cells.append("—")
            else:
                cells.append(f"{float(row['time']) * 1000:.2f} мс")
        if markdown:
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append(separator.join(f"{cell:>20s}" for cell in cells))
    return "\n".join(lines)


def format_slowdown(rows: Sequence[Dict[str, object]], markdown: bool = False) -> str:
    """Во сколько раз каждое ядро медленнее NumPy."""
    sizes = sorted({int(row["n"]) for row in rows})
    lookup = {(int(row["n"]), str(row["method"])): row for row in rows}
    methods = [m for m in {str(r["method"]) for r in rows} if m != "numpy.linalg.eigh"]
    methods.sort()

    header = ["n"] + [f"{m} / numpy" for m in methods]
    lines = []
    if markdown:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
    else:
        lines.append("  ".join(f"{title:>24s}" for title in header))
        lines.append("-" * (26 * len(header)))

    for n in sizes:
        base = lookup.get((n, "numpy.linalg.eigh"))
        if not base or base.get("time") is None:
            continue
        cells = [f"{n}"]
        for method in methods:
            row = lookup.get((n, method))
            if row is None or row.get("time") is None:
                cells.append("—")
            else:
                cells.append(f"x{float(row['time']) / float(base['time']):.0f}")
        if markdown:
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append("  ".join(f"{cell:>24s}" for cell in cells))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа скрипта."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="размеры матриц")
    parser.add_argument("--repeats", type=int, default=3, help="число повторов замера")
    parser.add_argument("--skip-slow-above", type=int, default=80,
                        help="не запускать ядра scalar/matrix на матрицах крупнее")
    parser.add_argument("--markdown", metavar="FILE", nargs="?", const="-",
                        help="вывести таблицы в формате Markdown (опционально — в файл)")
    args = parser.parse_args(argv)

    markdown = args.markdown is not None
    blocks: List[str] = []

    print("Замеры идут, это может занять пару минут...", file=sys.stderr)

    # Замеры самого алгоритма проводятся с отключёнными контрактами: иначе
    # измерялась бы скорость проверок, а не вычислений. Цена контрактов
    # измеряется отдельно, в конце отчёта.
    with contracts_disabled():
        kernel_rows = benchmark_kernels(args.sizes, args.repeats, args.skip_slow_above)
    blocks.append("## Время работы ядер\n" if markdown else "=== Время работы ядер ===")
    blocks.append(format_table(kernel_rows, markdown))
    blocks.append("\n## Замедление относительно numpy.linalg.eigh\n" if markdown
                  else "\n=== Замедление относительно numpy.linalg.eigh ===")
    blocks.append(format_slowdown(kernel_rows, markdown))

    strategy_sizes = [n for n in args.sizes if n <= 80]
    if strategy_sizes:
        with contracts_disabled():
            strategy_rows = benchmark_strategies(strategy_sizes, args.repeats)
        blocks.append("\n## Стратегии выбора элемента\n" if markdown
                      else "\n=== Стратегии выбора элемента ===")
        blocks.append(format_table(strategy_rows, markdown))
        lines = ["", "Число вращений до сходимости:"]
        for row in strategy_rows:
            lines.append(f"  n={row['n']:4d}  {row['method']:14s} "
                         f"вращений={row['rotations']:7d}  свипов={row['sweeps']}")
        blocks.append("\n".join(lines))

    overhead = benchmark_contracts(40, args.repeats)
    ratio = overhead["with"] / overhead["without"] if overhead["without"] else float("nan")
    blocks.append("\n## Накладные расходы контрактов (n = 40)\n" if markdown
                  else "\n=== Накладные расходы контрактов (n = 40) ===")
    blocks.append(
        f"с контрактами:  {overhead['with'] * 1000:.2f} мс\n"
        f"без контрактов: {overhead['without'] * 1000:.2f} мс\n"
        f"отношение:      x{ratio:.3f}"
    )

    text = "\n".join(blocks)
    if markdown and args.markdown != "-":
        Path(args.markdown).write_text(text + "\n", encoding="utf-8")
        print(f"Результаты записаны в {args.markdown}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
