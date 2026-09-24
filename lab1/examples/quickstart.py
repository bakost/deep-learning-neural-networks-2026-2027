#!/usr/bin/env python3
"""Примеры использования пакета :mod:`jacobi_eigen`.

Запуск::

    python3 examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jacobi_eigen import (  # noqa: E402
    ConvergenceError,
    JacobiEigensolver,
    JacobiError,
    jacobi_eigh,
)


def example_1_basic() -> None:
    """Простейший вызов: одна функция, один результат."""
    print("=" * 70)
    print("1. Базовое использование")
    print("=" * 70)

    a = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    result = jacobi_eigh(a)

    print("Матрица A:", a.tolist(), sep="\n")
    print("\nСобственные числа:", np.round(result.eigenvalues, 6))
    print("Собственные векторы (по столбцам):")
    print(np.round(result.eigenvectors, 4))
    print(f"\nСошлось за {result.sweeps} свипов, {result.rotations} вращений")
    print(f"Невязка ||A V - V L||_F = {result.residual_norm(a):.2e}")


def example_2_object_interface() -> None:
    """Объектный интерфейс: настройки задаются один раз."""
    print("\n" + "=" * 70)
    print("2. Объектный интерфейс и повторное использование")
    print("=" * 70)

    solver = JacobiEigensolver(tol=1e-14, kernel="vectorized", strategy="max")
    print(solver)

    for n in (3, 5, 8):
        generator = np.random.default_rng(n)
        m = generator.normal(size=(n, n))
        a = (m + m.T) / 2
        result = solver.solve(a)
        error = np.max(np.abs(result.eigenvalues - np.linalg.eigvalsh(a)))
        print(f"  n={n}: {result.rotations:4d} вращений, "
              f"расхождение с numpy {error:.2e}")


def example_3_error_handling() -> None:
    """Обработка ошибок: один except на все случаи жизни."""
    print("\n" + "=" * 70)
    print("3. Диагностика некорректных данных")
    print("=" * 70)

    bad_inputs = {
        "несимметричная":  [[1.0, 2.0], [3.0, 4.0]],
        "не квадратная":   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "комплексная":     [[1.0, 2j], [-2j, 1.0]],
        "с NaN":           [[1.0, np.nan], [np.nan, 1.0]],
        "строки вместо чисел": [["a", "b"], ["c", "d"]],
        "вообще не матрица":   {"ключ": "значение"},
    }
    for label, matrix in bad_inputs.items():
        try:
            jacobi_eigh(matrix)
        except JacobiError as exc:
            print(f"  {label:22s} -> {type(exc).__name__}")
            print(f"  {'':22s}    {exc}")


def example_4_convergence_control() -> None:
    """Контроль сходимости: метод не зацикливается."""
    print("\n" + "=" * 70)
    print("4. Контроль сходимости")
    print("=" * 70)

    generator = np.random.default_rng(42)
    m = generator.normal(size=(20, 20))
    a = (m + m.T) / 2

    for tol in (1e-2, 1e-8, 1e-14):
        result = jacobi_eigh(a, tol=tol)
        print(f"  tol={tol:.0e}: {result.sweeps} свипов, "
              f"{result.rotations:5d} вращений, off(A)={result.off_norm:.2e}")

    try:
        jacobi_eigh(a, tol=1e-15, max_sweeps=1)
    except ConvergenceError as exc:
        print(f"\n  При max_sweeps=1: {exc}")
        print(f"  Частичный результат доступен: converged={exc.partial.converged}")


def example_5_comparison() -> None:
    """Сравнение ядер и стратегий."""
    print("\n" + "=" * 70)
    print("5. Сравнение ядер вращения и стратегий")
    print("=" * 70)

    generator = np.random.default_rng(1)
    m = generator.normal(size=(30, 30))
    a = (m + m.T) / 2
    reference = np.linalg.eigvalsh(a)

    print(f"  {'конфигурация':30s} {'время, мс':>10s} {'вращений':>10s} {'ошибка':>12s}")
    print("  " + "-" * 64)
    for kernel in ("scalar", "vectorized", "matrix"):
        for strategy in ("cyclic", "max"):
            result = jacobi_eigh(a, kernel=kernel, strategy=strategy)
            error = np.max(np.abs(result.eigenvalues - reference))
            print(f"  {kernel + '/' + strategy:30s} {result.elapsed * 1000:10.2f} "
                  f"{result.rotations:10d} {error:12.2e}")


def example_6_properties() -> None:
    """Проверка математических свойств результата."""
    print("\n" + "=" * 70)
    print("6. Проверка качества результата")
    print("=" * 70)

    generator = np.random.default_rng(7)
    m = generator.normal(size=(10, 10))
    a = (m + m.T) / 2
    result = jacobi_eigh(a)

    print(f"  ||A V - V L||_F      = {result.residual_norm(a):.3e}   (определение с. вектора)")
    print(f"  ||V^T V - I||_F      = {result.orthogonality_error():.3e}   (ортогональность)")
    print(f"  ||A - V L V^T||_F    = {result.reconstruction_error(a):.3e}   (спектральное разложение)")
    print(f"  |sum(L) - tr(A)|     = {result.trace_error(a):.3e}   (сохранение следа)")
    print(f"  |prod(L) - det(A)|   = {abs(np.prod(result.eigenvalues) - np.linalg.det(a)):.3e}"
          "   (сохранение определителя)")


def main() -> int:
    example_1_basic()
    example_2_object_interface()
    example_3_error_handling()
    example_4_convergence_control()
    example_5_comparison()
    example_6_properties()
    print("\nВсе примеры выполнены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
