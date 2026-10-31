"""Командная строка: counts, equivariance, softmax, report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .counts import parameter_counts
from .experiments import equivariance_experiment, softmax_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ЛР6а: самовнимание и softmax")
    commands = parser.add_subparsers(dest="command", required=True)
    counts = commands.add_parser("counts", help="число параметров (задача 1)")
    counts.add_argument("-D", type=int, default=64)
    counts.add_argument("-N", type=int, default=16)
    counts.add_argument("-H", type=int, default=None, help="скрытая ширина MLP; по умолчанию DN")
    equiv = commands.add_parser("equivariance", help="численная проверка задачи 2")
    equiv.add_argument("-D", type=int, default=3)
    equiv.add_argument("-N", type=int, default=4)
    equiv.add_argument("--seed", type=int, default=2026)
    equiv.add_argument("--scaled", action="store_true")
    sm = commands.add_parser("softmax", help="25 производных из задачи 3")
    for command in (counts, equiv, sm):
        command.add_argument("--json", action="store_true")
    report = commands.add_parser("report", help="пересчитать таблицы, JSON, CSV и рисунки")
    report.add_argument("--output", type=Path, default=Path.cwd(), help="каталог результатов (по умолчанию текущий)")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            from .report import generate_report

            generate_report(args.output)
            print(f"Результаты и рисунки: {args.output.resolve()}")
            return 0
        if args.command == "counts":
            result = parameter_counts(args.D, args.N, args.H)
            message = (f'Q/K/V: {result["qkv_total"]} обучаемых параметров\n'
                       f'A(X): {result["attention_entries"]} коэффициентов, 0 собственных обучаемых параметров\n'
                       f'MLP (H = {result["H"]}): {result["mlp_total"]} обучаемых параметров')
        elif args.command == "equivariance":
            result = equivariance_experiment(args.D, args.N, args.seed, scaled=args.scaled)
            message = (f'max |Sa(XP) − Sa(X)P| = {result["output_max_abs_error"]:.3e}\n'
                       f'max |A(XP) − PᵀA(X)P| = {result["attention_max_abs_error"]:.3e}')
        else:
            result = softmax_experiment()
            message = ("y = " + np.array2string(np.array(result["probabilities"]), precision=8)
                       + "\nJ =\n" + np.array2string(np.array(result["jacobian"]), precision=8)
                       + f'\nJ33: обычная формула {result["naive_J33"]}, устойчивая {result["stable_J33"]:.8e}'
                       + f'\nМаксимальная относительная ошибка против Decimal: '
                       f'{result["max_relative_error_vs_decimal"]:.3e}')
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) if args.json else message)
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
