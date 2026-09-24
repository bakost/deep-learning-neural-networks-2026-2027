"""Интерфейс командной строки: ``python -m level_spacing``.

Примеры::

    # пункты 1–8 для GOE: 2×2, 4×4, 16×16 по 100 000 матриц
    python -m level_spacing run --ensemble goe --sizes 2 4 16 --matrices 100000

    # то же для матриц из ±1 и с сохранением гистограмм
    python -m level_spacing run --ensemble pm1 --plot pm1.png

    # собственные числа методом Якоби из ЛР1 (медленно, для проверки)
    python -m level_spacing run --sizes 4 --matrices 1000 --backend jacobi

    # точный закон для матриц 4×4 из ±1 полным перебором
    python -m level_spacing exact --size 4 --construction mirror

    # все рисунки и таблицы отчёта
    python -m level_spacing report

Ошибка в параметрах печатается одной строкой, код возврата — 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .ensembles import ENSEMBLES
from .exact import exact_pm1_spacing
from .experiments import DEFAULT_SEED, run_study
from .spectrum import BACKENDS

EXIT_OK = 0
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="level-spacing",
        description="Распределение расстояний между соседними собственными числами случайных матриц.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="пункты 1–8 задания для одного ансамбля")
    run.add_argument("-e", "--ensemble", default="goe", choices=sorted(ENSEMBLES),
                     help="ансамбль (по умолчанию goe)")
    run.add_argument("-n", "--sizes", type=int, nargs="+", default=[2, 4, 16], metavar="N",
                     help="размеры матриц (по умолчанию 2 4 16)")
    run.add_argument("-m", "--matrices", type=int, default=100_000, metavar="M",
                     help="число матриц в ансамбле (по умолчанию 100000)")
    run.add_argument("-k", "--pair", type=int, default=None, metavar="K",
                     help="номер пары (λ_K, λ_K+1), с единицы; по умолчанию центральная n/2")
    run.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED,
                     help=f"зерно генератора (по умолчанию {DEFAULT_SEED})")
    run.add_argument("-b", "--backend", default="numpy", choices=BACKENDS,
                     help="чем считать собственные числа (по умолчанию numpy)")
    run.add_argument("-p", "--plot", metavar="FILE", help="сохранить гистограммы в файл")
    run.add_argument("--json", action="store_true", help="вывести результат в JSON")

    exact = commands.add_parser("exact", help="точный закон для матриц из ±1 (полный перебор)")
    exact.add_argument("-n", "--size", type=int, default=2, metavar="N", help="размер матриц")
    exact.add_argument("-c", "--construction", default="mirror", choices=("mirror", "sum"),
                       help="mirror — элементы матрицы ±1; sum — H = A + Aᵀ, A из ±1")
    exact.add_argument("-k", "--pair", type=int, default=None, metavar="K", help="номер пары")

    report = commands.add_parser("report", help="пересчитать рисунки и таблицы отчёта")
    report.add_argument("-o", "--output", default=".", metavar="DIR",
                        help="куда сохранить figures/ и results/ (по умолчанию текущий каталог)")
    report.add_argument("--quick", action="store_true", help="уменьшенные ансамбли (≈ 10 с)")

    commands.add_parser("list", help="список ансамблей")
    return parser


def _run(args: argparse.Namespace) -> int:
    studies = [
        run_study(args.ensemble, n, args.matrices, args.seed, k=args.pair, backend=args.backend)
        for n in args.sizes
    ]
    summaries = [study.summary() for study in studies]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        ensemble = ENSEMBLES[args.ensemble]
        print(f"Ансамбль: {ensemble.name} — {ensemble.title}; M = {args.matrices}, зерно {args.seed}")
        header = (f"{'n':>3} {'пара':>8} {'<s> до норм.':>13} {'дисперсия':>10} "
                  f"{'D_KS(Вигнер)':>13} {'p':>8} {'D_KS(GOE∞)':>11} {'P(s=0)':>8}")
        print(header)
        print("-" * len(header))
        for s in summaries:
            # p-значение моделируется по 1000 выборкам, поэтому меньше 0.001 не бывает.
            p = "<0.001" if s["wigner_ks_p"] < 0.001 else f"{s['wigner_ks_p']:.3f}"
            print(f"{s['n']:>3} {'(' + str(s['k']) + ',' + str(s['k'] + 1) + ')':>8} "
                  f"{s['raw_mean']:>13.4f} {s['variance']:>10.4f} {s['wigner_ks_d']:>13.4f} "
                  f"{p:>8} {s['goe-limit_ks_d']:>11.4f} {s['degenerate_fraction']:>8.4f}")
        print("Догадка Вигнера: дисперсия 0.2732; точный закон GOE (n → ∞): 0.2855.")
    if args.plot:
        from .plots import plot_histogram_grid

        fig = plot_histogram_grid([studies], widths=[0.05 if args.matrices >= 20_000 else 0.2],
                                  title=ENSEMBLES[args.ensemble].title)
        fig.savefig(args.plot, dpi=150)
        print(f"Гистограммы сохранены в {args.plot}")
    return EXIT_OK


def _exact(args: argparse.Namespace) -> int:
    law = exact_pm1_spacing(args.size, args.construction, k=args.pair)
    normalized = law.normalized()
    count = 2 ** (args.size * (args.size + 1) // 2 if args.construction == "mirror" else args.size**2)
    print(f"Матрицы {args.size}x{args.size} из ±1 ({args.construction}), перебрано {count} матриц")
    print(f"<s> = {law.mean:.6f}; дисперсия после нормировки {normalized.variance:.6f}")
    print(f"{'s':>12} {'s/<s>':>12} {'вероятность':>12}")
    for raw, value, p in zip(law.values, normalized.values, law.probabilities):
        print(f"{raw:>12.6f} {value:>12.6f} {p:>12.6g}")
    return EXIT_OK


def _report(args: argparse.Namespace) -> int:
    from .report import ReportConfig, generate_report

    config = ReportConfig.quick() if args.quick else ReportConfig()
    generate_report(Path(args.output), config)
    return EXIT_OK


def _list(_: argparse.Namespace) -> int:
    for ensemble in ENSEMBLES.values():
        print(f"{ensemble.name:<14} {ensemble.title}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа; возвращает код завершения."""
    args = build_parser().parse_args(argv)
    handlers = {"run": _run, "exact": _exact, "report": _report, "list": _list}
    try:
        return handlers[args.command](args)
    except (ValueError, TypeError, ImportError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
