"""Командная строка: ``python -m sgd_lab <команда>``.

Команды::

    minimize   задача 1 — минимум f(z) методом золотого сечения или Брента
    train      задачи 2–5 — обучить одну из трёх моделей выбранным методом
    report     пересчитать все рисунки и таблицы отчёта
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from .linesearch import bracket_minimum, brent, exact_minimum, f_task, golden_section
from .models import MODELS, X, Y, init_params
from .training import METHODS, Objective, TrainConfig, train

EXIT_OK, EXIT_ERROR = 0, 2
LAB_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgd_lab",
                                     description="ЛР6: стохастический градиентный спуск и его модификации")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    mn = commands.add_parser("minimize", help="задача 1: минимум f(z)")
    mn.add_argument("-m", "--method", choices=("golden", "brent"), default="brent")
    mn.add_argument("-a", type=float, default=0.0, help="первая начальная точка для поиска отрезка (0)")
    mn.add_argument("-b", type=float, default=1.0, help="вторая начальная точка (1)")
    mn.add_argument("--tol", type=float, default=3e-8, help="относительная точность (3e-8 ≈ √ε_маш)")
    mn.add_argument("--plot", metavar="FILE", help="сохранить график f(z) и сходимость")

    tr = commands.add_parser("train", help="задачи 2–5: обучить модель")
    tr.add_argument("model", choices=tuple(MODELS), help="linear | quadratic | relu")
    tr.add_argument("-m", "--method", choices=tuple(METHODS), default="gd", help="gd | nesterov | adam")
    tr.add_argument("-e", "--lr", type=float, default=0.01, help="скорость обучения ε (0.01)")
    tr.add_argument("-b", "--batch-size", type=int, default=None, help="размер мини-пакета (по умолчанию вся выборка)")
    tr.add_argument("-n", "--epochs", type=int, default=10_000, help="число эпох (10000)")
    tr.add_argument("-d", "--delta", type=float, default=None, help="остановка по ‖DJ‖ < δ")
    tr.add_argument("--tau", type=float, default=None, help="ε_e = ε₀/(1 + e/τ)")
    tr.add_argument("--l2", type=float, default=0.0, help="коэффициент квадратичного штрафа λ (0)")
    tr.add_argument("-s", "--seed", type=int, default=0, help="зерно начальных весов и перемешивания (0)")
    tr.add_argument("--plot", metavar="FILE", help="сохранить данные и найденную зависимость")
    tr.add_argument("--json", action="store_true", help="вывести результат в JSON")

    rep = commands.add_parser("report", help="все рисунки и таблицы отчёта")
    rep.add_argument("-o", "--output", default=str(LAB_ROOT), help="каталог для figures/ и results/")
    rep.add_argument("--quick", action="store_true", help="уменьшенный вариант (≈ 20 с)")
    rep.add_argument("-j", "--workers", type=int, default=None, help="процессов (по числу ядер, до 8)")
    return parser


def _cmd_minimize(args) -> int:
    if not args.tol > 0:
        raise ValueError("точность должна быть положительной")
    trip, nb = bracket_minimum(f_task, args.a, args.b)
    res = (brent if args.method == "brent" else golden_section)(f_task, trip, tol=args.tol)
    z_star = exact_minimum()
    name = "метод Брента" if args.method == "brent" else "метод золотого сечения"
    print(f"Начальный отрезок (mnbrak): [{trip[0]:.6g}, {trip[2]:.6g}], внутренняя точка {trip[1]:.6g}, "
          f"{nb} вычислений f")
    print(f"{name}: z* = {res.x:.12f}, f(z*) = {res.fx:.12f}")
    print(f"  итераций {res.iterations}, вычислений f {res.nfev}; |z − z*| = {abs(res.x - z_star):.1e} "
          f"(эталон — корень f′(z) = 0: {z_star:.15f})")
    if args.plot:
        from . import plots

        plots.save(plots.plot_minimize(res), args.plot)
        print(f"  рисунок: {args.plot}")
    return EXIT_OK


def _cmd_train(args) -> int:
    model = MODELS[args.model]
    obj = Objective(model, X, Y, args.l2)
    cfg = TrainConfig(method=args.method, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                      delta=args.delta, tau=args.tau, seed=args.seed, record_every=max(1, args.epochs // 200))
    res = train(obj, init_params(model, args.seed), cfg)
    if args.json:
        print(json.dumps({"model": args.model, "method": args.method, "status": res.status, "epochs": res.epochs,
                          "updates": res.updates, "loss": res.loss, "sse": res.sse, "grad_norm": res.grad_norm,
                          "params": dict(zip(model.names, res.params.tolist()))}, ensure_ascii=False))
    else:
        b = args.batch_size or X.size
        print(f"Модель {model.formula}; {METHODS[args.method]}, ε = {args.lr:g}, мини-пакет {b}"
              + (f", λ = {args.l2:g}" if args.l2 else ""))
        status = {"converged": "сошёлся (‖DJ‖ < δ)", "max_epochs": "выполнены все эпохи",
                  "diverged": "разошёлся"}[res.status]
        print(f"  {status}: эпох {res.epochs}, обновлений {res.updates}, ‖DJ‖ = {res.grad_norm:.2e}")
        print(f"  сумма квадратов ошибок {res.sse:.6f}" + (f", J̃ = {res.loss:.6f}" if args.l2 else ""))
        print("  " + ", ".join(f"{n} = {v:.5f}" for n, v in zip(model.names, res.params)))
    if args.plot:
        from . import plots

        plots.save(plots.plot_fit(model, res.params), args.plot)
    return EXIT_OK


def _cmd_report(args) -> int:
    from .report import ReportConfig, generate_report

    cfg = ReportConfig.quick() if args.quick else ReportConfig()
    if args.workers is not None:
        cfg = ReportConfig(**{**cfg.__dict__, "workers": args.workers})
    generate_report(cfg, args.output)
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            if not args.lr > 0:
                raise ValueError("скорость обучения должна быть положительной")
            if args.batch_size is not None and not 1 <= args.batch_size <= X.size:
                raise ValueError(f"размер мини-пакета должен быть от 1 до {X.size}")
            if args.l2 < 0:
                raise ValueError("коэффициент штрафа не может быть отрицательным")
        with np.errstate(over="ignore", invalid="ignore"):
            return {"minimize": _cmd_minimize, "train": _cmd_train, "report": _cmd_report}[args.command](args)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
