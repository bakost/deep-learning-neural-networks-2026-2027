"""Интерфейс командной строки: ``python -m lipo_mlp``.

Примеры::

    # один перцептрон: H = 8, ε = 0.1, δ = 1e-4 (пункты 2–4)
    python -m lipo_mlp train --hidden 8

    # с уменьшением скорости обучения εⱼ = ε₀/(1 + j/τ)
    python -m lipo_mlp train --hidden 16 --schedule inverse --tau 10000

    # все размеры скрытого слоя, по 3 зерна (пункт 5, первый вопрос)
    python -m lipo_mlp sweep --seeds 3

    # сравнение с scikit-learn
    python -m lipo_mlp sklearn --hidden 8

    # все рисунки и таблицы отчёта
    python -m lipo_mlp report

Ошибка в параметрах печатается одной строкой, код возврата — 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from .data import Standardizer, load_lipo
from .experiments import DELTA, EPS, MAX_ITER, SIZES, Job, interpolation_floor, run_jobs
from .network import init_params, n_params
from .schedules import Schedule, constant, exponential, inverse_time, step
from .training import gradient_descent

EXIT_OK = 0
EXIT_ERROR = 2
LAB_ROOT = Path(__file__).resolve().parents[2]


def _add_training_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-e", "--lr", type=float, default=EPS, help=f"скорость обучения ε (по умолчанию {EPS})")
    p.add_argument("-d", "--delta", type=float, default=DELTA, help=f"точность δ для нормы градиента ({DELTA:g})")
    p.add_argument("-n", "--max-iter", type=int, default=MAX_ITER, help=f"предел итераций ({MAX_ITER})")
    p.add_argument("--schedule", choices=("constant", "inverse", "exponential", "step"), default="constant",
                   help="закон уменьшения ε: constant, inverse ε₀/(1+j/τ), exponential ε₀γʲ, step ε₀γ^⌊j/m⌋")
    p.add_argument("--tau", type=float, default=1e4, help="τ для закона inverse (1e4)")
    p.add_argument("--gamma", type=float, default=0.99999, help="γ для exponential/step (0.99999)")
    p.add_argument("--every", type=int, default=20_000, help="m для закона step (20000)")


def _schedule(args) -> Schedule:
    if args.schedule == "inverse":
        return inverse_time(args.lr, args.tau)
    if args.schedule == "exponential":
        return exponential(args.lr, args.gamma)
    if args.schedule == "step":
        return step(args.lr, args.gamma, args.every)
    return constant(args.lr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lipo-mlp",
        description="Двухслойный перцептрон для коэффициента липофильности, обучение градиентным спуском.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="обучить один перцептрон")
    train.add_argument("-H", "--hidden", type=int, default=8, help="нейронов скрытого слоя (8)")
    train.add_argument("-s", "--seed", type=int, default=0, help="зерно начальных весов (0)")
    train.add_argument("-b", "--batch-size", type=int, default=None, help="размер мини-пакета (по умолчанию весь)")
    _add_training_args(train)
    train.add_argument("--plot", metavar="FILE", help="сохранить кривые обучения в файл")
    train.add_argument("--json", action="store_true", help="вывести результат в JSON")

    sweep = commands.add_parser("sweep", help="все размеры скрытого слоя")
    sweep.add_argument("-H", "--sizes", type=int, nargs="+", default=list(SIZES), help="размеры (2 4 8 16 32)")
    sweep.add_argument("--seeds", type=int, default=3, help="зёрен на размер (3)")
    sweep.add_argument("-j", "--workers", type=int, default=None, help="процессов (по числу ядер, до 8)")
    _add_training_args(sweep)

    sk = commands.add_parser("sklearn", help="сравнение с MLPRegressor из scikit-learn")
    sk.add_argument("-H", "--hidden", type=int, default=8)
    sk.add_argument("--iterations", type=int, default=2000, help="итераций для сверки траекторий (2000)")

    rep = commands.add_parser("report", help="все рисунки и таблицы отчёта")
    rep.add_argument("-o", "--output", default=str(LAB_ROOT), help="каталог для figures/ и results/")
    rep.add_argument("--quick", action="store_true", help="уменьшенный вариант (≈ 20 с)")
    rep.add_argument("-j", "--workers", type=int, default=None, help="процессов (по числу ядер, до 8)")
    return parser


def _check(args) -> None:
    if args.lr <= 0 or args.delta <= 0 or args.max_iter < 0:
        raise ValueError("нужны ε > 0, δ > 0, max-iter ≥ 0")


def _cmd_train(args) -> int:
    _check(args)
    if args.hidden < 1:
        raise ValueError("число нейронов должно быть положительным")
    data = load_lipo()
    scaler = Standardizer.fit(data.X, data.y)
    Z, t = scaler.transform(data.X, data.y)
    sch = _schedule(args)
    t0 = time.perf_counter()
    res = gradient_descent(init_params(data.k, args.hidden, args.seed), Z, t, lr=sch, delta=args.delta,
                           max_iter=args.max_iter, record_every=100, batch_size=args.batch_size, seed=args.seed)
    elapsed = time.perf_counter() - t0
    rmse = float(np.sqrt(2 * res.loss) * scaler.y_std) if np.isfinite(res.loss) else float("inf")
    r2 = 1 - rmse**2 / float(np.var(data.y))
    out = {"hidden": args.hidden, "params": n_params(data.k, args.hidden), "schedule": sch.label(),
           "delta": args.delta, "status": res.status, "iterations": res.iterations, "grad_norm": res.grad_norm,
           "rmse": rmse, "r2": r2, "seconds": elapsed}
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        status = {"converged": "сошёлся (‖DJ‖ < δ)", "max_iter": "не сошёлся за отведённые итерации",
                  "diverged": "разошёлся"}[res.status]
        print(f"Перцептрон H = {args.hidden} ({out['params']} параметров), {sch.label()}, δ = {args.delta:g}")
        print(f"  {status}: итераций {res.iterations}, ‖DJ‖ = {res.grad_norm:.3e}, {elapsed:.1f} с")
        print(f"  RMSE на обучающей выборке = {rmse:.4f}, R² = {r2:.4f} "
              f"(нижняя граница RMSE {interpolation_floor(data):.4f})")
    if args.plot:
        from matplotlib.figure import Figure

        from .plots import _style, save

        with _style():
            fig = Figure(figsize=(8, 3.4), layout="constrained")
            a, b = fig.subplots(1, 2)
            it = res.history["iteration"] + 1
            a.loglog(it, np.sqrt(2 * res.history["loss"]) * scaler.y_std)
            a.set_xlabel("итерация")
            a.set_ylabel("RMSE обучения")
            b.loglog(it, res.history["grad_norm"])
            b.axhline(args.delta, color="#898781")
            b.set_xlabel("итерация")
            b.set_ylabel("‖DJ‖")
        save(fig, args.plot)
    return EXIT_OK


def _cmd_sweep(args) -> int:
    _check(args)
    sch = _schedule(args)
    jobs = [Job("sweep", h, s, sch, args.delta, args.max_iter, keep_params=False)
            for h in args.sizes for s in range(args.seeds)]
    t0 = time.perf_counter()
    runs = run_jobs(jobs, args.workers)
    print(f"{sch.label()}, δ = {args.delta:g}, предел {args.max_iter} итераций; {time.perf_counter() - t0:.0f} с")
    print(f"{'H':>3} {'параметров':>10} {'сошлось':>8} {'итераций (медиана)':>19} {'RMSE':>7} {'‖DJ‖':>9}")
    for h in args.sizes:
        rs = [r for r in runs if r.job.hidden == h]
        conv = [r.iterations for r in rs if r.converged]
        med = f"{int(np.median(conv))}" if conv else "—"
        print(f"{h:>3} {n_params(24, h):>10} {len(conv):>5}/{len(rs):<2} {med:>19} "
              f"{np.mean([r.rmse for r in rs]):7.3f} {np.median([r.grad_norm for r in rs]):9.1e}")
    return EXIT_OK


def _cmd_sklearn(args) -> int:
    from .sklearn_compare import SKLEARN_CONFIGS, sklearn_fit, sklearn_trajectory

    data = load_lipo()
    Z, t = Standardizer.fit(data.X, data.y).transform(data.X, data.y)
    p0 = init_params(data.k, args.hidden, 123)
    ours = gradient_descent(p0, Z, t, lr=EPS, delta=1e-300, max_iter=args.iterations, record_every=10**9).params
    theirs = sklearn_trajectory(p0, Z, t, EPS, args.iterations)
    print(f"Одинаковые начальные веса, {args.iterations} итераций полнопакетного спуска, ε = {EPS}:")
    print(f"  max |w_наш − w_sklearn| = {np.max(np.abs(ours.flat() - theirs.flat())):.2e}")
    for name, conf in SKLEARN_CONFIGS.items():
        info, _ = sklearn_fit(data.X, data.y, args.hidden, conf)
        print(f"  MLPRegressor {name}: итераций {info['n_iter']}, RMSE обучения {info['rmse']:.3f}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "sweep":
            return _cmd_sweep(args)
        if args.command == "sklearn":
            return _cmd_sklearn(args)
        if args.command == "report":
            from .report import ReportConfig, generate_report

            cfg = ReportConfig.quick() if args.quick else ReportConfig()
            if args.workers:
                cfg = ReportConfig(**{**vars(cfg), "workers": args.workers})
            generate_report(Path(args.output), cfg)
            return EXIT_OK
    except (ValueError, FileNotFoundError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
