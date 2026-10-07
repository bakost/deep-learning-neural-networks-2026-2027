"""Интерфейс командной строки: ``python -m lipo_ols``.

Примеры::

    # пункты 2–6: коэффициенты, s², s, R², доверительные интервалы
    python -m lipo_ols fit

    # то же для части регрессоров и 99%-х интервалов
    python -m lipo_ols fit --columns C PSA HBA1 HBA2 --level 0.99

    # МНК-остатки по веществам (пункт 3)
    python -m lipo_ols residuals

    # условия применимости МНК (пункт 7)
    python -m lipo_ols diagnostics

    # все рисунки и таблицы отчёта
    python -m lipo_ols report

Ошибка в параметрах печатается одной строкой, код возврата — 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from .data import load_lipo
from .diagnostics import (
    breusch_pagan,
    condition_number,
    durbin_watson,
    leverage,
    normality_tests,
    pure_error,
    reset_test,
    vif,
)
from .ols import METHODS, fit_ols_data
from .validation import kfold_cv, press_statistics

EXIT_OK = 0
EXIT_ERROR = 2
LAB_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="lipo-ols",
        description="Линейная регрессия коэффициента липофильности на структурные дескрипторы.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-d", "--data", metavar="CSV", help="путь к lipo.csv (по умолчанию — корень репозитория)")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="пункты 2–6: МНК-оценки, s², s, R², интервалы")
    fit.add_argument("-c", "--columns", nargs="+", metavar="X", help="регрессоры (по умолчанию все 24)")
    fit.add_argument("-l", "--level", type=float, default=0.95, help="доверительная вероятность (0.95)")
    fit.add_argument("-m", "--method", choices=METHODS, default="normal", help="способ решения МНК")
    fit.add_argument("--json", action="store_true", help="вывести результат в JSON")

    res = commands.add_parser("residuals", help="пункт 3: остатки по веществам")
    res.add_argument("-c", "--columns", nargs="+", metavar="X", help="регрессоры (по умолчанию все 24)")
    res.add_argument("--sort", action="store_true", help="упорядочить по |e|")

    diag = commands.add_parser("diagnostics", help="пункт 7: условия применимости МНК")
    diag.add_argument("-c", "--columns", nargs="+", metavar="X", help="регрессоры (по умолчанию все 24)")

    rep = commands.add_parser("report", help="все рисунки и таблицы отчёта")
    rep.add_argument("-o", "--output", default=str(LAB_ROOT), help="каталог для figures/ и results/")
    rep.add_argument("--repeats", type=int, default=200, help="повторений 5-кратной проверки (200)")
    return parser


def _cmd_fit(args, data) -> int:
    fit = fit_ols_data(data, args.columns, method=args.method)
    if not 0.0 < args.level < 1.0:
        raise ValueError("доверительная вероятность должна лежать в (0, 1)")
    if args.json:
        print(json.dumps({"summary": fit.summary(), "coefficients": fit.table(args.level)},
                         ensure_ascii=False, indent=2))
        return EXIT_OK
    s = fit.summary()
    print(f"МНК: n = {fit.n}, K = {fit.K} (константа и регрессоры: {fit.K - 1}), n − K = {fit.df_resid}")
    print(f"{'коэф.':>9} {'b':>10} {'SE(b)':>9} {'t':>7} {'p':>8}   {int(args.level * 100)}%-й интервал")
    print("-" * 70)
    for row in fit.table(args.level):
        star = "*" if row["p"] < 1 - args.level else " "
        print(f"{row['label']:>9} {row['b']:10.4f} {row['se']:9.4f} {row['t']:7.2f} {row['p']:8.4f}{star}  "
              f"[{row['low']:8.4f}; {row['high']:8.4f}]")
    print("-" * 70)
    print(f"t-квантиль t_α^(n−K) = {fit.t_critical(args.level):.4f};  * — интервал не содержит 0")
    print(f"SSR = {s['ssr']:.4f}   s² = {s['s2']:.4f}   s = {s['s']:.4f}   RMSE = {s['rmse']:.4f}")
    print(f"R² = {s['r2']:.4f}   R²adj = {s['r2_adj']:.4f}   R²uc = {s['r2_uncentered']:.4f}")
    print(f"F = {s['F']:.3f} (p = {s['F_p']:.3g}, критическое {s['F_critical']:.3f})")
    return EXIT_OK


def _cmd_residuals(args, data) -> int:
    fit = fit_ols_data(data, args.columns)
    h = leverage(fit.X)
    order = np.argsort(-np.abs(fit.residuals)) if args.sort else np.arange(data.n)
    print(f"{'вещество':<24} {'logP':>6} {'ŷ':>7} {'e':>7} {'h':>5}")
    for i in order:
        print(f"{data.names[i]:<24} {data.y[i]:6.2f} {fit.fitted[i]:7.3f} {fit.residuals[i]:7.3f} {h[i]:5.2f}")
    print(f"сумма остатков = {fit.residuals.sum():.2e}, max |Xᵀe| = {np.max(np.abs(fit.X.T @ fit.residuals)):.2e}")
    return EXIT_OK


def _cmd_diagnostics(args, data) -> int:
    fit = fit_ols_data(data, args.columns)
    sub = data.subset(args.columns) if args.columns else data
    norm = normality_tests(fit.residuals)
    bp = breusch_pagan(fit)
    reset = reset_test(fit)
    pe = pure_error(fit, sub)
    press = press_statistics(fit.X, fit.y)
    cv = kfold_cv(fit.X, fit.y, 5, 100)
    h = leverage(fit.X)
    exact = [data.names[i] for i in np.where(h > 1 - 1e-8)[0]]
    print(f"Ранг X: {np.linalg.matrix_rank(fit.X)} из {fit.K}; κ(X) = {condition_number(fit.X, False):.0f}, "
          f"κ(X) после нормировки столбцов = {condition_number(fit.X):.0f}")
    top = sorted(vif(sub).items(), key=lambda kv: -kv[1])[:5]
    print("Наибольшие VIF: " + ", ".join(f"{k} {v:.0f}" for k, v in top))
    print(f"Нормальность: Шапиро–Уилк W = {norm['shapiro'].statistic:.3f}, p = {norm['shapiro'].p:.3f}; "
          f"Харке–Бера = {norm['jarque_bera'].statistic:.2f}, p = {norm['jarque_bera'].p:.3f}")
    print(f"Гомоскедастичность: Бройш–Паган LM = {bp.statistic:.2f}, p = {bp.p:.3f}")
    print(f"Линейность: RESET F = {reset.statistic:.2f}, p = {reset.p:.3f}")
    print(f"Дарбин–Уотсон (порядок по алфавиту): {durbin_watson(fit.residuals):.2f}")
    print(f"Подогнаны точно (h = 1): {', '.join(exact) if exact else 'нет'}")
    print(f"Чистая ошибка: SS = {pe.ss_pe:.3f} ({pe.df_pe} ст. св.), σ_pe = {pe.sigma_pe:.3f}; "
          f"тест на неадекватность F = {pe.F:.2f}, p = {pe.p:.3f}")
    print(f"Скользящий контроль: RMSE = {press['rmse']:.3f}, Q² = {press['q2']:.3f}; "
          f"5-кратная проверка: RMSE = {cv.rmse_mean:.3f} ± {cv.rmse_std:.3f}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = load_lipo(args.data)
        if args.command == "fit":
            return _cmd_fit(args, data)
        if args.command == "residuals":
            return _cmd_residuals(args, data)
        if args.command == "diagnostics":
            return _cmd_diagnostics(args, data)
        if args.command == "report":
            from .report import generate_report

            generate_report(Path(args.output), data, repeats=args.repeats)
            return EXIT_OK
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
