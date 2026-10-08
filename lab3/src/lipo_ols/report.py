r"""Пересчёт всех чисел и рисунков отчёта одной командой.

.. code-block:: bash

    python -m lipo_ols report          # ≈ 10 с

Создаёт ``figures/*.png``, ``results/summary.json`` (все числа),
``results/residuals.csv`` (остатки по веществам, пункт 3) и
``results/tables.md`` (таблицы Markdown, из которых собран ``REPORT.md``).
Случайность есть только в разбиениях перекрёстной проверки; генератор
инициализирован фиксированным зерном, повторный запуск даёт те же числа.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contributions import contribution_table, matched_pairs
from .data import DESCRIPTORS, LipoData, duplicate_groups, load_lipo
from .diagnostics import (
    breusch_pagan,
    condition_number,
    cooks_distance,
    durbin_watson,
    leverage,
    normality_tests,
    pure_error,
    reset_test,
    studentized_residuals,
    vif,
)
from .distributions import fisher_quantile, student_quantile
from .literature import PAPER_MODELS, PAPER_PAIRWISE_R2, paper_columns
from .ols import METHODS, OLSResult, fit_ols_data
from .plots import (
    plot_coefficients,
    plot_correlations,
    plot_data_overview,
    plot_elimination,
    plot_fit,
    plot_matched_pairs,
    plot_residuals,
    save,
)
from .validation import backward_elimination, kfold_cv, press_statistics

__all__ = ["CV_REPEATS", "CV_SEED", "generate_report"]

CV_SEED = 2026
CV_REPEATS = 200
LEVEL = 0.95


def _f(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x != 0 and abs(x) < 10 ** (-digits):
        return f"{x:.1e}".replace("e-0", "e-")
    return f"{x:.{digits}f}"


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _md(headers: Sequence[str], rows: Sequence[Sequence[str]], align: Optional[str] = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    marks = {"l": "---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(marks[a] for a in align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _validation(fit: OLSResult, seed: int, repeats: int) -> Dict[str, float]:
    press = press_statistics(fit.X, fit.y)
    cv = kfold_cv(fit.X, fit.y, k=5, repeats=repeats, seed=seed)
    return {"loo_rmse": press["rmse"], "q2": press["q2"], "press": press["press"],
            "cv_rmse": cv.rmse_mean, "cv_rmse_std": cv.rmse_std, "cv_q2": cv.q2_mean}


def generate_report(root: Path, data: Optional[LipoData] = None, repeats: int = CV_REPEATS,
                    seed: int = CV_SEED, verbose: bool = True) -> Dict[str, object]:
    """Посчитать всё для отчёта и записать в ``root/figures`` и ``root/results``."""
    t0 = time.perf_counter()
    root = Path(root)
    figures = root / "figures"
    results = root / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.perf_counter() - t0:5.1f} с] {msg}")

    data = data or load_lipo()
    fit = fit_ols_data(data)
    summary = fit.summary()
    tables: List[str] = []
    out: Dict[str, object] = {"model": summary}

    # ------------------------------------------------------- пункт 1: данные
    counts = np.count_nonzero(data.X, axis=0)
    rows = []
    for j, name in enumerate(data.columns):
        col = data.X[:, j]
        r = float(np.corrcoef(col, data.y)[0, 1])
        rows.append([f"{j + 1}", f"`{name}`", DESCRIPTORS[name], f"{counts[j]}",
                     _f(col.min(), 2 if col.max() > 10 else 0), _f(col.max(), 2 if col.max() > 10 else 0),
                     _f(col.mean(), 2), f"{r:+.2f}"])
    tables.append("## Таблица 1. Регрессоры\n\n" + _md(
        ["#", "имя", "смысл", "≠ 0 у", "мин", "макс", "среднее", "r(x, logP)"], rows, "llllrrrr"))
    groups = duplicate_groups(data)
    out["data"] = {
        "n": data.n, "k": data.k,
        "y": {"mean": float(data.y.mean()), "std": float(data.y.std(ddof=1)),
              "min": float(data.y.min()), "max": float(data.y.max()),
              "argmin": data.names[int(np.argmin(data.y))], "argmax": data.names[int(np.argmax(data.y))]},
        "nonzero_counts": dict(zip(data.columns, map(int, counts))),
        "duplicate_groups": [[(data.names[i], float(data.y[i])) for i in g] for g in groups],
    }
    log("данные")

    # --------------------------------------- пункты 2–6: оценки и интервалы
    methods = {m: float(np.max(np.abs(fit_ols_data(data, method=m).b - fit.b))) for m in METHODS}
    contrib = contribution_table(fit, data, LEVEL)
    beta = {r["label"]: r for r in contrib}
    ci = fit.conf_int(LEVEL)
    rows = []
    for k, label in enumerate(fit.labels):
        extra = beta.get(label)
        rows.append([f"`{label}`", _f(fit.b[k], 4), _f(fit.se[k], 4), _f(fit.t[k], 2), _p(fit.p_values[k]),
                     f"[{_f(ci[k, 0], 3)}; {_f(ci[k, 1], 3)}]",
                     "" if extra is None else _f(extra["beta"], 2),
                     "" if extra is None else _f(extra["delta_r2"], 4),
                     "**да**" if fit.p_values[k] < 1 - LEVEL else "нет"])
    tables.append("## Таблица 2. Коэффициенты модели (2) и 95%-е доверительные интервалы\n\n" + _md(
        ["коэф.", "b", "SE(b)", "t", "p", "95%-й интервал", "β*", "ΔR²", "≠ 0?"], rows, "lrrrrrrrc"))
    t_lecture = student_quantile(LEVEL, fit.df_resid)
    f_lecture = fisher_quantile(LEVEL, fit.K - 1, fit.df_resid)
    out["coefficients"] = fit.table(LEVEL)
    out["contributions"] = contrib
    out["numerics"] = {
        "solver_max_diff": methods,
        "cond_X": condition_number(fit.X, scale=False),
        "cond_X_scaled": condition_number(fit.X, scale=True),
        "cond_XtX": float(np.linalg.cond(fit.X.T @ fit.X)),
        "orthogonality_max_abs_Xte": float(np.max(np.abs(fit.X.T @ fit.residuals))),
        "t_critical_scipy": fit.t_critical(LEVEL), "t_critical_lecture": t_lecture,
        "F_critical_scipy": summary["F_critical"], "F_critical_lecture": f_lecture,
    }

    # пункт 3: остатки по веществам
    h = leverage(fit.X)
    tstud = studentized_residuals(fit)
    cook = cooks_distance(fit)
    with open(results / "residuals.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["substance", "logP", "fitted", "residual", "leverage", "studentized", "cook"])
        for i in range(data.n):
            writer.writerow([data.names[i], f"{data.y[i]:.3f}", f"{fit.fitted[i]:.4f}", f"{fit.residuals[i]:.4f}",
                             f"{h[i]:.4f}", "" if np.isnan(tstud[i]) else f"{tstud[i]:.3f}",
                             "" if np.isnan(cook[i]) else f"{cook[i]:.4f}"])
    rows = [[data.names[i], _f(data.y[i], 2), _f(fit.fitted[i], 2),
             "0 (h = 1)" if h[i] > 1 - 1e-8 else _f(fit.residuals[i], 3), _f(h[i], 2)] for i in range(data.n)]
    tables.append("## Таблица П1. МНК-остатки\n\n" + _md(["вещество", "logP", "ŷ", "e", "h"], rows))
    log("оценки, интервалы, остатки")

    # ---------------------------------------------- пункт 7: адекватность
    val_full = _validation(fit, seed, repeats)
    out["validation"] = val_full
    norm = normality_tests(fit.residuals)
    bp = breusch_pagan(fit)
    reset = reset_test(fit)
    pe = pure_error(fit, data)
    vifs = vif(data)
    out["diagnostics"] = {
        "shapiro": vars(norm["shapiro"]), "jarque_bera": vars(norm["jarque_bera"]),
        "skewness": norm["skewness"], "excess_kurtosis": norm["excess_kurtosis"],
        "breusch_pagan": vars(bp), "reset": vars(reset), "durbin_watson": durbin_watson(fit.residuals),
        "vif": vifs, "exact_fit": [data.names[i] for i in np.where(h > 1 - 1e-8)[0]],
        "high_leverage": [(data.names[i], float(h[i])) for i in np.argsort(-h) if 2 * fit.K / fit.n < h[i] < 1 - 1e-8],
        "max_cook": [(data.names[i], float(cook[i])) for i in np.argsort(-np.nan_to_num(cook))[:5]],
        "max_abs_studentized": [(data.names[i], float(tstud[i]))
                                for i in np.argsort(-np.nan_to_num(np.abs(tstud)))[:5]],
        "pure_error": {"ss_pe": pe.ss_pe, "df_pe": pe.df_pe, "ss_lof": pe.ss_lof, "df_lof": pe.df_lof,
                       "F": pe.F, "p": pe.p, "sigma_pe": pe.sigma_pe},
    }
    rows = [[f"`{n}`", _f(v, 1)] for n, v in sorted(vifs.items(), key=lambda kv: -kv[1])]
    half = (len(rows) + 1) // 2
    rows2 = [a + b for a, b in zip(rows[:half], rows[half:] + [["", ""]] * (2 * half - len(rows)))]
    tables.append("## Таблица 3. Коэффициенты увеличения дисперсии VIF\n\n" + _md(
        ["регрессор", "VIF", "регрессор", "VIF"], rows2, "lrlr"))
    rows = []
    for g in groups:
        ys = data.y[g]
        rows.append([", ".join(data.names[i] for i in g), ", ".join(_f(v, 2) for v in ys), _f(ys.mean(), 3),
                     _f(float(np.sum((ys - ys.mean()) ** 2)), 4)])
    tables.append("## Таблица 4. Вещества с одинаковыми регрессорами\n\n" + _md(
        ["вещества", "logP", "среднее", "вклад в SS_pe"], rows, "lrrr"))
    log("диагностика")

    # --------------------------------------- пункт 7: сравнение со статьёй
    rows = []
    out["paper"] = {}
    for name, ref in PAPER_MODELS.items():
        m = fit_ols_data(data, paper_columns(name))
        s = m.summary()
        val = _validation(m, seed, repeats)
        out["paper"][name] = {"ours": s, "paper": ref, "validation": val}
        rows.append([name, f"{int(ref['k'])}",
                     f"{_f(s['r2'], 3)} / {ref['r2']:.2f}", f"{_f(s['r2_adj'], 3)} / {ref['r2_adj']:.2f}",
                     f"{_f(s['rmse'], 3)} / {ref['rmse']:.2f}", f"{_f(s['s'], 3)} / {ref['s']:.2f}",
                     f"{_f(s['F'], 2)} / {ref['F']:.1f}", f"{s['F_p']:.2e} / {ref['F_p']:.2e}",
                     _f(val["loo_rmse"], 3), _f(val["cv_rmse"], 3)])
    tables.append("## Таблица 5. Сравнение со статьёй [2] (наш расчёт / статья)\n\n" + _md(
        ["модель", "k", "R²", "R²adj", "RMSE", "s", "F", "p(F)", "RMSE LOO", "RMSE 5-fold"], rows))
    out["paper_pairwise_r2"] = {c: {"ours": float(np.corrcoef(data.column(c), data.y)[0, 1] ** 2), "paper": v}
                                for c, v in PAPER_PAIRWISE_R2.items()}
    nested = {"MLR-1 → MLR-3": fit.zero_test(["HBA1", "HBA2", "HBD", "PSA", "MR"]),
              "MLR-2 → MLR-3": fit.zero_test(["PSA", "MR"])}
    out["nested_tests"] = {k: vars(v) for k, v in nested.items()}
    log("сравнение со статьёй")

    # -------------------------------------------- исключение регрессоров
    steps = backward_elimination(data, 1 - LEVEL)
    loo_path, cv_path = [], []
    rows = []
    for s in steps:
        m = fit_ols_data(data, s.columns)
        val = _validation(m, seed, repeats)
        loo_path.append(val["loo_rmse"])
        cv_path.append(val["cv_rmse"])
        rows.append([str(len(s.columns)), "—" if s.removed is None else f"`{s.removed}`",
                     "" if s.removed is None else _f(s.p_removed, 3), _f(s.r2, 3), _f(s.r2_adj, 3),
                     _f(s.s, 3), _f(val["loo_rmse"], 3), _f(val["cv_rmse"], 3), _f(s.bic, 1)])
    tables.append("## Таблица 6. Пошаговое исключение регрессоров\n\n" + _md(
        ["k", "исключён", "его p", "R²", "R²adj", "s", "RMSE LOO", "RMSE 5-fold", "BIC"], rows))
    reduced = fit_ols_data(data, steps[-1].columns)
    dropped = [s.removed for s in steps[1:]]
    joint = fit.zero_test(dropped, LEVEL)
    rows = [[f"`{r['label']}`", _f(r["b"], 4), _f(r["se"], 4), _f(r["t"], 2), _p(r["p"]),
             f"[{_f(r['low'], 3)}; {_f(r['high'], 3)}]"] for r in reduced.table(LEVEL)]
    tables.append("## Таблица 7. Модель после исключения\n\n" + _md(
        ["коэф.", "b", "SE(b)", "t", "p", "95%-й интервал"], rows))
    out["elimination"] = {
        "steps": [{"k": len(s.columns), "removed": s.removed, "p_removed": s.p_removed, "r2": s.r2,
                   "r2_adj": s.r2_adj, "s": s.s, "bic": s.bic, "loo_rmse": lo, "cv_rmse": cv}
                  for s, lo, cv in zip(steps, loo_path, cv_path)],
        "reduced": reduced.summary(), "reduced_coefficients": reduced.table(LEVEL),
        "reduced_validation": _validation(reduced, seed, repeats),
        "joint_test_dropped": vars(joint), "dropped": dropped,
    }
    log("исключение регрессоров")

    # ---------------------------------------------------------- пары молекул
    pairs = matched_pairs(fit, data)
    pairs_red = matched_pairs(reduced, data.subset(reduced.labels[1:]))
    rows = []
    for p, q in zip(pairs, pairs_red):
        terms = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(p.terms.items(), key=lambda kv: -abs(kv[1])))
        rows.append([p.change, p.child, _f(p.delta_exp, 2), _f(p.delta_model, 2) + (" *" if p.exact_fit else ""),
                     _f(q.delta_model, 2), terms])
    tables.append("## Таблица 8. Пары молекул (* — подогнано точно, h = 1)\n\n" + _md(
        ["замена", "производное", "Δ эксп.", "Δ модель (2)", "Δ модель 7", "слагаемые Δ модели (2)"], rows,
        "llrrrl"))
    out["matched_pairs"] = [{"change": p.change, "parent": p.parent, "child": p.child, "delta_exp": p.delta_exp,
                             "delta_model": p.delta_model, "delta_reduced": q.delta_model,
                             "exact_fit": p.exact_fit, "terms": p.terms} for p, q in zip(pairs, pairs_red)]
    log("пары молекул")

    # --------------------------------------- свой расчёт против statsmodels
    try:
        from .package_compare import compare_with_statsmodels

        cmp = compare_with_statsmodels(data, LEVEL)
    except ImportError:
        cmp = None
    if cmp is not None:
        rows = [[r["item"], f"{r['diff']:.1e}" + (f" ({r['note']})" if r.get("note") else "")] for r in cmp["rows"]]
        tm = cmp["timing"]
        rows.append(["время: оценка + SE, p, интервалы, F", f"{tm['ours_ms']:.2f} мс (свой) / "
                     f"{tm['statsmodels_ms']:.2f} мс (statsmodels)"])
        tables.append(f"## Таблица 9. Свой расчёт и statsmodels {cmp['version']}: наибольшее расхождение\n\n"
                      + _md(["величина", "max \\|свой − statsmodels\\|"], rows, "lr"))
        out["statsmodels"] = cmp
    log("сравнение с statsmodels")

    # ---------------------------------------------------------------- рисунки
    press = press_statistics(fit.X, fit.y)
    save(plot_data_overview(data), figures / "data_overview.png")
    save(plot_correlations(data), figures / "correlations.png")
    save(plot_fit(fit, data, loo=press["predictions"]), figures / "fit.png")
    save(plot_residuals(fit, data), figures / "residuals.png")
    save(plot_coefficients(contrib, LEVEL), figures / "coefficients.png")
    save(plot_matched_pairs(pairs, pairs_red), figures / "matched_pairs.png")
    save(plot_elimination(steps, cv_path, loo_path), figures / "elimination.png")
    log("рисунки")

    out["config"] = {"cv_repeats": repeats, "cv_seed": seed, "level": LEVEL}
    with open(results / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2, default=float)
    header = ("# Таблицы отчёта\n\nСгенерировано командой `python -m lipo_ols report`; "
              "не редактировать вручную.\n\n")
    (results / "tables.md").write_text(header + "\n\n".join(tables) + "\n", encoding="utf-8")
    log(f"готово: {figures}, {results}")
    return out
