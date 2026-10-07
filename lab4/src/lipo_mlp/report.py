r"""Пересчёт всех чисел и рисунков отчёта одной командой.

.. code-block:: bash

    python -m lipo_mlp report            # ≈ 6 мин на 8 ядрах
    python -m lipo_mlp report --quick    # уменьшенный вариант для проверки (≈ 20 с)

Создаёт ``figures/*.png``, ``results/summary.json`` и ``results/tables.md``.
Все запуски детерминированы (зёрна фиксированы), параллельное выполнение
результат не меняет.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .data import Standardizer, load_lipo
from .experiments import (
    DELTA,
    EPS,
    MAX_ITER,
    REDUCED_OLS_COLUMNS,
    SIZES,
    Job,
    RunSummary,
    cv_splits,
    interpolation_floor,
    linear_stability_bound,
    ols_cv,
    run_jobs,
)
from .importance import ensemble_predict, ols_standardized, permutation_importance, sensitivity
from .network import init_params, n_params
from .schedules import Schedule, constant, exponential, inverse_time, step
from .sklearn_compare import SKLEARN_CONFIGS, sklearn_cv, sklearn_fit, sklearn_trajectory
from .training import gradient_descent

__all__ = ["ReportConfig", "compute", "generate_report", "write_outputs"]

LR_GRID = (0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5)
ALPHAS = (0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0)
LR_HIDDEN = 16


def smooth_schedules() -> List[Schedule]:
    """Законы уменьшения ε для H = 16 (гладкий режим), кроме постоянного."""
    return [inverse_time(EPS, 1e4), inverse_time(EPS, 1e3), exponential(EPS, 0.99999), exponential(EPS, 0.9999),
            step(EPS, 0.5, 20_000)]


def big_schedules() -> List[Schedule]:
    """Слишком большой начальный шаг ε₀ = 0.5 с уменьшением."""
    return [inverse_time(0.5, 1e3), exponential(0.5, 0.9999)]


def kink_schedules() -> List[Schedule]:
    """Законы уменьшения ε для H = 4 (режим «дребезга» на изломах ReLU)."""
    return [inverse_time(EPS, 1e3), exponential(EPS, 0.99999), step(EPS, 0.5, 20_000)]


def sgd_schedules() -> List[Schedule]:
    return [constant(0.05), inverse_time(0.05, 2e4)]


@dataclass(frozen=True)
class ReportConfig:
    seeds: int = 5
    max_iter: int = MAX_ITER
    cv_folds: int = 5
    cv_repeats: int = 2
    cv_max_iter: int = 100_000
    sgd_batch: int = 16
    workers: Optional[int] = None

    @classmethod
    def quick(cls) -> "ReportConfig":
        return cls(seeds=2, max_iter=20_000, cv_repeats=1, cv_max_iter=5_000)


def _jobs(cfg: ReportConfig, n: int) -> Dict[str, List[Job]]:
    seeds = range(cfg.seeds)
    mi = cfg.max_iter
    out: Dict[str, List[Job]] = {
        "sweep": [Job("sweep", h, s, max_iter=mi) for h in SIZES for s in seeds],
        "lr": [Job("lr", LR_HIDDEN, s, constant(e), max_iter=mi, keep_params=False)
               for e in LR_GRID if e != EPS for s in seeds],
        "decay": [Job("decay", LR_HIDDEN, s, sch, max_iter=mi, keep_params=False)
                  for sch in smooth_schedules() for s in seeds],
        "big": [Job("big", LR_HIDDEN, s, sch, max_iter=mi, keep_params=False)
                for sch in big_schedules() for s in seeds],
        "kink": [Job("kink", 4, s, sch, max_iter=mi, keep_params=False) for sch in kink_schedules() for s in seeds],
        "sgd": [Job("sgd", LR_HIDDEN, s, sch, max_iter=mi, batch_size=cfg.sgd_batch, record_every=1000,
                    keep_params=False) for sch in sgd_schedules() for s in seeds],
    }
    splits = cv_splits(n, cfg.cv_folds, cfg.cv_repeats)
    out["cv"] = [Job("cv", h, k, max_iter=cfg.cv_max_iter, record_every=10, train=tr, val=va, keep_params=False)
                 for h in SIZES for k, (tr, va) in enumerate(splits)]
    return out


def compute(cfg: Optional[ReportConfig] = None, log: Callable[[str], None] = print) -> Dict[str, object]:
    """Выполнить все эксперименты; возвращает словарь с результатами (объекты :class:`RunSummary` и массивы)."""
    cfg = cfg or ReportConfig()
    data = load_lipo()
    jobs = _jobs(cfg, data.n)
    order = [name for name in jobs]
    flat = [job for name in order for job in jobs[name]]
    log(f"запусков обучения: {len(flat)}")
    results = run_jobs(flat, cfg.workers)
    runs: Dict[str, List[RunSummary]] = {}
    k = 0
    for name in order:
        runs[name] = results[k:k + len(jobs[name])]
        k += len(jobs[name])
    log("обучение перцептронов")

    splits = cv_splits(data.n, cfg.cv_folds, cfg.cv_repeats)
    fpr = cfg.cv_folds
    out: Dict[str, object] = {"config": cfg, "runs": runs, "splits": splits}
    out["floor"] = interpolation_floor(data)
    out["stability"] = linear_stability_bound(data)
    out["ols_cv"] = {"МНК, 24 регрессора": ols_cv(splits, None, data),
                     "МНК, 7 регрессоров (ЛР3)": ols_cv(splits, REDUCED_OLS_COLUMNS, data)}
    X1 = np.column_stack([np.ones(data.n), data.X])
    coef, *_ = np.linalg.lstsq(X1, data.y, rcond=None)
    out["ols_train_rmse"] = float(np.sqrt(np.mean((data.y - X1 @ coef) ** 2)))
    Xr = np.column_stack([np.ones(data.n), data.X[:, [data.columns.index(c) for c in REDUCED_OLS_COLUMNS]]])
    coef_r, *_ = np.linalg.lstsq(Xr, data.y, rcond=None)
    out["ols7_train_rmse"] = float(np.sqrt(np.mean((data.y - Xr @ coef_r) ** 2)))

    # sklearn: совпадение траекторий и практические настройки
    scaler = Standardizer.fit(data.X, data.y)
    Z, t = scaler.transform(data.X, data.y)
    traj = {}
    for h in SIZES:
        p0 = init_params(data.k, h, 123)
        ours = gradient_descent(p0, Z, t, lr=EPS, delta=1e-300, max_iter=2000, record_every=10**9).params
        theirs = sklearn_trajectory(p0, Z, t, EPS, 2000)
        traj[h] = float(np.max(np.abs(ours.flat() - theirs.flat())))
    out["sklearn_trajectory"] = traj
    sk_fit: Dict[str, Dict[int, dict]] = {}
    sk_cv: Dict[str, Dict[int, np.ndarray]] = {}
    for name, conf in SKLEARN_CONFIGS.items():
        sk_fit[name], sk_cv[name] = {}, {}
        for h in SIZES:
            sk_fit[name][h], _ = sklearn_fit(data.X, data.y, h, conf)
            sk_cv[name][h] = sklearn_cv(data.X, data.y, h, conf, splits, fpr)
    out["sklearn_fit"], out["sklearn_cv"] = sk_fit, sk_cv
    reg: Dict[int, Dict[float, dict]] = {}
    for h in (8, 32):
        reg[h] = {}
        for alpha in ALPHAS:
            conf = {"solver": "lbfgs", "max_iter": 20_000, "alpha": alpha}
            info, _ = sklearn_fit(data.X, data.y, h, conf)
            pred = sklearn_cv(data.X, data.y, h, conf, splits, fpr)
            reg[h][alpha] = {"train_rmse": info["rmse"], "cv_rmse": _cv_rmse(pred, data.y).tolist()}
    out["regularization"] = reg
    log("scikit-learn")

    # важность регрессоров: ансамбли сошедшихся сетей на всей выборке
    imp = {"ols_beta": ols_standardized(Z, t)}
    for h in (8, 32):
        plist = [r.params for r in runs["sweep"] if r.job.hidden == h and r.params is not None]
        sens = sensitivity(plist, Z)
        imp[h] = {"mean": sens["mean"], "mean_abs": sens["mean_abs"],
                  "perm": permutation_importance(ensemble_predict(plist), Z, t, repeats=30, seed=0)}
    beta_ols = imp["ols_beta"]
    imp["ols_perm"] = permutation_importance(lambda M: M @ beta_ols + float(t.mean()), Z, t, repeats=30, seed=0)
    out["importance"] = imp
    out["columns"] = data.columns
    out["n"] = data.n
    out["y"] = data.y
    out["names"] = data.names
    log("важность регрессоров")
    return out


# ============================================================== вывод
def _f(x: float, d: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def _md(headers: Sequence[str], rows: Sequence[Sequence[str]], align: Optional[str] = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    marks = {"l": "---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(marks[a] for a in align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _iters(rs: Sequence[RunSummary]) -> str:
    conv = sorted(r.iterations for r in rs if r.converged)
    if not conv:
        return "—"
    if len(conv) == 1:
        return f"{conv[0]:,}".replace(",", " ")
    med = int(np.median(conv))
    return f"{med:,} ({conv[0]:,}–{conv[-1]:,})".replace(",", " ")


def _status(rs: Sequence[RunSummary]) -> str:
    c = sum(r.converged for r in rs)
    dv = sum(r.status == "diverged" for r in rs)
    s = f"{c} из {len(rs)}"
    return s + (f", разошлись {dv}" if dv else "")


def _cv_rmse(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RMSE для каждого повторения разбиения."""
    return np.sqrt(np.mean((pred - y[None, :]) ** 2, axis=1))


def _mlp_cv_pred(runs: Sequence[RunSummary], h: int, n: int, folds: int) -> np.ndarray:
    rs = [r for r in runs if r.job.hidden == h]
    reps = len(rs) // folds
    pred = np.full((reps, n), np.nan)
    for k, r in enumerate(rs):
        pred[k // folds, list(r.job.val)] = r.val_pred
    return pred


def _cv_curves(runs: Sequence[RunSummary], h: int, max_iter: int) -> Dict[str, np.ndarray]:
    """RMSE на контроле и на обучении (по всем блокам и повторениям) как функции номера итерации."""
    grid = np.unique(np.r_[0, np.geomspace(1, max_iter, 160).astype(int)])
    rs = [r for r in runs if r.job.hidden == h]
    sse_val = np.zeros(grid.size)
    sse_tr = np.zeros(grid.size)
    n_tr = 0
    for r in rs:
        it = r.history["iteration"]
        idx = np.searchsorted(it, grid, side="right") - 1
        idx = np.clip(idx, 0, it.size - 1)
        nv = len(r.job.val)
        nt = len(r.job.train)
        sse_val += 2 * nv * r.history["val_loss"][idx] * r.y_std**2
        sse_tr += 2 * nt * r.history["loss"][idx] * r.y_std**2
        n_tr += nt
    total_val = sum(len(r.job.val) for r in rs)
    return {"iteration": grid, "val_rmse": np.sqrt(sse_val / total_val), "train_rmse": np.sqrt(sse_tr / n_tr)}


def write_outputs(R: Dict[str, object], root: Path, log: Callable[[str], None] = print) -> Dict[str, object]:
    """Записать таблицы, рисунки и summary.json по результатам :func:`compute`."""
    from . import plots

    root = Path(root)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    cfg: ReportConfig = R["config"]  # type: ignore[assignment]
    runs: Dict[str, List[RunSummary]] = R["runs"]  # type: ignore[assignment]
    y = np.asarray(R["y"])
    n = int(R["n"])
    floor = float(R["floor"])
    tables: List[str] = []
    summary: Dict[str, object] = {"config": vars(cfg), "floor_rmse": floor, "stability": R["stability"],
                                  "ols_train_rmse": R["ols_train_rmse"], "ols7_train_rmse": R["ols7_train_rmse"]}

    # ---- таблица 1: размер скрытого слоя
    rows = []
    sweep_summary = {}
    for h in SIZES:
        rs = [r for r in runs["sweep"] if r.job.hidden == h]
        rm = np.array([r.rmse for r in rs])
        g = np.array([r.grad_norm for r in rs])
        flips = np.array([r.flips for r in rs])
        r2 = 1 - rm**2 / np.var(y)
        rows.append([str(h), str(n_params(24, h)), _status(rs), _iters(rs),
                     f"{rm.mean():.3f} ({rm.min():.3f}–{rm.max():.3f})",
                     f"{r2.mean():.3f}", f"{np.median(g):.1e}"])
        sweep_summary[h] = {"converged": int(sum(r.converged for r in rs)), "runs": len(rs),
                            "iterations": [r.iterations for r in rs], "status": [r.status for r in rs],
                            "rmse": rm.tolist(), "grad_norm": g.tolist(), "flips": flips.tolist(),
                            "near_kink": [r.near_kink for r in rs], "dead_units": [r.dead_units for r in rs]}
    summary["sweep"] = sweep_summary
    tables.append("## Таблица 1. Число нейронов скрытого слоя (ε = 0.1, δ = 1e-4)\n\n" + _md(
        ["H", "параметров", "сошлось", "итераций: медиана (мин–макс)", "RMSE обучения", "R²", "‖DJ‖ в конце"], rows))

    # ---- таблица 2: скорость обучения
    rows = []
    lr_summary = {}
    for e in LR_GRID:
        rs = ([r for r in runs["sweep"] if r.job.hidden == LR_HIDDEN] if e == EPS
              else [r for r in runs["lr"] if r.job.schedule.eps0 == e])
        rm = np.array([r.rmse for r in rs if r.status != "diverged"])
        rows.append([f"{e:g}", _status(rs), _iters(rs), "—" if rm.size == 0 else f"{rm.mean():.3f}"])
        lr_summary[str(e)] = {"status": [r.status for r in rs], "iterations": [r.iterations for r in rs]}
    summary["lr"] = lr_summary
    tables.append(f"## Таблица 2. Постоянная скорость обучения ε (H = {LR_HIDDEN})\n\n" + _md(
        ["ε", "сошлось", "итераций: медиана (мин–макс)", "RMSE обучения"], rows))

    # ---- таблица 3: уменьшение скорости обучения
    def sched_rows(groups):
        rows_ = []
        for title, sch, rs in groups:
            rm = np.array([r.rmse for r in rs if r.status != "diverged"])
            gn = np.array([r.grad_norm for r in rs if r.status != "diverged"])
            total = "∞" if not np.isfinite(sch.total()) else f"{sch.total():.0f}"
            rows_.append([title, sch.label(), total, _status(rs),
                          _iters(rs), "—" if rm.size == 0 else f"{rm.mean():.4f}",
                          "—" if gn.size == 0 else f"{np.median(gn):.1e}"])
        return rows_

    base16 = [r for r in runs["sweep"] if r.job.hidden == LR_HIDDEN]
    base4 = [r for r in runs["sweep"] if r.job.hidden == 4]
    big_const = [r for r in runs["lr"] if r.job.schedule.eps0 == 0.5]
    groups = [("H = 16", constant(EPS), base16)]
    groups += [("H = 16", s, [r for r in runs["decay"] if r.job.schedule == s]) for s in smooth_schedules()]
    groups += [("H = 16, большой ε₀", constant(0.5), big_const)]
    groups += [("H = 16, большой ε₀", s, [r for r in runs["big"] if r.job.schedule == s]) for s in big_schedules()]
    groups += [("H = 4", constant(EPS), base4)]
    groups += [("H = 4", s, [r for r in runs["kink"] if r.job.schedule == s]) for s in kink_schedules()]
    groups += [(f"H = 16, мини-пакеты по {cfg.sgd_batch}", s, [r for r in runs["sgd"] if r.job.schedule == s])
               for s in sgd_schedules()]
    tables.append("## Таблица 3. Уменьшение скорости обучения на каждой итерации\n\n" + _md(
        ["сеть", "закон ε_j", "Σ ε_j", "сошлось", "итераций: медиана (мин–макс)", "RMSE обучения", "‖DJ‖ в конце"],
        sched_rows(groups), "llrrrrr"))
    summary["decay"] = [{"group": g, "schedule": s.label(), "status": [r.status for r in rs],
                         "iterations": [r.iterations for r in rs], "grad_norm": [r.grad_norm for r in rs],
                         "rmse": [r.rmse for r in rs]} for g, s, rs in groups]

    # ---- таблица 4: перекрёстная проверка
    rows = []
    cv_summary = {}
    for name, pred in R["ols_cv"].items():  # type: ignore[union-attr]
        cv = _cv_rmse(pred, y)
        train = R["ols_train_rmse"] if "24" in name else R["ols7_train_rmse"]
        rows.append([name, "25" if "24" in name else "8", _f(train), f"{cv.mean():.3f}", "—"])
        cv_summary[name] = {"train_rmse": train, "cv_rmse": cv.tolist()}
    curves = {}
    for h in SIZES:
        pred = _mlp_cv_pred(runs["cv"], h, n, cfg.cv_folds)
        cv = _cv_rmse(pred, y)
        curve = _cv_curves(runs["cv"], h, cfg.cv_max_iter)
        curves[h] = curve
        best = int(np.argmin(curve["val_rmse"]))
        train = float(np.mean([r.rmse for r in runs["sweep"] if r.job.hidden == h]))
        rows.append([f"перцептрон H = {h}", str(n_params(24, h)), _f(train), f"{cv.mean():.3f}",
                     f"{curve['val_rmse'][best]:.3f} (итерация {int(curve['iteration'][best]):,})".replace(",", " ")])
        cv_summary[f"mlp{h}"] = {"train_rmse": train, "cv_rmse": cv.tolist(),
                                 "best_val_rmse": float(curve["val_rmse"][best]),
                                 "best_iteration": int(curve["iteration"][best])}
    for name in SKLEARN_CONFIGS:
        for h in (8, 32):
            cv = _cv_rmse(R["sklearn_cv"][name][h], y)  # type: ignore[index]
            info = R["sklearn_fit"][name][h]  # type: ignore[index]
            rows.append([f"sklearn {name}, H = {h}", str(n_params(24, h)), _f(info["rmse"]), f"{cv.mean():.3f}", "—"])
            cv_summary[f"sklearn {name} {h}"] = {"train_rmse": info["rmse"], "cv_rmse": cv.tolist()}
    summary["cv"] = cv_summary
    tables.append(f"## Таблица 4. Ошибка на обучении и на контроле ({cfg.cv_folds}-кратная проверка, "
                  f"{cfg.cv_repeats} повторения)\n\n" + _md(
                      ["модель", "параметров", "RMSE обучения", "RMSE контроля",
                       "лучшая RMSE контроля по ходу обучения"],
                      rows, "lrrrr"))

    # ---- таблица 4б: L2-регуляризация
    reg = R["regularization"]
    rows = []
    for alpha in ALPHAS:
        row = [f"{alpha:g}"]
        for h in (8, 32):
            item = reg[h][alpha]  # type: ignore[index]
            row += [_f(item["train_rmse"]), f"{np.mean(item['cv_rmse']):.3f}"]
        rows.append(row)
    tables.append("## Таблица 4б. L2-регуляризация (MLPRegressor, lbfgs): RMSE обучения и контроля\n\n" + _md(
        ["α", "H = 8: обучение", "H = 8: контроль", "H = 32: обучение", "H = 32: контроль"], rows))
    summary["regularization"] = {str(h): {str(a): v for a, v in reg[h].items()} for h in (8, 32)}  # type: ignore[index]

    # ---- таблица 5: scikit-learn
    rows = []
    for h in SIZES:
        row = [str(h), f"{R['sklearn_trajectory'][h]:.1e}"]  # type: ignore[index]
        for name in SKLEARN_CONFIGS:
            info = R["sklearn_fit"][name][h]  # type: ignore[index]
            row.append(f"{info['n_iter']} / {info['rmse']:.3f}")
        rows.append(row)
    tables.append("## Таблица 5. Сравнение с MLPRegressor (scikit-learn)\n\n" + _md(
        ["H", "max \\|Δw\\| после 2000 итераций"] + [f"{k}: итераций / RMSE" for k in SKLEARN_CONFIGS], rows))
    summary["sklearn"] = {"trajectory_max_diff": R["sklearn_trajectory"], "fit": R["sklearn_fit"]}

    # ---- таблица 6: важность регрессоров
    imp = R["importance"]
    cols = list(R["columns"])  # type: ignore[arg-type]
    beta = imp["ols_beta"]  # type: ignore[index]
    order = np.argsort(-np.abs(beta))
    rows = []
    for j in order:
        rows.append([f"`{cols[j]}`", f"{beta[j]:+.2f}", f"{imp['ols_perm'][j]:.3f}",  # type: ignore[index]
                     f"{imp[8]['mean'][j]:+.2f}", f"{imp[8]['mean_abs'][j]:.2f}", f"{imp[8]['perm'][j]:.3f}",  # type: ignore[index]
                     f"{imp[32]['mean'][j]:+.2f}", f"{imp[32]['mean_abs'][j]:.2f}", f"{imp[32]['perm'][j]:.3f}"])  # type: ignore[index]
    tables.append("## Таблица 6. Вклад регрессоров: МНК и перцептроны (стандартизованные единицы)\n\n" + _md(
        ["регрессор", "МНК β*", "МНК перест.", "H=8 ⟨∂ŷ/∂x⟩", "H=8 ⟨\\|∂ŷ/∂x\\|⟩", "H=8 перест.",
         "H=32 ⟨∂ŷ/∂x⟩", "H=32 ⟨\\|∂ŷ/∂x\\|⟩", "H=32 перест."], rows))
    from scipy.stats import spearmanr

    rank = {f"H={h} перест. vs МНК |β*|": float(spearmanr(imp[h]["perm"], np.abs(beta)).statistic)  # type: ignore[index]
            for h in (8, 32)}
    rank.update({f"H={h} ⟨|∂ŷ/∂x|⟩ vs МНК |β*|": float(spearmanr(imp[h]["mean_abs"], np.abs(beta)).statistic)  # type: ignore[index]
                 for h in (8, 32)})
    summary["importance"] = {"columns": cols, "ols_beta": beta.tolist(), "ols_perm": imp["ols_perm"].tolist(),  # type: ignore[index]
                             **{str(h): {k: v.tolist() for k, v in imp[h].items()} for h in (8, 32)},  # type: ignore[index]
                             "spearman": rank}
    log("таблицы")

    # ---- рисунки
    figs = root / "figures"
    plots.save(plots.plot_training_curves(runs["sweep"], floor, float(R["ols_train_rmse"]), DELTA),
               figs / "training_curves.png")
    plots.save(plots.plot_iterations(runs["sweep"], runs["lr"], LR_GRID, LR_HIDDEN, EPS, R["stability"]["eps_max"],  # type: ignore[index]
                                     cfg.max_iter), figs / "iterations.png")
    plots.save(plots.plot_kink(runs["sweep"]), figs / "kink.png")
    plots.save(plots.plot_decay(runs, groups, DELTA), figs / "decay.png")
    plots.save(plots.plot_generalization(curves, summary["cv"], floor), figs / "generalization.png")  # type: ignore[arg-type]
    plots.save(plots.plot_importance(cols, beta, imp), figs / "importance.png")
    log("рисунки")

    with open(root / "results" / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    header = "# Таблицы отчёта\n\nСгенерировано командой `python -m lipo_mlp report`; не редактировать вручную.\n\n"
    (root / "results" / "tables.md").write_text(header + "\n\n".join(tables) + "\n", encoding="utf-8")
    return summary


def generate_report(root: Path, cfg: Optional[ReportConfig] = None, verbose: bool = True) -> Dict[str, object]:
    cfg = cfg or ReportConfig()
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.perf_counter() - t0:6.1f} с] {msg}")

    R = compute(cfg, log)
    out = write_outputs(R, root, log)
    log(f"готово: {root}")
    return out
