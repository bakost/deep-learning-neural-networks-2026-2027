r"""Пересчёт всех чисел, таблиц и рисунков отчёта одной командой.

.. code-block:: bash

    python -m sgd_lab report            # полный вариант (≈ 5 мин на 8 ядрах)
    python -m sgd_lab report --quick    # уменьшенный вариант для проверки (≈ 20 с)

Создаёт ``figures/*.png``, ``results/summary.json`` и ``results/tables.md``.
Все запуски детерминированы (зёрна фиксированы).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .experiments import BATCHES, Series, Spec, excess, first_epoch_below, run_specs
from .linesearch import bracket_minimum, brent, df_task, exact_minimum, f_task, golden_section
from .models import MODELS, X, Y, init_params
from .numdiff import ridders_derivative
from .reference import linear_solution, loo_press, minibatch_lr_bound, multistart_minimum, stability
from .training import Objective, TrainConfig, train

__all__ = ["ReportConfig", "classify", "compute", "generate_report"]

DELTA = 1e-6               # точность по норме градиента (задача 2)
MAX_ITER = 100_000         # предел итераций полнопакетного спуска
LR_GRID = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
RELU_LR_GRID = (0.003, 0.01, 0.03)
RELU_LR = 0.01             # скорость спуска для сети ReLU в задачах 2–3
TARGET = 1e-6              # «сошёлся»: (J − J*)/J* ≤ 10⁻⁶
L2_GRID = (0.0, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
L2_DEMO = 0.1
METHOD_BATCHES = (1, 4, 12)
METHOD_TITLES = {"gd": "ГС / СГС", "nesterov": "Нестеров", "adam": "Adam"}

# Общая для всех размеров пакета скорость в задаче 3 — половина оценки
# границы устойчивости худшего пакета из одного наблюдения.
COMMON_LR = {"linear": 0.01, "quadratic": 0.0025}
TAU3 = 200.0
TAU4 = {"linear": 10.0, "quadratic": 70.0, "relu": 300.0}

OUTCOMES = (
    ("global", "глобальный минимум"),
    ("better_quad", "лучше квадратичной"),
    ("between", "между квадр. и линейной"),
    ("linear", "≈ линейная модель"),
    ("worse", "хуже линейной"),
    ("constant", "константа ȳ"),
    ("dead", "мёртвая сеть (ŷ ≡ 0)"),
)


@dataclass(frozen=True)
class ReportConfig:
    ref_starts: int = 100           # запусков Нелдера — Мида для эталона сети ReLU
    relu_seeds: int = 100           # начальных точек сети ReLU
    seeds: int = 20                 # зёрен для выпуклых моделей
    grid_seeds: int = 10            # зёрен при подборе скорости обучения
    relu_epochs: int = 3000
    relu_gd_iter: int = 50_000
    epochs3: int = 2000
    epochs4: Optional[Dict[str, int]] = None
    l5_starts: int = 40
    loo_inits: int = 30
    workers: Optional[int] = None

    @property
    def e4(self) -> Dict[str, int]:
        return self.epochs4 or {"linear": 300, "quadratic": 2000, "relu": self.relu_epochs}

    @classmethod
    def quick(cls) -> "ReportConfig":
        return cls(ref_starts=8, relu_seeds=16, seeds=4, grid_seeds=3, relu_epochs=300, relu_gd_iter=3000,
                   epochs3=200, epochs4={"linear": 100, "quadratic": 300, "relu": 300}, l5_starts=4,
                   loo_inits=4)


# ------------------------------------------------------------------ вспомогательное
def _f(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def baselines() -> Dict[str, float]:
    return {"zero": float(np.sum(Y * Y)), "constant": float(np.sum((Y - Y.mean()) ** 2)),
            "linear": float(np.sum((MODELS["linear"](linear_solution(MODELS["linear"], X, Y), X) - Y) ** 2)),
            "quadratic": float(np.sum((MODELS["quadratic"](linear_solution(MODELS["quadratic"], X, Y), X) - Y) ** 2))}


def classify(sse: np.ndarray, best: float, base: Dict[str, float]) -> np.ndarray:
    """Тип конечной точки сети ReLU по сумме квадратов ошибок."""
    sse = np.asarray(sse, dtype=float)
    out = np.full(sse.shape, "worse", dtype=object)
    out[sse <= 1.01 * best] = "global"
    out[(sse > 1.01 * best) & (sse < base["quadratic"])] = "better_quad"
    out[(sse >= base["quadratic"]) & (sse < 0.99 * base["linear"])] = "between"
    out[np.abs(sse - base["linear"]) <= 0.01 * base["linear"]] = "linear"
    out[np.abs(sse - base["constant"]) <= 0.01 * base["constant"]] = "constant"
    out[(np.abs(sse - base["zero"]) <= 0.01 * base["zero"]) | ~np.isfinite(sse)] = "dead"
    return out


def outcome_shares(sse, best, base) -> Dict[str, float]:
    c = classify(sse, best, base)
    return {k: float(np.mean(c == k)) for k, _ in OUTCOMES}


def _median(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def _median_epochs(a) -> float:
    """Медиана числа эпох; не достигшие порога считаются как ∞ (медиана ∞ — у большинства не достигнут)."""
    a = np.where(np.isnan(np.asarray(a, dtype=float)), np.inf, a)
    return float(np.median(a)) if a.size else float("nan")


# ------------------------------------------------------------------ задача 1
def compute_task1() -> dict:
    from scipy.optimize import minimize_scalar

    z_star = exact_minimum()
    f_star = float(f_task(z_star))
    br, nb = bracket_minimum(f_task, 0.0, 1.0)
    g = golden_section(f_task, br)
    b = brent(f_task, br)
    sg = minimize_scalar(f_task, bracket=br, method="golden")
    sb = minimize_scalar(f_task, bracket=br, method="brent")
    methods = [
        {"method": "золотое сечение (своё)", "x": g.x, "fx": g.fx, "nfev": g.nfev, "iter": g.iterations},
        {"method": "Брент (своё)", "x": b.x, "fx": b.fx, "nfev": b.nfev, "iter": b.iterations},
        {"method": "золотое сечение (SciPy)", "x": float(sg.x), "fx": float(sg.fun), "nfev": int(sg.nfev),
         "iter": int(sg.nit)},
        {"method": "Брент (SciPy)", "x": float(sb.x), "fx": float(sb.fun), "nfev": int(sb.nfev), "iter": int(sb.nit)},
    ]
    for row in methods:
        row["err"] = abs(row["x"] - z_star)
    tols = []
    for tol in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 3e-8, 1e-9, 1e-10, 1e-12):
        gg, bb = golden_section(f_task, br, tol=tol), brent(f_task, br, tol=tol)
        tols.append({"tol": tol, "golden_err": abs(gg.x - z_star), "golden_nfev": gg.nfev,
                     "brent_err": abs(bb.x - z_star), "brent_nfev": bb.nfev})
    starts = []
    for a0, b0 in ((0.0, 1.0), (0.6, 0.7), (-3.0, -2.0), (2.0, 3.0), (5.0, 6.0)):
        trip, n0 = bracket_minimum(f_task, a0, b0)
        bb = brent(f_task, trip)
        gg = golden_section(f_task, trip)
        starts.append({"start": [a0, b0], "bracket": list(trip), "bracket_nfev": n0,
                       "brent_x": bb.x, "brent_nfev": bb.nfev, "golden_x": gg.x, "golden_nfev": gg.nfev})
    zz = np.linspace(0.45, 0.9, 20001)
    d = df_task(zz)
    k = int(np.argmin(d))
    kinds = [h[4] for h in b.history]
    f2 = float((df_task(z_star + 1e-6) - df_task(z_star - 1e-6)) / 2e-6)
    return {
        "z_star": z_star, "f_star": f_star, "f2_star": f2,
        "floor": math.sqrt(2 * np.finfo(float).eps * abs(f_star) / f2),
        "bracket": list(br), "bracket_nfev": nb, "methods": methods, "tols": tols, "starts": starts,
        "shoulder": {"z": float(zz[k]), "min_df": float(d[k]), "f": float(f_task(zz[k]))},
        "brent_parabolic_steps": kinds.count("parabolic"), "brent_golden_steps": kinds.count("golden"),
        "golden_widths": g.widths.tolist(), "brent_widths": b.widths.tolist(),
        "golden_x": [h[2] for h in g.history], "brent_x": [h[2] for h in b.history],
    }


def gradient_check() -> List[dict]:
    rows = []
    for key, model in MODELS.items():
        obj = Objective(model, X, Y)
        errs, rel = [], []
        for s in range(20):
            w = init_params(model, s)
            g_num = obj.grad(w)
            g_an = model.analytic_grad(w, X, Y)
            errs.append(np.max(np.abs(g_num - g_an)))
            rel.append(np.max(np.abs(g_num - g_an)) / max(np.max(np.abs(g_an)), 1e-12))
        w = init_params(model, 0)
        # Риддерс для первой компоненты — точность, недостижимая простой разностью
        e1 = np.zeros(model.n_params)
        e1[0] = 1.0
        d_r, err_r = ridders_derivative(lambda t, obj=obj, w=w, e1=e1: float(obj(w + t * e1)), 0.0, h=0.1)
        rows.append({"model": key, "max_abs_err": float(np.max(errs)), "max_rel_err": float(np.max(rel)),
                     "ridders_err": abs(d_r - model.analytic_grad(w, X, Y)[0]), "ridders_est": err_r})
    return rows


# ------------------------------------------------------------------ задача 2
def compute_task2_convex(base) -> dict:
    out = {}
    for key in ("linear", "quadratic"):
        model = MODELS[key]
        obj = Objective(model, X, Y)
        st = stability(model, X)
        w_ols = linear_solution(model, X, Y)
        w0 = init_params(model, 0)
        g0 = float(np.linalg.norm(obj.grad(w0)))
        res = train(obj, w0, TrainConfig(lr=st["lr_opt"], epochs=MAX_ITER, delta=DELTA))
        k_theory = math.ceil(math.log(DELTA / g0) / math.log(st["rate_opt"]))
        sweep = []
        for frac in (0.05, 0.1, 0.25, 0.5, "opt", 0.9, 0.99, 1.02):
            lr = st["lr_opt"] if frac == "opt" else frac * st["lr_max"]
            rho = max(abs(1 - lr * st["lambda_min"]), abs(1 - lr * st["lambda_max"]))
            r = train(obj, w0, TrainConfig(lr=lr, epochs=MAX_ITER, delta=DELTA, record_every=MAX_ITER))
            sweep.append({"frac": frac if frac != "opt" else lr / st["lr_max"], "is_opt": frac == "opt", "lr": lr,
                          "rho": rho, "status": r.status, "iterations": r.epochs,
                          "theory": math.ceil(math.log(DELTA / g0) / math.log(rho)) if rho < 1 else None})
        out[key] = {
            **st, "lr": st["lr_opt"], "w0": w0.tolist(), "g0": g0, "status": res.status, "iterations": res.epochs,
            "theory": k_theory, "params": res.params.tolist(), "w_ols": w_ols.tolist(),
            "err_ols": float(np.max(np.abs(res.params - w_ols))), "sse": res.sse, "sse_ols": base[key],
            "history": {k: res.history[k].tolist() for k in ("epoch", "loss", "grad_norm")}, "sweep": sweep,
        }
    return out


def compute_relu_reference(cfg: ReportConfig, l2: float = 0.0, starts: Optional[int] = None) -> dict:
    model = MODELS["relu"]
    w, f, vals = multistart_minimum(model, X, Y, l2, starts=starts or cfg.ref_starts)
    return {"params": w.tolist(), "loss": f, "sse": float(np.sum((model(w, X) - Y) ** 2)),
            "distinct": sorted({round(v, 4) for v in vals.tolist()})[:12]}


def _ref_job(args) -> dict:
    cfg, lam, starts = args
    return compute_relu_reference(cfg, lam, starts=starts)


def _parallel_refs(cfg: ReportConfig, lams: Sequence[float], starts: int) -> List[dict]:
    from concurrent.futures import ProcessPoolExecutor

    from .experiments import default_workers

    workers = cfg.workers or default_workers()
    jobs = [(cfg, lam, starts) for lam in lams]
    if workers == 1:
        return [_ref_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_ref_job, jobs))


def compute_task2_relu(cfg: ReportConfig, ref: dict, base: dict) -> dict:
    seeds = tuple(range(cfg.relu_seeds))
    spec = Spec("relu", "gd", RELU_LR, 12, cfg.relu_gd_iter, seeds, delta=DELTA, record_every=100)
    s, = run_specs([spec], cfg.workers)
    cats = classify(s.sse, ref["sse"], base)
    best = int(np.argmin(s.sse))
    examples = {}
    for k, _ in OUTCOMES:
        idx = np.nonzero(cats == k)[0]
        if idx.size:
            j = idx[np.argmin(s.sse[idx])]
            examples[k] = {"seed": int(seeds[j]), "sse": float(s.sse[j]), "params": s.params[j].tolist()}
    conv = s.status == "converged"
    # доля «мёртвых» запусков при наивном смещении выхода 0.1
    model = MODELS["relu"]
    W0 = np.stack([init_params(model, i, output_bias=0.1) for i in seeds])
    z0 = model(W0, X)
    dead_naive = float(np.mean(np.all(z0 <= 0, axis=1)))
    W1 = np.stack([init_params(model, i) for i in seeds])
    dead_ybar = float(np.mean(np.all(model(W1, X) <= 0, axis=1)))
    return {
        "lr": RELU_LR, "seeds": len(seeds), "status": {k: int(np.sum(s.status == k)) for k in np.unique(s.status)},
        "iterations_converged_median": _median(s.epochs[conv]),
        "iterations_converged_range": ([int(np.min(s.epochs[conv])), int(np.max(s.epochs[conv]))]
                                       if conv.any() else None),
        "shares": outcome_shares(s.sse, ref["sse"], base),
        "counts": {k: int(np.sum(cats == k)) for k, _ in OUTCOMES},
        "sse": s.sse.tolist(), "best": {"seed": int(seeds[best]), "sse": float(s.sse[best]),
                                        "params": s.params[best].tolist()},
        "examples": examples, "dead_init_naive": dead_naive, "dead_init_ybar": dead_ybar,
        "dead_converged_at_0": int(np.sum(conv & (s.epochs == 0))),
    }


# ------------------------------------------------------------------ задача 3
def compute_task3(cfg: ReportConfig, t2: dict, ref_relu: dict, base: dict) -> dict:
    seeds = tuple(range(cfg.seeds))
    E = cfg.epochs3
    specs, keys = [], []
    for key in ("linear", "quadratic"):
        for b in BATCHES:
            specs.append(Spec(key, "gd", t2[key]["lr"], b, E, seeds, record_every=E))
            keys.append((key, b, "t2"))
            specs.append(Spec(key, "gd", COMMON_LR[key], b, E, seeds, record_every=1))
            keys.append((key, b, "const"))
            specs.append(Spec(key, "gd", COMMON_LR[key], b, E, seeds, tau=TAU3, record_every=1))
            keys.append((key, b, "decay"))
    rseeds = tuple(range(cfg.relu_seeds))
    for b in BATCHES:
        for tau in (None, TAU4["relu"]):
            specs.append(Spec("relu", "gd", RELU_LR, b, cfg.relu_epochs, rseeds, tau=tau,
                              record_every=cfg.relu_epochs))
            keys.append(("relu", b, "const" if tau is None else "decay"))
    res = dict(zip(keys, run_specs(specs, cfg.workers)))
    out: dict = {"epochs": E, "tau": TAU3, "common_lr": COMMON_LR, "relu_tau": TAU4["relu"], "rows": {}, "curves": {}}
    for key in ("linear", "quadratic"):
        best = base[key]
        rows = []
        for b in BATCHES:
            s_t2, s_c, s_d = res[(key, b, "t2")], res[(key, b, "const")], res[(key, b, "decay")]
            ex_c = excess(s_c.history["loss"], best)
            ex_d = excess(s_d.history["loss"], best)
            last = max(1, E // 10)
            rows.append({
                "batch": b, "updates_per_epoch": math.ceil(12 / b),
                "bound": minibatch_lr_bound(MODELS[key], X, b),
                "t2_diverged": int(np.sum(s_t2.status == "diverged")),
                "t2_excess": _median(excess(s_t2.loss, best)),
                "const_excess": _median(ex_c[-1]),
                "const_floor": _median(np.mean(ex_c[-last:], axis=0)),
                "const_epochs": _median_epochs(first_epoch_below(s_c, best, 1e-3)),
                "decay_excess": _median(ex_d[-1]),
                "decay_epochs": _median_epochs(first_epoch_below(s_d, best, 1e-3)),
            })
            out["curves"][f"{key}/{b}/const"] = np.median(ex_c, axis=1).tolist()
            out["curves"][f"{key}/{b}/decay"] = np.median(ex_d, axis=1).tolist()
        out["rows"][key] = rows
    relu_rows = []
    for b in BATCHES:
        for sched in ("const", "decay"):
            s = res[("relu", b, sched)]
            relu_rows.append({"batch": b, "schedule": sched, "median_sse": _median(s.sse),
                              **outcome_shares(s.sse, ref_relu["sse"], base)})
    out["relu"] = relu_rows
    return out


# ------------------------------------------------------------------ задача 4
def _select(series: Sequence[Series], score) -> Series:
    return min(series, key=score)


def compute_task4(cfg: ReportConfig, ref_relu: dict, base: dict) -> dict:
    E4 = cfg.e4
    gseeds = tuple(range(cfg.grid_seeds))
    specs = []
    for key in ("linear", "quadratic"):
        for meth in METHOD_TITLES:
            for b in METHOD_BATCHES:
                for tau in ((None,) if b == 12 else (None, TAU4[key])):
                    for lr in LR_GRID:
                        specs.append(Spec(key, meth, lr, b, E4[key], gseeds, tau=tau,
                                          record_every=max(1, E4[key] // 200)))
    rseeds = tuple(range(cfg.relu_seeds))
    for meth in METHOD_TITLES:
        for b in METHOD_BATCHES:
            for tau in ((None,) if b == 12 else (None, TAU4["relu"])):
                for lr in RELU_LR_GRID:
                    specs.append(Spec("relu", meth, lr, b, E4["relu"], rseeds, tau=tau, record_every=E4["relu"]))
    grid = run_specs(specs, cfg.workers)
    chosen: Dict[tuple, Series] = {}
    grid_table = []
    for key in MODELS:
        best = base[key] if key != "relu" else ref_relu["sse"]
        for meth in METHOD_TITLES:
            for b in METHOD_BATCHES:
                cand = [s for s in grid if s.spec.model == key and s.spec.method == meth and s.spec.batch == b]
                if key == "relu":
                    def score(s):
                        return (-float(np.mean(s.sse < base["quadratic"])), _median(s.sse))
                else:
                    # сначала точность (до уровня округления 10⁻¹²), затем — как быстро она устойчиво достигнута
                    def score(s, best=best):
                        v = _median(excess(s.loss, best))
                        v = max(v, 1e-12) if np.isfinite(v) else np.inf
                        return (v, _median_epochs(first_epoch_below(s, best, TARGET)), s.spec.lr)
                sel = _select(cand, score)
                chosen[(key, meth, b)] = sel
                for s in cand:
                    grid_table.append({"model": key, "method": meth, "batch": b, "lr": s.spec.lr, "tau": s.spec.tau,
                                       "median_excess": _median(excess(s.loss, best)),
                                       "share_better_quad": float(np.mean(s.sse < base["quadratic"])),
                                       "diverged": int(np.sum(s.status == "diverged"))})
    # финальные прогоны выбранных настроек с полной историей
    seeds = tuple(range(cfg.seeds))
    final_specs, final_keys = [], []
    for (key, meth, b), sel in chosen.items():
        if key == "relu":
            final_specs.append(Spec(key, meth, sel.spec.lr, b, E4[key], rseeds, tau=sel.spec.tau, record_every=10))
        else:
            final_specs.append(Spec(key, meth, sel.spec.lr, b, E4[key], seeds, tau=sel.spec.tau, record_every=1))
        final_keys.append((key, meth, b))
    final = dict(zip(final_keys, run_specs(final_specs, cfg.workers)))
    rows, curves = [], {}
    for (key, meth, b), s in final.items():
        best = base[key] if key != "relu" else ref_relu["sse"]
        row = {"model": key, "method": meth, "batch": b, "lr": s.spec.lr, "tau": s.spec.tau,
               "updates": int(s.updates.max())}
        if key == "relu":
            row.update({"median_sse": _median(s.sse), **outcome_shares(s.sse, best, base)})
            curves[f"{key}/{meth}/{b}"] = np.median(s.history["sse"], axis=1).tolist()
        else:
            ex = excess(s.history["loss"], best)
            row.update({"median_excess": _median(ex[-1]),
                        "epochs_1e3": _median_epochs(first_epoch_below(s, best, 1e-3)),
                        "epochs_target": _median_epochs(first_epoch_below(s, best, TARGET)),
                        "reached_target": int(np.sum(np.isfinite(first_epoch_below(s, best, TARGET))))})
            curves[f"{key}/{meth}/{b}"] = np.median(ex, axis=1).tolist()
        rows.append(row)
    curve_epochs = {key: final[(key, "gd", 12)].history["epoch"].tolist() for key in MODELS}
    return {"epochs": E4, "rows": rows, "grid": grid_table, "curves": curves, "curve_epochs": curve_epochs,
            "chosen": {f"{k[0]}/{k[1]}/{k[2]}": {"lr": v.spec.lr, "tau": v.spec.tau} for k, v in chosen.items()}}


def compute_paths(t2: dict, t4: dict) -> dict:
    """Траектории методов на плоскости (w₁, b₁) для линейной модели (рисунок задачи 4)."""
    model = MODELS["linear"]
    obj = Objective(model, X, Y)
    w0 = np.array([-1.0, 3.0])
    n_upd = 60
    paths = {}
    ch = t4["chosen"]
    configs = {
        "gd": TrainConfig(method="gd", lr=t2["linear"]["lr"], epochs=n_upd, record_path=True),
        "nesterov": TrainConfig(method="nesterov", lr=ch["linear/nesterov/12"]["lr"], epochs=n_upd, record_path=True),
        "adam": TrainConfig(method="adam", lr=ch["linear/adam/12"]["lr"], epochs=n_upd, record_path=True),
        "sgd1": TrainConfig(method="gd", lr=COMMON_LR["linear"], batch_size=1, epochs=n_upd // 12 + 1,
                            record_path=True, seed=0),
    }
    for k, c in configs.items():
        r = train(obj, w0, c)
        paths[k] = {"lr": c.lr, "path": r.path[: n_upd + 1].tolist()}
    return {"w0": w0.tolist(), "w_star": linear_solution(model, X, Y).tolist(), "paths": paths, "updates": n_upd}


# ------------------------------------------------------------------ задача 5
def compute_task5(cfg: ReportConfig, t4: dict, base: dict) -> dict:
    out: dict = {"grid": list(L2_GRID), "demo": L2_DEMO, "convex": {}, "relu": []}
    for key in ("linear", "quadratic"):
        model = MODELS[key]
        rows = []
        for lam in L2_GRID:
            w = linear_solution(model, X, Y, lam)
            st = stability(model, X, lam)
            obj = Objective(model, X, Y, lam)
            r = train(obj, init_params(model, 0), TrainConfig(lr=st["lr_opt"], epochs=MAX_ITER, delta=DELTA,
                                                              record_every=MAX_ITER))
            rows.append({"l2": lam, "params": w.tolist(), "sse": float(np.sum((model(w, X) - Y) ** 2)),
                         "loss": float(obj(w)), "weight_norm2": float(np.sum(model.mask * w * w)),
                         "kappa": st["kappa"], "lr_opt": st["lr_opt"], "gd_iterations": r.epochs,
                         "gd_err": float(np.max(np.abs(r.params - w))), "press": loo_press(model, X, Y, lam)})
        out["convex"][key] = rows
    # Сеть ReLU: эталон (Нелдер — Мид), Adam из многих точек, скользящий контроль
    model = MODELS["relu"]
    rseeds = tuple(range(cfg.relu_seeds))
    specs = [Spec("relu", "adam", 0.01, 12, cfg.relu_epochs, rseeds, l2=lam, record_every=cfg.relu_epochs)
             for lam in L2_GRID]
    loo_seeds = tuple(range(cfg.loo_inits))
    loo_specs, loo_keys = [], []
    for lam in L2_GRID:
        for i in range(X.size):
            loo_specs.append(Spec("relu", "adam", 0.01, 11, cfg.relu_epochs, loo_seeds, l2=lam, exclude=i,
                                  record_every=cfg.relu_epochs))
            loo_keys.append((lam, i))
    adam_runs = run_specs(specs, cfg.workers)
    loo_runs = dict(zip(loo_keys, run_specs(loo_specs, cfg.workers)))
    refs = _parallel_refs(cfg, L2_GRID, cfg.l5_starts)
    for lam, s, ref in zip(L2_GRID, adam_runs, refs):
        k = int(np.argmin(s.loss))
        # лучший из Adam и Нелдера — Мида
        w_best = s.params[k] if s.loss[k] < ref["loss"] else np.array(ref["params"])
        j_best = min(float(s.loss[k]), ref["loss"])
        press = 0.0
        for i in range(X.size):
            r = loo_runs[(lam, i)]
            j = int(np.argmin(r.loss))
            press += float((model(r.params[j], X[i:i + 1])[0] - Y[i]) ** 2)
        mask = model.mask
        out["relu"].append({
            "l2": lam, "loss": j_best, "sse": float(np.sum((model(w_best, X) - Y) ** 2)), "params": w_best.tolist(),
            "weight_norm2": float(np.sum(mask * w_best * w_best)),
            "adam_share_near_best": float(np.mean(s.loss <= j_best * 1.01)),
            "adam_share_better_quad": float(np.mean(s.sse < base["quadratic"])),
            "adam_median_sse": _median(s.sse), "press": press,
        })
    out["demo_runs"] = compute_task5_demo(cfg, t4, out, base)
    return out


def compute_task5_demo(cfg: ReportConfig, t4: dict, t5: dict, base: dict) -> List[dict]:
    """Задачи 2–4 с квадратичным штрафом λ = L2_DEMO: все методы на одной функции J̃."""
    lam = L2_DEMO
    ch = t4["chosen"]
    seeds = tuple(range(cfg.seeds))
    rseeds = tuple(range(cfg.relu_seeds))
    specs, keys = [], []
    for key in MODELS:
        E = cfg.e4[key]
        sd = rseeds if key == "relu" else seeds
        if key == "relu":
            gd_lr = RELU_LR
        else:
            gd_lr = stability(MODELS[key], X, lam)["lr_opt"]
        specs.append(Spec(key, "gd", gd_lr, 12, MAX_ITER if key != "relu" else cfg.relu_gd_iter, sd, l2=lam,
                          delta=DELTA, record_every=1000))
        keys.append((key, "gd", 12))
        for meth in METHOD_TITLES:
            for b in (1, 12) if meth != "gd" else (1,):
                c = ch[f"{key}/{meth}/{b}"]
                specs.append(Spec(key, meth, c["lr"], b, E, sd, tau=c["tau"], l2=lam, record_every=E))
                keys.append((key, meth, b))
    runs = dict(zip(keys, run_specs(specs, cfg.workers)))
    rows = []
    for (key, meth, b), s in runs.items():
        if key == "relu":
            ref = next(r for r in t5["relu"] if r["l2"] == lam)
            best = ref["loss"]
            w_ref = np.array(ref["params"])
        else:
            w_ref = linear_solution(MODELS[key], X, Y, lam)
            best = float(Objective(MODELS[key], X, Y, lam)(w_ref))
        dist = np.max(np.abs(s.params - w_ref), axis=1)
        rows.append({"model": key, "method": meth, "batch": b, "lr": s.spec.lr, "tau": s.spec.tau,
                     "epochs_max": s.spec.epochs, "median_epochs": _median(s.epochs),
                     "converged": int(np.sum(s.status == "converged")),
                     "median_excess": _median(excess(s.loss, best)),
                     "median_dist": _median(dist) if key != "relu" else None,
                     "share_near_best": float(np.mean(s.loss <= 1.01 * best)),
                     "median_sse": _median(s.sse)})
    return rows


# ------------------------------------------------------------------ всё вместе
def compute(cfg: ReportConfig, log=print) -> dict:
    t0 = time.time()
    base = baselines()
    R: dict = {"data": {"x": X.tolist(), "y": Y.tolist()}, "baselines": base,
               "settings": {"delta": DELTA, "max_iter": MAX_ITER, "relu_lr": RELU_LR, "target": TARGET,
                            "l2_demo": L2_DEMO, "config": {k: v for k, v in cfg.__dict__.items()}}}
    log("задача 1: золотое сечение и метод Брента")
    R["task1"] = compute_task1()
    log("численный градиент")
    R["gradcheck"] = gradient_check()
    log("эталон сети ReLU (Нелдер — Мид из многих точек)")
    R["relu_ref"] = compute_relu_reference(cfg)
    log("задача 2: градиентный спуск")
    R["task2"] = compute_task2_convex(base)
    R["task2"]["relu"] = compute_task2_relu(cfg, R["relu_ref"], base)
    log("задача 3: мини-пакеты")
    R["task3"] = compute_task3(cfg, R["task2"], R["relu_ref"], base)
    log("задача 4: Нестеров и Adam (подбор скоростей обучения)")
    R["task4"] = compute_task4(cfg, R["relu_ref"], base)
    R["paths"] = compute_paths(R["task2"], R["task4"])
    log("задача 5: квадратичный штраф")
    R["task5"] = compute_task5(cfg, R["task4"], base)
    R["elapsed_s"] = round(time.time() - t0, 1)
    return R


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _clean(o):
    """NaN и бесконечности → None (JSON их не допускает)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_outputs(R: dict, out_dir: Path) -> None:
    from . import plots, tables

    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    plots.save_all(R, out_dir / "figures")
    (out_dir / "results" / "tables.md").write_text(tables.render(R), encoding="utf-8")
    slim = json.loads(json.dumps(R, default=_json_default))
    for key in ("curves",):
        slim["task3"].pop(key, None)
        slim["task4"].pop(key, None)
    slim["task2"]["linear"].pop("history", None)
    slim["task2"]["quadratic"].pop("history", None)
    (out_dir / "results" / "summary.json").write_text(
        json.dumps(_clean(slim), ensure_ascii=False, indent=1), encoding="utf-8")


def generate_report(cfg: ReportConfig, out_dir, log=print) -> dict:
    R = compute(cfg, log)
    write_outputs(R, Path(out_dir))
    log(f"готово за {R['elapsed_s']} с")
    return R
