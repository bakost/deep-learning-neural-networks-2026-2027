r"""Рисунки отчёта.

Оформление как в предыдущих ЛР: тонкие линии, светлая сплошная сетка,
текст нейтральными цветами. Методы обучения — категории (фиксированная
категориальная палитра), размер мини-пакета — упорядоченная величина
(одна шкала синего: светлее — меньше пакет).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator

from .linesearch import LineSearchResult, df_task, f_task
from .models import MODELS, Model, X, Y

__all__ = ["plot_fit", "plot_minimize", "save", "save_all"]

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, VIOLET, RED = CATEGORICAL
METHOD_COLORS = {"gd": BLUE, "nesterov": ORANGE, "adam": AQUA}
METHOD_NAMES = {"gd": "ГС / СГС", "nesterov": "Нестеров", "adam": "Adam"}
MODEL_COLORS = {"linear": BLUE, "quadratic": ORANGE, "relu": AQUA}
MODEL_NAMES = {"linear": "1: линейная", "quadratic": "2: квадратичная", "relu": "3: сеть ReLU"}
BATCH_COLORS = {1: "#b7d3f5", 2: "#86b6ef", 3: "#5598e7", 4: "#2a78d6", 6: "#1c5cab", 12: "#0d366b"}
OUTCOME_COLORS = {"global": "#0d366b", "better_quad": "#2a78d6", "between": "#86b6ef", "linear": "#eda100",
                  "worse": "#f3c97a", "constant": "#c3c2b7", "dead": "#52514e"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.8, "axes.grid": True,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
}


@contextmanager
def _style() -> Iterator[None]:
    with rc_context(STYLE):
        yield


def save(fig: Figure, path) -> None:
    with _style():
        fig.savefig(path)


def _data(ax, label=True) -> None:
    ax.scatter(X, Y, s=22, color=INK, zorder=5, label="данные" if label else None)


def _smooth(y: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Скользящее среднее log y по окну [e(1 − frac), e] — для шумных кривых СГС на логарифмической оси."""
    ly = np.log10(np.maximum(y, 1e-16))
    c = np.concatenate([[0.0], np.cumsum(ly)])
    e = np.arange(y.size)
    lo = np.floor(e * (1 - frac)).astype(int)
    return 10 ** ((c[e + 1] - c[lo]) / (e + 1 - lo))


def _grid_x():
    return np.linspace(0.0, 2.0, 801)


# ------------------------------------------------------------------ задача 1
def plot_minimize(res: LineSearchResult) -> Figure:
    """График f(z) и найденный минимум (для командной строки)."""
    with _style():
        fig = Figure(figsize=(5.2, 3.3), layout="constrained")
        ax = fig.subplots()
        z = np.linspace(-0.5, 1.5, 1001)
        ax.plot(z, f_task(z), color=BLUE)
        ax.scatter([res.x], [res.fx], color=ORANGE, zorder=5, label=f"минимум z* = {res.x:.6f}")
        ax.set_xlabel("z")
        ax.set_ylabel("f(z)")
        ax.legend(loc="lower right")
        return fig


def plot_task1(t1: dict) -> Figure:
    with _style():
        fig = Figure(figsize=(9.6, 3.7), layout="constrained")
        a, b = fig.subplots(1, 2)
        z = np.linspace(-0.6, 1.6, 2001)
        a.plot(z, f_task(z), color=BLUE, label="f(z)")
        a.plot(z, 1 - 0.5 * np.exp(-10 * (z - 0.65) ** 2), color=MUTED, lw=0.9, ls="--",
               label="1 − ½·exp(−10(z − 0.65)²)")
        a.plot(z, 1 - 0.45 * np.exp(-50 * (z - 0.35) ** 2), color=MUTED, lw=0.9, ls=":",
               label="1 − 0.45·exp(−50(z − 0.35)²)")
        a.scatter([t1["z_star"]], [t1["f_star"]], color=ORANGE, zorder=6,
                  label=f"минимум z* = {t1['z_star']:.6f}")
        sh = t1["shoulder"]
        a.annotate("«полка»: f′ > 0,\nвторого минимума нет", xy=(sh["z"], sh["f"]), xytext=(-0.55, 0.62),
                   fontsize=8, color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        ax2 = a.inset_axes([0.68, 0.07, 0.30, 0.30])
        zz = np.linspace(0.2, 1.0, 401)
        ax2.plot(zz, df_task(zz), color=BLUE, lw=1.0)
        ax2.axhline(0, color=MUTED, lw=0.6)
        ax2.set_title("f′(z)", fontsize=7, pad=2)
        ax2.tick_params(labelsize=6)
        a.set_xlabel("z")
        a.set_ylabel("f(z)")
        a.set_title("а) функция задачи 1")
        a.legend(loc="lower left", fontsize=7)
        a.set_ylim(0.28, 1.05)
        for key, color, name in (("golden", BLUE, "золотое сечение"), ("brent", ORANGE, "Брент")):
            w = np.array(t1[f"{key}_widths"])
            err = np.abs(np.array(t1[f"{key}_x"]) - t1["z_star"])
            # номер вычисления f: первые вычисления — начальные точки
            n0 = 2 if key == "golden" else 1
            k = np.arange(1, len(w) + 1) + n0
            b.semilogy(k, w, color=color, label=f"{name}: длина отрезка")
            b.semilogy(k, np.maximum(err, 1e-17), color=color, ls="--", lw=1.1, label=f"{name}: |z − z*|")
        b.axhline(t1["floor"], color=MUTED, lw=0.8, ls=":")
        b.text(2, t1["floor"] * 1.8, "предел √(2ε|f|/f″)", fontsize=7, color=INK_2)
        b.set_xlabel("вычислений f")
        b.set_title("б) сходимость (начальный отрезок [0, 2.618])")
        b.legend(fontsize=7, loc="upper right")
        return fig


# ------------------------------------------------------------------ задача 2
def plot_fit(model: Model, params) -> Figure:
    with _style():
        fig = Figure(figsize=(5.2, 3.4), layout="constrained")
        ax = fig.subplots()
        _data(ax)
        g = _grid_x()
        ax.plot(g, model(params, g), color=BLUE, label=model.formula)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower right")
        return fig


def plot_task2_fits(R: dict) -> Figure:
    t2 = R["task2"]
    with _style():
        fig = Figure(figsize=(6.6, 3.9), layout="constrained")
        ax = fig.subplots()
        g = _grid_x()
        for key in ("linear", "quadratic"):
            ax.plot(g, MODELS[key](t2[key]["params"], g), color=MODEL_COLORS[key],
                    label=f"{MODEL_NAMES[key]}, SSE = {t2[key]['sse']:.4f}")
        best = t2["relu"]["best"]
        ax.plot(g, MODELS["relu"](best["params"], g), color=MODEL_COLORS["relu"],
                label=f"{MODEL_NAMES['relu']} (лучший из {t2['relu']['seeds']} запусков ГС), SSE = {best['sse']:.4f}")
        ref = R["relu_ref"]
        ax.plot(g, MODELS["relu"](ref["params"], g), color=AQUA, ls=":", lw=1.2,
                label=f"сеть ReLU: эталонный минимум, SSE = {ref['sse']:.4f}")
        _data(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower right", fontsize=7.5)
        return fig


def plot_task2_gd(R: dict) -> Figure:
    t2 = R["task2"]
    with _style():
        fig = Figure(figsize=(9.6, 3.7), layout="constrained")
        a, b = fig.subplots(1, 2)
        for key in ("linear", "quadratic"):
            d = t2[key]
            h = d["history"]
            k = np.array(h["epoch"])
            a.loglog(k[1:], np.array(h["grad_norm"])[1:], color=MODEL_COLORS[key],
                     label=f"{MODEL_NAMES[key]}: ε = {d['lr']:.4f}, κ = {d['kappa']:.0f}")
            a.loglog(k[1:], d["g0"] * d["rate_opt"] ** k[1:], color=MODEL_COLORS[key], ls=":", lw=1.0)
        a.axhline(R["settings"]["delta"], color=MUTED, lw=0.8, ls="--")
        a.text(1.2, R["settings"]["delta"] * 1.8, "δ", color=INK_2)
        a.set_xlabel("итерация")
        a.set_ylabel("‖DJ‖")
        a.set_title("а) норма градиента (пунктир — ‖DJ₀‖·ρᵏ)")
        a.legend(fontsize=7.5, loc="lower left")
        for key in ("linear", "quadratic"):
            rows = sorted(t2[key]["sweep"], key=lambda r: r["frac"])
            ok = [r for r in rows if r["status"] == "converged"]
            b.semilogy([r["frac"] for r in ok], [r["iterations"] for r in ok], "o-", ms=3.5,
                       color=MODEL_COLORS[key], label=MODEL_NAMES[key])
            th = [r for r in rows if r["theory"]]
            b.semilogy([r["frac"] for r in th], [r["theory"] for r in th], ":", color=MODEL_COLORS[key], lw=1.0)
            for r in rows:
                if r["status"] == "diverged":
                    b.scatter([r["frac"]], [ok[-1]["iterations"]], marker="x", color=RED, zorder=6)
            opt = next(r for r in rows if r["is_opt"])
            b.scatter([opt["frac"]], [opt["iterations"]], s=40, facecolor="none", edgecolor=INK, zorder=6)
        b.axvline(1.0, color=RED, lw=0.8, ls="--")
        b.text(1.0, b.get_ylim()[1] * 0.6, " ε = 2/λmax", color=RED, fontsize=8)
        b.set_xlabel("ε / (2/λmax)")
        b.set_ylabel("итераций до ‖DJ‖ < δ")
        b.set_title("б) ○ — ε = 2/(λmin+λmax), × — расходится")
        b.legend(fontsize=7.5)
        return fig


def plot_task2_relu(R: dict) -> Figure:
    t = R["task2"]["relu"]
    base = R["baselines"]
    best = R["relu_ref"]["sse"]
    from .report import OUTCOMES

    with _style():
        fig = Figure(figsize=(9.6, 3.8), layout="constrained")
        a, b = fig.subplots(1, 2)
        sse = np.array(t["sse"], dtype=float)
        rng = np.random.default_rng(0)
        a.scatter(sse, rng.uniform(-0.3, 0.3, sse.size), s=10, color=BLUE, alpha=0.7)
        a.set_xscale("log")
        for v, name in ((best, "глоб. минимум"), (base["quadratic"], "квадратичная"), (base["linear"], "линейная"),
                        (base["constant"], "ŷ = ȳ"), (base["zero"], "ŷ ≡ 0")):
            a.axvline(v, color=MUTED, lw=0.8, ls="--")
            a.text(v, 0.42, name, rotation=90, fontsize=7, color=INK_2, ha="right", va="bottom")
        a.set_ylim(-0.5, 1.2)
        a.yaxis.set_major_locator(NullLocator())
        a.set_xlabel("сумма квадратов ошибок в конце обучения")
        a.set_title(f"а) {t['seeds']} начальных точек, ГС ε = {t['lr']}")
        g = _grid_x()
        _data(b)
        names = dict(OUTCOMES)
        for k, ex in t["examples"].items():
            b.plot(g, MODELS["relu"](ex["params"], g), color=OUTCOME_COLORS[k],
                   label=f"{names[k]}: SSE = {ex['sse']:.3f}")
        b.set_xlabel("x")
        b.set_ylabel("y")
        b.set_title("б) куда приходит спуск: примеры")
        b.legend(fontsize=7, loc="lower right")
        return fig


# ------------------------------------------------------------------ задача 3
def plot_task3(R: dict) -> Figure:
    t3 = R["task3"]
    with _style():
        fig = Figure(figsize=(9.6, 6.4), layout="constrained")
        axes = fig.subplots(2, 2, sharex=True)
        for j, key in enumerate(("linear", "quadratic")):
            for i, sched in enumerate(("const", "decay")):
                ax = axes[i, j]
                for b in sorted(BATCH_COLORS):
                    y = np.array(t3["curves"].get(f"{key}/{b}/{sched}", []), dtype=float)
                    if not y.size:
                        continue
                    e = np.arange(y.size)
                    ax.loglog(e[1:], _smooth(y)[1:], color=BATCH_COLORS[b], label=f"m = {b}")
                lr = t3["common_lr"][key]
                title = f"ε = {lr}" if sched == "const" else f"ε = {lr}/(1 + e/{t3['tau']:.0f})"
                ax.set_title(f"{MODEL_NAMES[key]}: {title}")
                ax.set_ylabel("(J − J*)/J*, медиана (сглаж.)")
                if i == 1:
                    ax.set_xlabel("эпоха")
                if i == 0 and j == 0:
                    ax.legend(fontsize=7.5, ncol=2)
        return fig


def plot_relu_outcomes(R: dict) -> Figure:
    from .report import OUTCOMES

    rows = [("задача 2: ГС, m = 12, до ‖DJ‖ < δ", R["task2"]["relu"]["shares"])]
    for r in R["task3"]["relu"]:
        s = "ε const" if r["schedule"] == "const" else "ε убыв."
        rows.append((f"задача 3: СГС, m = {r['batch']}, {s}", r))
    for r in R["task4"]["rows"]:
        if r["model"] != "relu":
            continue
        sched = "" if r["tau"] is None else ", ε убыв."
        rows.append((f"задача 4: {METHOD_NAMES[r['method']]}, m = {r['batch']}, ε = {r['lr']:g}{sched}", r))
    with _style():
        fig = Figure(figsize=(9.6, 0.24 * len(rows) + 1.2), layout="constrained")
        ax = fig.subplots()
        y = np.arange(len(rows))[::-1]
        left = np.zeros(len(rows))
        for k, name in OUTCOMES:
            v = np.array([r[1].get(k, 0.0) for r in rows])
            ax.barh(y, v, left=left, color=OUTCOME_COLORS[k], height=0.72, label=name, edgecolor="white", lw=0.4)
            left += v
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"доля из {R['task2']['relu']['seeds']} начальных точек")
        ax.grid(axis="y", visible=False)
        ax.legend(ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.4, 1.0))
        return fig


# ------------------------------------------------------------------ задача 4
def plot_paths(R: dict) -> Figure:
    p = R["paths"]
    model = MODELS["linear"]
    with _style():
        fig = Figure(figsize=(6.6, 4.8), layout="constrained")
        ax = fig.subplots()
        w1 = np.linspace(-1.6, 2.6, 241)
        b1 = np.linspace(-0.6, 3.4, 241)
        W1, B1 = np.meshgrid(w1, b1)
        P = np.stack([W1, B1], axis=-1)
        J = np.sum((model(P, X) - Y) ** 2, axis=-1)
        levels = R["baselines"]["linear"] + np.geomspace(0.01, 200, 14)
        ax.contour(W1, B1, J, levels=levels, colors=AXIS, linewidths=0.7)
        names = {"gd": "ГС, m = 12", "nesterov": "Нестеров, m = 12", "adam": "Adam, m = 12", "sgd1": "СГС, m = 1"}
        colors = {"gd": BLUE, "nesterov": ORANGE, "adam": AQUA, "sgd1": PINK}
        for k, d in p["paths"].items():
            path = np.array(d["path"])
            ax.plot(path[:, 0], path[:, 1], "-o", ms=2.2, lw=1.0, color=colors[k],
                    label=f"{names[k]}, ε = {d['lr']:.3g}")
        ax.scatter(*p["w0"], marker="s", color=INK, zorder=6, label="начало")
        ax.scatter(*p["w_star"], marker="*", s=90, color=RED, zorder=6, label="минимум (МНК)")
        ax.set_xlabel("w₁")
        ax.set_ylabel("b₁")
        ax.set_title(f"Модель 1: первые {p['updates']} обновлений весов")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.set_xlim(w1[0], w1[-1])
        ax.set_ylim(b1[0], b1[-1])
        return fig


def plot_task4_curves(R: dict) -> Figure:
    t4 = R["task4"]
    styles = {1: ":", 4: "--", 12: "-"}
    with _style():
        fig = Figure(figsize=(10.2, 3.6), layout="constrained")
        axes = fig.subplots(1, 3)
        for ax, key in zip(axes, MODELS):
            for meth in METHOD_NAMES:
                for b in (1, 4, 12):
                    y = np.array(t4["curves"].get(f"{key}/{meth}/{b}", []), dtype=float)
                    if not y.size:
                        continue
                    if key == "relu":
                        e = np.arange(y.size) * 10
                    else:
                        e = np.arange(y.size)
                    ax.plot(e[1:], _smooth(y)[1:], ls=styles[b], color=METHOD_COLORS[meth], lw=1.3,
                            label=f"{METHOD_NAMES[meth]}, m = {b}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("эпоха")
            ax.set_title(MODEL_NAMES[key])
            ax.set_ylabel("сумма квадратов ошибок, медиана" if key == "relu" else "(J − J*)/J*, медиана")
        axes[0].legend(fontsize=6.8, ncol=1)
        return fig


# ------------------------------------------------------------------ задача 5
def plot_task5_path(R: dict) -> Figure:
    t5 = R["task5"]
    lam = np.array(t5["grid"], dtype=float)
    x = np.where(lam > 0, lam, lam[1] / 3)            # λ = 0 — слева от сетки на логарифмической оси
    ticks = list(x)
    labels = ["0"] + [f"{v:g}" for v in lam[1:]]
    with _style():
        fig = Figure(figsize=(10.2, 3.5), layout="constrained")
        a, b, c = fig.subplots(1, 3)
        q = np.array([r["params"] for r in t5["convex"]["quadratic"]])
        for k, name, col in ((0, "w₁ (при x²)", BLUE), (1, "w₂ (при x)", ORANGE), (2, "b₁ (не штрафуется)", MUTED)):
            a.plot(x, q[:, k], "o-", ms=3, color=col, label=name)
        a.set_title("а) модель 2: коэффициенты")
        r = np.array([row["params"] for row in t5["relu"]])
        names = MODELS["relu"].names
        for k, col in zip((0, 2, 4, 5), (BLUE, ORANGE, AQUA, PINK)):
            b.plot(x, r[:, k], "o-", ms=3, color=col, label=names[k])
        b.set_title("б) модель 3: веса (лучший минимум)")
        for key in MODELS:
            rows = t5["convex"][key] if key != "relu" else t5["relu"]
            c.plot(x, [row["sse"] for row in rows], "o-", ms=3, color=MODEL_COLORS[key],
                   label=f"{MODEL_NAMES[key]}: SSE")
            c.plot(x, [row["press"] for row in rows], "s--", ms=3, lw=1.0, color=MODEL_COLORS[key],
                   label=f"{MODEL_NAMES[key]}: LOO")
        c.set_yscale("log")
        c.set_title("в) ошибка: обучение и скользящий контроль")
        for ax in (a, b, c):
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_xlabel("λ")
            ax.legend(fontsize=6.8)
        return fig


def plot_task5_fits(R: dict) -> Figure:
    t5 = R["task5"]
    g = _grid_x()
    shown = (0.0, t5["demo"], 1.0)
    shades = {0.0: 1.0, t5["demo"]: 0.65, 1.0: 0.35}
    with _style():
        fig = Figure(figsize=(10.2, 3.3), layout="constrained")
        axes = fig.subplots(1, 3, sharey=True)
        for ax, key in zip(axes, MODELS):
            rows = t5["convex"][key] if key != "relu" else t5["relu"]
            for row in rows:
                if row["l2"] in shown:
                    ax.plot(g, MODELS[key](row["params"], g), color=MODEL_COLORS[key], alpha=shades[row["l2"]],
                            lw=1.6 if row["l2"] == 0 else 1.3, ls="-" if row["l2"] == 0 else "--",
                            label=f"λ = {row['l2']:g}: SSE = {row['sse']:.3f}")
            _data(ax, label=False)
            ax.set_title(MODEL_NAMES[key])
            ax.set_xlabel("x")
            ax.legend(fontsize=7, loc="lower right")
        axes[0].set_ylabel("y")
        return fig


def save_all(R: dict, out: Path) -> Dict[str, str]:
    out = Path(out)
    figs = {
        "task1.png": plot_task1(R["task1"]),
        "task2_fits.png": plot_task2_fits(R),
        "task2_gd.png": plot_task2_gd(R),
        "task2_relu.png": plot_task2_relu(R),
        "task3_batches.png": plot_task3(R),
        "relu_outcomes.png": plot_relu_outcomes(R),
        "task4_paths.png": plot_paths(R),
        "task4_curves.png": plot_task4_curves(R),
        "task5_path.png": plot_task5_path(R),
        "task5_fits.png": plot_task5_fits(R),
    }
    for name, fig in figs.items():
        save(fig, out / name)
    return {k: str(out / k) for k in figs}
