r"""Рисунки отчёта.

Оформление как в ЛР3: тонкие линии, светлая сплошная сетка, текст
нейтральными цветами. Размер скрытого слоя :math:`H` — упорядоченная
величина, поэтому кривые для разных :math:`H` окрашены по одной шкале синего
(светлее — меньше нейронов). Разные законы изменения скорости обучения —
категории, для них используется категориальная палитра в фиксированном порядке.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure

from .experiments import RunSummary
from .network import forward, gradients

__all__ = [
    "plot_decay",
    "plot_generalization",
    "plot_importance",
    "plot_iterations",
    "plot_kink",
    "plot_training_curves",
    "save",
]

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA = CATEGORICAL[:3]
H_COLORS = {2: "#86b6ef", 4: "#5598e7", 8: "#2a78d6", 16: "#1c5cab", 32: "#0d366b"}
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


def _seed0(runs: Sequence[RunSummary], hidden: int) -> RunSummary:
    return min((r for r in runs if r.job.hidden == hidden), key=lambda r: r.job.seed)


# ------------------------------------------------------------------ рисунок 1
def plot_training_curves(sweep: Sequence[RunSummary], floor: float, ols_rmse: float, delta: float) -> Figure:
    """RMSE на обучающей выборке и норма градиента по итерациям для H = 2…32 (зерно 0)."""
    with _style():
        fig = Figure(figsize=(9.4, 3.9), layout="constrained")
        a, b = fig.subplots(1, 2)
        for h, color in H_COLORS.items():
            r = _seed0(sweep, h)
            it = r.history["iteration"] + 1
            a.plot(it, r.rmse_curve(), color=color, label=f"H = {h}")
            b.plot(it, r.history["grad_norm"], color=color, label=f"H = {h}")
            if r.converged:
                b.scatter([it[-1]], [r.history["grad_norm"][-1]], s=24, color=color, edgecolors="white", zorder=3)
        a.axhline(ols_rmse, color=ORANGE, linewidth=1.2)
        a.text(1.3, ols_rmse * 1.05, f"МНК (ЛР3): {ols_rmse:.3f}", color=INK_2, fontsize=7.5, va="bottom")
        a.axhline(floor, color=MUTED, linewidth=1.2)
        a.text(1.3, floor * 0.95, f"нижняя граница √(SS_pe/n) = {floor:.3f}", color=INK_2, fontsize=7.5, va="top")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_ylim(bottom=floor * 0.8)
        a.set_xlabel("итерация j + 1")
        a.set_ylabel(r"RMSE на обучающей выборке, ед. $\log P$")
        a.set_title("(а) Ошибка на обучающей выборке", loc="left")
        a.legend(loc="upper right", ncols=2)
        b.axhline(delta, color=MUTED, linewidth=1.2)
        b.text(1.3, delta * 1.3, f"δ = {delta:g}", color=INK_2, fontsize=7.5)
        b.set_xscale("log")
        b.set_yscale("log")
        b.set_xlabel("итерация j + 1")
        b.set_ylabel(r"$\Vert DJ \Vert$")
        b.set_title("(б) Норма градиента (точка — остановка по δ)", loc="left")
    return fig


# ------------------------------------------------------------------ рисунок 2
def plot_iterations(sweep: Sequence[RunSummary], lr_runs: Sequence[RunSummary], lr_grid: Sequence[float],
                    lr_hidden: int, eps: float, eps_max_linear: float, max_iter: int) -> Figure:
    """Число итераций до сходимости: по размеру скрытого слоя и по скорости обучения."""
    with _style():
        fig = Figure(figsize=(9.4, 3.7), layout="constrained")
        a, b = fig.subplots(1, 2)
        sizes = sorted({r.job.hidden for r in sweep})
        for k, h in enumerate(sizes):
            rs = sorted((r for r in sweep if r.job.hidden == h), key=lambda r: r.job.seed)
            xs = k + np.linspace(-0.18, 0.18, len(rs))
            for x, r in zip(xs, rs):
                if r.converged:
                    a.scatter([x], [r.iterations], s=30, color=H_COLORS.get(h, BLUE), edgecolors="white", zorder=3)
                else:
                    a.scatter([x], [max_iter], s=34, facecolors="none", edgecolors=INK_2, linewidths=1.1, zorder=3)
        a.axhline(max_iter, color=GRID, linewidth=1.0)
        a.text(len(sizes) - 0.5, max_iter * 1.04, "предел итераций; ○ — не сошёлся", ha="right", va="bottom",
               fontsize=7.5, color=INK_2)
        a.set_xticks(range(len(sizes)), [f"H = {h}" for h in sizes])
        a.set_ylim(0, max_iter * 1.15)
        a.grid(axis="x", visible=False)
        a.set_ylabel("итераций до ‖DJ‖ < δ")
        a.set_title(f"(а) Размер скрытого слоя (ε = {eps:g})", loc="left")

        for k, e in enumerate(lr_grid):
            rs = ([r for r in sweep if r.job.hidden == lr_hidden] if e == eps
                  else [r for r in lr_runs if r.job.schedule.eps0 == e])
            rs = sorted(rs, key=lambda r: r.job.seed)
            xs = k + np.linspace(-0.18, 0.18, len(rs))
            for x, r in zip(xs, rs):
                if r.converged:
                    b.scatter([x], [r.iterations], s=30, color=BLUE, edgecolors="white", zorder=3)
                elif r.status == "diverged":
                    b.scatter([x], [0.03 * max_iter], s=36, marker="x", color=INK_2, linewidths=1.2, zorder=3)
                else:
                    b.scatter([x], [max_iter], s=34, facecolors="none", edgecolors=INK_2, linewidths=1.1, zorder=3)
        pos = np.interp(eps_max_linear, lr_grid, np.arange(len(lr_grid)))
        b.axvline(pos, color=ORANGE, linewidth=1.2)
        b.text(pos - 0.08, max_iter * 1.04, f"2/λmax линейной модели = {eps_max_linear:.2f}", ha="right",
               va="bottom", fontsize=7.5, color=INK_2)
        b.set_xticks(range(len(lr_grid)), [f"{e:g}" for e in lr_grid])
        b.set_ylim(0, max_iter * 1.15)
        b.grid(axis="x", visible=False)
        b.set_xlabel("скорость обучения ε")
        b.set_ylabel("итераций до ‖DJ‖ < δ")
        b.set_title(f"(б) Скорость обучения (H = {lr_hidden}); × — разошёлся", loc="left")
    return fig


# ------------------------------------------------------------------ рисунок 3
def plot_kink(sweep: Sequence[RunSummary], steps: int = 300, lr: float = 0.1) -> Figure:
    r"""«Дребезг» на изломах ReLU: продолжение спуска из конечной точки на ``steps`` шагов.

    Когда нейрон :math:`k` переключается на веществе :math:`i`, градиент
    скачком меняется на величину порядка :math:`|L_i w^2_k|\,\lVert x_i\rVert`
    (слагаемое с :math:`H(z_{ik})` в формулах лекции). Панель (б) показывает
    средний множитель :math:`|L_i w^2_k|` по переключившимся парам.
    """
    from .data import Standardizer, load_lipo

    data = load_lipo()
    Z, t = Standardizer.fit(data.X, data.y).transform(data.X, data.y)
    with _style():
        fig = Figure(figsize=(9.4, 3.5), layout="constrained")
        a, b = fig.subplots(1, 2)
        for h in (2, 4, 16):
            r = _seed0(sweep, h)
            p = r.params.copy()
            norms, jumps, steps_with_flip = [], [], []
            pattern = forward(p, Z)[1] >= 0
            for k in range(steps):
                _, g = gradients(p, Z, t)
                L = t - forward(p, Z)[0]
                norms.append(np.linalg.norm(g.flat()))
                p.W1 -= lr * g.W1
                p.b1 -= lr * g.b1
                p.w2 -= lr * g.w2
                p.b2 -= lr * g.b2
                new = forward(p, Z)[1] >= 0
                ii, kk = np.nonzero(new != pattern)
                if ii.size:
                    jumps.append(float(np.mean(np.abs(L[ii] * p.w2[kk]))))
                    steps_with_flip.append(k)
                pattern = new
            label = f"H = {h}" + (" (сошёлся)" if r.converged else "")
            a.plot(np.arange(steps), norms, color=H_COLORS[h], label=label, linewidth=1.2)
            b.scatter(steps_with_flip, jumps, s=6, color=H_COLORS[h], label=label, zorder=3)
        a.set_yscale("log")
        a.set_xlabel("шаг после окончания обучения")
        a.set_ylabel(r"$\Vert DJ \Vert$")
        a.set_title("(а) Норма градиента в конечной точке", loc="left")
        a.legend(loc="center right")
        b.set_yscale("log")
        b.set_xlabel("шаг после окончания обучения")
        b.set_ylabel(r"среднее $|L_i\, w^2_k|$ по переключениям")
        b.set_title("(б) Скачок градиента при переключении нейрона", loc="left")
        b.legend(loc="center right", markerscale=2.5)
    return fig


# ------------------------------------------------------------------ рисунок 4
def plot_decay(runs: Dict[str, List[RunSummary]], groups, delta: float) -> Figure:
    """Уменьшение скорости обучения: H = 16, H = 4 и мини-пакеты (зерно 0)."""
    with _style():
        fig = Figure(figsize=(9.4, 7.2), layout="constrained")
        (a, b), (c, d) = fig.subplots(2, 2)
        panels = {"H = 16": a, "H = 4": c, "мини-пакеты": d}
        counters = {k: 0 for k in panels}
        for title, sch, rs in groups:
            key = "мини-пакеты" if "мини-пакеты" in title else title
            if key not in panels or not rs:
                continue
            r = min(rs, key=lambda r: r.job.seed)
            color = CATEGORICAL[counters[key] % len(CATEGORICAL)]
            counters[key] += 1
            ax = panels[key]
            ax.plot(r.history["iteration"] + 1, r.history["grad_norm"], color=color, linewidth=1.2,
                    label=sch.label())
            if key == "H = 16":
                it = np.unique(np.r_[np.geomspace(1, 3e5, 200).astype(int), 0])
                b.plot(it + 1, [sch(int(j)) for j in it], color=color, label=sch.label())
        for ax, title in ((a, "(а) H = 16: норма градиента"), (c, "(в) H = 4: норма градиента"),
                          (d, "(г) H = 16, мини-пакеты: норма полного градиента")):
            ax.axhline(delta, color=MUTED, linewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("итерация j + 1")
            ax.set_ylabel(r"$\Vert DJ \Vert$")
            ax.set_title(title, loc="left")
            ax.legend(loc="lower left", fontsize=7)
        b.set_xscale("log")
        b.set_yscale("log")
        b.set_ylim(1e-6, 0.3)
        b.set_xlabel("итерация j + 1")
        b.set_ylabel(r"$\varepsilon_j$")
        b.set_title("(б) Законы уменьшения скорости обучения", loc="left")
    return fig


# ------------------------------------------------------------------ рисунок 5
def plot_generalization(curves: Dict[int, Dict[str, np.ndarray]], cv: Dict[str, dict], floor: float) -> Figure:
    """Ошибка на обучении и на контроле: итоговая и по ходу обучения."""
    with _style():
        fig = Figure(figsize=(9.4, 4.0), layout="constrained")
        a, b = fig.subplots(1, 2, width_ratios=[1, 1.25])
        names = ["МНК, 24 регрессора", "МНК, 7 регрессоров (ЛР3)"] + [f"mlp{h}" for h in sorted(curves)]
        labels = ["МНК, 24", "МНК, 7"] + [f"H = {h}" for h in sorted(curves)]
        y = np.arange(len(names))[::-1]
        for yy, name in zip(y, names):
            item = cv[name]
            cvm = float(np.mean(item["cv_rmse"]))
            color = ORANGE if name.startswith("МНК") else H_COLORS[int(name[3:])]
            a.plot([item["train_rmse"], cvm], [yy, yy], color=GRID, linewidth=2.0, zorder=1)
            a.scatter([item["train_rmse"]], [yy], s=40, facecolors="white", edgecolors=color, linewidths=1.5, zorder=3)
            a.scatter([cvm], [yy], s=40, color=color, edgecolors="white", zorder=3)
            a.text(cvm + 0.03, yy, f"{cvm:.2f}", va="center", fontsize=7.5, color=INK_2)
        a.axvline(floor, color=MUTED, linewidth=1.0)
        a.set_yticks(y, labels)
        a.grid(axis="y", visible=False)
        a.set_xlim(0, max(float(np.mean(cv[name]["cv_rmse"])) for name in names) + 0.25)
        a.set_xlabel(r"RMSE, ед. $\log P$")
        a.set_title("(а) ○ обучение, ● контроль", loc="left")

        for h, curve in sorted(curves.items()):
            keep = curve["iteration"] >= 10          # точки записываются каждые 10 итераций
            b.plot(curve["iteration"][keep], curve["val_rmse"][keep], color=H_COLORS[h], label=f"H = {h}")
        for name, color in (("МНК, 24 регрессора", ORANGE), ("МНК, 7 регрессоров (ЛР3)", AQUA)):
            level = float(np.mean(cv[name]["cv_rmse"]))
            b.axhline(level, color=color, linewidth=1.2)
            b.text(12, level + 0.01, name, fontsize=7.5, color=INK_2, va="bottom")
        b.set_xscale("log")
        low = min(min(float(np.min(c["val_rmse"])) for c in curves.values()),
                  min(float(np.mean(cv[k]["cv_rmse"])) for k in ("МНК, 24 регрессора", "МНК, 7 регрессоров (ЛР3)")))
        b.set_ylim(0.9 * low, 2.2)
        b.set_xlabel("итерация j")
        b.set_ylabel(r"RMSE на контроле, ед. $\log P$")
        b.set_title("(б) Ошибка на контроле по ходу обучения", loc="left")
        b.legend(loc="upper right", ncols=2)
    return fig


# ------------------------------------------------------------------ рисунок 6
def plot_importance(columns: Sequence[str], beta: np.ndarray, imp: dict) -> Figure:
    """Вклад регрессоров: МНК против перцептронов."""
    with _style():
        order = np.argsort(np.abs(beta))
        fig = Figure(figsize=(9.4, 6.6), layout="constrained")
        a, b = fig.subplots(1, 2, sharey=True)
        y = np.arange(len(columns))
        series = [("МНК, β*", beta, ORANGE, "o"), ("перцептрон H = 8, ⟨∂ŷ/∂x⟩", imp[8]["mean"], H_COLORS[8], "s"),
                  ("перцептрон H = 32, ⟨∂ŷ/∂x⟩", imp[32]["mean"], H_COLORS[32], "D")]
        for k, (label, values, color, marker) in enumerate(series):
            a.scatter(values[order], y + (k - 1) * 0.22, s=22, marker=marker, color=color, edgecolors="white",
                      linewidths=0.6, label=label, zorder=3)
        a.axvline(0, color=MUTED, linewidth=1.0)
        a.set_yticks(y, [columns[j] for j in order], fontsize=8)
        a.grid(axis="y", visible=False)
        a.set_xlabel("эффект регрессора (стандартизованные единицы)")
        a.set_title("(а) Средний эффект со знаком", loc="left")
        a.legend(loc="lower right", fontsize=7)

        perm = [("МНК", imp["ols_perm"], ORANGE), ("H = 8", imp[8]["perm"], H_COLORS[8]),
                ("H = 32", imp[32]["perm"], H_COLORS[32])]
        height = 0.26
        for k, (label, values, color) in enumerate(perm):
            share = values / np.sum(np.clip(values, 0, None)) * 100
            b.barh(y + (1 - k) * height, share[order], height=height * 0.9, color=color, label=label)
        b.set_xlabel("перестановочная важность, % от суммы")
        b.set_title("(б) Рост ошибки при перемешивании регрессора", loc="left")
        b.grid(axis="y", visible=False)
        b.legend(loc="lower right")
    return fig
