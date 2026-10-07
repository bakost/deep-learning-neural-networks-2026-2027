r"""Рисунки отчёта.

Все функции строят :class:`matplotlib.figure.Figure` через объектный API и не
трогают глобальное состояние ``pyplot``. Оформление общее: тонкие линии,
сплошная светлая сетка, текст — нейтральными цветами, данные — цветами
категориальной палитры в фиксированном порядке (синий, оранжевый,
бирюзовый). Корреляции — расходящаяся шкала «синий — серый — красный».
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence

import numpy as np
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy import stats

from .contributions import MatchedPair
from .data import LipoData
from .diagnostics import cooks_distance, leverage, studentized_residuals
from .ols import OLSResult

__all__ = [
    "plot_coefficients",
    "plot_correlations",
    "plot_data_overview",
    "plot_elimination",
    "plot_fit",
    "plot_matched_pairs",
    "plot_residuals",
]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


@contextmanager
def _style() -> Iterator[None]:
    with rc_context(STYLE):
        yield


def _scatter(ax, x, y, color=BLUE, size=26, **kw):
    """Точки с белым ободком (чтобы были видны там, где перекрываются)."""
    return ax.scatter(x, y, s=size, color=color, edgecolors="white", linewidths=0.8, zorder=3, **kw)


# ------------------------------------------------------------------ данные
def plot_data_overview(data: LipoData) -> Figure:
    """Распределение :math:`\\log P` и число веществ, у которых регрессор ненулевой."""
    with _style():
        fig = Figure(figsize=(9.0, 3.6), layout="constrained")
        ax1, ax2 = fig.subplots(1, 2, width_ratios=[1, 1.6])
        ax1.hist(data.y, bins=np.arange(-2.0, 7.5, 0.5), color=BLUE, edgecolor="white", linewidth=1.0)
        ax1.set_xlabel(r"$\log P$")
        ax1.set_ylabel("число веществ")
        ax1.set_title(r"Распределение $\log P$ (n = 82)", loc="left")
        ax1.grid(axis="x", visible=False)

        counts = np.count_nonzero(data.X, axis=0)
        rare = counts <= 2
        x = np.arange(data.k)
        ax2.bar(x[~rare], counts[~rare], width=0.62, color=BLUE, label="регрессор ненулевой у > 2 веществ")
        ax2.bar(x[rare], counts[rare], width=0.62, color=ORANGE, label="у 1–2 веществ")
        for i in np.where(rare)[0]:
            ax2.text(i, counts[i] + 1.5, str(counts[i]), ha="center", va="bottom", color=INK_2, fontsize=7)
        ax2.set_xticks(x, data.columns, rotation=60, ha="right", fontsize=7)
        ax2.set_ylabel("веществ с ненулевым значением")
        ax2.set_title("Сколько веществ «несут» каждый регрессор", loc="left")
        ax2.grid(axis="x", visible=False)
        ax2.legend(loc="upper left")
    return fig


def plot_correlations(data: LipoData) -> Figure:
    """Матрица корреляций регрессоров и :math:`\\log P`."""
    with _style():
        table = np.column_stack([data.X, data.y])
        names = [*data.columns, "logP"]
        corr = np.corrcoef(table, rowvar=False)
        shown = corr.copy()
        np.fill_diagonal(shown, np.nan)
        cmap = DIVERGING.with_extremes(bad="#ffffff")
        fig = Figure(figsize=(7.4, 6.4), layout="constrained")
        ax = fig.subplots()
        im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j and abs(corr[i, j]) >= 0.9:
                    ax.text(j, i, f"{corr[i, j]:.2f}"[1:] if corr[i, j] > 0 else f"{corr[i, j]:.1f}",
                            ha="center", va="center", fontsize=5.5, color="white")
        bar = fig.colorbar(im, ax=ax, shrink=0.75)
        bar.set_label("коэффициент корреляции")
        bar.outline.set_visible(False)
        ax.set_title("Корреляции регрессоров (подписаны |r| ≥ 0.9)", loc="left")
    return fig


# ---------------------------------------------------------------- подгонка
def plot_fit(fit: OLSResult, data: LipoData, loo: Optional[np.ndarray] = None, label_top: int = 6) -> Figure:
    """Предсказанные значения против экспериментальных."""
    with _style():
        fig = Figure(figsize=(5.4, 5.0), layout="constrained")
        ax = fig.subplots()
        lim = (min(data.y.min(), fit.fitted.min()) - 0.5, max(data.y.max(), fit.fitted.max()) + 0.5)
        ax.plot(lim, lim, color=MUTED, linewidth=1.0, zorder=1)
        if loo is not None:
            _scatter(ax, data.y, loo, color=ORANGE, size=18, alpha=0.9, label="скользящий контроль (без вещества)")
        _scatter(ax, data.y, fit.fitted, color=BLUE, label="МНК по всей выборке")
        h = leverage(fit.X)
        exact = h > 1 - 1e-8
        ax.scatter(data.y[exact], fit.fitted[exact], s=90, facecolors="none", edgecolors=INK, linewidths=1.0,
                   zorder=4, label="рычаг h = 1 (подогнаны точно)")
        # Подписи — выносками в пустой правый нижний угол, по порядку прогноза (линии не пересекаются).
        order = np.argsort(-np.abs(fit.residuals))[:label_top]
        order = order[np.argsort(-fit.fitted[order])]
        text_y = np.linspace(1.6, lim[0] + 0.6, len(order))
        for i, ty in zip(order, text_y):
            ax.annotate(f"{data.names[i]} (e = {fit.residuals[i]:+.2f})", (data.y[i], fit.fitted[i]),
                        xytext=(4.2, ty), textcoords="data", fontsize=7, color=INK_2, va="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkB": 3})
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"экспериментальный $\log P$")
        ax.set_ylabel(r"предсказанный $\log P$")
        ax.set_title(f"Модель (2): R² = {fit.r2:.3f}, s = {fit.s:.3f}", loc="left")
        ax.legend(loc="upper left")
    return fig


def plot_residuals(fit: OLSResult, data: LipoData) -> Figure:
    """Четыре диагностических графика остатков."""
    with _style():
        fig = Figure(figsize=(9.0, 7.0), layout="constrained")
        (a, b), (c, d) = fig.subplots(2, 2)
        e = fit.residuals
        h = leverage(fit.X)
        ok = h < 1 - 1e-8

        _scatter(a, fit.fitted[ok], e[ok])
        a.axhline(0.0, color=MUTED, linewidth=1.0)
        for sign in (-2, 2):
            a.axhline(sign * fit.s, color=GRID, linewidth=1.0)
        a.set_xlabel(r"подогнанное $\hat y$")
        a.set_ylabel("остаток e")
        a.set_title("(а) Остатки против прогноза (линии ±2s)", loc="left")

        z = np.sort(studentized_residuals(fit, external=False)[ok])
        q = stats.norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
        _scatter(b, q, z)
        b.plot([-3, 3], [-3, 3], color=MUTED, linewidth=1.0, zorder=1)
        b.set_xlabel("квантили N(0, 1)")
        b.set_ylabel("стьюдентизированный остаток")
        b.set_title("(б) Квантиль-квантиль (нормальность)", loc="left")

        grid = np.linspace(-2.6, 2.6, 200)
        c.hist(e[ok], bins=np.arange(-2.25, 2.0, 0.25), density=True, color=BLUE, edgecolor="white", linewidth=1.0)
        c.plot(grid, stats.norm.pdf(grid, 0.0, fit.s), color=ORANGE, label=f"N(0, s²), s = {fit.s:.2f}")
        c.set_xlabel("остаток e")
        c.set_ylabel("плотность")
        c.set_title("(в) Гистограмма остатков", loc="left")
        c.grid(axis="x", visible=False)
        c.legend(loc="upper left")

        t = studentized_residuals(fit)
        cook = cooks_distance(fit)
        _scatter(d, h[ok], t[ok])
        hh = np.linspace(0.02, 0.98, 200)
        for level in (0.5, 1.0):
            r = np.sqrt(level * fit.K * (1 - hh) / hh)
            d.plot(hh, r, color=GRID, linewidth=1.0, zorder=1)
            d.plot(hh, -r, color=GRID, linewidth=1.0, zorder=1)
            d.text(0.9, np.sqrt(level * fit.K * 0.1 / 0.9) + 0.15, f"D = {level:g}", fontsize=7, color=MUTED)
        top = np.argsort(-np.nan_to_num(cook))[:3]
        for i in top:
            d.annotate(data.names[i], (h[i], t[i]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK_2)
        d.axvline(2 * fit.K / fit.n, color=MUTED, linewidth=1.0)
        d.set_xlim(0, 1.02)
        d.set_ylim(-4, 4)
        d.set_xlabel("рычаг h")
        d.set_ylabel("внешний стьюдентиз. остаток")
        d.set_title("(г) Влиятельные наблюдения (линия: h = 2K/n)", loc="left")
        d.text(0.02, 3.6, f"не показаны {int((~ok).sum())} точки с h = 1 (остаток ≡ 0)", ha="left", va="top",
               fontsize=7, color=MUTED)
    return fig


def plot_coefficients(rows: List[dict], level: float = 0.95) -> Figure:
    """Стандартизованные коэффициенты с доверительными интервалами (из :func:`contribution_table`)."""
    with _style():
        rows = sorted(rows, key=lambda r: r["beta"])
        fig = Figure(figsize=(6.4, 6.6), layout="constrained")
        ax = fig.subplots()
        y = np.arange(len(rows))
        sig = np.array([r["p"] < 1 - level for r in rows])
        for k, r in enumerate(rows):
            color = BLUE if sig[k] else NEUTRAL
            ax.plot([r["beta_low"], r["beta_high"]], [k, k], color=color, linewidth=2.0, solid_capstyle="round")
            ax.scatter([r["beta"]], [k], s=30, color=color if sig[k] else MUTED, edgecolors="white",
                       linewidths=0.8, zorder=3)
        ax.axvline(0.0, color=MUTED, linewidth=1.0)
        ax.set_yticks(y, [r["label"] for r in rows], fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(r"стандартизованный коэффициент $\beta^*_j = b_j\,\sigma_{x_j}/\sigma_y$")
        ax.set_title(f"Коэффициенты и {int(level * 100)}%-е доверительные интервалы", loc="left")
        ax.plot([], [], color=BLUE, linewidth=2.0, label="интервал не содержит 0 (значим)")
        ax.plot([], [], color=NEUTRAL, linewidth=2.0, label="интервал содержит 0")
        ax.legend(loc="lower right")
    return fig


def plot_matched_pairs(pairs: Sequence[MatchedPair], reduced: Optional[Sequence[MatchedPair]] = None) -> Figure:
    """Изменение :math:`\\log P` при замене группы: эксперимент и модель."""
    with _style():
        fig = Figure(figsize=(7.0, 5.6), layout="constrained")
        ax = fig.subplots()
        n = len(pairs)
        y = np.arange(n)[::-1]
        series = [("эксперимент", [p.delta_exp for p in pairs], INK_2),
                  ("модель (2), 24 регрессора", [p.delta_model for p in pairs], BLUE)]
        if reduced is not None:
            series.append(("модель после исключения, 7 регрессоров", [p.delta_model for p in reduced], ORANGE))
        height = 0.8 / len(series)
        for s, (label, values, color) in enumerate(series):
            offset = (len(series) - 1) / 2 - s
            ax.barh(y + offset * height, values, height=height * 0.85, color=color, label=label)
        for k, p in enumerate(pairs):
            if p.exact_fit:
                ax.text(1.95, y[k], "h = 1", va="center", ha="right", fontsize=7, color=MUTED)
        ax.axvline(0.0, color=MUTED, linewidth=1.0)
        ax.set_yticks(y, [f"{p.change}  ({p.child})" for p in pairs], fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-1.5, 2.0)
        ax.set_xlabel(r"$\Delta \log P$ = log P(производное) − log P(исходное)")
        ax.set_title("Пары молекул: эффект замены одной группы", loc="left")
        fig.legend(loc="outside lower center", ncols=len(series))
    return fig


def plot_elimination(steps: Sequence, cv_rmse: Sequence[float], loo_rmse: Sequence[float]) -> Figure:
    """Путь пошагового исключения: качество на обучающей выборке и на контроле."""
    with _style():
        k = np.array([len(s.columns) for s in steps])
        fig = Figure(figsize=(9.0, 3.6), layout="constrained")
        a, b = fig.subplots(1, 2)
        a.plot(k, [s.r2 for s in steps], color=BLUE, marker="o", markersize=4, label="R²")
        a.plot(k, [s.r2_adj for s in steps], color=ORANGE, marker="o", markersize=4, label="скорректированный R²")
        a.set_xlabel("число регрессоров в модели")
        a.set_ylabel("доля объяснённой дисперсии")
        a.set_title("(а) Качество на обучающей выборке", loc="left")
        a.invert_xaxis()
        a.legend(loc="lower left")

        b.plot(k, [s.s for s in steps], color=BLUE, marker="o", markersize=4, label="s (обучающая выборка)")
        b.plot(k, loo_rmse, color=AQUA, marker="o", markersize=4, label="RMSE скользящего контроля")
        b.plot(k, cv_rmse, color=ORANGE, marker="o", markersize=4, label="RMSE 5-кратной проверки")
        b.set_xlabel("число регрессоров в модели")
        b.set_ylabel(r"ошибка, ед. $\log P$")
        b.set_title("(б) Ошибка на обучении и на контроле", loc="left")
        b.invert_xaxis()
        b.legend(loc="upper right")
        for ax in (a, b):
            ax.set_xticks(range(int(k.max()), int(k.min()) - 1, -2))
    return fig


def save(fig: Figure, path) -> None:
    """Сохранить рисунок с параметрами оформления пакета."""
    with _style():
        fig.savefig(path)
