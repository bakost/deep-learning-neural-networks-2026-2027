r"""Рисунки отчёта.

Все функции строят :class:`matplotlib.figure.Figure` через объектный API, не
трогая глобальное состояние ``pyplot``: модуль можно импортировать и из
скрипта без дисплея, и из Jupyter. Сохранение — ``fig.savefig(path)``.

Цвета во всех рисунках одинаковы: гистограмма — голубая, догадка Вигнера
(15) — красная, точный закон GOE при :math:`n \to \infty` — зелёный пунктир,
закон Пуассона :math:`e^{-s}` — серые точки.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from .distributions import (
    GOE2_RAW_MEAN,
    WIGNER_VARIANCE,
    DiscreteDistribution,
    goe2_raw_pdf,
    goe_limit_cdf,
    goe_limit_pdf,
    goe_limit_table,
    normal_mirror_2x2_pdf,
    poisson_cdf,
    poisson_pdf,
    wigner_cdf,
    wigner_pdf,
)
from .experiments import SpacingStudy, pair_scan, pooled_spacings
from .stats import histogram, uniform_bins

__all__ = [
    "plot_2x2_geometry",
    "plot_discrete_ensemble",
    "plot_histogram_grid",
    "plot_ks_vs_size",
    "plot_normal_mirror",
    "plot_pair_choice",
    "plot_residuals",
    "plot_small_s",
]

HIST_FACE = "#a6cee3"
HIST_EDGE = "#3b7fb6"
WIGNER = "#d62728"
GOE_LIMIT = "#2ca02c"
POISSON = "#7f7f7f"
EXACT = "#ff7f0e"
SIZE_COLORS = {2: "#1f77b4", 4: "#9467bd", 16: "#e377c2"}

_S = np.linspace(0.0, 4.0, 801)


def _goe_limit_variance() -> float:
    grid, density, _ = goe_limit_table()
    step = grid[1] - grid[0]
    return float(np.sum(grid**2 * density) * step - 1.0)


def _format_size(m: int) -> str:
    return f"{m:,}".replace(",", " ")


def _pair_label(study: SpacingStudy) -> str:
    return f"$n = {study.n}$, пара $(\\lambda_{{{study.k}}}, \\lambda_{{{study.k + 1}}})$"


def _draw_histogram(ax, samples: np.ndarray, edges: np.ndarray, label: str = "гистограмма"):
    hist = histogram(samples, edges)
    ax.bar(hist.edges[:-1], hist.density, width=hist.widths, align="edge",
           color=HIST_FACE, edgecolor=HIST_EDGE, linewidth=0.4, label=label)
    return hist


def _draw_references(ax, poisson: bool = True, goe_limit: bool = False) -> None:
    ax.plot(_S, wigner_pdf(_S), color=WIGNER, lw=2.0, label="догадка Вигнера (15)")
    if goe_limit:
        ax.plot(_S, goe_limit_pdf(_S), color=GOE_LIMIT, lw=1.6, ls="--",
                label="точный GOE, $n\\to\\infty$")
    if poisson:
        ax.plot(_S, poisson_pdf(_S), color=POISSON, lw=1.2, ls=":", label="Пуассон $e^{-s}$")


def _figure_legend(fig, axes) -> None:
    """Одна общая легенда под рисунком: без повторов и без наложения на данные."""
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=9,
               frameon=False)


def _decorate(ax, xlabel: bool = True, ylabel: bool = True) -> None:
    ax.set_xlim(0.0, 3.6)
    ax.grid(alpha=0.25)
    if xlabel:
        ax.set_xlabel("нормированное расстояние $s$")
    if ylabel:
        ax.set_ylabel("плотность $\\rho(s)$")


def plot_2x2_geometry(matrices: np.ndarray, s: float = 3.0, delta: float = 1.0,
                      points: int = 4000) -> Figure:
    r"""Рисунок к разделу 1.2–1.3: точки :math:`(d, b)` и распределение (13).

    Слева — облако точек :math:`(d, b)` для ансамбля 2×2 и кольцо рис. 1
    методички. Плотность точек максимальна в нуле, но расстояние
    :math:`s = 2\sqrt{b^2 + d^2}` — это удвоенный *радиус*, и доля точек в
    кольце радиуса :math:`s/2` пропорциональна его площади
    :math:`\tfrac12 \pi s \Delta`, поэтому малые :math:`s` редки.
    """
    b = matrices[:, 0, 1]
    d = 0.5 * (matrices[:, 1, 1] - matrices[:, 0, 0])
    raw = 2.0 * np.hypot(b, d)

    fig = Figure(figsize=(11.0, 4.6), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.25]})

    shown_b, shown_d, shown_s = b[:points], d[:points], raw[:points]
    ring = (shown_s >= s) & (shown_s < s + delta)
    ax1.scatter(shown_d[~ring], shown_b[~ring], s=3, color=HIST_EDGE, alpha=0.35, lw=0)
    ax1.scatter(shown_d[ring], shown_b[ring], s=4, color=EXACT, alpha=0.8, lw=0,
                label=f"${s:g} \\leq s < {s + delta:g}$: {ring.mean():.1%} точек")
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    for radius in (s / 2.0, (s + delta) / 2.0):
        ax1.plot(radius * np.cos(angle), radius * np.sin(angle), color="k", lw=1.0)
    ax1.set_aspect("equal")
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axvline(0, color="k", lw=0.5)
    ax1.set_xlabel("$d = (c - a)/2$")
    ax1.set_ylabel("$b$")
    ax1.set_title(f"(а) точки $(d, b)$, первые {_format_size(points)} матриц")
    ax1.legend(loc="upper right", fontsize=8)

    edges = uniform_bins(0.25, 14.0)
    _draw_histogram(ax2, raw, edges, label=f"гистограмма, $M = {_format_size(raw.size)}$")
    grid = np.linspace(0.0, 14.0, 600)
    ax2.plot(grid, goe2_raw_pdf(grid), color=WIGNER, lw=2.0,
             label="(13): $\\frac{s}{8}\\,e^{-s^2/16}$")
    ax2.axvline(GOE2_RAW_MEAN, color="k", ls="--", lw=1.0,
                label=f"(14): $2\\sqrt{{\\pi}} = {GOE2_RAW_MEAN:.4f}$")
    ax2.axvline(raw.mean(), color=EXACT, ls=":", lw=1.6,
                label=f"выборочное среднее ${raw.mean():.4f}$")
    ax2.set_xlim(0, 14)
    ax2.set_xlabel("$s = \\lambda_2 - \\lambda_1$ (до нормировки)")
    ax2.set_ylabel("плотность")
    ax2.set_title("(б) распределение расстояния до нормировки, $\\sigma = 1$")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    return fig


def plot_histogram_grid(rows: Sequence[Sequence[SpacingStudy]], widths: Sequence[float],
                        title: Optional[str] = None, goe_limit: bool = False) -> Figure:
    """Сетка гистограмм (пункты 5 и 7): строки — объёмы ансамбля, столбцы — размеры матриц."""
    nrows, ncols = len(rows), len(rows[0])
    fig = Figure(figsize=(4.2 * ncols, 3.5 * nrows), layout="constrained")
    axes = np.atleast_2d(fig.subplots(nrows, ncols, sharex=True, sharey=True))
    for r, (row, width) in enumerate(zip(rows, widths)):
        for c, study in enumerate(row):
            ax = axes[r, c]
            _draw_histogram(ax, study.normalized, uniform_bins(width, 4.0))
            _draw_references(ax, goe_limit=goe_limit)
            d = study.compare("wigner")["ks_d"]
            ax.text(0.97, 0.96,
                    f"{_pair_label(study)}\n$M = {_format_size(study.size)}$\n$D_{{KS}} = {d:.4f}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": 0.9})
            _decorate(ax, xlabel=r == nrows - 1, ylabel=c == 0)
            ax.set_ylim(0, 1.05)
    _figure_legend(fig, axes.ravel())
    if title:
        fig.suptitle(title)
    return fig


def plot_small_s(studies: Sequence[SpacingStudy]) -> Figure:
    r"""Пункт 6: поведение около нуля.

    (а) начало гистограмм в линейном масштабе: плотность растёт от нуля
    линейно, :math:`\rho(s) \approx \frac{\pi}{2}s`, а не стартует с 1, как у
    Пуассона; (б) функция распределения в логарифмических осях: наклон 2
    означает :math:`F(s) \propto s^2`, то есть :math:`\rho(s) \propto s`.
    """
    fig = Figure(figsize=(11.0, 4.4), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)

    edges = uniform_bins(0.04, 0.8)
    for study in studies:
        hist = histogram(study.normalized, edges)
        color = SIZE_COLORS.get(study.n, None)
        ax1.errorbar(hist.centers, hist.density, yerr=hist.errors, fmt="o", ms=3.5,
                     color=color, capsize=2, lw=0.8, label=f"$n = {study.n}$")
    grid = np.linspace(0.0, 0.8, 200)
    ax1.plot(grid, wigner_pdf(grid), color=WIGNER, lw=2.0, label="догадка Вигнера (15)")
    ax1.plot(grid, 0.5 * np.pi * grid, color=WIGNER, lw=1.0, ls="--",
             label="касательная $\\frac{\\pi}{2}s$")
    ax1.plot(grid, poisson_pdf(grid), color=POISSON, lw=1.2, ls=":", label="Пуассон $e^{-s}$")
    ax1.set_xlim(0, 0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("нормированное расстояние $s$")
    ax1.set_ylabel("плотность $\\rho(s)$")
    ax1.set_title("(а) окрестность нуля: линейный рост")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="lower right")

    grid = np.geomspace(1e-3, 2.0, 200)
    for study in studies:
        values = np.sort(study.normalized)
        ecdf = np.searchsorted(values, grid, side="right") / values.size
        mask = ecdf > 0
        ax2.loglog(grid[mask], ecdf[mask], color=SIZE_COLORS.get(study.n), lw=1.6,
                   label=f"$n = {study.n}$, $M = {_format_size(study.size)}$")
    ax2.loglog(grid, wigner_cdf(grid), color=WIGNER, lw=1.0, ls="--",
               label="Вигнер: $F \\approx \\frac{\\pi}{4}s^2$ (наклон 2)")
    ax2.loglog(grid, poisson_cdf(grid), color=POISSON, lw=1.2, ls=":",
               label="Пуассон: $F \\approx s$ (наклон 1)")
    ax2.axhline(1.0 / studies[0].size, color="k", lw=0.6, ls="-.")
    ax2.text(1.1e-3, 1.4 / studies[0].size, "$1/M$ — одна точка выборки", fontsize=8)
    ax2.set_xlim(1e-3, 2.0)
    ax2.set_ylim(0.3 / studies[0].size, 1.5)
    ax2.set_xlabel("$s$")
    ax2.set_ylabel("$F(s) = P(\\,s_i < s\\,)$")
    ax2.set_title("(б) функция распределения, логарифмические оси")
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=8, loc="lower right")
    return fig


def plot_residuals(studies: Sequence[SpacingStudy], width: float = 0.1) -> Figure:
    r"""Пункт 8: разность «гистограмма минус догадка Вигнера» с погрешностями.

    Если бы догадка Вигнера была точной, точки разбросались бы около нуля в
    пределах погрешностей. Систематическое отклонение сравнивается с
    разностью «точный закон GOE при :math:`n\to\infty` минус догадка Вигнера».
    """
    fig = Figure(figsize=(4.2 * len(studies), 3.6), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(studies), sharey=True))
    edges = uniform_bins(width, 3.5)
    for ax, study in zip(axes, studies):
        hist = histogram(study.normalized, edges)
        residual = hist.density - hist.expected_density(wigner_cdf)
        ax.errorbar(hist.centers, residual, yerr=hist.errors, fmt="o", ms=3, capsize=2,
                    color=SIZE_COLORS.get(study.n, HIST_EDGE), lw=0.8,
                    label="гистограмма − Вигнер")
        exact_residual = hist.expected_density(goe_limit_cdf) - hist.expected_density(wigner_cdf)
        ax.plot(hist.centers, exact_residual, color=GOE_LIMIT, lw=1.6, ls="--",
                label="точный GOE ($n\\to\\infty$) − Вигнер")
        ax.axhline(0.0, color=WIGNER, lw=1.2)
        chi_w = np.sum((residual / np.maximum(hist.errors, 1e-12)) ** 2)
        ax.set_title(f"$n = {study.n}$, $M = {_format_size(study.size)}$, "
                     f"$\\chi^2/N_{{bin}} = {chi_w / residual.size:.1f}$", fontsize=10)
        ax.set_xlabel("$s$")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 3.5)
    axes[0].set_ylabel("$h(s) - \\rho_{W}(s)$")
    axes[0].legend(fontsize=8, loc="lower left")
    return fig


def plot_ks_vs_size(data: Mapping[int, Mapping[int, np.ndarray]], plateau: float,
                    noise: float) -> Figure:
    r"""Пункт 8: расстояние Колмогорова–Смирнова до догадки Вигнера в зависимости от :math:`M`.

    ``data[n][M]`` — значения :math:`D` по независимым блокам; ``plateau`` —
    :math:`\sup_s |F_{GOE,\infty}(s) - F_W(s)|`, собственная погрешность
    догадки Вигнера; ``noise`` — константа :math:`c` в уровне шума
    :math:`c/\sqrt{M}` (см. :func:`~level_spacing.experiments.ks_noise_constant`).
    """
    fig = Figure(figsize=(7.2, 4.6), layout="constrained")
    ax = fig.subplots()
    all_sizes: List[int] = []
    for n, by_size in data.items():
        sizes = np.array(sorted(by_size))
        all_sizes.extend(sizes)
        median = np.array([np.median(by_size[m]) for m in sizes])
        low = np.array([np.percentile(by_size[m], 25) for m in sizes])
        high = np.array([np.percentile(by_size[m], 75) for m in sizes])
        color = SIZE_COLORS.get(n)
        ax.fill_between(sizes, low, high, color=color, alpha=0.15, lw=0)
        ax.loglog(sizes, median, "o-", color=color, lw=1.6, ms=4, label=f"$n = {n}$")
    grid = np.geomspace(min(all_sizes), max(all_sizes), 50)
    ax.loglog(grid, noise / np.sqrt(grid), color="k", ls="--", lw=1.0,
              label=f"${noise:.2f}/\\sqrt{{M}}$ — шум, если бы закон (15) был точным")
    ax.axhline(plateau, color=GOE_LIMIT, ls=":", lw=1.4,
               label=f"$\\sup|F_{{GOE,\\infty}} - F_W| = {plateau:.4f}$")
    ax.set_xlabel("число матриц в ансамбле $M$")
    ax.set_ylabel("$D_{KS}$ до догадки Вигнера")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5)
    ax.set_title("Медиана по независимым ансамблям, полоса — квартили")
    return fig


def _annotate_clipped(ax, hist, top: float) -> None:
    for i in np.flatnonzero(hist.density > top):
        mass = hist.counts[i] / hist.total
        ax.annotate(f"$P = {mass:.3f}$ ↑", xy=(hist.edges[i + 1], top), xytext=(2, -4),
                    textcoords="offset points", ha="left", va="top", fontsize=8, rotation=90,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})


def plot_discrete_ensemble(studies: Sequence[SpacingStudy],
                           exact: Mapping[int, DiscreteDistribution],
                           width: float = 0.05, top: float = 4.0) -> Figure:
    r"""Пункт 9: гистограммы для матриц из ±1 и точный закон (где перебор возможен).

    Для малых :math:`n` распределение дискретно, и столбцы гистограммы — это
    отдельные атомы высотой :math:`p/\Delta`. Оранжевый контур — точный закон
    из полного перебора, разнесённый по тем же бинам: если генератор и
    вычисления верны, столбцы должны совпасть с контуром. Столбцы выше ``top``
    обрезаны, рядом подписана их вероятность.
    """
    fig = Figure(figsize=(4.2 * len(studies), 4.0), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(studies)))
    edges = uniform_bins(width, 4.0)
    for ax, study in zip(axes, studies):
        hist = _draw_histogram(ax, study.normalized, edges)
        _draw_references(ax, goe_limit=study.n >= 16)
        law = exact.get(study.n)
        if law is not None:
            mass, _ = np.histogram(law.values / study.raw_mean, bins=edges,
                                   weights=law.probabilities)
            expected = mass / hist.widths
            ax.stairs(np.minimum(expected, top * 1.02), edges, color=EXACT, lw=1.3,
                      label="точный закон (полный перебор)")
            _annotate_clipped(ax, hist, top)
            ax.set_ylim(0, top)
        else:
            ax.set_ylim(0, 1.05)
        degenerate = np.mean(study.raw <= 1e-9)
        ax.text(0.97, 0.96,
                f"{_pair_label(study)}\n$M = {_format_size(study.size)}$\n"
                f"$P(s = 0) = {degenerate:.4f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": 0.9})
        _decorate(ax, ylabel=ax is axes[0])
    _figure_legend(fig, axes)
    fig.suptitle(studies[0].ensemble.title)
    return fig


def plot_pair_choice(eigvals: np.ndarray, radius: float) -> Figure:
    r"""Сноски 2 и 4 методички: почему фиксируется одна пара соседних уровней.

    (а) плотность уровней и закон полукруга радиуса ``radius``;
    (б) среднее расстояние :math:`\langle\lambda_{k+1} - \lambda_k\rangle` для
    каждой пары; (в) дисперсия нормированного расстояния для каждой пары;
    (г) все расстояния одной выборкой: с общей нормировкой и после
    нормировки каждой пары на своё среднее.
    """
    n = eigvals.shape[1]
    scan = pair_scan(eigvals)
    fig = Figure(figsize=(11.0, 8.0), layout="constrained")
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

    lam = np.linspace(-radius, radius, 400)
    ax1.hist(eigvals.ravel(), bins=120, density=True, color=HIST_FACE, edgecolor=HIST_EDGE,
             lw=0.3, label="все собственные числа")
    ax1.plot(lam, 2.0 / (np.pi * radius**2) * np.sqrt(radius**2 - lam**2), color=WIGNER, lw=2.0,
             label=f"полукруг, $R = 2\\sqrt{{2n}} = {radius:.2f}$")
    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("плотность уровней")
    ax1.set_title(f"(а) спектр GOE ${n}\\times{n}$: уровни гуще в центре")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.bar(scan["k"], scan["mean"], color=HIST_FACE, edgecolor=HIST_EDGE)
    ax2.set_xlabel("номер пары $k$")
    ax2.set_ylabel("$\\langle \\lambda_{k+1} - \\lambda_k \\rangle$")
    ax2.set_title("(б) среднее расстояние зависит от $k$")
    ax2.set_xticks(scan["k"])
    ax2.grid(alpha=0.25, axis="y")

    # Погрешность выборочной дисперсии ≈ σ² √(2/M) с поправкой на эксцесс; для
    # этих законов μ4/σ⁴ ≈ 3, поэтому оценка σ²√(2/M) достаточно точна.
    errors = scan["variance"] * np.sqrt(2.0 / eigvals.shape[0])
    ax3.errorbar(scan["k"], scan["variance"], yerr=errors, fmt="o", ms=5, color=HIST_EDGE,
                 ecolor=HIST_EDGE, capsize=3, zorder=3)
    ax3.plot([n // 2], [scan["variance"][n // 2 - 1]], "o", ms=7, color=WIGNER, zorder=4,
             label=f"центральная пара $k = {n // 2}$")
    ax3.axhline(WIGNER_VARIANCE, color=WIGNER, lw=1.4, label="догадка Вигнера: 0.273")
    ax3.axhline(_goe_limit_variance(), color=GOE_LIMIT, lw=1.4, ls="--",
                label="точный GOE, $n\\to\\infty$: 0.286")
    ax3.set_xlabel("номер пары $k$")
    ax3.set_ylabel("дисперсия $s_k / \\langle s_k \\rangle$")
    ax3.set_title("(в) форма распределения почти не зависит от $k$")
    ax3.set_xticks(scan["k"])
    ax3.set_ylim(0.265, 0.30)
    ax3.legend(fontsize=8, loc="upper center")
    ax3.grid(alpha=0.25, axis="y")

    edges = uniform_bins(0.05, 4.0)
    pooled = histogram(pooled_spacings(eigvals, unfold=False), edges)
    unfolded = histogram(pooled_spacings(eigvals, unfold=True), edges)
    ax4.stairs(pooled.density, edges, color=EXACT, lw=1.6, label="общая нормировка")
    ax4.stairs(unfolded.density, edges, color=HIST_EDGE, lw=1.6,
               label="каждая пара на своё среднее")
    ax4.plot(_S, wigner_pdf(_S), color=WIGNER, lw=2.0, label="догадка Вигнера (15)")
    ax4.set_xlim(0, 3.6)
    ax4.set_xlabel("нормированное расстояние $s$")
    ax4.set_ylabel("плотность")
    ax4.set_title("(г) все пары одной выборкой")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25)
    return fig


def plot_normal_mirror(studies: Sequence[SpacingStudy], goe_2x2: SpacingStudy) -> Figure:
    r"""Дополнение: симметричная матрица с элементами N(0, 1) без удвоения диагонали.

    Для 2×2 догадка Вигнера перестаёт быть точной — гистограмма следует
    формуле с функцией Бесселя :func:`~level_spacing.distributions.normal_mirror_2x2_pdf`;
    для 16×16 различие исчезает (универсальность).
    """
    fig = Figure(figsize=(4.2 * (len(studies) + 1), 3.8), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(studies) + 1))
    edges = uniform_bins(0.1, 3.5)

    ax = axes[0]
    for study, color, label in ((goe_2x2, HIST_EDGE, "GOE ($A + A^T$)"),
                                (studies[0], EXACT, "одинаковые дисперсии")):
        hist = histogram(study.normalized, edges)
        ax.errorbar(hist.centers, hist.density - hist.expected_density(wigner_cdf),
                    yerr=hist.errors, fmt="o", ms=3, capsize=2, lw=0.8, color=color,
                    label=f"{label}: гистограмма − Вигнер")
    grid = np.linspace(0.0, 3.5, 300)
    ax.plot(grid, normal_mirror_2x2_pdf(grid) - wigner_pdf(grid), color=EXACT, lw=1.4,
            ls="--", label="формула с $I_0$ − Вигнер")
    ax.axhline(0.0, color=WIGNER, lw=1.2)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$h(s) - \\rho_W(s)$")
    ax.set_title("(а) $2\\times2$: отклонение от догадки Вигнера")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 3.5)

    for ax, study in zip(axes[1:], studies):
        _draw_histogram(ax, study.normalized, uniform_bins(0.05, 4.0))
        _draw_references(ax, poisson=False, goe_limit=study.n > 2)
        if study.n == 2:
            ax.plot(_S, normal_mirror_2x2_pdf(_S), color=EXACT, lw=1.6, ls="--",
                    label="формула с $I_0$")
        ax.text(0.97, 0.96, f"{_pair_label(study)}\n$M = {_format_size(study.size)}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": 0.9})
        _decorate(ax)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7.5, loc="center right")
    fig.suptitle(studies[0].ensemble.title)
    return fig
