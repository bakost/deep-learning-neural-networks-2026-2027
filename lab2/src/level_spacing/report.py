r"""Пересчёт всех чисел и рисунков отчёта одной командой.

.. code-block:: bash

    python -m level_spacing report            # ≈ 1 минута
    python -m level_spacing report --quick    # уменьшенные ансамбли, для проверки

Создаёт ``figures/*.png``, ``results/summary.json`` (все числа) и
``results/tables.md`` (таблицы в формате Markdown, из которых собран
``REPORT.md``). Все генераторы инициализируются от одного зерна
:data:`~level_spacing.experiments.DEFAULT_SEED`, поэтому повторный запуск
даёт те же числа до последнего знака.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .distributions import (
    GOE2_RAW_MEAN,
    NORMAL_MIRROR_2X2_RAW_MEAN,
    WIGNER_VARIANCE,
    goe_limit_table,
    pm1_2x2_distribution,
    wigner_cdf,
)
from .ensembles import get_ensemble
from .exact import exact_pm1_spacing
from .experiments import (
    DEFAULT_SEED,
    ks_by_ensemble_size,
    ks_noise_constant,
    orthogonal_invariance,
    pair_scan,
    pooled_spacings,
    run_study,
    study_rng,
    variance_vs_angle,
)
from .plots import (
    plot_2x2_geometry,
    plot_discrete_ensemble,
    plot_histogram_grid,
    plot_ks_vs_size,
    plot_normal_mirror,
    plot_orthogonal_invariance,
    plot_pair_choice,
    plot_residuals,
    plot_small_s,
)
from .spectrum import eigenvalues, pair_spacings, simulate_eigenvalues
from .stats import ks_statistic

__all__ = ["ReportConfig", "generate_report"]

SIZES = (2, 4, 16)


@dataclass(frozen=True)
class ReportConfig:
    """Объёмы ансамблей для отчёта."""

    size: int = 100_000          # основной ансамбль (пункт 1: «1000 или более»)
    minimal_size: int = 1_000    # минимально допустимый по заданию
    big_size: int = 1_000_000    # для остатков и зависимости D(M)
    jacobi_size: int = 1_000     # проверка методом Якоби из ЛР1
    seed: int = DEFAULT_SEED

    @classmethod
    def quick(cls) -> "ReportConfig":
        return cls(size=20_000, minimal_size=1_000, big_size=100_000, jacobi_size=100)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (value != 0.0 and abs(value) < 10 ** (-digits)):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def _fmt_p(value: float, floor: float = 0.0) -> str:
    """«p = …» или «p < …»; ``floor`` — наименьшее p, которое метод вообще может дать."""
    if floor > 0.0 and value <= floor * 1.0001:
        return f"p < {floor:.3f}"
    if value < 1e-10:
        return "p < 1e-10"
    if value < 0.001:
        return f"p = {value:.1e}"
    return f"p = {value:.3f}" if value < 0.01 else f"p = {value:.2f}"


_KS_FLOOR = 1.0 / 1001.0   # 1000 повторов в ks_null_distribution


def _study_row(summary: Dict[str, object]) -> str:
    s = summary
    size = f"{s['size']:,}".replace(",", " ")
    return (
        f"| {s['n']} | ({s['k']}, {s['k'] + 1}) | {size} "
        f"| {_fmt(s['raw_mean'])} ± {_fmt(s['raw_mean_error'])} "
        f"| {_fmt(s['variance'])} ± {_fmt(s['variance_error'])} "
        f"| {_fmt(s['wigner_ks_d'])} ({_fmt_p(s['wigner_ks_p'], _KS_FLOOR)}) "
        f"| {s['wigner_chi2']:.0f} / {s['wigner_chi2_dof']} ({_fmt_p(s['wigner_chi2_p'])}) "
        f"| {_fmt(s['wigner_l1'], 3)} "
        f"| {_fmt(s['goe-limit_ks_d'])} ({_fmt_p(s['goe-limit_ks_p'], _KS_FLOOR)}) "
        f"| {_fmt(s['small_s_exponent'], 2)} |"
    )


_STUDY_HEADER = (
    "| n | пара | M | ⟨s⟩ до нормировки | дисперсия | D_KS до Вигнера | χ² до Вигнера / ст. св. "
    "| L1 до Вигнера | D_KS до GOE∞ | γ |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def _study_table(summaries: Sequence[Dict[str, object]], extra: Optional[str] = None) -> str:
    rows = [_STUDY_HEADER] + [_study_row(s) for s in summaries]
    if extra:
        rows.append(extra)
    return "\n".join(rows)


def _atoms_table(law, digits: int = 4) -> str:
    normalized = law.normalized()
    lines = ["| s (до нормировки) | s / ⟨s⟩ | вероятность |", "|---|---|---|"]
    for raw, value, p in zip(law.values, normalized.values, law.probabilities):
        lines.append(f"| {raw:.{digits}f} | {value:.{digits}f} | {p:.6g} |")
    return "\n".join(lines)


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=150)
    return path.name


def generate_report(
    root: Path = Path("."),
    config: Optional[ReportConfig] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Посчитать всё для отчёта и сохранить в ``root/figures`` и ``root/results``."""
    config = ReportConfig() if config is None else config
    root = Path(root)
    figures = root / "figures"
    results_dir = root / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    started = time.perf_counter()
    results: Dict[str, object] = {"config": config.__dict__, "figures": []}
    tables: List[str] = ["# Таблицы для отчёта (сгенерировано `python -m level_spacing report`)\n"]

    def step(message: str) -> None:
        log(f"[{time.perf_counter() - started:6.1f} с] {message}")

    # --- Ортогональность ансамбля ------------------------------------------------
    step("ортогональность ансамбля: Q^T H Q против H")
    invariance_names = ("goe", "pm1-sum", "normal-mirror", "pm1")
    angles = np.radians(np.arange(0.0, 90.1, 7.5))
    angle_data = {name: variance_vs_angle(name, angles, config.size, seed) for name in invariance_names}
    checks = {n: {name: orthogonal_invariance(name, n, config.size, seed=seed) for name in invariance_names}
              for n in (4, 16)}
    results["orthogonality"] = {str(n): value for n, value in checks.items()}
    results["figures"].append(_save(plot_orthogonal_invariance(angle_data, checks[16]),
                                    figures / "orthogonal_invariance.png"))
    lines = ["| n | ансамбль | Var диагонали до → после | Var вне диагонали до → после "
             "| KS H11 до/после | KS H12 до/после | ‖QᵀQ − I‖ | расхождение спектров |",
             "|---|---|---|---|---|---|---|---|"]
    for n, by_name in checks.items():
        for name, c in by_name.items():
            lines.append(
                f"| {n} | {name} | {c['diag_var_before']:.3f} → {c['diag_var_after']:.3f} "
                f"| {c['off_var_before']:.3f} → {c['off_var_after']:.3f} "
                f"| {_fmt_p(c['ks_diag_p'])} | {_fmt_p(c['ks_off_p'])} "
                f"| {c['orthogonality_error']:.1e} | {c['spectrum_error']:.1e} |")
    tables.append("## Ортогональность ансамблей\n\n" + "\n".join(lines) + "\n")

    # --- 2×2: геометрия и формулы (13)–(14) ----------------------------------
    step("GOE 2×2: проверка формул (13)–(14)")
    goe2_matrices = get_ensemble("goe").sample(2, config.size, study_rng(seed, "goe", 2))
    results["figures"].append(_save(plot_2x2_geometry(goe2_matrices), figures / "geometry_2x2.png"))
    raw2 = pair_spacings(eigenvalues(goe2_matrices))
    b = goe2_matrices[:, 0, 1]
    d = 0.5 * (goe2_matrices[:, 1, 1] - goe2_matrices[:, 0, 0])
    results["goe2_check"] = {
        "raw_mean": float(raw2.mean()),
        "raw_mean_error": float(raw2.std() / np.sqrt(raw2.size)),
        "theory_mean": GOE2_RAW_MEAN,
        "var_b": float(b.var()),
        "var_d": float(d.var()),
        "var_a": float(goe2_matrices[:, 0, 0].var()),
        "formula5_max_error": float(np.max(np.abs(raw2 - 2.0 * np.hypot(b, d)))),
    }

    # --- Пункты 1–8 для GOE ----------------------------------------------------
    step("GOE: основные ансамбли")
    main = {n: run_study("goe", n, config.size, seed) for n in SIZES}
    minimal = {n: run_study("goe", n, config.minimal_size, seed + 1) for n in SIZES}
    results["goe_main"] = [main[n].summary() for n in SIZES]
    results["goe_minimal"] = [minimal[n].summary() for n in SIZES]
    results["figures"].append(_save(
        plot_histogram_grid([[minimal[n] for n in SIZES], [main[n] for n in SIZES]],
                            widths=(0.2, 0.05),
                            title="Гауссов ортогональный ансамбль: гистограммы и догадка Вигнера"),
        figures / "goe_histograms.png"))
    results["figures"].append(_save(plot_small_s([main[n] for n in SIZES]),
                                    figures / "small_s.png"))
    tables.append("## GOE, основной ансамбль\n\n" + _study_table(results["goe_main"]) + "\n")
    tables.append("## GOE, минимальный ансамбль\n\n" + _study_table(results["goe_minimal"]) + "\n")

    step("GOE: большие ансамбли для остатков и D(M)")
    big = {n: run_study("goe", n, config.big_size, seed + 2) for n in SIZES}
    results["goe_big"] = [big[n].summary() for n in SIZES]
    tables.append("## GOE, большой ансамбль\n\n" + _study_table(results["goe_big"]) + "\n")
    results["figures"].append(_save(plot_residuals([big[n] for n in SIZES]),
                                    figures / "residuals.png"))
    grid, _, cdf_limit = goe_limit_table()
    plateau = float(np.max(np.abs(cdf_limit - wigner_cdf(grid))))
    noise = ks_noise_constant()
    results["ks_noise_constant"] = noise
    sizes = [m for m in (300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
             if m <= config.big_size]
    ks_data = {n: ks_by_ensemble_size(big[n].raw, sizes) for n in SIZES}
    results["ks_vs_size"] = {
        str(n): {str(m): float(np.median(v)) for m, v in by_size.items()}
        for n, by_size in ks_data.items()
    }
    results["wigner_vs_goe_limit_sup"] = plateau
    results["figures"].append(_save(plot_ks_vs_size(ks_data, plateau, noise),
                                    figures / "ks_vs_size.png"))

    # --- Выбор пары k -----------------------------------------------------------
    step("GOE 16×16: все пары k")
    ev16 = simulate_eigenvalues("goe", 16, config.size, study_rng(seed + 3, "goe", 16))
    radius = 2.0 * np.sqrt(2.0 * 16)
    results["figures"].append(_save(plot_pair_choice(ev16, radius), figures / "pair_choice.png"))
    scan = pair_scan(ev16)
    results["pair_scan"] = {key: value.tolist() for key, value in scan.items()}
    results["pooled"] = {
        "global_ks_d": ks_statistic(pooled_spacings(ev16, unfold=False), wigner_cdf),
        "unfolded_ks_d": ks_statistic(pooled_spacings(ev16, unfold=True), wigner_cdf),
        "global_variance": float(pooled_spacings(ev16, unfold=False).var()),
        "unfolded_variance": float(pooled_spacings(ev16, unfold=True).var()),
    }
    lines = ["| k | ⟨s_k⟩ | дисперсия s_k/⟨s_k⟩ | D_KS до Вигнера |", "|---|---|---|---|"]
    for k, mean, var, dks in zip(scan["k"], scan["mean"], scan["variance"], scan["ks_d"]):
        lines.append(f"| {k} | {mean:.4f} | {var:.4f} | {dks:.4f} |")
    tables.append("## GOE 16×16, все пары\n\n" + "\n".join(lines) + "\n")

    # --- Пункт 9: матрицы из ±1 --------------------------------------------------
    exact_laws: Dict[str, Dict[int, object]] = {}
    for name, construction, figure_name in (("pm1", "mirror", "pm1.png"),
                                             ("pm1-sum", "sum", "pm1_sum.png")):
        step(f"±1 ({name}): ансамбли и полный перебор")
        studies = {n: run_study(name, n, config.size, seed) for n in SIZES}
        results[f"{name}_main"] = [studies[n].summary() for n in SIZES]
        laws = {n: exact_pm1_spacing(n, construction) for n in (2, 4)}
        exact_laws[name] = laws
        results[f"{name}_exact"] = {
            str(n): {
                "values": law.values.tolist(),
                "normalized": law.normalized().values.tolist(),
                "probabilities": law.probabilities.tolist(),
                "mean": law.mean,
                "variance_normalized": law.normalized().variance,
            }
            for n, law in laws.items()
        }
        results["figures"].append(_save(
            plot_discrete_ensemble([studies[n] for n in SIZES], laws), figures / figure_name))
        tables.append(f"## {name}: Монте-Карло\n\n" + _study_table(results[f"{name}_main"]) + "\n")
        degenerate = " | ".join(f"n={s['n']}: {s['degenerate_fraction']:.5f}"
                                for s in results[f"{name}_main"])
        tables.append(f"Доля вырожденных пар P(s = 0): {degenerate}\n")
        for n, law in laws.items():
            tables.append(f"### {name}: точный закон, n = {n} (перебор)\n\n" + _atoms_table(law) + "\n")

        # Пункт 10: аналитика 2×2 против перебора и Монте-Карло.
        analytic = pm1_2x2_distribution(construction)
        mc = studies[2].raw
        rows = ["| s | P аналитически | P перебором | частота Монте-Карло |", "|---|---|---|---|"]
        for value, p in zip(analytic.values, analytic.probabilities):
            rows.append(
                f"| {value:.6f} | {p:.4f} | {laws[2].probability_of(value, 1e-9):.4f} "
                f"| {np.mean(np.abs(mc - value) < 1e-9):.4f} |"
            )
        tables.append(f"### {name}: 2×2, аналитика против перебора и Монте-Карло\n\n"
                      + "\n".join(rows) + "\n")
        results[f"{name}_2x2_analytic"] = {
            "values": analytic.values.tolist(),
            "probabilities": analytic.probabilities.tolist(),
            "mean": analytic.mean,
            "normalized": analytic.normalized().values.tolist(),
            "normalized_variance": analytic.normalized().variance,
            "mc_frequencies": [float(np.mean(np.abs(mc - v) < 1e-9)) for v in analytic.values],
        }

    # --- Дополнение: почему A + Aᵀ ------------------------------------------------
    step("normal-mirror: симметричная матрица без удвоения диагонали")
    mirror = {n: run_study("normal-mirror", n, config.size, seed) for n in (2, 16)}
    results["normal_mirror"] = [mirror[n].summary() for n in (2, 16)]
    results["normal_mirror_theory"] = {"raw_mean": NORMAL_MIRROR_2X2_RAW_MEAN}
    results["figures"].append(_save(plot_normal_mirror([mirror[2], mirror[16]], main[2]),
                                    figures / "normal_mirror.png"))
    tables.append("## normal-mirror\n\n" + _study_table(results["normal_mirror"]) + "\n")

    # --- Проверка методом Якоби из ЛР1 --------------------------------------------
    step("проверка собственных чисел методом Якоби (ЛР1)")
    try:
        check = {}
        for n in SIZES:
            matrices = get_ensemble("goe").sample(n, config.jacobi_size, study_rng(seed + 4, "goe", n))
            lapack = eigenvalues(matrices, backend="numpy")
            jacobi = eigenvalues(matrices, backend="jacobi")
            check[str(n)] = {
                "max_abs_diff": float(np.max(np.abs(lapack - jacobi))),
                "ks_d_jacobi": ks_statistic(pair_spacings(jacobi) / pair_spacings(jacobi).mean(),
                                            wigner_cdf),
            }
        results["jacobi_check"] = check
    except ImportError as exc:
        results["jacobi_check"] = {"skipped": str(exc)}

    results["theory"] = {"wigner_variance": WIGNER_VARIANCE,
                         "goe_limit_variance": _goe_limit_variance()}
    results["elapsed_seconds"] = time.perf_counter() - started
    (results_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (results_dir / "tables.md").write_text("\n".join(tables), encoding="utf-8")
    step(f"готово: {len(results['figures'])} рисунков, results/summary.json, results/tables.md")
    return results


def _goe_limit_variance() -> float:
    grid, density, _ = goe_limit_table()
    step = grid[1] - grid[0]
    return float(np.sum(grid**2 * density) * step - (np.sum(grid * density) * step) ** 2)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"не сериализуется в JSON: {type(value).__name__}")
