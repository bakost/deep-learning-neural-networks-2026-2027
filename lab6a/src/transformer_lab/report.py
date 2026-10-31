"""Пересчёт чисел, таблиц и рисунков отчёта без доступа к сети."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .experiments import run_experiments


def markdown_table(headers: list, rows: list) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def make_tables(results: dict) -> str:
    task3 = results["task3"]
    tables = [
        ("Параметры при D = 64 и H = DN", markdown_table(
            ["N", "Q/K/V: веса + смещения", "Элементы A(X)", "MLP: веса + смещения", "MLP / QKV"],
            [[r["N"], r["qkv_total"], r["attention_entries"], r["mlp_total"],
              f'{r["mlp_total"] / r["qkv_total"]:.3f}'] for r in results["task1"]],
        )),
        ("Проверка эквивариантности", markdown_table(
            ["D", "N", "seed", "Делитель", "max abs ΔSa", "max abs ΔA"],
            [[r["D"], r["N"], r["seed"], "√D" if r["scaled"] else "1",
              f'{r["output_max_abs_error"]:.3e}', f'{r["attention_max_abs_error"]:.3e}']
             for r in results["task2"]],
        )),
        ("Значения softmax", markdown_table(
            ["i", "zᵢ", "yᵢ (округление float64)"],
            [[i + 1, int(z), f"{y:.10e}"]
             for i, (z, y) in enumerate(zip(task3["z"], task3["probabilities"]))],
        )),
        ("Все 25 производных ∂yᵢ/∂zⱼ", markdown_table(
            ["i / j", "1", "2", "3", "4", "5"],
            [[i + 1] + [f"{v:.8e}" for v in row] for i, row in enumerate(task3["jacobian"])],
        )),
        ("Масштабирование заданного вектора логитов", markdown_table(
            ["s в softmax(z/s)", "max yᵢ", "max abs ∂yᵢ/∂zⱼ"],
            [[f'{r["scale"]:.6g}', f'{r["max_probability"]:.10f}', f'{r["max_abs_dy_dz"]:.8e}']
             for r in task3["scaling"]],
        )),
    ]
    return "\n".join(f"## Таблица {i}. {title}\n\n{table}" for i, (title, table) in enumerate(tables, 1))


def make_figures(results: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})

    n = np.arange(1, 257)
    d = 64
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(n, np.full_like(n, 3 * d * (d + 1)), label="Q/K/V: 3D(D+1)", lw=2)
    ax.loglog(n, 2 * d * n * (d * n + 1), label="MLP: 2DN(DN+1), H=DN", lw=2)
    ax.loglog(n, n * n, "--", label="N² коэффициентов A(X) (вычисляемые)")
    ax.set(xlabel="Длина последовательности N", ylabel="Количество", title="Задача 1: D = 64")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(output / "parameters.png")
    plt.close(fig)

    r = results["task2"][0]
    a, ap, p = (np.array(r[key]) for key in ("A", "A_permuted", "P"))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3), constrained_layout=True)
    for ax, values, title in zip(axes, (a, ap, p.T @ a @ p), ("A(X)", "A(XP)", "Pᵀ A(X) P")):
        im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
        ax.set(title=title, xlabel="Запрос n", ylabel="Ключ m", xticks=range(4), yticks=range(4),
               xticklabels=range(1, 5), yticklabels=range(1, 5))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        color="white" if values[i, j] > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(output / "equivariance.png")
    plt.close(fig)

    jac = np.array(results["task3"]["jacobian"])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.log10(np.abs(jac)), cmap="viridis", vmin=-89, vmax=-41)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{jac[i, j]:.2e}", ha="center", va="center", fontsize=8,
                    color="black" if np.log10(abs(jac[i, j])) > -55 else "white")
    ax.set(title="Задача 3: якобиан softmax (все элементы ненулевые)",
           xlabel="Вход j", ylabel="Выход i", xticks=range(5), yticks=range(5),
           xticklabels=range(1, 6), yticklabels=range(1, 6))
    fig.colorbar(im, ax=ax, label="log₁₀ |∂yᵢ/∂zⱼ|")
    fig.tight_layout()
    fig.savefig(output / "softmax_jacobian.png")
    plt.close(fig)


def generate_report(output: Path) -> dict:
    output = Path(output)
    for directory in ("results", "figures"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    results = run_experiments()
    (output / "results" / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output / "results" / "tables.md").write_text(make_tables(results), encoding="utf-8")
    np.savetxt(output / "results" / "jacobian.csv", results["task3"]["jacobian"], delimiter=",",
               header="dy_i/dz_1,dy_i/dz_2,dy_i/dz_3,dy_i/dz_4,dy_i/dz_5", comments="", fmt="%.17e")
    make_figures(results, output / "figures")
    return results
