"""Таблицы отчёта (Markdown) из словаря результатов :func:`sgd_lab.report.compute`.

Каждая таблица — под заголовком «## Таблица N. …»; ``tools/fill_tables.py``
переносит их в REPORT.md по меткам ``<!-- table:N -->``.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

__all__ = ["render"]

METHOD = {"gd": "ГС / СГС", "nesterov": "Нестеров", "adam": "Adam"}
MODEL = {"linear": "1 (линейная)", "quadratic": "2 (квадратичная)", "relu": "3 (сеть ReLU)"}


def sci(x: Optional[float], d: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "∞"
    if x == 0:
        return "0"
    return f"{x:.{d}e}".replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e")


def num(x: Optional[float], d: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—" if x is None or math.isnan(x) else "∞"
    return f"{x:.{d}f}"


def intf(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "не дост."
    return f"{round(x):,}".replace(",", " ")


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def table(head: Sequence[str], rows: Iterable[Sequence[str]], align: Optional[str] = None) -> List[str]:
    align = align or ("l" + "r" * (len(head) - 1))
    sep = ["---" if a == "l" else "---:" for a in align]

    def esc(c) -> str:
        return str(c).replace("|", "\\|")      # «|» внутри ячейки — не граница столбца

    out = ["| " + " | ".join(map(esc, head)) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(map(esc, r)) + " |" for r in rows]
    return out


def schedule(lr: float, tau) -> str:
    return f"{lr:.4g}" if tau is None else f"{lr:.4g}/(1 + e/{tau:g})"


def render(R: dict) -> str:
    from .report import OUTCOMES

    L: List[str] = ["# Таблицы отчёта", "",
                    "Сгенерировано командой `python -m sgd_lab report`; не редактировать вручную.", ""]
    n = [0]

    def add(title: str, lines: List[str]) -> None:
        n[0] += 1
        L.extend([f"## Таблица {n[0]}. {title}", "", *lines, ""])

    cfg = R["settings"]["config"]
    t1 = R["task1"]
    add("Задача 1: поиск минимума f(z) из начального отрезка [0, 2.618] (tol = 3·10⁻⁸)", table(
        ["метод", "z*", "f(z*)", "|z − z*|", "итераций", "вычислений f"],
        [[m["method"], f"{m['x']:.10f}", f"{m['fx']:.10f}", sci(m["err"]), m["iter"], m["nfev"]]
         for m in t1["methods"]]))
    add("Задача 1: точность и число вычислений f в зависимости от tol", table(
        ["tol", "золотое сечение: |z − z*|", "вычислений f", "Брент: |z − z*|", "вычислений f"],
        [[sci(r["tol"], 0), sci(r["golden_err"]), r["golden_nfev"], sci(r["brent_err"]), r["brent_nfev"]]
         for r in t1["tols"]]))
    add("Задача 1: разные начальные точки поиска отрезка", table(
        ["начальные точки", "отрезок (a, b, c)", "вычислений f (mnbrak)", "Брент: z", "золотое сечение: z"],
        [[f"{s['start'][0]:g}, {s['start'][1]:g}", ", ".join(f"{v:.4g}" for v in s["bracket"]),
          s["bracket_nfev"], f"{s['brent_x']:.8f}", f"{s['golden_x']:.8f}"] for s in t1["starts"]]))
    add("Численный градиент (центральные разности, NR 5.7) против аналитического, 20 случайных точек", table(
        ["модель", "max |ошибка|", "max отн. ошибка", "метод Риддерса: |ошибка| ∂J/∂w₁"],
        [[MODEL[r["model"]], sci(r["max_abs_err"]), sci(r["max_rel_err"]), sci(r["ridders_err"])]
         for r in R["gradcheck"]]))

    t2 = R["task2"]
    add(f"Задача 2: классический градиентный спуск для моделей 1 и 2 (δ = {sci(R['settings']['delta'], 0)})", table(
        ["модель", "λmin(H)", "λmax(H)", "κ", "2/λmax", "ε = 2/(λmin+λmax)", "итераций", "оценка ⌈ln(δ/‖DJ₀‖)/ln ρ⌉",
         "max |w − w_МНК|", "SSE"],
        [[MODEL[k], num(t2[k]["lambda_min"], 3), num(t2[k]["lambda_max"], 2), num(t2[k]["kappa"], 1),
          num(t2[k]["lr_max"], 5), num(t2[k]["lr"], 5), intf(t2[k]["iterations"]), intf(t2[k]["theory"]),
          sci(t2[k]["err_ols"]), num(t2[k]["sse"], 5)] for k in ("linear", "quadratic")]))

    def order(rows):   # сначала доли 2/λmax по возрастанию, последней — оптимальная ε
        return sorted(rows, key=lambda r: (r["is_opt"], r["frac"]))

    def cell(r):
        if r["status"] == "diverged":
            return "расходится"
        return f"{intf(r['iterations'])} ({intf(r['theory'])})"

    rows = []
    for a, b in zip(order(t2["linear"]["sweep"]), order(t2["quadratic"]["sweep"])):
        label = f"2/(λmin+λmax): {a['frac']:.3f} и {b['frac']:.4f}" if a["is_opt"] else f"{a['frac']:g}"
        rows.append([label, num(a["lr"], 5), num(a["rho"], 4), cell(a), num(b["lr"], 5), num(b["rho"], 4), cell(b)])
    add("Задача 2: скорость обучения и число итераций (в скобках — оценка по ρ = max|1 − ελᵢ|)", table(
        ["ε / (2/λmax)", "модель 1: ε", "ρ", "итераций", "модель 2: ε", "ρ", "итераций"], rows))

    tr = t2["relu"]
    add(f"Задача 2: сеть ReLU, градиентный спуск из {tr['seeds']} начальных точек (ε = {tr['lr']})", table(
        ["куда пришёл спуск", "SSE", "запусков", "доля"],
        [[name, _outcome_range(k, R), tr["counts"][k], pct(tr["shares"][k])] for k, name in OUTCOMES]))

    t3 = R["task3"]
    for key in ("linear", "quadratic"):
        rows = []
        for r in t3["rows"][key]:
            t2c = sci(r["t2_excess"]) + (f", разошлись {r['t2_diverged']}" if r["t2_diverged"] else "")
            rows.append([r["batch"], r["updates_per_epoch"], num(r["bound"], 4), t2c, sci(r["const_floor"]),
                         intf(r["const_epochs"]), sci(r["decay_excess"]), intf(r["decay_epochs"])])
        add(f"Задача 3: СГС с мини-пакетами, {MODEL[key]} — медианы по {cfg['seeds']} "
            f"запускам, {t3['epochs']} эпох", table(
                ["m", "обновлений за эпоху", "граница ε (худший пакет)",
                 f"ε = {num(t2[key]['lr'], 4)} (из задачи 2): (J−J*)/J*",
                 f"ε = {t3['common_lr'][key]}: уровень шума (J−J*)/J*", "≤ 10⁻³ начиная с эпохи",
                 f"ε = {t3['common_lr'][key]}/(1 + e/{t3['tau']:g}): (J−J*)/J*", "≤ 10⁻³ начиная с эпохи"], rows))
    rows = []
    for r in t3["relu"]:
        rows.append([r["batch"], "постоянная" if r["schedule"] == "const" else f"/(1 + e/{t3['relu_tau']:g})",
                     pct(r["global"]), pct(r["global"] + r["better_quad"]), num(r["median_sse"], 4),
                     pct(r["constant"] + r["dead"])])
    add(f"Задача 3: сеть ReLU, СГС (ε₀ = {R['settings']['relu_lr']}, {cfg['relu_epochs']} эпох, "
        f"{tr['seeds']} начальных точек)", table(
            ["m", "ε", "глобальный минимум", "лучше квадратичной модели", "медиана SSE", "ŷ = const"], rows))

    t4 = R["task4"]
    rows = []
    for r in t4["rows"]:
        if r["model"] == "relu":
            continue
        rows.append([MODEL[r["model"]], METHOD[r["method"]], r["batch"], schedule(r["lr"], r["tau"]),
                     sci(r["median_excess"]), intf(r["epochs_1e3"]), intf(r["epochs_target"]),
                     f"{r['reached_target']} из {cfg['seeds']}"])
    add(f"Задача 4: сравнение методов на моделях 1 и 2 (лучшая ε из сетки, медианы; бюджет "
        f"{t4['epochs']['linear']} и {t4['epochs']['quadratic']} эпох)", table(
            ["модель", "метод", "m", "ε", "(J−J*)/J* в конце", "≤ 10⁻³ начиная с эпохи", "≤ 10⁻⁶ начиная с эпохи",
             "дошли до 10⁻⁶"], rows, "lllrrrrr"))
    rows = []
    for r in t4["rows"]:
        if r["model"] != "relu":
            continue
        rows.append([METHOD[r["method"]], r["batch"], schedule(r["lr"], r["tau"]), pct(r["global"]),
                     pct(r["global"] + r["better_quad"]), num(r["median_sse"], 4), pct(r["constant"] + r["dead"])])
    add(f"Задача 4: сравнение методов на сети ReLU ({tr['seeds']} начальных точек, {t4['epochs']['relu']} эпох)",
        table(["метод", "m", "ε", "глобальный минимум", "лучше квадратичной модели", "медиана SSE", "ŷ = const"],
              rows, "llrrrrr"))

    t5 = R["task5"]
    rows = []
    for a, b in zip(t5["convex"]["linear"], t5["convex"]["quadratic"]):
        rows.append([f"{a['l2']:g}", num(a["params"][0], 4), num(a["sse"], 4), num(a["press"], 4),
                     num(b["params"][0], 4), num(b["params"][1], 4), num(b["params"][2], 4), num(b["sse"], 4),
                     num(b["press"], 4), num(b["kappa"], 0), intf(b["gd_iterations"])])
    add("Задача 5: квадратичный штраф λΣw², модели 1 и 2 (точное решение, LOO — скользящий контроль)", table(
        ["λ", "мод. 1: w₁", "SSE", "LOO", "мод. 2: w₁", "w₂", "b₁", "SSE", "LOO", "κ", "итераций ГС"], rows))
    rows = []
    for r in t5["relu"]:
        p = r["params"]
        rows.append([f"{r['l2']:g}", num(r["sse"], 4), num(r["loss"], 4), num(r["weight_norm2"], 2),
                     f"{p[0]:.2f}; {p[2]:.2f}", f"{p[4]:.2f}; {p[5]:.2f}", pct(r["adam_share_near_best"]),
                     num(r["press"], 4)])
    add("Задача 5: квадратичный штраф, сеть ReLU (лучший найденный минимум J̃)", table(
        ["λ", "SSE", "J̃", "Σw²", "w₁₁; w₁₂", "w₂₁; w₂₂", "Adam: доля дошедших до J̃*", "LOO"], rows))
    rows = []
    for r in t5["demo_runs"]:
        seeds = cfg["relu_seeds" if r["model"] == "relu" else "seeds"]
        conv = f"{r['converged']} из {seeds}" if r["method"] == "gd" and r["batch"] == 12 else "—"
        rows.append([MODEL[r["model"]], METHOD[r["method"]], r["batch"], schedule(r["lr"], r["tau"]),
                     sci(r["median_excess"]), sci(r["median_dist"]) if r["median_dist"] is not None else "—",
                     pct(r["share_near_best"]), num(r["median_sse"], 4), conv])
    add(f"Задача 5: задачи 2–4 со штрафом λ = {t5['demo']} (J̃* — точное решение или лучший минимум)", table(
        ["модель", "метод", "m", "ε", "(J̃−J̃*)/J̃*, медиана", "max|w − w*|, медиана", "дошли до 1 % от J̃*",
         "SSE, медиана", "сошлось по δ"], rows, "lllrrrrrr"))
    return "\n".join(L)


def _outcome_range(key: str, R: dict) -> str:
    b, best = R["baselines"], R["relu_ref"]["sse"]
    return {
        "global": f"≤ {1.01 * best:.4f}",
        "better_quad": f"{1.01 * best:.4f}–{b['quadratic']:.4f}",
        "between": f"{b['quadratic']:.4f}–{0.99 * b['linear']:.4f}",
        "linear": f"≈ {b['linear']:.4f}",
        "worse": "прочие",
        "constant": f"≈ {b['constant']:.4f}",
        "dead": f"≈ {b['zero']:.2f}",
    }[key]
