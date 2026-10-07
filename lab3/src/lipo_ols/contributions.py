r"""Какие регрессоры вносят наибольший вклад (пункт 7, второй вопрос).

Сами коэффициенты :math:`b_j` сравнивать нельзя: регрессоры измерены в
разных единицах (число атомов углерода — до 40, площадь PSA — до 200 Å²,
индикаторы групп — 0 или 1). Поэтому используются несколько мер:

* **стандартизованный коэффициент** :math:`\beta^*_j = b_j\,\sigma_{x_j}/\sigma_y` —
  на сколько стандартных отклонений меняется :math:`\log P` при изменении
  :math:`x_j` на одно своё стандартное отклонение (при прочих равных);
* **вклад при исключении** :math:`\Delta R^2_j` — насколько упадёт :math:`R^2`,
  если убрать :math:`x_j` из модели; :math:`\Delta R^2_j = t_j^2 (1 - R^2)/(n - K)`;
* **парная корреляция** :math:`r(x_j, y)` — связь без учёта прочих регрессоров;
* **пары молекул** (matched molecular pairs): разность прогнозов модели для
  двух веществ, отличающихся одной группой, против экспериментальной разности.
  Для замещённых бензолов экспериментальная разность — это константа
  гидрофобности Ганча :math:`\pi_X = \log P(\mathrm{C_6H_5X}) - \log P(\mathrm{C_6H_6})`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import LipoData
from .ols import OLSResult

__all__ = [
    "MATCHED_PAIRS",
    "MatchedPair",
    "contribution_table",
    "drop_one_r2",
    "matched_pairs",
    "pairwise_correlations",
    "standardized_coefficients",
]

#: Пары «исходное вещество → производное» и подпись изменения.
MATCHED_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("Benzene", "Fluorobenzene", "H → F"),
    ("Benzene", "Chlorobenzene", "H → Cl"),
    ("Benzene", "Bromobenzene", "H → Br"),
    ("Benzene", "Phenol", "H → OH"),
    ("Benzene", "Aniline", "H → NH₂"),
    ("Benzene", "Anisole", "H → OCH₃"),
    ("Benzene", "Benzaldehyde", "H → CHO"),
    ("Benzene", "Acetophenone", "H → COCH₃"),
    ("Benzene", "Benzoic acid", "H → COOH"),
    ("Diethyl ether", "Diethyl thioether", "O → S (эфир)"),
    ("THF", "Tetrahydrothiophene", "O → S (цикл)"),
    ("Furan", "Thiophene", "O → S (аромат.)"),
    ("Ethanol", "Ethylamine", "OH → NH₂"),
)


def standardized_coefficients(fit: OLSResult, data: LipoData) -> Dict[str, float]:
    r""":math:`\beta^*_j = b_j\,\sigma_{x_j}/\sigma_y` для всех регрессоров модели (без константы)."""
    sy = data.y.std(ddof=1)
    out = {}
    for label, b in zip(fit.labels, fit.b):
        if label == "const":
            continue
        out[label] = float(b * data.column(label).std(ddof=1) / sy)
    return out


def drop_one_r2(fit: OLSResult) -> Dict[str, float]:
    r"""Падение :math:`R^2` при исключении регрессора: :math:`t_j^2 (1 - R^2)/(n - K)`."""
    factor = (1.0 - fit.r2) / fit.df_resid
    return {label: float(t * t * factor) for label, t in zip(fit.labels, fit.t) if label != "const"}


def pairwise_correlations(data: LipoData) -> Dict[str, float]:
    """Коэффициент корреляции Пирсона каждого регрессора с :math:`\\log P`."""
    return {name: float(np.corrcoef(data.column(name), data.y)[0, 1]) for name in data.columns}


def contribution_table(fit: OLSResult, data: LipoData, level: float = 0.95) -> List[dict]:
    """Сводная таблица: b, CI, p, стандартизованный коэффициент с CI, ΔR², r(x, y)."""
    std = standardized_coefficients(fit, data)
    drop = drop_one_r2(fit)
    corr = pairwise_correlations(data)
    ci = fit.conf_int(level)
    sy = data.y.std(ddof=1)
    rows = []
    for k, label in enumerate(fit.labels):
        if label == "const":
            continue
        scale = data.column(label).std(ddof=1) / sy
        rows.append({
            "label": label, "b": float(fit.b[k]), "low": float(ci[k, 0]), "high": float(ci[k, 1]),
            "p": float(fit.p_values[k]), "beta": std[label],
            "beta_low": float(ci[k, 0] * scale), "beta_high": float(ci[k, 1] * scale),
            "delta_r2": drop[label], "r": corr[label],
        })
    return rows


@dataclass(frozen=True)
class MatchedPair:
    """Пара веществ: экспериментальная и предсказанная разности :math:`\\log P` и разложение по регрессорам."""

    parent: str
    child: str
    change: str
    delta_exp: float
    delta_model: float
    terms: Dict[str, float]
    exact_fit: bool

    @property
    def error(self) -> float:
        return self.delta_model - self.delta_exp


def matched_pairs(fit: OLSResult, data: LipoData,
                  pairs: Sequence[Tuple[str, str, str]] = MATCHED_PAIRS) -> List[MatchedPair]:
    r"""Разности :math:`\Delta\log P` для пар веществ: эксперимент и модель.

    Прогноз модели для разности: :math:`\Delta\hat y = \sum_j b_j (x_{Bj} - x_{Aj})`;
    ``terms`` — слагаемые этой суммы (ненулевые). ``exact_fit`` отмечает
    пары, где одно из веществ подогнано точно (рычаг :math:`h = 1`): там
    совпадение с экспериментом заложено в оценку и ничего не доказывает.
    """
    from .diagnostics import leverage

    h = leverage(fit.X)
    labels = list(fit.labels)
    out = []
    for parent, child, change in pairs:
        i, j = data.index(parent), data.index(child)
        diff = fit.X[j] - fit.X[i]
        terms = {labels[k]: float(fit.b[k] * diff[k]) for k in range(len(labels)) if diff[k] != 0.0}
        out.append(MatchedPair(parent, child, change, float(data.y[j] - data.y[i]),
                               float(diff @ fit.b), terms, bool(max(h[i], h[j]) > 1 - 1e-8)))
    return out
