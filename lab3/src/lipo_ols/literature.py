r"""Результаты работы [2] для сравнения (пункт 7, четвёртый вопрос).

K. López, S. Pinheiro, W. J. Zamora. *Multiple linear regression models for
predicting the n-octanol/water partition coefficients in the SAMPL7 blind
challenge* // J. Comput.-Aided Mol. Des. 2021. V. 35. P. 923–931.
doi:10.1007/s10822-021-00409-2 (открытый доступ: PMC8273033).

Авторы построили три вложенные модели на одной обучающей выборке из 82
веществ: MLR-1 — регрессоры 1–19 табл. 1 (функциональные группы, атомы
углерода, циклы), MLR-2 — плюс HBA1, HBA2, HBD, MLR-3 — все 24. Числа ниже
переписаны из таблицы статистик моделей в статье с той же точностью, с
какой они там напечатаны. Таблица коэффициентов (Table S3) вынесена в
приложение к статье и здесь не используется: МНК-решение при полном ранге
:math:`X` единственно, так что совпадение :math:`R^2`, :math:`s`, :math:`F` и
:math:`p` до напечатанных знаков уже означает совпадение модели.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .data import DESCRIPTOR_NAMES

__all__ = ["PAPER_MODELS", "PAPER_PAIRWISE_R2", "PAPER_REFERENCE", "PAPER_TEST", "paper_columns"]

PAPER_REFERENCE = (
    "K. López, S. Pinheiro, W. J. Zamora. Multiple linear regression models for predicting "
    "the n-octanol/water partition coefficients in the SAMPL7 blind challenge // "
    "J. Comput.-Aided Mol. Des. 2021. V. 35. P. 923–931"
)

#: Статистики моделей на обучающей выборке (n = 82), как напечатано в статье.
PAPER_MODELS: Dict[str, Dict[str, float]] = {
    "MLR-1": {"k": 19, "r2": 0.79, "r2_adj": 0.73, "rmse": 0.72, "s": 0.83, "F": 12.6, "F_p": 1.02e-14},
    "MLR-2": {"k": 22, "r2": 0.82, "r2_adj": 0.75, "rmse": 0.68, "s": 0.80, "F": 12.2, "F_p": 1.30e-14},
    "MLR-3": {"k": 24, "r2": 0.84, "r2_adj": 0.77, "rmse": 0.64, "s": 0.77, "F": 12.3, "F_p": 9.00e-15},
}

#: Коэффициенты детерминации парных регрессий, которые приводятся в тексте статьи.
PAPER_PAIRWISE_R2: Dict[str, float] = {"C": 0.50, "MR": 0.41, "AROMATIC": 0.34, "HBA1": 0.11, "HBA2": 0.10}

#: Качество MLR-3 вне обучающей выборки (по статье): 5 отложенных сульфонамидов и 22 вещества SAMPL7.
PAPER_TEST: Dict[str, Dict[str, float]] = {
    "test (5 сульфонамидов)": {"n": 5, "rmse": 0.20, "mae": 0.12},
    "SAMPL7 (22 N-ацилсульфонамида)": {"n": 22, "rmse": 0.58, "mae": 0.41},
}


def paper_columns(model: str) -> Tuple[str, ...]:
    """Регрессоры модели MLR-1/2/3: первые ``k`` из табл. 1."""
    return DESCRIPTOR_NAMES[: int(PAPER_MODELS[model]["k"])]
