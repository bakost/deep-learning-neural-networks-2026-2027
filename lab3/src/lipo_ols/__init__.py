r"""lipo_ols — зависимость коэффициента липофильности от структуры молекул, линейная регрессия.

Лабораторная работа 3. Пакет оценивает регрессию

.. math::

    \log P = \beta_0 + \beta_1 x_1 + \ldots + \beta_{24} x_{24} + \varepsilon

методом наименьших квадратов по выборке ``lipo.csv`` (82 вещества), находит
остатки, :math:`s^2`, :math:`s`, :math:`R^2`, доверительные интервалы
коэффициентов по t-статистике и проверяет условия применимости МНК.

Быстрый старт
-------------
>>> from lipo_ols import load_lipo, fit_ols_data
>>> data = load_lipo()                      # 1. выборка
>>> fit = fit_ols_data(data)                # 2. b = (XᵀX)⁻¹Xᵀy
>>> e = fit.residuals                       # 3. МНК-остатки
>>> round(fit.s2, 4), round(fit.s, 4)       # 4. s² и s
(0.591, 0.7687)
>>> round(fit.r2, 4)                        # 5. R²
0.838
>>> ci = fit.conf_int(0.95)                 # 6. доверительные интервалы, форма (25, 2)
>>> ci.shape
(25, 2)

Состав пакета
-------------
==================================  ===========================================
Модуль                              Назначение
==================================  ===========================================
:mod:`lipo_ols.data`                чтение ``lipo.csv``, описание регрессоров
:mod:`lipo_ols.ols`                 МНК, s², R², t-тесты, интервалы, F-тесты
:mod:`lipo_ols.distributions`       плотности Стьюдента и Фишера по лекции 3
:mod:`lipo_ols.diagnostics`         условия применимости МНК
:mod:`lipo_ols.contributions`       вклад регрессоров, пары молекул
:mod:`lipo_ols.validation`          скользящий контроль, перекрёстная проверка
:mod:`lipo_ols.literature`          числа из статьи [2]
:mod:`lipo_ols.plots`               рисунки отчёта
:mod:`lipo_ols.report`              пересчёт всех чисел и рисунков отчёта
==================================  ===========================================
"""

from __future__ import annotations

from .contributions import (
    MATCHED_PAIRS,
    MatchedPair,
    contribution_table,
    drop_one_r2,
    matched_pairs,
    pairwise_correlations,
    standardized_coefficients,
)
from .data import DESCRIPTOR_NAMES, DESCRIPTORS, LipoData, default_data_path, duplicate_groups, load_lipo
from .diagnostics import (
    PureError,
    TestResult,
    breusch_pagan,
    condition_number,
    cooks_distance,
    durbin_watson,
    jarque_bera,
    leverage,
    normality_tests,
    pure_error,
    reset_test,
    studentized_residuals,
    vif,
)
from .distributions import fisher_pdf, fisher_quantile, student_pdf, student_quantile
from .literature import PAPER_MODELS, PAPER_PAIRWISE_R2, PAPER_TEST, paper_columns
from .ols import METHODS, FTest, OLSResult, fit_ols, fit_ols_data, solve_ols
from .validation import CVResult, backward_elimination, final_model, kfold_cv, loo_predictions, press_statistics

__version__ = "1.0.0"

__all__ = [
    "DESCRIPTORS",
    "DESCRIPTOR_NAMES",
    "MATCHED_PAIRS",
    "METHODS",
    "PAPER_MODELS",
    "PAPER_PAIRWISE_R2",
    "PAPER_TEST",
    "CVResult",
    "FTest",
    "LipoData",
    "MatchedPair",
    "OLSResult",
    "PureError",
    "TestResult",
    "__version__",
    "backward_elimination",
    "breusch_pagan",
    "condition_number",
    "contribution_table",
    "cooks_distance",
    "default_data_path",
    "drop_one_r2",
    "duplicate_groups",
    "durbin_watson",
    "final_model",
    "fisher_pdf",
    "fisher_quantile",
    "fit_ols",
    "fit_ols_data",
    "jarque_bera",
    "kfold_cv",
    "leverage",
    "load_lipo",
    "loo_predictions",
    "matched_pairs",
    "normality_tests",
    "pairwise_correlations",
    "paper_columns",
    "press_statistics",
    "pure_error",
    "reset_test",
    "solve_ols",
    "standardized_coefficients",
    "student_pdf",
    "student_quantile",
    "studentized_residuals",
    "vif",
]
