r"""level_spacing — распределение расстояний между соседними уровнями случайных матриц.

Лабораторная работа 2 «Свойства случайных матриц». Пакет генерирует
ансамбли симметричных матриц, находит расстояния между фиксированной парой
соседних собственных чисел, нормирует их и сравнивает с догадкой Вигнера
:math:`\rho(s) = \frac{\pi s}{2} e^{-\pi s^2/4}`.

Быстрый старт
-------------
>>> from level_spacing import run_study
>>> study = run_study("goe", n=16, size=10_000, seed=2026)
>>> study.k                                  # центральная пара (λ8, λ9)
8
>>> round(float(study.normalized.mean()), 12)
1.0
>>> study.compare("wigner")["ks_d"] < 0.02
True

Пошагово, по пунктам задания:

>>> import numpy as np
>>> from level_spacing import sample_matrices, eigenvalues, pair_spacings, normalize
>>> H = sample_matrices("goe", n=4, size=1000, rng=0)   # 1. ансамбль
>>> ev = eigenvalues(H)                                 # 2. спектры по возрастанию
>>> s = pair_spacings(ev, k=2)                          # 3. λ3 − λ2
>>> x = normalize(s)                                    # 4. среднее = 1
>>> x.shape
(1000,)

Состав пакета
-------------
==================================  ===========================================
Модуль                              Назначение
==================================  ===========================================
:mod:`level_spacing.ensembles`      ансамбли матриц (GOE, ±1, ...)
:mod:`level_spacing.spectrum`       спектры, расстояния, нормировка
:mod:`level_spacing.distributions`  теоретические законы (Вигнер, GOE, 2×2 ±1)
:mod:`level_spacing.exact`          точные законы для ±1 полным перебором
:mod:`level_spacing.stats`          гистограммы, критерии согласия, моменты
:mod:`level_spacing.experiments`    сценарии эксперимента
:mod:`level_spacing.plots`          рисунки отчёта
:mod:`level_spacing.report`         пересчёт всех чисел и рисунков отчёта
==================================  ===========================================
"""

from __future__ import annotations

from .distributions import (
    GOE2_RAW_MEAN,
    WIGNER_SECOND_MOMENT,
    WIGNER_VARIANCE,
    DiscreteDistribution,
    goe2_raw_cdf,
    goe2_raw_pdf,
    goe_limit_cdf,
    goe_limit_pdf,
    normal_mirror_2x2_cdf,
    normal_mirror_2x2_pdf,
    pm1_2x2_distribution,
    poisson_cdf,
    poisson_pdf,
    wigner_cdf,
    wigner_pdf,
    wigner_sample,
)
from .ensembles import ENSEMBLES, Ensemble, available_ensembles, get_ensemble, sample_matrices
from .exact import enumerate_pm1_matrices, exact_pm1_spacing
from .experiments import (
    DEFAULT_SEED,
    SpacingStudy,
    ks_by_ensemble_size,
    pair_scan,
    pooled_spacings,
    run_study,
)
from .spectrum import (
    all_spacings,
    central_pair,
    eigenvalues,
    normalize,
    pair_spacings,
    simulate_eigenvalues,
    simulate_spacings,
)
from .stats import (
    Histogram,
    chi2_test,
    histogram,
    ks_pvalue_normalized,
    ks_statistic,
    ks_test,
    l1_distance,
    moments,
    small_s_exponent,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_SEED",
    "DiscreteDistribution",
    "ENSEMBLES",
    "Ensemble",
    "GOE2_RAW_MEAN",
    "Histogram",
    "SpacingStudy",
    "WIGNER_SECOND_MOMENT",
    "WIGNER_VARIANCE",
    "__version__",
    "all_spacings",
    "available_ensembles",
    "central_pair",
    "chi2_test",
    "eigenvalues",
    "enumerate_pm1_matrices",
    "exact_pm1_spacing",
    "get_ensemble",
    "goe2_raw_cdf",
    "goe2_raw_pdf",
    "goe_limit_cdf",
    "goe_limit_pdf",
    "histogram",
    "ks_by_ensemble_size",
    "ks_pvalue_normalized",
    "ks_statistic",
    "ks_test",
    "l1_distance",
    "moments",
    "normal_mirror_2x2_cdf",
    "normal_mirror_2x2_pdf",
    "normalize",
    "pair_scan",
    "pair_spacings",
    "pm1_2x2_distribution",
    "poisson_cdf",
    "poisson_pdf",
    "pooled_spacings",
    "run_study",
    "sample_matrices",
    "simulate_eigenvalues",
    "simulate_spacings",
    "small_s_exponent",
    "wigner_cdf",
    "wigner_pdf",
    "wigner_sample",
]
