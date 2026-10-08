r"""Свой расчёт против пакета ``statsmodels`` (дополнение к заданию).

Все величины пунктов 2–6 и диагностика пункта 7 считаются двумя способами:
собственными функциями пакета (по формулам лекции 3) и стандартным
пакетом ``statsmodels`` (``sm.OLS(y, X).fit()`` и модуль
``statsmodels.stats``). Для каждой величины — наибольшее расхождение.

Два отличия заранее известны и не являются ошибками:

* **AIC/BIC.** ``statsmodels`` не считает дисперсию ошибки :math:`\sigma^2`
  оцениваемым параметром (:math:`k = K`), здесь она учитывается
  (:math:`k = K + 1`). Разница ровно :math:`2` и :math:`\ln n`.
* **Точки с рычагом** :math:`h = 1`. Стьюдентизированный остаток и
  расстояние Кука для них не определены (деление на :math:`1 - h = 0`).
  Здесь возвращается ``nan``; ``statsmodels`` получает :math:`h` с
  ошибкой округления :math:`1 - 10^{-15}` и выдаёт для части таких точек
  конечные, но бессмысленные числа.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List

import numpy as np

from .data import LipoData
from .diagnostics import (
    breusch_pagan,
    cooks_distance,
    durbin_watson,
    jarque_bera,
    leverage,
    reset_test,
    studentized_residuals,
    vif,
)
from .ols import fit_ols_data

__all__ = ["compare_with_statsmodels"]


def _ms(fn: Callable[[], object], repeats: int) -> float:
    fn()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        best = min(best, (time.perf_counter() - t0) / repeats * 1e3)
    return best


def compare_with_statsmodels(data: LipoData, level: float = 0.95) -> Dict[str, object]:
    """Сравнить свой расчёт с ``statsmodels``: расхождения по величинам, особые точки, время."""
    import warnings

    import statsmodels.api as sm
    from statsmodels.stats import stattools
    from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    fit = fit_ols_data(data)
    X, y = fit.X, fit.y
    res = sm.OLS(y, X).fit()
    h = leverage(X)
    ok = h < 1 - 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        infl = res.get_influence()
        sm_cook = infl.cooks_distance[0]
        sm_stud = infl.resid_studentized_external
    overall = fit.overall_f_test()
    bp = het_breuschpagan(res.resid, X)
    reset = linear_reset(res, power=3, use_f=True)
    sm_vif = [variance_inflation_factor(X, j) for j in range(1, X.shape[1])]

    def diff(a, b) -> float:
        return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))

    rows: List[dict] = [
        {"item": "МНК-оценки b (пункт 2)", "diff": diff(res.params, fit.b)},
        {"item": "остатки e (пункт 3)", "diff": diff(res.resid, fit.residuals)},
        {"item": "s² (пункт 4)", "diff": diff(res.scale, fit.s2)},
        {"item": "R² и скорректированный R² (пункт 5)",
         "diff": max(diff(res.rsquared, fit.r2), diff(res.rsquared_adj, fit.r2_adj))},
        {"item": "SE(b), t, p (пункт 6)",
         "diff": max(diff(res.bse, fit.se), diff(res.tvalues, fit.t), diff(res.pvalues, fit.p_values))},
        {"item": f"{int(level * 100)}%-е доверительные интервалы (пункт 6)",
         "diff": diff(res.conf_int(1 - level), fit.conf_int(level))},
        {"item": "F-статистика регрессии и её p",
         "diff": max(diff(res.fvalue, overall.F), diff(res.f_pvalue, overall.p))},
        {"item": "логарифм правдоподобия", "diff": diff(res.llf, fit.loglik)},
        {"item": "AIC, BIC (разные соглашения о числе параметров)",
         "diff": max(diff(res.aic + 2, fit.aic), diff(res.bic + np.log(fit.n), fit.bic)),
         "note": f"сырая разница {fit.aic - res.aic:.0f} и {fit.bic - res.bic:.2f} = ln n"},
        {"item": "VIF (относительная)",
         "diff": float(np.max(np.abs(np.array(sm_vif) / np.array(list(vif(data).values())) - 1)))},
        {"item": "рычаги h", "diff": diff(infl.hat_matrix_diag, h)},
        {"item": "расстояние Кука (h < 1)", "diff": diff(sm_cook[ok], cooks_distance(fit)[ok])},
        {"item": "внешние стьюдентизированные остатки (h < 1)",
         "diff": diff(sm_stud[ok], studentized_residuals(fit)[ok])},
        {"item": "тест Бройша–Пагана (LM)", "diff": diff(bp[0], breusch_pagan(fit).statistic)},
        {"item": "тест RESET (F)", "diff": diff(float(reset.fvalue), reset_test(fit).statistic)},
        {"item": "тест Харке–Бера",
         "diff": diff(stattools.jarque_bera(res.resid)[0], jarque_bera(fit.residuals).statistic)},
        {"item": "статистика Дарбина–Уотсона",
         "diff": diff(stattools.durbin_watson(res.resid), durbin_watson(fit.residuals))},
    ]
    special = [{"name": data.names[i], "leverage_ours": float(h[i]), "leverage_sm": float(infl.hat_matrix_diag[i]),
                "cook_sm": float(sm_cook[i]), "stud_sm": float(sm_stud[i])} for i in np.where(~ok)[0]]

    def ours() -> object:
        f = fit_ols_data(data)
        return f.se, f.p_values, f.conf_int(level), f.overall_f_test()

    def theirs() -> object:
        r = sm.OLS(y, X).fit()
        return r.bse, r.pvalues, r.conf_int(1 - level), r.fvalue

    timing = {"ours_ms": _ms(ours, 200), "statsmodels_ms": _ms(theirs, 200),
              "statsmodels_summary_ms": _ms(lambda: sm.OLS(y, X).fit().summary().as_text(), 20)}
    import statsmodels

    return {"rows": rows, "special": special, "timing": timing, "version": statsmodels.__version__}
