r"""Проверка условий применимости МНК (пункт 7, третий вопрос).

Лекция 3 перечисляет предположения классической линейной модели:
линейность, строгая экзогенность, отсутствие мультиколлинеарности,
гомоскедастичность, отсутствие корреляции между наблюдениями; для t- и
F-тестов — ещё нормальность ошибок. Здесь собраны инструменты для каждого:

==========================  ===============================================
Условие                     Проверка
==========================  ===============================================
линейность                  тест Рамсея (RESET): добавить :math:`\hat y^2, \hat y^3`
мультиколлинеарность        ранг, число обусловленности, VIF
гомоскедастичность          тест Бройша–Пагана (вариант Кёнкера)
некоррелированность         статистика Дарбина–Уотсона
нормальность                тесты Шапиро–Уилка и Харке–Бера, асимметрия, эксцесс
влиятельные наблюдения      рычаги :math:`h_{ii}`, стьюдентизированные остатки, расстояние Кука
«чистая ошибка»             повторяющиеся строки X: нижняя граница SSR, тест на неадекватность
==========================  ===============================================

Строгую экзогенность :math:`\langle\varepsilon|X\rangle = 0` по одной
выборке проверить нельзя: МНК-остатки ортогональны регрессорам по
построению (:math:`X^Te = 0`). Её можно только обсуждать содержательно.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .data import LipoData, duplicate_groups
from .ols import OLSResult, fit_ols

__all__ = [
    "PureError",
    "TestResult",
    "breusch_pagan",
    "condition_number",
    "cooks_distance",
    "durbin_watson",
    "jarque_bera",
    "leverage",
    "normality_tests",
    "pure_error",
    "reset_test",
    "studentized_residuals",
    "vif",
]


@dataclass(frozen=True)
class TestResult:
    """Статистика критерия, её p-значение и (необязательно) число степеней свободы."""

    name: str
    statistic: float
    p: float
    df: Optional[tuple] = None


# ------------------------------------------------------------ влияние точек
def leverage(X: np.ndarray) -> np.ndarray:
    r"""Диагональ матрицы-проектора :math:`P = X(X^TX)^{-1}X^T` (рычаги :math:`h_{ii}`).

    Считается через QR: :math:`P = QQ^T`, :math:`h_{ii} = \sum_j Q_{ij}^2`.
    Сумма рычагов равна :math:`K`; :math:`h_{ii} = 1` означает, что
    наблюдение подогнано точно и остаток равен нулю при любом :math:`y_i`.
    """
    q, _ = np.linalg.qr(np.asarray(X, dtype=float))
    return np.sum(q * q, axis=1)


def studentized_residuals(fit: OLSResult, external: bool = True) -> np.ndarray:
    r"""Стьюдентизированные остатки.

    Внутренние :math:`r_i = e_i / (s\sqrt{1 - h_{ii}})`, внешние (с оценкой
    :math:`s_{(i)}` без :math:`i`-го наблюдения)
    :math:`t_i = r_i\sqrt{(n-K-1)/(n-K-r_i^2)}`; при верной модели
    :math:`t_i \sim t_{n-K-1}`. Для :math:`h_{ii} = 1` не определены (``nan``).
    """
    h = leverage(fit.X)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = fit.residuals / (fit.s * np.sqrt(1.0 - h))
        r[h > 1.0 - 1e-10] = np.nan
        if not external:
            return r
        nu = fit.df_resid
        return r * np.sqrt((nu - 1) / (nu - r * r))


def cooks_distance(fit: OLSResult) -> np.ndarray:
    r"""Расстояние Кука :math:`D_i = \frac{r_i^2}{K}\frac{h_{ii}}{1 - h_{ii}}` (``nan`` при :math:`h_{ii}=1`)."""
    h = leverage(fit.X)
    r = studentized_residuals(fit, external=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        return r * r / fit.K * h / (1.0 - h)


# ------------------------------------------------------ мультиколлинеарность
def condition_number(X: np.ndarray, scale: bool = True) -> float:
    r"""Число обусловленности :math:`\kappa(X) = \sigma_{max}/\sigma_{min}`.

    При ``scale=True`` столбцы предварительно нормируются на единичную длину
    (так предлагают Белсли, Кух и Уэлш): тогда число не зависит от единиц
    измерения регрессоров. Порог «сильной» мультиколлинеарности — 30.
    """
    X = np.asarray(X, dtype=float)
    if scale:
        X = X / np.linalg.norm(X, axis=0)
    sv = np.linalg.svd(X, compute_uv=False)
    return float(sv[0] / sv[-1])


def vif(data: LipoData) -> Dict[str, float]:
    r"""Коэффициенты увеличения дисперсии :math:`VIF_j = 1/(1 - R_j^2)`.

    :math:`R_j^2` — коэффициент детерминации регрессии :math:`x_j` на
    константу и остальные регрессоры. :math:`VIF_j` во столько раз больше
    дисперсия :math:`b_j`, чем была бы при регрессоре, ортогональном
    остальным; :math:`VIF > 10` считают признаком сильной мультиколлинеарности.
    """
    out: Dict[str, float] = {}
    for j, name in enumerate(data.columns):
        others = np.column_stack([np.ones(data.n), np.delete(data.X, j, axis=1)])
        target = data.X[:, j]
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        tss = float(np.sum((target - target.mean()) ** 2))
        r2 = 1.0 - float(resid @ resid) / tss
        out[name] = float(np.inf) if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
    return out


# ----------------------------------------------------------- остатки: тесты
def jarque_bera(e: np.ndarray) -> TestResult:
    r"""Тест Харке–Бера: :math:`JB = \frac{n}{6}\left(S^2 + \frac{(\kappa - 3)^2}{4}\right) \sim \chi^2_2`."""
    e = np.asarray(e, dtype=float)
    n = e.size
    d = e - e.mean()
    m2 = np.mean(d**2)
    skew = np.mean(d**3) / m2**1.5
    kurt = np.mean(d**4) / m2**2
    jb = n / 6.0 * (skew**2 + (kurt - 3.0) ** 2 / 4.0)
    return TestResult("Харке–Бера", float(jb), float(stats.chi2.sf(jb, 2)), (2,))


def normality_tests(e: np.ndarray) -> Dict[str, object]:
    """Шапиро–Уилк, Харке–Бера, выборочные асимметрия и эксцесс остатков."""
    e = np.asarray(e, dtype=float)
    sw = stats.shapiro(e)
    return {
        "shapiro": TestResult("Шапиро–Уилк", float(sw.statistic), float(sw.pvalue)),
        "jarque_bera": jarque_bera(e),
        "skewness": float(stats.skew(e)),
        "excess_kurtosis": float(stats.kurtosis(e)),
    }


def breusch_pagan(fit: OLSResult) -> TestResult:
    r"""Тест Бройша–Пагана в форме Кёнкера (устойчив к ненормальности).

    Вспомогательная регрессия :math:`e_i^2` на те же регрессоры;
    :math:`LM = nR^2_{aux} \sim \chi^2_{K-1}` при гомоскедастичности.
    """
    u = fit.residuals**2
    coef, *_ = np.linalg.lstsq(fit.X, u, rcond=None)
    resid = u - fit.X @ coef
    r2 = 1.0 - float(resid @ resid) / float(np.sum((u - u.mean()) ** 2))
    df = fit.K - 1
    lm = fit.n * r2
    return TestResult("Бройш–Паган (Кёнкер)", float(lm), float(stats.chi2.sf(lm, df)), (df,))


def reset_test(fit: OLSResult, powers: Sequence[int] = (2, 3)) -> TestResult:
    r"""Тест Рамсея на пропущенную нелинейность.

    В модель добавляются :math:`\hat y^2, \hat y^3`; F-тест гипотезы, что
    их коэффициенты равны нулю (та же формула F-отношения, что в лекции 3).
    """
    yhat = fit.fitted
    scale = np.std(yhat)
    extra = np.column_stack([(yhat / scale) ** p for p in powers])
    big = fit_ols(np.column_stack([fit.X, extra]), fit.y,
                  list(fit.labels) + [f"yhat^{p}" for p in powers])
    test = big.zero_test([f"yhat^{p}" for p in powers])
    return TestResult("RESET", test.F, test.p, (test.df1, test.df2))


def durbin_watson(e: np.ndarray) -> float:
    r""":math:`DW = \sum_{i \ge 2}(e_i - e_{i-1})^2 / \sum e_i^2`; около 2 — автокорреляции нет."""
    e = np.asarray(e, dtype=float)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e * e))


# ------------------------------------------------------------- чистая ошибка
@dataclass(frozen=True)
class PureError:
    r"""Разложение :math:`SSR = SS_{pe} + SS_{lof}` по группам одинаковых строк ``X``.

    Attributes
    ----------
    groups:
        номера наблюдений в каждой группе повторов;
    ss_pe, df_pe:
        «чистая ошибка» — разброс :math:`y` вокруг средних внутри групп;
        это минимально возможная SSR для *любой* функции :math:`f(x)`;
    ss_lof, df_lof:
        неадекватность (lack of fit) — остальная часть SSR;
    F, p:
        тест на неадекватность :math:`F = \frac{SS_{lof}/df_{lof}}{SS_{pe}/df_{pe}}`.
    """

    groups: List[List[int]]
    ss_pe: float
    df_pe: int
    ss_lof: float
    df_lof: int
    F: float
    p: float

    @property
    def sigma_pe(self) -> float:
        """Оценка стандартного отклонения «неустранимой» ошибки по повторам."""
        return float(np.sqrt(self.ss_pe / self.df_pe))


def pure_error(fit: OLSResult, data: LipoData) -> PureError:
    """Тест на неадекватность по группам веществ с одинаковыми регрессорами."""
    groups = duplicate_groups(data)
    ss_pe = sum(float(np.sum((data.y[g] - data.y[g].mean()) ** 2)) for g in groups)
    df_pe = sum(len(g) - 1 for g in groups)
    ss_lof = fit.ssr - ss_pe
    df_lof = fit.df_resid - df_pe
    if df_pe == 0 or df_lof <= 0:
        return PureError(groups, ss_pe, df_pe, ss_lof, df_lof, float("nan"), float("nan"))
    F = (ss_lof / df_lof) / (ss_pe / df_pe)
    return PureError(groups, ss_pe, df_pe, ss_lof, df_lof, float(F), float(stats.f.sf(F, df_lof, df_pe)))
