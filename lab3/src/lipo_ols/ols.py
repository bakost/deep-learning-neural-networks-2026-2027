r"""Метод наименьших квадратов и выводы о коэффициентах (пункты 2–6 задания).

Обозначения лекции 3: :math:`y = X\beta + \varepsilon`, :math:`X` — матрица
:math:`n \times K` (первый столбец — единицы), МНК-оценка

.. math::

    b = \arg\min_{\tilde\beta} (y - X\tilde\beta)^T (y - X\tilde\beta),
    \qquad X^T X\, b = X^T y \;\Rightarrow\; b = (X^T X)^{-1} X^T y = X^+ y.

Остатки :math:`e = y - Xb`, дисперсия МНК-ошибки
:math:`s^2 = e^T e / (n - K)`, стандартная ошибка регрессии :math:`s`,
коэффициент детерминации
:math:`R^2 = 1 - \langle e^2\rangle / (\langle y^2\rangle - \langle y\rangle^2)`,
стандартные ошибки коэффициентов
:math:`SE(b_k) = \sqrt{s^2 \left((X^TX)^{-1}\right)_{kk}}`, доверительный
интервал :math:`b_k \pm SE(b_k)\, t^{n-K}_\alpha`.

>>> from lipo_ols import load_lipo
>>> fit = fit_ols_data(load_lipo())
>>> fit.n, fit.K, fit.df_resid
(82, 25, 57)
>>> round(fit.r2, 4), round(fit.s, 4)
(0.838, 0.7687)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .data import LipoData

__all__ = ["METHODS", "FTest", "OLSResult", "fit_ols", "fit_ols_data", "solve_ols"]

#: Способы решения: нормальные уравнения (как в лекции), QR-разложение, псевдообратная.
METHODS = ("normal", "qr", "pinv")


def solve_ols(X: np.ndarray, y: np.ndarray, method: str = "normal") -> np.ndarray:
    r"""МНК-оценка :math:`b` тремя способами.

    * ``normal`` — нормальные уравнения :math:`X^T X b = X^T y`
      (решаются без явного обращения, :func:`numpy.linalg.solve`);
    * ``qr`` — :math:`X = QR`, :math:`R b = Q^T y`; число обусловленности
      системы равно :math:`\kappa(X)`, а не :math:`\kappa(X)^2`;
    * ``pinv`` — :math:`b = X^+ y` через SVD (лекция 3, «Сложно!»).

    При полном столбцовом ранге все три дают одно и то же.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "normal":
        return np.linalg.solve(X.T @ X, X.T @ y)
    if method == "qr":
        q, r = np.linalg.qr(X)
        return np.linalg.solve(r, q.T @ y)
    if method == "pinv":
        return np.linalg.pinv(X) @ y
    raise ValueError(f"неизвестный метод {method!r}; допустимы: {', '.join(METHODS)}")


@dataclass(frozen=True)
class FTest:
    r"""Результат F-теста линейной гипотезы :math:`H_0: R\beta = r`.

    ``critical`` — :math:`F_\alpha(\#r, n - K)` для ``level``; :math:`H_0`
    отвергается, если ``F > critical`` (эквивалентно ``p < 1 − level``).
    """

    F: float
    df1: int
    df2: int
    p: float
    level: float
    critical: float

    @property
    def rejected(self) -> bool:
        return self.F > self.critical


@dataclass(frozen=True)
class OLSResult:
    """Результат МНК-оценивания и все производные величины.

    Все свойства вычисляются из ``X``, ``y`` и ``b``; объект неизменяемый.
    """

    X: np.ndarray
    y: np.ndarray
    b: np.ndarray
    labels: Tuple[str, ...]
    method: str = "normal"
    _cache: Dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    # ---------------------------------------------------------------- размеры
    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def K(self) -> int:
        """Число оцениваемых коэффициентов (вместе с константой)."""
        return self.X.shape[1]

    @property
    def df_resid(self) -> int:
        """Число степеней свободы остатков :math:`n - K`."""
        return self.n - self.K

    @property
    def has_intercept(self) -> bool:
        return bool(np.allclose(self.X[:, 0], 1.0))

    # ------------------------------------------------------- пункты 3, 4, 5
    @property
    def fitted(self) -> np.ndarray:
        r"""Подогнанные значения :math:`\hat y = Xb`."""
        return self.X @ self.b

    @property
    def residuals(self) -> np.ndarray:
        r"""МНК-остатки :math:`e = y - Xb` (пункт 3)."""
        return self.y - self.fitted

    @property
    def ssr(self) -> float:
        r"""Сумма квадратов остатков :math:`SSR = e^T e`."""
        e = self.residuals
        return float(e @ e)

    @property
    def tss(self) -> float:
        r"""Полная сумма квадратов :math:`\sum (y_i - \bar y)^2`."""
        d = self.y - self.y.mean()
        return float(d @ d)

    @property
    def s2(self) -> float:
        r"""Дисперсия МНК-ошибки :math:`s^2 = e^T e / (n - K)` (пункт 4)."""
        return self.ssr / self.df_resid

    @property
    def s(self) -> float:
        """Стандартная ошибка регрессии :math:`s = \\sqrt{s^2}` (SER, пункт 4)."""
        return float(np.sqrt(self.s2))

    @property
    def sigma2_ml(self) -> float:
        r"""ML-оценка дисперсии :math:`e^Te/n = \frac{n-K}{n} s^2` (смещённая)."""
        return self.ssr / self.n

    @property
    def rmse(self) -> float:
        r"""Среднеквадратичная ошибка на обучающей выборке :math:`\sqrt{e^Te/n}`."""
        return float(np.sqrt(self.sigma2_ml))

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.residuals)))

    @property
    def r2(self) -> float:
        r"""Коэффициент детерминации (пункт 5).

        :math:`R^2 = 1 - \langle e^2\rangle/(\langle y^2\rangle - \langle y\rangle^2)`, как в лекции 3.
        """
        e = self.residuals
        return float(1.0 - np.mean(e * e) / (np.mean(self.y**2) - np.mean(self.y) ** 2))

    @property
    def r2_uncentered(self) -> float:
        r"""Нецентрированный :math:`R^2_{uc} = 1 - e^Te / y^Ty`."""
        return float(1.0 - self.ssr / (self.y @ self.y))

    @property
    def r2_adj(self) -> float:
        r"""Скорректированный :math:`\bar R^2 = 1 - (1 - R^2)\frac{n-1}{n-K}`."""
        return float(1.0 - (1.0 - self.r2) * (self.n - 1) / self.df_resid)

    @property
    def loglik(self) -> float:
        r"""Максимум логарифма правдоподобия при нормальных ошибках (лекция 3)."""
        n = self.n
        return float(-n / 2 * (np.log(2 * np.pi) + np.log(self.sigma2_ml) + 1.0))

    @property
    def aic(self) -> float:
        return float(-2 * self.loglik + 2 * (self.K + 1))

    @property
    def bic(self) -> float:
        return float(-2 * self.loglik + np.log(self.n) * (self.K + 1))

    # -------------------------------------------------------------- пункт 6
    @property
    def xtx_inv(self) -> np.ndarray:
        r""":math:`(X^TX)^{-1}`."""
        if "xtx_inv" not in self._cache:
            self._cache["xtx_inv"] = np.linalg.inv(self.X.T @ self.X)
        return self._cache["xtx_inv"]  # type: ignore[return-value]

    @property
    def cov(self) -> np.ndarray:
        r"""Оценка ковариационной матрицы :math:`\widehat{\mathrm{Var}}(b) = s^2 (X^TX)^{-1}`."""
        return self.s2 * self.xtx_inv

    @property
    def se(self) -> np.ndarray:
        r""":math:`SE(b_k) = \sqrt{s^2 ((X^TX)^{-1})_{kk}}`."""
        return np.sqrt(np.diag(self.cov))

    def t_values(self, null: Optional[Sequence[float]] = None) -> np.ndarray:
        r"""t-отношения :math:`t_k = (b_k - \bar\beta_k)/SE(b_k)`; по умолчанию :math:`\bar\beta = 0`."""
        null = np.zeros(self.K) if null is None else np.asarray(null, dtype=float)
        return (self.b - null) / self.se

    @property
    def t(self) -> np.ndarray:
        """t-отношения для гипотез :math:`\\beta_k = 0`."""
        return self.t_values()

    @property
    def p_values(self) -> np.ndarray:
        """Двусторонние p-значения t-тестов :math:`\\beta_k = 0`."""
        return 2.0 * stats.t.sf(np.abs(self.t), self.df_resid)

    def t_critical(self, level: float = 0.95) -> float:
        r"""Квантиль :math:`t^{n-K}_\alpha`, :math:`\alpha` = ``level``."""
        return float(stats.t.ppf(0.5 + level / 2, self.df_resid))

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        r"""Доверительные интервалы :math:`[b_k - SE\,t_\alpha,\; b_k + SE\,t_\alpha]`, форма ``(K, 2)``."""
        half = self.se * self.t_critical(level)
        return np.column_stack([self.b - half, self.b + half])

    # ---------------------------------------------------------------- F-тесты
    def f_test(self, R: np.ndarray, r: Optional[np.ndarray] = None, level: float = 0.95) -> FTest:
        r"""F-тест линейной гипотезы :math:`H_0: R\beta = r` (лекция 3).

        .. math::

            F = \frac{(Rb - r)^T \left[R (X^TX)^{-1} R^T\right]^{-1} (Rb - r)}{\#r \cdot s^2}
        """
        R = np.atleast_2d(np.asarray(R, dtype=float))
        if R.shape[1] != self.K:
            raise ValueError(f"в R должно быть {self.K} столбцов")
        r = np.zeros(R.shape[0]) if r is None else np.asarray(r, dtype=float)
        d = R @ self.b - r
        middle = R @ self.xtx_inv @ R.T
        F = float(d @ np.linalg.solve(middle, d) / (R.shape[0] * self.s2))
        df1, df2 = R.shape[0], self.df_resid
        return FTest(F, df1, df2, float(stats.f.sf(F, df1, df2)), level,
                     float(stats.f.ppf(level, df1, df2)))

    def zero_test(self, labels: Sequence[str], level: float = 0.95) -> FTest:
        """F-тест гипотезы «коэффициенты при ``labels`` все равны нулю»."""
        idx = [self.labels.index(label) for label in labels]
        R = np.zeros((len(idx), self.K))
        R[np.arange(len(idx)), idx] = 1.0
        return self.f_test(R, level=level)

    def overall_f_test(self, level: float = 0.95) -> FTest:
        """F-тест значимости регрессии в целом: все коэффициенты, кроме константы, равны нулю."""
        start = 1 if self.has_intercept else 0
        return self.zero_test(self.labels[start:], level=level)

    # ---------------------------------------------------------------- прочее
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        return np.asarray(X_new, dtype=float) @ self.b

    def coefficient(self, label: str) -> float:
        return float(self.b[self.labels.index(label)])

    def table(self, level: float = 0.95) -> list:
        """Строки таблицы коэффициентов: метка, b, SE, t, p, нижняя и верхняя границы."""
        ci = self.conf_int(level)
        return [
            {"label": lab, "b": float(b), "se": float(se), "t": float(t), "p": float(p),
             "low": float(lo), "high": float(hi)}
            for lab, b, se, t, p, (lo, hi) in zip(self.labels, self.b, self.se, self.t, self.p_values, ci)
        ]

    def summary(self) -> Dict[str, float]:
        """Основные числа модели одним словарём."""
        overall = self.overall_f_test()
        return {
            "n": self.n, "K": self.K, "df_resid": self.df_resid,
            "ssr": self.ssr, "s2": self.s2, "s": self.s, "rmse": self.rmse, "mae": self.mae,
            "r2": self.r2, "r2_adj": self.r2_adj, "r2_uncentered": self.r2_uncentered,
            "F": overall.F, "F_p": overall.p, "F_critical": overall.critical,
            "t_critical": self.t_critical(), "loglik": self.loglik, "aic": self.aic, "bic": self.bic,
        }


def fit_ols(X: np.ndarray, y: np.ndarray, labels: Optional[Sequence[str]] = None,
            method: str = "normal") -> OLSResult:
    """Оценить линейную регрессию по матрице плана ``X`` (константу добавляет вызывающий).

    Raises
    ------
    ValueError
        если наблюдений не больше, чем коэффициентов, или ранг ``X`` неполный
        (нарушено условие отсутствия мультиколлинеарности).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("ожидаются X формы (n, K) и y формы (n,)")
    n, K = X.shape
    if n <= K:
        raise ValueError(f"наблюдений ({n}) должно быть больше, чем коэффициентов ({K})")
    rank = np.linalg.matrix_rank(X)
    if rank < K:
        raise ValueError(f"ранг X равен {rank} < {K}: столбцы линейно зависимы")
    labels = tuple(labels) if labels is not None else tuple(f"b{i}" for i in range(K))
    if len(labels) != K:
        raise ValueError("число меток не совпадает с числом столбцов X")
    return OLSResult(X, y, solve_ols(X, y, method), labels, method)


def fit_ols_data(data: LipoData, columns: Optional[Sequence[str]] = None,
                 method: str = "normal") -> OLSResult:
    """Регрессия :math:`\\log P` на константу и регрессоры ``columns`` (по умолчанию — все 24)."""
    if columns is not None:
        data = data.subset(columns)
    return fit_ols(data.design(), data.y, data.labels(), method)
