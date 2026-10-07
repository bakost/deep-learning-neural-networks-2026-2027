r"""Сценарии экспериментов отчёта.

Каждый запуск обучения описывается :class:`Job` (архитектура, зерно,
закон изменения скорости обучения, разбиение на обучение/контроль) и
выполняется функцией :func:`run_job`. Запуски независимы и детерминированы
(всё случайное определяется зерном), поэтому их можно выполнять в
нескольких процессах (:func:`run_jobs`) — результат от этого не меняется.

Основные параметры отчёта (пункт 3 задания — «зафиксируйте скорость
обучения и точность»): :data:`EPS` :math:`= 0.1`, :data:`DELTA`
:math:`= 10^{-4}`, :data:`MAX_ITER` :math:`= 3\cdot10^5`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import LipoData, Standardizer, duplicate_groups, load_lipo
from .network import Params, forward, gradients, init_params, n_params
from .schedules import Schedule, constant
from .training import gradient_descent

__all__ = [
    "DELTA",
    "EPS",
    "MAX_ITER",
    "REDUCED_OLS_COLUMNS",
    "SIZES",
    "Job",
    "RunSummary",
    "cv_splits",
    "interpolation_floor",
    "kink_activity",
    "linear_stability_bound",
    "ols_cv",
    "run_job",
    "run_jobs",
]

SIZES: Tuple[int, ...] = (2, 4, 8, 16, 32)
EPS = 0.1
DELTA = 1e-4
MAX_ITER = 300_000

#: Регрессоры модели после пошагового исключения в ЛР3 (7 значимых).
REDUCED_OLS_COLUMNS = ("R3N", "RCOR", "ROR", "C", "HBA1", "HBA2", "PSA")

_DATA: Optional[LipoData] = None


def _data() -> LipoData:
    global _DATA
    if _DATA is None:
        _DATA = load_lipo()
    return _DATA


@dataclass(frozen=True)
class Job:
    """Один запуск обучения.

    ``train``/``val`` — номера веществ обучающей и контрольной частей
    (``None`` — обучение на всей выборке без контроля).
    """

    tag: str
    hidden: int
    seed: int
    schedule: Schedule = field(default_factory=lambda: constant(EPS))
    delta: float = DELTA
    max_iter: int = MAX_ITER
    record_every: int = 100
    batch_size: Optional[int] = None
    train: Optional[Tuple[int, ...]] = None
    val: Optional[Tuple[int, ...]] = None
    init: str = "he"
    keep_params: bool = True


@dataclass
class RunSummary:
    """Итог запуска: статус, число итераций, ошибки в единицах :math:`\\log P`, история."""

    job: Job
    status: str
    iterations: int
    loss: float
    grad_norm: float
    rmse: float
    history: Dict[str, np.ndarray]
    y_std: float
    flips: int = 0
    near_kink: int = 0
    dead_units: int = 0
    params: Optional[Params] = None
    val_pred: Optional[np.ndarray] = None

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    def rmse_curve(self, key: str = "loss") -> np.ndarray:
        r"""История RMSE в единицах :math:`\log P`: :math:`\sqrt{2J}\,\sigma_y`."""
        return np.sqrt(2.0 * self.history[key]) * self.y_std


def kink_activity(params: Params, X: np.ndarray, y: np.ndarray, lr: float, steps: int = 100,
                  tol: float = 1e-3) -> Tuple[int, int, int]:
    """Поведение около изломов ReLU в конечной точке.

    Делается ещё ``steps`` шагов градиентного спуска (на копии весов) и
    считается, сколько раз за них нейроны переключались между активным и
    неактивным состоянием на каком-либо веществе. Также возвращаются число
    пар «вещество–нейрон» с :math:`|z| <` ``tol`` и число «мёртвых» нейронов
    (неактивных на всех веществах).
    """
    p = params.copy()
    _, Z, _ = forward(p, X)
    near = int(np.sum(np.abs(Z) < tol))
    dead = int(np.sum(np.all(Z < 0, axis=0)))
    pattern = Z >= 0
    flips = 0
    for _ in range(steps):
        _, g = gradients(p, X, y)
        p.W1 -= lr * g.W1
        p.b1 -= lr * g.b1
        p.w2 -= lr * g.w2
        p.b2 -= lr * g.b2
        _, Z, _ = forward(p, X)
        new = Z >= 0
        flips += int(np.sum(new != pattern))
        pattern = new
    return flips, near, dead


def run_job(job: Job) -> RunSummary:
    """Выполнить запуск: стандартизация по обучающей части, инициализация, градиентный спуск."""
    data = _data()
    train = np.arange(data.n) if job.train is None else np.asarray(job.train)
    scaler = Standardizer.fit(data.X[train], data.y[train])
    X, y = scaler.transform(data.X[train], data.y[train])
    X_val = y_val = None
    if job.val is not None:
        val = np.asarray(job.val)
        X_val, y_val = scaler.transform(data.X[val], data.y[val])
    params0 = init_params(data.k, job.hidden, job.seed, job.init)
    res = gradient_descent(params0, X, y, lr=job.schedule, delta=job.delta, max_iter=job.max_iter,
                           record_every=job.record_every, X_val=X_val, y_val=y_val,
                           batch_size=job.batch_size, seed=job.seed)
    flips = near = dead = 0
    if job.batch_size is None and res.status != "diverged":
        flips, near, dead = kink_activity(res.params, X, y, job.schedule(res.iterations))
    rmse = float(np.sqrt(2.0 * res.loss) * scaler.y_std) if np.isfinite(res.loss) else float("inf")
    val_pred = None
    if X_val is not None and res.status != "diverged":
        val_pred = scaler.inverse_y(forward(res.params, X_val)[0])
    return RunSummary(job, res.status, res.iterations, res.loss, res.grad_norm, rmse, res.history,
                      scaler.y_std, flips, near, dead, res.params if job.keep_params else None, val_pred)


def run_jobs(jobs: Sequence[Job], workers: Optional[int] = None) -> List[RunSummary]:
    """Выполнить запуски параллельно (порядок результатов совпадает с порядком ``jobs``).

    ``workers=1`` — последовательно в текущем процессе. В рабочих процессах
    BLAS ограничивается одним потоком: матрицы маленькие, и многопоточность
    внутри процесса только мешает.
    """
    workers = workers or min(8, os.cpu_count() or 1)
    if workers <= 1 or len(jobs) <= 1:
        return [run_job(job) for job in jobs]
    for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_job, jobs, chunksize=1))


# ------------------------------------------------------------------ опорные величины
def interpolation_floor(data: Optional[LipoData] = None) -> float:
    r"""Наименьшая возможная RMSE на обучающей выборке для *любой* функции :math:`f(x)`.

    Вещества с одинаковыми признаками сеть не различает; лучшее, что она
    может, — выдать для них среднее по группе. Отсюда
    :math:`\min RMSE = \sqrt{SS_{pe}/n}` (в ЛР3 — «чистая ошибка»).
    """
    data = data or _data()
    ss = sum(float(np.sum((data.y[g] - data.y[g].mean()) ** 2)) for g in duplicate_groups(data.X))
    return float(np.sqrt(ss / data.n))


def linear_stability_bound(data: Optional[LipoData] = None) -> Dict[str, float]:
    r"""Граница устойчивости градиентного спуска для *линейной* модели на тех же признаках.

    Для :math:`J = \frac{1}{2n}\lVert y - X\theta\rVert^2` гессиан равен
    :math:`X^TX/n`; спуск с постоянным шагом сходится при
    :math:`\varepsilon < 2/\lambda_{max}`, а скорость сходимости по самому
    «пологому» направлению определяется :math:`\lambda_{min}`.
    """
    data = data or _data()
    scaler = Standardizer.fit(data.X, data.y)
    Z = np.column_stack([np.ones(data.n), scaler.transform_x(data.X)])
    lam = np.linalg.eigvalsh(Z.T @ Z / data.n)
    return {"lambda_max": float(lam[-1]), "lambda_min": float(lam[0]), "condition": float(lam[-1] / lam[0]),
            "eps_max": float(2.0 / lam[-1])}


def cv_splits(n: int, k: int = 5, repeats: int = 2, seed: int = 2026) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Разбиения для повторной k-кратной перекрёстной проверки: список пар (обучение, контроль)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        for fold in np.array_split(rng.permutation(n), k):
            train = np.setdiff1d(np.arange(n), fold)
            out.append((tuple(int(i) for i in train), tuple(int(i) for i in np.sort(fold))))
    return out


def ols_cv(splits: Sequence[Tuple[Sequence[int], Sequence[int]]], columns: Optional[Sequence[str]] = None,
           data: Optional[LipoData] = None) -> np.ndarray:
    """Прогнозы МНК на контрольных частях (для сравнения с перцептроном на тех же разбиениях).

    Возвращает массив ``(repeats, n)``: прогноз каждого вещества в каждом повторении.
    """
    data = data or _data()
    X = data.X if columns is None else data.X[:, [data.columns.index(c) for c in columns]]
    X = np.column_stack([np.ones(data.n), X])
    folds_per_rep = _folds_per_repeat(splits, data.n)
    reps = len(splits) // folds_per_rep
    pred = np.empty((reps, data.n))
    for s, (train, val) in enumerate(splits):
        coef, *_ = np.linalg.lstsq(X[list(train)], data.y[list(train)], rcond=None)
        pred[s // folds_per_rep, list(val)] = X[list(val)] @ coef
    return pred


def _folds_per_repeat(splits, n: int) -> int:
    total = 0
    for k, (_, val) in enumerate(splits, start=1):
        total += len(val)
        if total == n:
            return k
    raise ValueError("разбиения не покрывают выборку целиком")


def param_count_table(d: int = 24, sizes: Sequence[int] = SIZES) -> Dict[int, int]:
    """Число параметров перцептрона для каждого размера скрытого слоя."""
    return {h: n_params(d, h) for h in sizes}
