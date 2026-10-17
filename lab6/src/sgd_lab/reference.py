r"""Эталонные решения для проверки обучения.

* Линейная и квадратичная модели линейны по параметрам:
  :math:`\hat y = A w`, где строки :math:`A` — :math:`(x_i, 1)` или
  :math:`(x_i^2, x_i, 1)`. Минимум :math:`\tilde J = \lVert Aw - y\rVert^2 + \lambda w^T D w`
  (:math:`D` — диагональная маска весов) — решение нормальных уравнений
  :math:`(A^TA + \lambda D) w = A^T y` (гребневая регрессия; при
  :math:`\lambda = 0` — МНК, Гудфеллоу и др., разд. 4.5, 7.1.1).
  Гессиан :math:`H = 2(A^TA + \lambda D)` постоянен; градиентный спуск устойчив
  при :math:`\varepsilon < 2/\lambda_{max}(H)` и быстрее всего сходится при
  :math:`\varepsilon = 2/(\lambda_{min} + \lambda_{max})`, множитель
  уменьшения ошибки за шаг :math:`(\kappa - 1)/(\kappa + 1)`,
  :math:`\kappa = \lambda_{max}/\lambda_{min}` (разд. 4.3.1, 8.2.1).
* Сеть ReLU: глобальный минимум ищется многократным запуском симплекс-метода
  Нелдера — Мида из SciPy из случайных точек (лекция 6: «в простых моделях
  можно … запустить алгоритм из нескольких точек и выбрать минимальное
  значение»). Это только эталон — сеть обучается градиентными методами.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .models import Model, init_params
from .training import Objective

__all__ = ["design_matrix", "hessian", "linear_solution", "loo_press", "minibatch_lr_bound", "multistart_minimum",
           "stability"]


def design_matrix(model: Model, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if model.key == "linear":
        return np.column_stack([x, np.ones_like(x)])
    if model.key == "quadratic":
        return np.column_stack([x * x, x, np.ones_like(x)])
    raise ValueError("модель нелинейна по параметрам")


def hessian(model: Model, x: np.ndarray, l2: float = 0.0) -> np.ndarray:
    A = design_matrix(model, x)
    return 2.0 * (A.T @ A + l2 * np.diag(model.mask))


def linear_solution(model: Model, x: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
    A = design_matrix(model, x)
    return np.linalg.solve(A.T @ A + l2 * np.diag(model.mask), A.T @ np.asarray(y, dtype=float))


def stability(model: Model, x: np.ndarray, l2: float = 0.0) -> Dict[str, float]:
    """Собственные числа гессиана и характерные скорости обучения для спуска."""
    ev = np.linalg.eigvalsh(hessian(model, x, l2))
    lo, hi = float(ev[0]), float(ev[-1])
    kappa = hi / lo
    return {
        "lambda_min": lo, "lambda_max": hi, "kappa": kappa,
        "lr_max": 2.0 / hi, "lr_opt": 2.0 / (lo + hi), "rate_opt": (kappa - 1.0) / (kappa + 1.0),
    }


def loo_press(model: Model, x: np.ndarray, y: np.ndarray, l2: float = 0.0) -> float:
    """Сумма квадратов ошибок скользящего контроля (каждое наблюдение прогнозируется моделью без него)."""
    press = 0.0
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        w = linear_solution(model, x[keep], y[keep], l2)
        press += float((model(w, x[i:i + 1])[0] - y[i]) ** 2)
    return press


def minibatch_lr_bound(model: Model, x: np.ndarray, batch: int) -> float:
    r"""Оценка границы устойчивости СГС по худшему мини-пакету.

    Гессиан оценки :math:`\tfrac{n}{m}\sum_{i\in B}` равен
    :math:`\tfrac{2n}{m}A_B^TA_B`. Берётся максимум его наибольшего
    собственного числа по пакетам из :math:`m` *самых больших* по
    :math:`\lVert a_i\rVert` наблюдений (оценка сверху для любого пакета).
    """
    A = design_matrix(model, x)
    n = len(x)
    order = np.argsort(-np.sum(A * A, axis=1))[:batch]
    Ab = A[order]
    lam = np.linalg.eigvalsh(2.0 * n / batch * Ab.T @ Ab)[-1]
    return float(2.0 / lam)


def multistart_minimum(model: Model, x: np.ndarray, y: np.ndarray, l2: float = 0.0,
                       starts: int = 100, seed: int = 12345,
                       x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """Наименьший найденный минимум :math:`\\tilde J` (Нелдер — Мид из ``starts`` точек).

    Возвращает лучшие параметры, значение :math:`\\tilde J` и отсортированный
    массив значений во всех запусках.
    """
    from scipy.optimize import minimize

    obj = Objective(model, x, y, l2)
    rng = np.random.default_rng(seed)
    best_w, best_f, vals = None, np.inf, []
    inits = [init_params(model, int(s)) for s in rng.integers(0, 2**31, starts)]
    if x0 is not None:
        inits.insert(0, np.asarray(x0, dtype=float))
    for w0 in inits:
        w = w0
        for _ in range(3):  # перезапуск симплекса уточняет минимум
            res = minimize(lambda v: float(obj(v)), w, method="Nelder-Mead",
                           options={"maxiter": 20_000, "maxfev": 40_000, "xatol": 1e-10, "fatol": 1e-14})
            w = res.x
        f = float(obj(w))
        vals.append(f)
        if f < best_f:
            best_w, best_f = w, f
    return best_w, best_f, np.sort(np.array(vals))
