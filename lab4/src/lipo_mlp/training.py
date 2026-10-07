r"""Градиентный спуск для двухслойного перцептрона (алгоритм лекции 4).

1. Задать скорость обучения :math:`\varepsilon > 0` и точность :math:`\delta > 0`.
2. Задать начальные веса, :math:`j = 0`.
3. Цикл:

   3.1–3.3. вычислить :math:`L_i[j]`, :math:`V_i[j]` и градиент :math:`DJ[j]`;

   3.4. если :math:`\lVert DJ[j]\rVert < \delta` — выход, результат — текущие веса;

   3.5. иначе :math:`\theta[j+1] = \theta[j] - \varepsilon_j\,DJ[j]`, :math:`j \to j + 1`.

Норма — евклидова норма вектора всех :math:`H(d+2)+1` производных. В
лекции градиент взят с обратным знаком (без множителя :math:`-2`) и
прибавляется; здесь :math:`DJ` — настоящий градиент, и он вычитается, это
то же самое.

Кроме условия 3.4 цикл прерывается, если число итераций достигло
``max_iter`` (алгоритм не сошёлся за отведённое время) или функция потерь
стала бесконечной / выросла в :math:`10^6` раз (спуск разошёлся — шаг
слишком велик).

Режим ``batch_size`` (мини-пакеты, лекция 4) нужен только для одного
эксперимента в отчёте: веса обновляются по градиенту случайного
мини-пакета, а условие остановки проверяется по *полному* градиенту в
точках записи истории.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from .network import Params, gradients, loss
from .schedules import Schedule, constant

__all__ = ["TrainResult", "gradient_descent"]


@dataclass
class TrainResult:
    """Результат обучения.

    Attributes
    ----------
    params:
        веса после последней итерации;
    iterations:
        число выполненных обновлений весов (для сошедшегося спуска — номер
        итерации :math:`j`, на которой выполнилось :math:`\\lVert DJ\\rVert < \\delta`);
    status:
        ``"converged"``, ``"max_iter"`` или ``"diverged"``;
    loss, grad_norm:
        :math:`J` и :math:`\\lVert DJ\\rVert` в момент остановки;
    history:
        массивы ``iteration``, ``loss``, ``grad_norm``, ``lr`` (и ``val_loss``,
        если задана контрольная выборка), записанные каждые ``record_every``
        итераций и в момент остановки.
    """

    params: Params
    iterations: int
    status: str
    loss: float
    grad_norm: float
    history: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _norm(g: Params) -> float:
    return float(np.sqrt(g.b2 * g.b2 + g.w2 @ g.w2 + g.b1 @ g.b1 + np.sum(g.W1 * g.W1)))


def gradient_descent(params: Params, X: np.ndarray, y: np.ndarray,
                     lr: Union[float, Schedule] = 0.05, delta: float = 1e-4, max_iter: int = 100_000,
                     record_every: int = 100, X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None, batch_size: Optional[int] = None,
                     seed: Union[int, np.random.Generator, None] = None,
                     divergence_factor: float = 1e6) -> TrainResult:
    """Обучить перцептрон градиентным спуском, начиная с ``params`` (исходный объект не меняется)."""
    if delta <= 0 or max_iter < 0 or record_every < 1:
        raise ValueError("нужны δ > 0, max_iter ≥ 0, record_every ≥ 1")
    schedule = lr if isinstance(lr, Schedule) else constant(float(lr))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    p = params.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if batch_size else None
    cursor = 0
    hist: Dict[str, list] = {"iteration": [], "loss": [], "grad_norm": [], "lr": []}
    if X_val is not None:
        hist["val_loss"] = []

    def record(j: int, J: float, g: float) -> None:
        hist["iteration"].append(j)
        hist["loss"].append(J)
        hist["grad_norm"].append(g)
        hist["lr"].append(schedule(j))
        if X_val is not None:
            hist["val_loss"].append(loss(p, X_val, y_val))

    J0 = None
    status = "max_iter"
    j = 0
    J, grad = gradients(p, X, y)
    g = _norm(grad)
    while True:
        if J0 is None:
            J0 = J
        if not np.isfinite(J) or J > divergence_factor * max(J0, 1e-12):
            status = "diverged"
            record(j, J, g)
            break
        if g < delta:
            status = "converged"
            record(j, J, g)
            break
        if j >= max_iter:
            record(j, J, g)
            break
        if j % record_every == 0:
            record(j, J, g)
        # шаг 3.5: θ[j+1] = θ[j] − ε_j DJ[j]
        eps = schedule(j)
        if batch_size:
            if cursor + batch_size > n:
                order = rng.permutation(n)
                cursor = 0
            idx = order[cursor:cursor + batch_size]
            cursor += batch_size
            _, step_grad = gradients(p, X[idx], y[idx])
        else:
            step_grad = grad
        p.W1 -= eps * step_grad.W1
        p.b1 -= eps * step_grad.b1
        p.w2 -= eps * step_grad.w2
        p.b2 -= eps * step_grad.b2
        j += 1
        if batch_size and j % record_every != 0 and j < max_iter:
            continue          # полный градиент для мини-пакетов считаем только в точках записи
        J, grad = gradients(p, X, y)
        g = _norm(grad)
    history = {k: np.asarray(v, dtype=float) for k, v in hist.items()}
    return TrainResult(p, j, status, float(J), float(g), history)
