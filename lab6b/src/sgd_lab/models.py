r"""Обучающая выборка и три модели задачи 2.

1. :math:`\hat y = w_1 x + b_1` — параметры ``[w1, b1]``;
2. :math:`\hat y = w_1 x^2 + w_2 x + b_1` — ``[w1, w2, b1]``;
3. :math:`\hat y = \mathrm{ReLU}(w_{21}\,\mathrm{ReLU}(w_{11}x + b_{11})
   + w_{22}\,\mathrm{ReLU}(w_{12}x + b_{12}) + b_{21})`
   — ``[w11, b11, w12, b12, w21, w22, b21]``.

Все функции ``predict`` *векторизованы по параметрам*: массив параметров
формы ``(..., p)`` и вход ``x`` формы ``(n,)`` дают прогнозы формы
``(..., n)``. Так численный градиент (:mod:`sgd_lab.numdiff`) вычисляет
все :math:`2p` возмущённых значений функции потерь одним вызовом NumPy.

``weight_mask`` отмечает *веса* (не смещения) — только они входят в
квадратичный штраф задачи 5 (лекция 6, «:math:`L^2`-регуляризация»;
Гудфеллоу и др., разд. 7.1).

Аналитические градиенты (``analytic_grad``) для обучения не используются —
по заданию производные вычисляются численно; они нужны только в тестах,
чтобы проверить численное дифференцирование.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = ["MODELS", "X", "Y", "Model", "get_model", "init_params", "relu"]

X = np.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90])
Y = np.array([0.67, 0.85, 1.05, 1.00, 1.40, 1.50, 1.30, 1.54, 1.55, 1.68, 1.73, 1.60])


def relu(t):
    return np.maximum(t, 0.0)


def _linear(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    return p[..., 0, None] * x + p[..., 1, None]


def _quadratic(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    return p[..., 0, None] * x * x + p[..., 1, None] * x + p[..., 2, None]


def _relu_net(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    h1 = relu(p[..., 0, None] * x + p[..., 1, None])
    h2 = relu(p[..., 2, None] * x + p[..., 3, None])
    return relu(p[..., 4, None] * h1 + p[..., 5, None] * h2 + p[..., 6, None])


def _sse_grad_linear(p, x, y):
    r = _linear(p, x) - y
    return 2.0 * np.array([r @ x, r.sum()])


def _sse_grad_quadratic(p, x, y):
    r = _quadratic(p, x) - y
    return 2.0 * np.array([r @ (x * x), r @ x, r.sum()])


def _sse_grad_relu(p, x, y):
    w11, b11, w12, b12, w21, w22, b21 = p
    z1, z2 = w11 * x + b11, w12 * x + b12
    h1, h2 = relu(z1), relu(z2)
    z = w21 * h1 + w22 * h2 + b21
    r = 2.0 * (relu(z) - y) * (z > 0)            # dJ/dz по каждому наблюдению
    d1, d2 = r * w21 * (z1 > 0), r * w22 * (z2 > 0)
    return np.array([d1 @ x, d1.sum(), d2 @ x, d2.sum(), r @ h1, r @ h2, r.sum()])


@dataclass(frozen=True)
class Model:
    key: str
    title: str
    formula: str
    names: Tuple[str, ...]
    weight_mask: Tuple[bool, ...]
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]
    analytic_grad: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

    @property
    def n_params(self) -> int:
        return len(self.names)

    @property
    def mask(self) -> np.ndarray:
        return np.array(self.weight_mask, dtype=float)

    def __call__(self, p, x):
        return self.predict(np.asarray(p, dtype=float), np.asarray(x, dtype=float))


MODELS: Dict[str, Model] = {
    "linear": Model("linear", "линейная", "ŷ = w₁x + b₁", ("w1", "b1"), (True, False),
                    _linear, _sse_grad_linear),
    "quadratic": Model("quadratic", "квадратичная", "ŷ = w₁x² + w₂x + b₁", ("w1", "w2", "b1"),
                       (True, True, False), _quadratic, _sse_grad_quadratic),
    "relu": Model("relu", "сеть ReLU 1–2–1",
                  "ŷ = ReLU(w₂₁ReLU(w₁₁x + b₁₁) + w₂₂ReLU(w₁₂x + b₁₂) + b₂₁)",
                  ("w11", "b11", "w12", "b12", "w21", "w22", "b21"),
                  (True, False, True, False, True, True, False), _relu_net, _sse_grad_relu),
}


def get_model(key: str) -> Model:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"неизвестная модель {key!r}; есть: {', '.join(MODELS)}") from None


def init_params(model: Model, seed: int, output_bias: Optional[float] = None) -> np.ndarray:
    r"""Начальные параметры.

    Линейная и квадратичная модели — веса и смещение из
    :math:`\mathcal N(0, 1)` (функция потерь выпукла, начало не важно).

    Сеть ReLU — инициализация Хе (Гудфеллоу и др., разд. 8.4):
    :math:`w \sim \mathcal N(0, 2/n_{вх})`, т. е. дисперсия 2 для первого
    слоя (один вход) и 1 для второго (два входа). Смещения скрытых
    нейронов равны 0.1 — небольшое положительное значение, чтобы нейроны
    ReLU в начале были активны; смещение выходного нейрона по умолчанию
    равно среднему отклику :math:`\bar y` («смещение выхода — по маргинальной
    статистике выхода», там же). Без этого выходной ReLU примерно в четверти
    случаев «мёртв» с самого начала: градиент равен нулю и обучение не идёт.
    """
    rng = np.random.default_rng(seed)
    if model.key != "relu":
        return rng.normal(0.0, 1.0, model.n_params)
    w11, w12 = rng.normal(0.0, np.sqrt(2.0), 2)
    w21, w22 = rng.normal(0.0, 1.0, 2)
    b21 = float(Y.mean()) if output_bias is None else float(output_bias)
    return np.array([w11, 0.1, w12, 0.1, w21, w22, b21])
