r"""Двухслойный перцептрон (2) методички и его градиент, выведенный вручную.

Модель (лекция 4, «Простой двухслойный перцептрон»):

.. math::

    \hat y(x) = b^2_1 + (w^2)^T \mathrm{relu}(w^1 x + b^1),

где :math:`x \in \mathbb R^{d}` (:math:`d = 24`), :math:`w^1` — матрица
:math:`H \times d`, :math:`b^1, w^2 \in \mathbb R^H`, :math:`b^2_1` — число;
:math:`H` — число нейронов скрытого слоя. Всего :math:`H(d + 2) + 1`
параметров: 53 при :math:`H = 2` и 833 при :math:`H = 32` — при :math:`H \ge 4`
это больше числа наблюдений (82).

Функция потерь — среднеквадратичная:

.. math::

    J = \frac{1}{2n}\sum_i L_i^2, \qquad L_i = y_i - \hat y(x_i).

Множитель :math:`1/(2n)` вместо суммы из лекции не меняет точку минимума, но
делает градиент не зависящим от числа наблюдений (тогда разумная скорость
обучения :math:`\varepsilon` одна и та же для всей выборки и для её частей
при перекрёстной проверке) и совпадает с функцией потерь ``MLPRegressor`` из
scikit-learn. Градиент (лекция 4, с учётом этого множителя):

.. math::

    D_{b^2} J = -\frac1n\sum_i L_i, \quad
    D_{w^2} J = -\frac1n\sum_i \mathrm{relu}(w^1x_i + b^1)\,L_i, \\
    D_{b^1} J = -\frac1n\sum_i V_i L_i, \quad
    D_{w^1} J = -\frac1n\sum_i V_i L_i \otimes x_i, \qquad
    V_i = w^2 \circ H(w^1 x_i + b^1),

:math:`H(z)` — функция Хевисайда (производная ReLU, :math:`H(0) = 1`, как в лекции).

>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X, y = rng.normal(size=(10, 3)), rng.normal(size=10)
>>> p = init_params(3, 4, rng)
>>> p.n_params                                   # 4·(3 + 2) + 1
21
>>> max_gradient_error(p, X, y) < 1e-7           # сверка с конечными разностями
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

__all__ = [
    "Params",
    "forward",
    "gradients",
    "init_params",
    "input_gradients",
    "loss",
    "max_gradient_error",
    "n_params",
    "numerical_gradients",
    "predict",
    "relu",
]

Seed = Union[int, np.random.Generator, None]


def relu(z: np.ndarray) -> np.ndarray:
    """:math:`\\mathrm{relu}(z) = \\max(0, z)` поэлементно."""
    return np.maximum(z, 0.0)


def n_params(d: int, hidden: int) -> int:
    """Число параметров :math:`H(d + 2) + 1`."""
    return hidden * (d + 2) + 1


@dataclass
class Params:
    r"""Веса перцептрона в обозначениях лекции.

    Attributes
    ----------
    W1:
        :math:`w^1`, форма ``(H, d)``;
    b1:
        :math:`b^1`, форма ``(H,)``;
    w2:
        :math:`w^2`, форма ``(H,)``;
    b2:
        :math:`b^2_1`, число.
    """

    W1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float

    @property
    def hidden(self) -> int:
        return self.W1.shape[0]

    @property
    def d(self) -> int:
        return self.W1.shape[1]

    @property
    def n_params(self) -> int:
        return n_params(self.d, self.hidden)

    def copy(self) -> "Params":
        return Params(self.W1.copy(), self.b1.copy(), self.w2.copy(), float(self.b2))

    def flat(self) -> np.ndarray:
        """Все параметры одним вектором в порядке лекции: :math:`[b^2, w^2, b^1, w^1]`."""
        return np.concatenate([[self.b2], self.w2, self.b1, self.W1.ravel()])

    @classmethod
    def from_flat(cls, vector: np.ndarray, d: int, hidden: int) -> "Params":
        v = np.asarray(vector, dtype=float)
        if v.size != n_params(d, hidden):
            raise ValueError(f"ожидается {n_params(d, hidden)} параметров, получено {v.size}")
        h = hidden
        return cls(v[1 + 2 * h:].reshape(h, d).copy(), v[1 + h:1 + 2 * h].copy(), v[1:1 + h].copy(), float(v[0]))

    def norm(self) -> float:
        return float(np.linalg.norm(self.flat()))


def init_params(d: int, hidden: int, seed: Seed = None, scheme: str = "he") -> Params:
    r"""Начальные веса :math:`w^2[0], w^1[0], b^2[0], b^1[0]` (шаг 2 алгоритма лекции).

    * ``he`` — :math:`w^1_{ij} \sim N(0, 2/d)` (инициализация Хе для ReLU:
      дисперсия сигнала не растёт и не гаснет от слоя к слою),
      :math:`w^2_j \sim N(0, 1/H)`, смещения нулевые;
    * ``glorot`` — равномерное распределение Глоро, как в scikit-learn:
      :math:`U(-a, a)`, :math:`a = \sqrt{6/(n_{in} + n_{out})}`, смещения тоже из
      :math:`U(-a, a)`.

    Нулевые начальные веса не годятся: тогда все скрытые нейроны одинаковы и
    остаются одинаковыми на всех итерациях (симметрия не нарушается).
    """
    rng = np.random.default_rng(seed)
    if scheme == "he":
        return Params(rng.normal(0.0, np.sqrt(2.0 / d), size=(hidden, d)), np.zeros(hidden),
                      rng.normal(0.0, np.sqrt(1.0 / hidden), size=hidden), 0.0)
    if scheme == "glorot":
        a1 = np.sqrt(6.0 / (d + hidden))
        a2 = np.sqrt(6.0 / (hidden + 1))
        return Params(rng.uniform(-a1, a1, size=(hidden, d)), rng.uniform(-a1, a1, size=hidden),
                      rng.uniform(-a2, a2, size=hidden), float(rng.uniform(-a2, a2)))
    raise ValueError(f"неизвестная схема инициализации {scheme!r} (he, glorot)")


def forward(params: Params, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прямой проход: прогноз :math:`\\hat y`, входы скрытого слоя и их выходы.

    :math:`Z = Xw^{1T} + b^1` (форма ``(n, H)``), :math:`A = relu(Z)`.
    """
    Z = X @ params.W1.T + params.b1
    A = relu(Z)
    return A @ params.w2 + params.b2, Z, A


def predict(params: Params, X: np.ndarray) -> np.ndarray:
    return forward(params, X)[0]


def loss(params: Params, X: np.ndarray, y: np.ndarray) -> float:
    r""":math:`J = \frac{1}{2n}\sum_i (y_i - \hat y_i)^2`."""
    r = y - predict(params, X)
    return float(r @ r) / (2 * len(y))


def gradients(params: Params, X: np.ndarray, y: np.ndarray) -> Tuple[float, Params]:
    """Значение :math:`J` и его градиент по формулам лекции 4 (векторизовано по наблюдениям)."""
    n = len(y)
    yhat, Z, A = forward(params, X)
    L = y - yhat                                   # L_i
    V = params.w2 * (Z >= 0.0)                     # V_i = w² ∘ H(w¹x_i + b¹), форма (n, H)
    VL = V * L[:, None]
    grad = Params(W1=-(VL.T @ X) / n,              # D_w1 J = −(1/n) Σ V_i L_i ⊗ x_i
                  b1=-VL.sum(axis=0) / n,          # D_b1 J = −(1/n) Σ V_i L_i
                  w2=-(A.T @ L) / n,               # D_w2 J = −(1/n) Σ relu(·) L_i
                  b2=-float(L.sum()) / n)          # D_b2 J = −(1/n) Σ L_i
    return float(L @ L) / (2 * n), grad


def numerical_gradients(params: Params, X: np.ndarray, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Градиент центральными разностями (для проверки ручных формул), вектор в порядке :meth:`Params.flat`."""
    theta = params.flat()
    out = np.empty_like(theta)
    for k in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[k] += h
        minus[k] -= h
        out[k] = (loss(Params.from_flat(plus, params.d, params.hidden), X, y)
                  - loss(Params.from_flat(minus, params.d, params.hidden), X, y)) / (2 * h)
    return out


def max_gradient_error(params: Params, X: np.ndarray, y: np.ndarray) -> float:
    """Наибольшее расхождение аналитического и численного градиентов."""
    _, grad = gradients(params, X, y)
    return float(np.max(np.abs(grad.flat() - numerical_gradients(params, X, y))))


def input_gradients(params: Params, X: np.ndarray) -> np.ndarray:
    r"""Чувствительность прогноза к входам, форма ``(n, d)``.

    :math:`\partial\hat y/\partial x = (w^1)^T (w^2 \circ H(w^1x + b^1))`.
    """
    _, Z, _ = forward(params, X)
    return (params.w2 * (Z >= 0.0)) @ params.W1
