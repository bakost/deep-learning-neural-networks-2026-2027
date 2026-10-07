r"""Скорость обучения :math:`\varepsilon_j` как функция номера итерации :math:`j`.

Лекция 4 («Проблемы градиентного спуска»): если шаг в сторону антиградиента
слишком велик, скорость обучения можно уменьшать на каждой итерации. Здесь
собраны распространённые законы уменьшения:

=================  ==========================================  ====================
Закон              :math:`\varepsilon_j`                       :math:`\sum_j \varepsilon_j`
=================  ==========================================  ====================
постоянный         :math:`\varepsilon_0`                       :math:`\infty`
обратный           :math:`\varepsilon_0 / (1 + j/\tau)`        :math:`\infty` (как :math:`\ln j`)
степенной          :math:`\varepsilon_0 / (1 + j/\tau)^{p}`    :math:`\infty` при :math:`p \le 1`
экспоненциальный   :math:`\varepsilon_0\,\gamma^{j}`           :math:`\varepsilon_0/(1-\gamma) < \infty`
ступенчатый        :math:`\varepsilon_0\,\gamma^{\lfloor j/m\rfloor}`  :math:`< \infty`
=================  ==========================================  ====================

Последний столбец важен: сумма шагов ограничивает путь, который может
пройти алгоритм. Если она конечна, спуск может «застрять» на полпути, так и
не дойдя до точки с малым градиентом.

>>> s = inverse_time(0.1, tau=100)
>>> s(0), round(s(100), 3)
(0.1, 0.05)
>>> round(exponential(0.1, 0.99).total(), 6)       # ε₀/(1 − γ)
10.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Schedule", "constant", "exponential", "inverse_time", "power", "step"]


@dataclass(frozen=True)
class Schedule:
    """Закон изменения скорости обучения; вызывается как ``schedule(j)``."""

    kind: str
    eps0: float
    tau: float = 1.0
    gamma: float = 1.0
    p: float = 1.0
    every: int = 1

    def __post_init__(self) -> None:
        if self.eps0 <= 0:
            raise ValueError("начальная скорость обучения должна быть положительной")
        if self.kind not in ("constant", "inverse", "power", "exponential", "step"):
            raise ValueError(f"неизвестный закон {self.kind!r}")
        if self.tau <= 0 or not 0 < self.gamma <= 1 or self.p <= 0 or self.every < 1:
            raise ValueError("недопустимые параметры закона")

    def __call__(self, j: int) -> float:
        if self.kind == "constant":
            return self.eps0
        if self.kind == "inverse":
            return self.eps0 / (1.0 + j / self.tau)
        if self.kind == "power":
            return self.eps0 / (1.0 + j / self.tau) ** self.p
        if self.kind == "exponential":
            return self.eps0 * self.gamma**j
        return self.eps0 * self.gamma ** (j // self.every)

    def total(self) -> float:
        """:math:`\\sum_{j \\ge 0} \\varepsilon_j` (``inf`` для расходящихся рядов)."""
        if self.kind in ("constant", "inverse") or (self.kind == "power" and self.p <= 1):
            return math.inf
        if self.kind == "exponential":
            return self.eps0 / (1.0 - self.gamma) if self.gamma < 1 else math.inf
        if self.kind == "step":
            return self.eps0 * self.every / (1.0 - self.gamma) if self.gamma < 1 else math.inf
        return math.fsum(self(j) for j in range(10**6))  # pragma: no cover - степенной с p > 1

    def label(self) -> str:
        """Короткая подпись для таблиц и легенд."""
        if self.kind == "constant":
            return f"постоянная ε = {self.eps0:g}"
        if self.kind == "inverse":
            return f"ε₀/(1 + j/τ), ε₀ = {self.eps0:g}, τ = {self.tau:g}"
        if self.kind == "power":
            return f"ε₀/(1 + j/τ)^{self.p:g}, ε₀ = {self.eps0:g}, τ = {self.tau:g}"
        if self.kind == "exponential":
            return f"ε₀·γ^j, ε₀ = {self.eps0:g}, γ = {self.gamma:g}"
        return f"ε₀·γ^⌊j/m⌋, ε₀ = {self.eps0:g}, γ = {self.gamma:g}, m = {self.every}"


def constant(eps0: float) -> Schedule:
    return Schedule("constant", eps0)


def inverse_time(eps0: float, tau: float) -> Schedule:
    return Schedule("inverse", eps0, tau=tau)


def power(eps0: float, tau: float, p: float) -> Schedule:
    return Schedule("power", eps0, tau=tau, p=p)


def exponential(eps0: float, gamma: float) -> Schedule:
    return Schedule("exponential", eps0, gamma=gamma)


def step(eps0: float, gamma: float, every: int) -> Schedule:
    return Schedule("step", eps0, gamma=gamma, every=every)
