r"""Обучение перцептрона эволюционными алгоритмами (без градиента).

Эволюционный алгоритм минимизирует ту же функцию потерь
:math:`J = \frac{1}{2n}\sum_i L_i^2`, что и градиентный спуск, но использует
только её *значения*: веса — это «особи», :math:`J` — «приспособленность».
Производная ReLU не нужна, поэтому изломы ReLU, на которых застревает
критерий :math:`\lVert DJ\rVert < \delta` (раздел 3 отчёта), алгоритму не мешают.

Два варианта:

* :func:`evolution_strategy` — **свой**: эволюционная стратегия
  :math:`(\mu/\mu, \lambda)` с самоадаптацией шага мутации (Рехенберг,
  Швефель). У каждой особи свой шаг :math:`\sigma`; потомок получает
  :math:`\sigma' = \bar\sigma\, e^{\tau N(0,1)}` и
  :math:`\theta' = \bar\theta + \sigma' N(0, I)`, где :math:`\bar\theta,
  \bar\sigma` — среднее :math:`\mu` лучших родителей (промежуточная
  рекомбинация; для :math:`\sigma` — среднее геометрическое). Из
  :math:`\lambda` потомков в следующее поколение проходят :math:`\mu`
  лучших (отбор «через запятую»: родители не выживают, поэтому шаг
  :math:`\sigma` может уменьшаться). Хорошие :math:`\sigma` «выживают»
  вместе с хорошими :math:`\theta` — так шаг настраивается сам.
* :func:`cma_es` — **пакет** ``cma`` (Н. Хансен): CMA-ES, эволюционная
  стратегия, которая адаптирует не только длину шага, но и полную
  ковариационную матрицу мутаций — то есть выучивает форму «долины»
  функции потерь. Это стандартный эволюционный метод для непрерывной
  оптимизации.

Приспособленность всей популяции считается одним векторизованным проходом
(:func:`population_loss`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from .network import Params, init_params, n_params

__all__ = ["ESResult", "cma_es", "evolution_strategy", "population_loss"]

Seed = Union[int, np.random.Generator, None]


def _unpack(thetas: np.ndarray, d: int, hidden: int):
    """Разрезать матрицу особей ``(λ, P)`` на веса в порядке :meth:`Params.flat`: [b², w², b¹, w¹]."""
    h = hidden
    b2 = thetas[:, 0]
    w2 = thetas[:, 1:1 + h]
    b1 = thetas[:, 1 + h:1 + 2 * h]
    W1 = thetas[:, 1 + 2 * h:].reshape(-1, h, d)
    return W1, b1, w2, b2


def population_loss(thetas: np.ndarray, X: np.ndarray, y: np.ndarray, hidden: int) -> np.ndarray:
    r""":math:`J(\theta)` для каждой строки ``thetas`` (форма ``(λ, P)``) — без цикла по особям."""
    thetas = np.atleast_2d(thetas)
    W1, b1, w2, b2 = _unpack(thetas, X.shape[1], hidden)
    Z = np.einsum("nd,lhd->lnh", X, W1) + b1[:, None, :]
    yhat = np.einsum("lnh,lh->ln", np.maximum(Z, 0.0), w2) + b2[:, None]
    r = y[None, :] - yhat
    return np.einsum("ln,ln->l", r, r) / (2 * len(y))


@dataclass
class ESResult:
    """Итог эволюционного обучения.

    ``history`` — массивы ``evaluations`` (сколько раз вычислена :math:`J`),
    ``best_loss`` (лучшая найденная :math:`J`) и ``sigma`` (средний шаг
    мутации), записанные после каждого поколения.
    """

    params: Params
    loss: float
    evaluations: int
    generations: int
    seconds: float
    history: Dict[str, np.ndarray] = field(default_factory=dict)


def _record(hist, evals, best, sigma):
    hist["evaluations"].append(evals)
    hist["best_loss"].append(best)
    hist["sigma"].append(sigma)


def evolution_strategy(X: np.ndarray, y: np.ndarray, hidden: int, max_evals: int = 200_000,
                       lam: int = 60, mu: Optional[int] = None, sigma0: float = 0.1, tau_factor: float = 0.2,
                       seed: Seed = None, target: Optional[float] = None) -> ESResult:
    r"""Своя :math:`(\mu/\mu, \lambda)`-эволюционная стратегия с самоадаптацией шага.

    Parameters
    ----------
    lam, mu:
        число потомков и родителей (по умолчанию :math:`\mu = \lambda/4`);
    sigma0:
        начальный шаг мутации; начальное среднее — веса из :func:`init_params`
        (инициализация Хе, как у градиентного спуска);
    tau_factor:
        скорость самоадаптации :math:`\tau = \text{tau\_factor}/\sqrt{P}`.
        Классическое значение 1 (или :math:`1/\sqrt2`) здесь приводит к
        преждевременному «замерзанию»: :math:`\sigma` падает до
        :math:`10^{-5}`–:math:`10^{-9}` задолго до минимума. Значение 0.2
        подобрано на :math:`H = 2, 8` (см. отчёт, раздел 9);
    max_evals:
        бюджет вычислений функции потерь;
    target:
        остановиться, как только лучшая :math:`J` станет не больше ``target``.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    P = n_params(d, hidden)
    mu = mu or max(1, lam // 4)
    tau = tau_factor / math.sqrt(P)            # скорость самоадаптации шага
    mean = init_params(d, hidden, rng).flat()
    log_sigma = math.log(sigma0)
    best_theta, best = mean.copy(), float(population_loss(mean, X, y, hidden)[0])
    evals, gen = 1, 0
    hist: Dict[str, list] = {"evaluations": [], "best_loss": [], "sigma": []}
    _record(hist, evals, best, sigma0)
    while evals + lam <= max_evals and not (target is not None and best <= target):
        sig = np.exp(log_sigma + tau * rng.standard_normal(lam))        # мутация шага
        offspring = mean + sig[:, None] * rng.standard_normal((lam, P))  # мутация весов
        fit = population_loss(offspring, X, y, hidden)
        evals += lam
        gen += 1
        order = np.argsort(fit)[:mu]                                     # отбор (μ, λ)
        mean = offspring[order].mean(axis=0)                             # рекомбинация
        log_sigma = float(np.log(sig[order]).mean())
        if fit[order[0]] < best:
            best, best_theta = float(fit[order[0]]), offspring[order[0]].copy()
        _record(hist, evals, best, math.exp(log_sigma))
    params = Params.from_flat(best_theta, d, hidden)
    return ESResult(params, best, evals, gen, time.perf_counter() - t0,
                    {k: np.asarray(v, dtype=float) for k, v in hist.items()})


def cma_es(X: np.ndarray, y: np.ndarray, hidden: int, max_evals: int = 200_000, lam: Optional[int] = None,
           sigma0: float = 0.1, seed: int = 0, target: Optional[float] = None,
           diagonal: Optional[bool] = None) -> ESResult:
    """CMA-ES из пакета ``cma`` на той же функции потерь и с тем же стартом, что и своя стратегия.

    ``diagonal=True`` — вариант sep-CMA (диагональная ковариация) для больших
    размерностей; по умолчанию включается при числе весов > 300.
    """
    import cma

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    x0 = init_params(d, hidden, rng).flat()
    P = x0.size
    diagonal = P > 300 if diagonal is None else diagonal
    opts = {"maxfevals": max_evals, "seed": seed + 1, "verbose": -9, "CMA_diagonal": diagonal,
            "tolfun": 0, "tolfunhist": 0, "tolx": 0, "tolstagnation": 10**9, "tolflatfitness": 10**9}
    if lam:
        opts["popsize"] = lam
    if target is not None:
        opts["ftarget"] = target
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    best_theta, best = x0.copy(), float(population_loss(x0, X, y, hidden)[0])
    evals = 1
    hist: Dict[str, list] = {"evaluations": [], "best_loss": [], "sigma": []}
    _record(hist, evals, best, sigma0)
    while not es.stop():
        sols = es.ask()
        fit = population_loss(np.asarray(sols), X, y, hidden)
        es.tell(sols, fit.tolist())
        evals += len(sols)
        k = int(np.argmin(fit))
        if fit[k] < best:
            best, best_theta = float(fit[k]), np.asarray(sols[k]).copy()
        _record(hist, evals, best, float(es.sigma))
    params = Params.from_flat(best_theta, d, hidden)
    return ESResult(params, best, evals, es.countiter, time.perf_counter() - t0,
                    {k: np.asarray(v, dtype=float) for k, v in hist.items()})
