r"""sgd_lab — стохастический градиентный спуск и его модификации (лабораторная работа 6).

* задача 1 — минимум функции одной переменной методами золотого сечения и
  Брента (:mod:`sgd_lab.linesearch`, Numerical Recipes, гл. 10);
* задачи 2–4 — обучение трёх моделей (линейной, квадратичной и сети ReLU
  1–2–1) классическим градиентным спуском, стохастическим спуском с
  мини-пакетами, с импульсом Нестерова и методом Adam; производные
  вычисляются численно (:mod:`sgd_lab.numdiff`, NR 5.7);
* задача 5 — то же с квадратичным штрафом на веса.

Быстрый старт
-------------
>>> from sgd_lab import MODELS, X, Y, Objective, TrainConfig, train, init_params
>>> obj = Objective(MODELS["linear"], X, Y)
>>> res = train(obj, init_params(MODELS["linear"], 0), TrainConfig(lr=0.037, epochs=10_000, delta=1e-6))
>>> res.status, round(res.sse, 4)
('converged', 0.1999)
>>> from sgd_lab import bracket_minimum, brent, f_task
>>> round(brent(f_task, bracket_minimum(f_task, 0.0, 1.0)[0]).x, 6)
0.380322
"""

from __future__ import annotations

from .linesearch import (
    GOLDEN_RATIO,
    LineSearchResult,
    bracket_minimum,
    brent,
    df_task,
    exact_minimum,
    f_task,
    golden_section,
)
from .models import MODELS, Model, X, Y, get_model, init_params, relu
from .numdiff import central_gradient, ridders_derivative
from .reference import design_matrix, linear_solution, loo_press, minibatch_lr_bound, multistart_minimum, stability
from .training import METHODS, BatchResult, Objective, TrainConfig, TrainResult, train, train_batch

__version__ = "1.0.0"

__all__ = [
    "GOLDEN_RATIO",
    "METHODS",
    "MODELS",
    "X",
    "Y",
    "BatchResult",
    "LineSearchResult",
    "Model",
    "Objective",
    "TrainConfig",
    "TrainResult",
    "bracket_minimum",
    "brent",
    "central_gradient",
    "design_matrix",
    "df_task",
    "exact_minimum",
    "f_task",
    "get_model",
    "golden_section",
    "init_params",
    "linear_solution",
    "loo_press",
    "minibatch_lr_bound",
    "multistart_minimum",
    "relu",
    "ridders_derivative",
    "stability",
    "train",
    "train_batch",
]
