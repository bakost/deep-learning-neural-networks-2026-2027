r"""Сравнение с ``sklearn.neural_network.MLPRegressor`` (пункт 5, третий вопрос).

``MLPRegressor(hidden_layer_sizes=(H,), activation="relu")`` — та же модель
(2): скрытый слой ReLU и линейный выход. Его функция потерь
:math:`\frac{1}{2n}\sum (y - \hat y)^2 + \frac{\alpha}{2n}\lVert W\rVert^2` при
:math:`\alpha = 0` совпадает с нашей. С ``solver="sgd"``,
``batch_size = n`` (полный пакет), ``momentum=0`` и постоянной скоростью
обучения библиотека выполняет ровно тот же градиентный спуск. Поэтому при
одинаковых начальных весах траектории должны совпасть до ошибок округления
(:func:`sklearn_trajectory`). Это независимая проверка ручных формул
градиента.

Отличия, которые проявляются в практических настройках (:func:`sklearn_fit`):

* начальные веса — равномерное распределение Глоро, а не Хе;
* критерий остановки — не норма градиента, а отсутствие уменьшения потерь
  больше чем на ``tol`` в течение ``n_iter_no_change`` эпох подряд;
* по умолчанию ``solver="adam"`` (адаптивные шаги по каждому весу) и
  L2-регуляризация :math:`\alpha = 10^{-4}`; ``solver="lbfgs"`` — квазиньютоновский
  метод, который для малых выборок обычно сходится за сотни итераций.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .data import Standardizer
from .network import Params

__all__ = ["SKLEARN_CONFIGS", "params_from_sklearn", "sklearn_fit", "sklearn_trajectory"]

#: Настройки MLPRegressor для сравнения.
SKLEARN_CONFIGS: Dict[str, dict] = {
    "sgd (как у нас, остановка sklearn)": {"solver": "sgd", "learning_rate_init": 0.1, "momentum": 0.0,
                                           "nesterovs_momentum": False, "alpha": 0.0, "max_iter": 300_000},
    "adam (по умолчанию)": {"solver": "adam", "max_iter": 20_000},
    "lbfgs": {"solver": "lbfgs", "max_iter": 20_000},
}


def _regressor(hidden: int, n: int, **kwargs):
    from sklearn.neural_network import MLPRegressor

    base = {"hidden_layer_sizes": (hidden,), "activation": "relu", "batch_size": n, "shuffle": False,
            "random_state": 0}
    base.update(kwargs)
    return MLPRegressor(**base)


def params_from_sklearn(model) -> Params:
    """Веса обученного ``MLPRegressor`` в обозначениях лекции."""
    W1, W2 = model.coefs_
    b1, b2 = model.intercepts_
    return Params(W1.T.copy(), b1.copy(), W2[:, 0].copy(), float(b2[0]))


def sklearn_trajectory(params0: Params, X: np.ndarray, y: np.ndarray, lr: float, n_iter: int) -> Params:
    """``n_iter`` итераций полнопакетного SGD библиотеки, начиная с весов ``params0``.

    Начальные веса подставляются через ``warm_start``: первый вызов ``fit``
    с одной итерацией только создаёт внутренние массивы, затем они
    заменяются на ``params0``, и второй ``fit`` продолжает с них.
    """
    from sklearn.exceptions import ConvergenceWarning

    n = len(y)
    model = _regressor(params0.hidden, n, solver="sgd", learning_rate="constant", learning_rate_init=lr,
                       momentum=0.0, nesterovs_momentum=False, alpha=0.0, max_iter=1, tol=0.0,
                       n_iter_no_change=n_iter + 10, early_stopping=False, warm_start=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
        model.coefs_ = [params0.W1.T.copy(), params0.w2[:, None].copy()]
        model.intercepts_ = [params0.b1.copy(), np.array([params0.b2])]
        model.max_iter = n_iter
        model.fit(X, y)
    return params_from_sklearn(model)


def sklearn_fit(X_raw: np.ndarray, y_raw: np.ndarray, hidden: int, config: dict,
                X_test: Optional[np.ndarray] = None, seed: int = 0) -> Tuple[dict, Optional[np.ndarray]]:
    """Обучить ``MLPRegressor`` (стандартизация — как у нас) и вернуть сводку и прогноз для ``X_test``."""
    from sklearn.exceptions import ConvergenceWarning

    scaler = Standardizer.fit(X_raw, y_raw)
    X, y = scaler.transform(X_raw, y_raw)
    model = _regressor(hidden, len(y), random_state=seed, **config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    not_converged = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    resid = y_raw - scaler.inverse_y(model.predict(X))
    info = {"n_iter": int(model.n_iter_), "rmse": float(np.sqrt(np.mean(resid**2))),
            "converged": not not_converged}
    pred = None if X_test is None else scaler.inverse_y(model.predict(scaler.transform_x(X_test)))
    return info, pred


def sklearn_cv(X_raw: np.ndarray, y_raw: np.ndarray, hidden: int, config: dict,
               splits: Sequence[Tuple[Sequence[int], Sequence[int]]], folds_per_repeat: int) -> np.ndarray:
    """Прогнозы перекрёстной проверки ``(repeats, n)`` для одной настройки MLPRegressor."""
    n = len(y_raw)
    reps = len(splits) // folds_per_repeat
    pred = np.empty((reps, n))
    for s, (train, val) in enumerate(splits):
        train, val = list(train), list(val)
        _, p = sklearn_fit(X_raw[train], y_raw[train], hidden, config, X_raw[val], seed=s)
        pred[s // folds_per_repeat, val] = p
    return pred
