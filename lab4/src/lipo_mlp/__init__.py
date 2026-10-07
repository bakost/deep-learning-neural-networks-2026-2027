r"""lipo_mlp — зависимость коэффициента липофильности от структуры молекул, двухслойный перцептрон.

Лабораторная работа 4. Модель (2) методички

.. math::

    \log P = b^2_1 + (w^2)^T \mathrm{relu}(w^1 x + b^1)

обучается градиентным спуском с градиентом, выведенным вручную (лекция 4),
для 2, 4, 8, 16 и 32 нейронов скрытого слоя; исследуются сходимость,
уменьшение скорости обучения, обобщающая способность и вклад регрессоров;
результат сравнивается с ``MLPRegressor`` из scikit-learn и с линейной
регрессией из ЛР3.

Быстрый старт
-------------
>>> from lipo_mlp import load_lipo, Standardizer, init_params, gradient_descent
>>> data = load_lipo()
>>> scaler = Standardizer.fit(data.X, data.y)
>>> Z, t = scaler.transform(data.X, data.y)
>>> res = gradient_descent(init_params(24, 8, seed=0), Z, t, lr=0.1, delta=1e-4, max_iter=2000)
>>> res.status, res.iterations
('max_iter', 2000)
>>> res.loss < 0.05
True
"""

from __future__ import annotations

from .data import DESCRIPTORS, LipoData, Standardizer, default_data_path, duplicate_groups, load_lipo
from .experiments import (
    DELTA,
    EPS,
    MAX_ITER,
    SIZES,
    Job,
    RunSummary,
    cv_splits,
    interpolation_floor,
    kink_activity,
    linear_stability_bound,
    ols_cv,
    run_job,
    run_jobs,
)
from .importance import ensemble_predict, ols_standardized, permutation_importance, sensitivity
from .network import (
    Params,
    forward,
    gradients,
    init_params,
    input_gradients,
    loss,
    max_gradient_error,
    n_params,
    numerical_gradients,
    predict,
    relu,
)
from .schedules import Schedule, constant, exponential, inverse_time, power, step
from .training import TrainResult, gradient_descent

__version__ = "1.0.0"

__all__ = [
    "DELTA",
    "DESCRIPTORS",
    "EPS",
    "MAX_ITER",
    "SIZES",
    "Job",
    "LipoData",
    "Params",
    "RunSummary",
    "Schedule",
    "Standardizer",
    "TrainResult",
    "__version__",
    "constant",
    "cv_splits",
    "default_data_path",
    "duplicate_groups",
    "ensemble_predict",
    "exponential",
    "forward",
    "gradient_descent",
    "gradients",
    "init_params",
    "input_gradients",
    "interpolation_floor",
    "inverse_time",
    "kink_activity",
    "linear_stability_bound",
    "load_lipo",
    "loss",
    "max_gradient_error",
    "n_params",
    "numerical_gradients",
    "ols_cv",
    "ols_standardized",
    "permutation_importance",
    "power",
    "predict",
    "relu",
    "run_job",
    "run_jobs",
    "sensitivity",
    "step",
]
