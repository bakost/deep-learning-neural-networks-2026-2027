"""Совпадение с MLPRegressor из scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from lipo_mlp import gradient_descent, init_params
from lipo_mlp.sklearn_compare import SKLEARN_CONFIGS, sklearn_fit, sklearn_trajectory


@pytest.mark.parametrize("hidden", [2, 8])
def test_same_trajectory(std, hidden):
    """Полнопакетный SGD библиотеки с теми же начальными весами повторяет наш спуск до округления."""
    Z, t, _ = std
    p0 = init_params(24, hidden, 7)
    ours = gradient_descent(p0, Z, t, lr=0.1, delta=1e-300, max_iter=500, record_every=10**9).params
    theirs = sklearn_trajectory(p0, Z, t, 0.1, 500)
    assert np.max(np.abs(ours.flat() - theirs.flat())) < 1e-10


def test_sklearn_fit_runs(data):
    info, pred = sklearn_fit(data.X, data.y, 8, SKLEARN_CONFIGS["lbfgs"], X_test=data.X[:5])
    assert info["rmse"] < 0.2 and pred.shape == (5,)
