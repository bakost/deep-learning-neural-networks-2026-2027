"""Функция потерь и методы обучения (задачи 2–5)."""

from __future__ import annotations

import numpy as np
import pytest

from sgd_lab import (
    MODELS,
    Objective,
    TrainConfig,
    X,
    Y,
    init_params,
    linear_solution,
    stability,
    train,
    train_batch,
)


def test_sse_and_penalty(model):
    w = init_params(model, 3)
    obj = Objective(model, X, Y, l2=0.5)
    sse = np.sum((model(w, X) - Y) ** 2)
    assert obj.sse(w) == pytest.approx(sse)
    assert obj(w) == pytest.approx(sse + 0.5 * np.sum(model.mask * w * w))


def test_minibatch_estimate_is_unbiased(model):
    """Среднее оценок n/m·Σ_B по всем пакетам эпохи равно полной функции и полному градиенту."""
    obj = Objective(model, X, Y)
    w = init_params(model, 1)
    for m in (1, 3, 4, 6):
        batches = np.arange(12).reshape(-1, m)
        assert np.mean([obj(w, b) for b in batches]) == pytest.approx(float(obj(w)))
        g = np.mean([obj.grad(w, b) for b in batches], axis=0)
        assert np.allclose(g, obj.grad(w), atol=1e-8)


@pytest.mark.parametrize("key", ["linear", "quadratic"])
def test_gd_converges_to_least_squares(key):
    model = MODELS[key]
    st = stability(model, X)
    res = train(Objective(model, X, Y), init_params(model, 0),
                TrainConfig(lr=st["lr_opt"], epochs=20_000, delta=1e-8))
    assert res.converged
    assert np.allclose(res.params, linear_solution(model, X, Y), atol=1e-8)


@pytest.mark.parametrize("key", ["linear", "quadratic"])
def test_gd_diverges_above_stability_bound(key):
    model = MODELS[key]
    st = stability(model, X)
    obj = Objective(model, X, Y)
    assert train(obj, init_params(model, 0), TrainConfig(lr=1.05 * st["lr_max"], epochs=5000)).status == "diverged"
    assert train(obj, init_params(model, 0), TrainConfig(lr=0.95 * st["lr_max"], epochs=5000)).status != "diverged"


def test_nesterov_and_adam_follow_lecture_formulas():
    model = MODELS["quadratic"]
    obj = Objective(model, X, Y)
    w0 = init_params(model, 2)
    eps, beta, gamma = 0.01, 0.9, 0.999
    # два шага вручную
    w, m = w0.copy(), np.zeros(3)
    for _ in range(2):
        m = beta * m + (1 - beta) * obj.grad(w - eps * beta * m)
        w = w - eps * m
    assert np.allclose(train(obj, w0, TrainConfig("nesterov", eps, epochs=2)).params, w, atol=1e-15)
    w, m, v = w0.copy(), np.zeros(3), np.zeros(3)
    for t in (1, 2):
        g = obj.grad(w)
        m = beta * m + (1 - beta) * g
        v = gamma * v + (1 - gamma) * g * g
        w = w - eps * (m / (1 - beta ** t)) / (np.sqrt(v / (1 - gamma ** t)) + 1e-8)
    assert np.allclose(train(obj, w0, TrainConfig("adam", eps, epochs=2)).params, w, atol=1e-15)


def test_adam_first_step_has_size_lr(model):
    """После коррекции смещения первый шаг Adam — ε·sign(g) по каждой координате."""
    obj = Objective(model, X, Y)
    w0 = init_params(model, 0)
    g = obj.grad(w0)
    w1 = train(obj, w0, TrainConfig("adam", 0.05, epochs=1)).params
    assert np.allclose(w1 - w0, -0.05 * np.sign(g), atol=1e-6)


@pytest.mark.parametrize("method", ["gd", "nesterov", "adam"])
@pytest.mark.parametrize("batch", [1, 5, 12])
def test_train_batch_equals_train(model, method, batch):
    seeds = [0, 1, 2, 7]
    cfg = TrainConfig(method=method, lr=0.005, batch_size=batch, epochs=40, tau=20.0,
                      delta=1e-4 if batch == 12 else None, record_every=3)
    obj = Objective(model, X, Y, l2=0.01)
    W0 = np.stack([init_params(model, s) for s in seeds])
    br = train_batch(obj, W0, cfg, seeds)
    for i, s in enumerate(seeds):
        r = train(obj, W0[i], cfg.with_(seed=s))
        assert np.array_equal(r.params, br.params[i])
        assert (r.status, r.epochs, r.updates) == (br.status[i], br.epochs[i], br.updates[i])


def test_minibatches_cover_epoch_without_repetition():
    seen = []

    class Spy(Objective):
        def grad(self, w, idx=None):
            if idx is not None:
                seen.extend(np.asarray(idx).tolist())
            return super().grad(w, idx)

    obj = Spy(MODELS["linear"], X, Y)
    train(obj, init_params(MODELS["linear"], 0), TrainConfig(lr=0.001, batch_size=5, epochs=3, seed=4))
    epochs = [seen[k * 12:(k + 1) * 12] for k in range(3)]
    assert all(sorted(e) == list(range(12)) for e in epochs)
    assert epochs[0] != epochs[1]


def test_decaying_lr_beats_noise_floor():
    model = MODELS["linear"]
    obj = Objective(model, X, Y)
    best = float(obj(linear_solution(model, X, Y)))
    const, decay = [], []
    for s in range(5):
        cfg = TrainConfig(lr=0.01, batch_size=1, epochs=1000, seed=s, record_every=1000)
        const.append(train(obj, init_params(model, s), cfg).loss - best)
        decay.append(train(obj, init_params(model, s), cfg.with_(tau=50.0)).loss - best)
    assert np.median(decay) < np.median(const) / 10


@pytest.mark.parametrize("method", ["gd", "nesterov", "adam"])
def test_l2_penalty_converges_to_ridge(method):
    model = MODELS["quadratic"]
    obj = Objective(model, X, Y, l2=0.1)
    lr = {"gd": stability(model, X, 0.1)["lr_opt"], "nesterov": 0.05, "adam": 0.05}[method]
    res = train(obj, init_params(model, 0), TrainConfig(method, lr, epochs=6000))
    ridge = linear_solution(model, X, Y, 0.1)
    assert np.allclose(res.params, ridge, atol=1e-4 if method == "adam" else 1e-7)
    # штраф не трогает смещение: ∂R/∂b = 0
    assert Objective(model, X, Y, l2=5.0).penalty(np.array([0.0, 0.0, 3.0])) == 0.0


def test_dead_relu_stops_at_plateau():
    model = MODELS["relu"]
    w = np.array([1.0, 0.1, 1.0, 0.1, -1.0, -1.0, -0.5])   # выход ReLU отрицателен на всей выборке
    res = train(Objective(model, X, Y), w, TrainConfig(lr=0.01, epochs=100, delta=1e-6))
    assert res.converged and res.epochs == 0 and res.sse == pytest.approx(np.sum(Y ** 2))


def test_errors():
    obj = Objective(MODELS["linear"], X, Y)
    w = init_params(MODELS["linear"], 0)
    with pytest.raises(ValueError):
        train(obj, w, TrainConfig(lr=-1))
    with pytest.raises(ValueError):
        train(obj, w, TrainConfig(batch_size=13))
    with pytest.raises(ValueError):
        train(obj, w, TrainConfig(method="sgd"))
