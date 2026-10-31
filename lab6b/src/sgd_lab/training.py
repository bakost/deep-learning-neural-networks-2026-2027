r"""Функция потерь и методы обучения задач 2–5.

Функция потерь задания — сумма квадратов ошибок, в задаче 5 к ней
добавляется квадратичный штраф на веса (без смещений):

.. math::

    \tilde J(w) = \sum_{i=1}^{n} (y_i - \hat y(x_i; w))^2 + \lambda \sum_{k \in \text{веса}} w_k^2 .

По мини-пакету :math:`B` размера :math:`m` вычисляется несмещённая оценка
той же функции — сумма по пакету умножается на :math:`n/m`:
:math:`\tilde J_B(w) = \tfrac{n}{m}\sum_{i \in B}(y_i - \hat y_i)^2 + \lambda R(w)`.
Тогда :math:`\mathbb E\,D_w \tilde J_B = D_w \tilde J`, и одна и та же
скорость обучения :math:`\varepsilon` имеет один смысл при любом размере
пакета (Гудфеллоу и др., разд. 8.1.3 — среднее по пакету оценивает
градиент полной функции).

Мини-пакеты выбираются без повторений, пока не будет обойдена вся выборка
(одна эпоха), затем выборка перемешивается заново (лекция 6, «Минипакеты»).
При :math:`m = n` получается классический (полнопакетный) спуск задачи 2.

Методы — точно по формулам лекции 6 (обозначения: :math:`\varepsilon` —
скорость обучения, :math:`g = D_w\tilde J_B`):

* ``"gd"`` — :math:`w \leftarrow w - \varepsilon g`;
* ``"nesterov"`` — :math:`m \leftarrow \beta m + (1-\beta)\,g(w - \varepsilon\beta m)`,
  :math:`w \leftarrow w - \varepsilon m`;
* ``"adam"`` — :math:`m \leftarrow \beta m + (1-\beta) g`,
  :math:`v \leftarrow \gamma v + (1-\gamma) g^2`,
  :math:`w \leftarrow w - \varepsilon\,\tilde m / (\sqrt{\tilde v} + \epsilon)`, где
  :math:`\tilde m = m/(1-\beta^t)`, :math:`\tilde v = v/(1-\gamma^t)`.

Импульс Нестерова в лекции записан со множителем :math:`(1-\beta)` при
градиенте. У Гудфеллоу (алгоритм 8.3) множителя нет:
:math:`v \leftarrow \alpha v - \varepsilon' g(\theta + \alpha v)`. Записи
эквивалентны при :math:`v = -\varepsilon m`, :math:`\alpha = \beta`,
:math:`\varepsilon' = (1-\beta)\varepsilon`, т. е. скорость обучения
«лекционного» метода в :math:`1/(1-\beta)` раз больше той же по смыслу
величины у Гудфеллоу.

Скорость обучения может уменьшаться по эпохам:
:math:`\varepsilon_e = \varepsilon_0 / (1 + e/\tau)` (лекция 6, «Сходимость
СГС»; условия Роббинса — Монро :math:`\sum\varepsilon = \infty`,
:math:`\sum \varepsilon^2 < \infty` у Гудфеллоу, (8.12)–(8.13), для этого
закона выполняются).

Все градиенты вычисляются численно (:func:`sgd_lab.numdiff.central_gradient`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np

from .models import Model
from .numdiff import central_gradient

__all__ = ["METHODS", "BatchResult", "Objective", "TrainConfig", "TrainResult", "train", "train_batch"]

METHODS = {"gd": "градиентный спуск", "nesterov": "импульс Нестерова", "adam": "Adam"}


class Objective:
    r"""Функция потерь :math:`\tilde J` модели на выборке, векторизованная по параметрам."""

    def __init__(self, model: Model, x: np.ndarray, y: np.ndarray, l2: float = 0.0):
        self.model = model
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.l2 = float(l2)
        self.mask = model.mask
        self.n = self.x.size

    def sse(self, w: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Сумма квадратов ошибок (на пакете ``idx`` — с множителем n/m).

        ``idx`` формы ``(m,)`` — один пакет; формы ``(S, m)`` — свой пакет для
        каждого из ``S`` независимых запусков (тогда ``w`` имеет форму ``(S, K, p)``).
        """
        if idx is None:
            x, y = self.x, self.y
        else:
            idx = np.asarray(idx)
            x, y = self.x[idx], self.y[idx]
            if idx.ndim == 2:
                x, y = x[:, None, :], y[:, None, :]
        r = self.model.predict(np.asarray(w, dtype=float), x) - y
        s = np.sum(r * r, axis=-1)
        return s if idx is None else s * (self.n / idx.shape[-1])

    def penalty(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        return self.l2 * np.sum(self.mask * w * w, axis=-1)

    def __call__(self, w: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        return self.sse(w, idx) + self.penalty(w) if self.l2 else self.sse(w, idx)

    def grad(self, w: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Численный градиент (центральные разности, NR 5.7); ``w`` формы ``(p,)`` или ``(S, p)``."""
        return central_gradient(lambda W: self(W, idx), w)


@dataclass(frozen=True)
class TrainConfig:
    method: str = "gd"
    lr: float = 0.01
    batch_size: Optional[int] = None     # None — вся выборка
    epochs: int = 1000
    beta: float = 0.9                    # импульс (Нестеров, Adam)
    gamma: float = 0.999                 # Adam: усреднение квадратов
    adam_eps: float = 1e-8
    tau: Optional[float] = None          # ε_e = ε₀/(1 + e/τ); None — постоянная
    delta: Optional[float] = None        # остановка по ‖DJ‖ < δ (полный градиент)
    record_every: int = 1                # запись истории раз в столько эпох
    record_path: bool = False            # сохранять веса после каждого обновления
    seed: int = 0                        # перемешивание мини-пакетов

    def lr_at(self, epoch: int) -> float:
        return self.lr if self.tau is None else self.lr / (1.0 + epoch / self.tau)

    def with_(self, **kw) -> "TrainConfig":
        return replace(self, **kw)


@dataclass
class TrainResult:
    params: np.ndarray
    status: str              # "converged" | "max_epochs" | "diverged"
    epochs: int              # выполнено эпох
    updates: int             # выполнено обновлений весов
    loss: float              # полная J̃ в конце
    sse: float               # сумма квадратов ошибок в конце
    grad_norm: float
    history: Dict[str, np.ndarray] = field(default_factory=dict)
    path: Optional[np.ndarray] = None

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _check(cfg: TrainConfig, n: int) -> int:
    if cfg.method not in METHODS:
        raise ValueError(f"неизвестный метод {cfg.method!r}; есть: {', '.join(METHODS)}")
    if not cfg.lr > 0:
        raise ValueError("скорость обучения должна быть положительной")
    m = n if cfg.batch_size is None else int(cfg.batch_size)
    if not 1 <= m <= n:
        raise ValueError(f"размер мини-пакета должен быть от 1 до {n}")
    if cfg.epochs < 0:
        raise ValueError("число эпох должно быть неотрицательным")
    return m


def train(obj: Objective, w0: np.ndarray, cfg: TrainConfig) -> TrainResult:
    """Обучить модель методом ``cfg.method`` начиная с ``w0``.

    Одна эпоха — :math:`\\lceil n/m \\rceil` обновлений. После каждой
    ``record_every``-й эпохи записываются полная функция потерь, сумма
    квадратов ошибок, норма полного градиента и текущая скорость обучения.
    Если задано ``delta``, обучение останавливается, как только норма
    полного градиента в конце эпохи меньше ``delta`` (для полнопакетного
    спуска это условие лекции 4 на каждой итерации).
    """
    n = obj.n
    m = _check(cfg, n)
    rng = np.random.default_rng(cfg.seed)
    w = np.array(w0, dtype=float)
    mom = np.zeros_like(w)
    vel = np.zeros_like(w)
    t = 0
    full = m == n
    idx_all = np.arange(n)
    hist = {"epoch": [], "loss": [], "sse": [], "grad_norm": [], "lr": []}
    path = [w.copy()] if cfg.record_path else None
    j0 = float(obj(w))
    limit = 1e6 * max(j0, 1.0)

    def record(epoch: int, g: np.ndarray, j: float) -> None:
        hist["epoch"].append(epoch)
        hist["loss"].append(j)
        hist["sse"].append(float(obj.sse(w)))
        hist["grad_norm"].append(float(np.linalg.norm(g)))
        hist["lr"].append(cfg.lr_at(epoch))

    g_full = obj.grad(w)
    record(0, g_full, j0)
    status, epoch = "max_epochs", 0
    if cfg.delta is not None and np.linalg.norm(g_full) < cfg.delta:
        status = "converged"
    while status == "max_epochs" and epoch < cfg.epochs:
        lr = cfg.lr_at(epoch)
        order = idx_all if full else rng.permutation(n)
        for start in range(0, n, m):
            idx = None if full else order[start:start + m]
            t += 1
            if cfg.method == "gd":
                w -= lr * obj.grad(w, idx)
            elif cfg.method == "nesterov":
                g = obj.grad(w - lr * cfg.beta * mom, idx)
                mom = cfg.beta * mom + (1.0 - cfg.beta) * g
                w -= lr * mom
            else:  # adam
                g = obj.grad(w, idx)
                mom = cfg.beta * mom + (1.0 - cfg.beta) * g
                vel = cfg.gamma * vel + (1.0 - cfg.gamma) * g * g
                m_hat = mom / (1.0 - cfg.beta ** t)
                v_hat = vel / (1.0 - cfg.gamma ** t)
                w -= lr * m_hat / (np.sqrt(v_hat) + cfg.adam_eps)
            if path is not None:
                path.append(w.copy())
        epoch += 1
        need_check = cfg.delta is not None
        if need_check or epoch % cfg.record_every == 0 or epoch == cfg.epochs:
            j = float(obj(w))
            if not np.isfinite(j) or j > limit or not np.all(np.isfinite(w)):
                status = "diverged"
                hist["epoch"].append(epoch)
                for key in ("loss", "sse", "grad_norm"):
                    hist[key].append(np.inf)
                hist["lr"].append(lr)
                break
            g_full = obj.grad(w)
            if cfg.delta is not None and np.linalg.norm(g_full) < cfg.delta:
                status = "converged"
            if epoch % cfg.record_every == 0 or status == "converged" or epoch == cfg.epochs:
                record(epoch, g_full, j)
    history = {k: np.asarray(v, dtype=float) for k, v in hist.items()}
    final_j = float(obj(w)) if status != "diverged" else np.inf
    return TrainResult(
        params=w, status=status, epochs=epoch, updates=t, loss=final_j,
        sse=float(obj.sse(w)) if status != "diverged" else np.inf,
        grad_norm=float(np.linalg.norm(obj.grad(w))) if status != "diverged" else np.inf,
        history=history, path=None if path is None else np.array(path),
    )


@dataclass
class BatchResult:
    """Результаты ``S`` независимых запусков (первая ось — номер запуска)."""

    params: np.ndarray       # (S, p)
    status: np.ndarray       # (S,) строки
    epochs: np.ndarray       # (S,)
    updates: np.ndarray      # (S,)
    loss: np.ndarray         # (S,)
    sse: np.ndarray          # (S,)
    grad_norm: np.ndarray    # (S,)
    history: Dict[str, np.ndarray] = field(default_factory=dict)  # epoch (R,), остальное (R, S)


def train_batch(obj: Objective, W0: np.ndarray, cfg: TrainConfig, seeds) -> BatchResult:
    """То же, что :func:`train`, но для ``S`` запусков сразу (векторизовано по запускам).

    Строка ``W0[s]`` — начальные параметры, ``seeds[s]`` — зерно порядка
    мини-пакетов запуска ``s``. Каждый запуск идёт ровно так же, как
    ``train(obj, W0[s], cfg.with_(seed=seeds[s]))`` (проверяется тестом):
    свой генератор перестановок, своя остановка по δ или расходимости;
    остановившиеся запуски дальше не меняются. Так 100 запусков из разных
    начальных точек стоят почти как один — накладные расходы NumPy на
    малых массивах делятся между ними.
    """
    n = obj.n
    m = _check(cfg, n)
    W = np.array(W0, dtype=float)
    if W.ndim != 2:
        raise ValueError("W0 должен иметь форму (S, p)")
    S = W.shape[0]
    rngs = [np.random.default_rng(int(s)) for s in seeds]
    if len(rngs) != S:
        raise ValueError("число зёрен должно совпадать с числом запусков")
    full = m == n
    mom = np.zeros_like(W)
    vel = np.zeros_like(W)
    active = np.ones(S, dtype=bool)
    status = np.array(["max_epochs"] * S, dtype=object)
    stop_epoch = np.full(S, cfg.epochs)
    stop_update = np.zeros(S, dtype=int)
    hist = {"epoch": [], "loss": [], "sse": [], "grad_norm": []}
    with np.errstate(all="ignore"):
        j0 = obj(W)
        limit = 1e6 * np.maximum(j0, 1.0)
        g_full = obj.grad(W)
        last = {"loss": j0.copy(), "sse": obj.sse(W), "grad_norm": np.linalg.norm(g_full, axis=1)}

        def record(epoch: int) -> None:
            hist["epoch"].append(epoch)
            for key in ("loss", "sse", "grad_norm"):
                hist[key].append(last[key].copy())

        record(0)
        if cfg.delta is not None:
            done = last["grad_norm"] < cfg.delta
            status[done], stop_epoch[done], active[done] = "converged", 0, False
        t = 0
        epoch = 0
        while epoch < cfg.epochs and active.any():
            lr = cfg.lr_at(epoch)
            perms = None if full else np.stack([r.permutation(n) for r in rngs])
            for start in range(0, n, m):
                act = np.nonzero(active)[0]
                idx = None if full else perms[act, start:start + m]
                t += 1
                Wa = W[act]
                if cfg.method == "gd":
                    W[act] = Wa - lr * obj.grad(Wa, idx)
                elif cfg.method == "nesterov":
                    g = obj.grad(Wa - lr * cfg.beta * mom[act], idx)
                    mom[act] = cfg.beta * mom[act] + (1.0 - cfg.beta) * g
                    W[act] = Wa - lr * mom[act]
                else:
                    g = obj.grad(Wa, idx)
                    mom[act] = cfg.beta * mom[act] + (1.0 - cfg.beta) * g
                    vel[act] = cfg.gamma * vel[act] + (1.0 - cfg.gamma) * g * g
                    m_hat = mom[act] / (1.0 - cfg.beta ** t)
                    v_hat = vel[act] / (1.0 - cfg.gamma ** t)
                    W[act] = Wa - lr * m_hat / (np.sqrt(v_hat) + cfg.adam_eps)
            epoch += 1
            stop_update[active] = t
            want = epoch % cfg.record_every == 0 or epoch == cfg.epochs
            if cfg.delta is not None or want:
                act = np.nonzero(active)[0]
                j = obj(W[act])
                bad = ~np.isfinite(j) | (j > limit[act]) | ~np.all(np.isfinite(W[act]), axis=1)
                for key in ("loss", "sse", "grad_norm"):
                    last[key][act[bad]] = np.inf
                status[act[bad]], stop_epoch[act[bad]], active[act[bad]] = "diverged", epoch, False
                ok = act[~bad]
                if ok.size:
                    g = obj.grad(W[ok])
                    last["loss"][ok] = j[~bad]
                    last["sse"][ok] = obj.sse(W[ok])
                    last["grad_norm"][ok] = np.linalg.norm(g, axis=1)
                    if cfg.delta is not None:
                        conv = ok[last["grad_norm"][ok] < cfg.delta]
                        status[conv], stop_epoch[conv], active[conv] = "converged", epoch, False
                if want or not active.any():
                    record(epoch)
        if not hist["epoch"] or hist["epoch"][-1] != epoch:
            record(epoch)
    history = {"epoch": np.asarray(hist["epoch"], dtype=float)}
    for key in ("loss", "sse", "grad_norm"):
        history[key] = np.array(hist[key])
    return BatchResult(W, status.astype(str), stop_epoch, stop_update, last["loss"], last["sse"],
                       last["grad_norm"], history)
