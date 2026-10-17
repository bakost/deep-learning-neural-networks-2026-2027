r"""Запуски обучения для задач 2–5 (векторизованно по зёрнам, параллельно по сериям).

Серия :class:`Spec` — одна настройка обучения (модель, метод, скорость,
размер мини-пакета, число эпох, закон скорости, штраф), прогнанная для
набора зёрен. Зерно ``s`` задаёт и начальные веса (:func:`init_params`), и
порядок мини-пакетов. Серия выполняется функцией
:func:`~sgd_lab.training.train_batch` (все зёрна одним массивом), серии —
в нескольких процессах. Результат от числа процессов не зависит.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import MODELS, X, Y, init_params
from .training import Objective, TrainConfig, train_batch

__all__ = ["BATCHES", "Series", "Spec", "default_workers", "excess", "first_epoch_below", "run_specs"]

BATCHES = (1, 2, 3, 4, 6, 12)


@dataclass(frozen=True)
class Spec:
    model: str
    method: str = "gd"
    lr: float = 0.01
    batch: int = 12
    epochs: int = 1000
    seeds: Tuple[int, ...] = (0,)
    tau: Optional[float] = None
    l2: float = 0.0
    delta: Optional[float] = None
    record_every: int = 1
    beta: float = 0.9
    exclude: Optional[int] = None      # номер наблюдения, исключённого из обучения (скользящий контроль)

    def config(self) -> TrainConfig:
        return TrainConfig(method=self.method, lr=self.lr, batch_size=self.batch, epochs=self.epochs,
                           beta=self.beta, tau=self.tau, delta=self.delta, record_every=self.record_every)


@dataclass
class Series:
    """Результаты серии; массивы — по зёрнам (история — эпохи × зёрна)."""

    spec: Spec
    params: np.ndarray
    status: np.ndarray
    epochs: np.ndarray
    updates: np.ndarray
    loss: np.ndarray
    sse: np.ndarray
    grad_norm: np.ndarray
    history: Dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.spec.__dict__.items()
                     if k != "seeds"},
            "seeds": len(self.spec.seeds),
            "status": {s: int(np.sum(self.status == s)) for s in np.unique(self.status)},
            "loss": self.loss.tolist(),
            "sse": self.sse.tolist(),
            "epochs": self.epochs.tolist(),
        }


def _run(spec: Spec) -> Series:
    model = MODELS[spec.model]
    keep = np.arange(X.size) != spec.exclude if spec.exclude is not None else slice(None)
    obj = Objective(model, X[keep], Y[keep], spec.l2)
    W0 = np.stack([init_params(model, s) for s in spec.seeds])
    r = train_batch(obj, W0, spec.config(), spec.seeds)
    return Series(spec, r.params, r.status, r.epochs, r.updates, r.loss, r.sse, r.grad_norm, r.history)


def _merge(spec: Spec, parts: List[Series]) -> Series:
    if len(parts) == 1:
        return replace(parts[0], spec=spec)
    cat = {k: np.concatenate([getattr(p, k) for p in parts]) for k in
           ("params", "status", "epochs", "updates", "loss", "sse", "grad_norm")}
    hist = {"epoch": parts[0].history["epoch"]}
    # Серии одной настройки останавливаются в разное время — история
    # дополняется последним значением до общей длины.
    length = max(len(p.history["epoch"]) for p in parts)
    for p in parts:
        if len(p.history["epoch"]) == length:
            hist["epoch"] = p.history["epoch"]
    for key in ("loss", "sse", "grad_norm"):
        cols = []
        for p in parts:
            h = p.history[key]
            if h.shape[0] < length:
                h = np.vstack([h, np.repeat(h[-1:], length - h.shape[0], axis=0)])
            cols.append(h)
        hist[key] = np.hstack(cols)
    return Series(spec, history=hist, **cat)


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def run_specs(specs: Sequence[Spec], workers: Optional[int] = None, chunk: int = 25) -> List[Series]:
    """Выполнить серии; длинные наборы зёрен делятся на части по ``chunk``."""
    workers = default_workers() if workers is None else max(1, workers)
    pieces, owners = [], []
    for k, spec in enumerate(specs):
        for start in range(0, len(spec.seeds), chunk):
            pieces.append(replace(spec, seeds=tuple(spec.seeds[start:start + chunk])))
            owners.append(k)
    if workers == 1 or len(pieces) < 2:
        results = [_run(p) for p in pieces]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run, pieces))
    grouped: List[List[Series]] = [[] for _ in specs]
    for k, res in zip(owners, results):
        grouped[k].append(res)
    return [_merge(spec, parts) for spec, parts in zip(specs, grouped)]


def excess(values, best: float):
    """Относительный избыток :math:`(J - J^*)/J^*`."""
    return (np.asarray(values, dtype=float) - best) / best


def first_epoch_below(series: Series, best: float, rel: float, key: str = "loss") -> np.ndarray:
    """По каждому зерну — эпоха, *начиная с которой* :math:`(J - J^*)/J^* \\le` ``rel`` до конца обучения.

    Для шумного СГС «первое касание» порога ничего не говорит: траектория
    может случайно пройти рядом с минимумом и снова уйти. NaN — порог к
    концу обучения не достигнут (или снова превышен).
    """
    ex = excess(series.history[key], best)            # (R, S)
    above = ~(ex <= rel)
    R = above.shape[0]
    # последняя запись выше порога; settled — следующая за ней
    last_above = np.where(above.any(axis=0), R - 1 - np.argmax(above[::-1], axis=0), -1)
    out = np.full(above.shape[1], np.nan)
    ok = last_above < R - 1
    out[ok] = series.history["epoch"][last_above[ok] + 1]
    return out
