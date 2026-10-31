"""Воспроизводимые численные иллюстрации к трём аналитическим решениям."""

from __future__ import annotations

import numpy as np

from .attention import init_params, permutation_matrix, self_attention
from .counts import parameter_counts
from .softmax import TASK_LOGITS, decimal_reference, softmax, softmax_jacobian


def equivariance_experiment(d: int = 3, n: int = 4, seed: int = 2026, *, scaled: bool = False) -> dict:
    params = init_params(d, seed)
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(size=(d, n))
    # Именно матрица P из задания при N=4 (нулевая индексация).
    order = np.array([1, 3, 0, 2]) if n == 4 else rng.permutation(n)
    p = permutation_matrix(order)
    y, a = self_attention(x, params, scaled=scaled)
    yp, ap = self_attention(x @ p, params, scaled=scaled)
    return {
        "D": d, "N": n, "seed": seed, "scaled": scaled, "order": order.tolist(),
        "P": p.tolist(), "X": x.tolist(), "A": a.tolist(), "A_permuted": ap.tolist(),
        "Sa": y.tolist(), "Sa_permuted": yp.tolist(),
        "output_max_abs_error": float(np.max(np.abs(yp - y @ p))),
        "attention_max_abs_error": float(np.max(np.abs(ap - p.T @ a @ p))),
    }


def softmax_experiment() -> dict:
    z = np.array(TASK_LOGITS, dtype=float)
    y = softmax(z)
    jac = softmax_jacobian(z)
    reference = decimal_reference()
    exact = np.array(reference["jacobian"], dtype=float)
    naive = np.diag(y) - np.outer(y, y)
    scales = [1.0, np.sqrt(5.0), 10.0, 100.0]
    scaling = []
    for scale in scales:
        ys = softmax(z / scale)
        # Производная по исходным z: цепное правило даёт ещё один множитель 1/s.
        js = softmax_jacobian(z / scale) / scale
        scaling.append({
            "scale": float(scale), "max_probability": float(np.max(ys)),
            "max_abs_dy_dz": float(np.max(np.abs(js))),
        })
    return {
        "z": z.tolist(), "probabilities": y.tolist(), "jacobian": jac.tolist(),
        "decimal": reference,
        "max_relative_error_vs_decimal": float(np.max(np.abs((jac - exact) / exact))),
        "naive_J33": float(naive[2, 2]), "stable_J33": float(jac[2, 2]),
        "max_abs_row_sum": float(np.max(np.abs(jac.sum(axis=1)))),
        "scaling": scaling,
    }


def run_experiments() -> dict:
    return {
        "task1": [parameter_counts(64, n) for n in (1, 4, 16, 64, 256)],
        "task2": [equivariance_experiment(d, n, seed, scaled=scaled)
                  for d, n, seed in ((3, 4, 2026), (8, 7, 7), (16, 32, 42))
                  for scaled in (False, True)],
        "task3": softmax_experiment(),
    }
