"""Сценарии экспериментов."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_mlp import Job, constant, cv_splits, interpolation_floor, linear_stability_bound, ols_cv, run_job, run_jobs


def test_linear_stability_bound():
    b = linear_stability_bound()
    assert b["eps_max"] == pytest.approx(0.2408, abs=1e-3)
    assert b["condition"] > 1e4                   # мультиколлинеарность (ЛР3)


def test_cv_splits_partition():
    splits = cv_splits(82, 5, 2, seed=1)
    assert len(splits) == 10
    for rep in range(2):
        vals = np.concatenate([splits[rep * 5 + k][1] for k in range(5)])
        assert sorted(vals.tolist()) == list(range(82))
    for train, val in splits:
        assert not set(train) & set(val) and len(train) + len(val) == 82


def test_ols_cv_matches_direct(data):
    splits = cv_splits(82, 5, 1, seed=2)
    pred = ols_cv(splits)
    X = np.column_stack([np.ones(82), data.X])
    train, val = splits[0]
    coef, *_ = np.linalg.lstsq(X[list(train)], data.y[list(train)], rcond=None)
    assert np.allclose(pred[0, list(val)], X[list(val)] @ coef)


def test_run_job_deterministic_and_parallel():
    jobs = [Job("t", 4, s, constant(0.1), max_iter=300, record_every=100) for s in (0, 1)]
    seq = [run_job(j) for j in jobs]
    par = run_jobs(jobs, workers=2)
    for a, b in zip(seq, par):
        assert np.array_equal(a.params.flat(), b.params.flat()) and a.rmse == b.rmse
    assert seq[0].rmse != seq[1].rmse


def test_run_job_with_validation():
    splits = cv_splits(82, 5, 1)
    tr, va = splits[0]
    r = run_job(Job("cv", 4, 0, max_iter=200, record_every=100, train=tr, val=va, keep_params=False))
    assert r.params is None and r.val_pred.shape == (len(va),)
    assert "val_loss" in r.history and r.rmse_curve().shape == r.history["loss"].shape


def test_floor_value(data):
    assert interpolation_floor(data) == pytest.approx(np.sqrt(1.2555166666666666 / 82))
